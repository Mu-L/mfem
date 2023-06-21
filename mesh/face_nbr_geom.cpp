// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "face_nbr_geom.hpp"
#include "../general/forall.hpp"

namespace mfem
{

FaceNeighborGeometricFactors::FaceNeighborGeometricFactors(
   const GeometricFactors &geom_) : geom(geom_)
{
#ifdef MFEM_USE_MPI
   if (const ParMesh *par_mesh = dynamic_cast<const ParMesh*>(geom.mesh))
   {
      const int flags = geom.computed_factors;
      const int nq = geom.IntRule->Size();
      const int dim = par_mesh->Dimension();
      const int sdim = par_mesh->SpaceDimension();

      num_neighbor_elems = par_mesh->GetNFaceNeighborElements();

      const_cast<ParMesh*>(par_mesh)->ExchangeFaceNbrData();

      if (flags & GeometricFactors::COORDINATES)
      {
         ExchangeFaceNbrData(geom.X, X, nq*sdim);
      }
      if (flags & GeometricFactors::JACOBIANS)
      {
         ExchangeFaceNbrData(geom.J, J, nq*dim*sdim);
      }
      if (flags & GeometricFactors::DETERMINANTS)
      {
         ExchangeFaceNbrData(geom.detJ, detJ, nq);
      }
   }
#endif
}

void FaceNeighborGeometricFactors::ExchangeFaceNbrData(const Vector &x_local,
                                                       Vector &x_shared, const int ndof_per_el)
{
#ifdef MFEM_USE_MPI
   const ParMesh *mesh = static_cast<const ParMesh*>(geom.mesh);

   const int n_face_nbr = mesh->GetNFaceNeighbors();
   if (n_face_nbr == 0) { return; }

   const int ne_send = mesh->send_face_nbr_elements.Size_of_connections();

   x_shared.SetSize(ndof_per_el * num_neighbor_elems);
   send_offsets.SetSize(n_face_nbr + 1);
   send_data.SetSize(ndof_per_el * ne_send);
   auto d_send_data = Reshape(send_data.Write(), ndof_per_el, ne_send);
   const auto d_x_local = Reshape(x_local.Read(), ndof_per_el, mesh->GetNE());
   int idx = 0;
   Array<int> row;
   for (int i = 0; i < n_face_nbr; ++i)
   {
      send_offsets[i] = ndof_per_el*idx;
      mesh->send_face_nbr_elements.GetRow(i, row);
      for (const int el : row)
      {
         for (int j = 0; j < ndof_per_el; ++j)
         {
            d_send_data(j, idx) = d_x_local(j, el);
         }
         ++idx;
      }
   }
   send_offsets[n_face_nbr] = ndof_per_el*idx;

   recv_offsets.SetSize(n_face_nbr + 1);
   for (int i = 0; i < n_face_nbr + 1; ++i)
   {
      recv_offsets[i] = ndof_per_el * mesh->face_nbr_elements_offset[i];
   }

   MPI_Comm comm = mesh->GetComm();
   std::vector<MPI_Request> send_reqs(n_face_nbr);
   std::vector<MPI_Request> recv_reqs(n_face_nbr);
   std::vector<MPI_Status> statuses(n_face_nbr);

   bool mpi_gpu_aware = Device::GetGPUAwareMPI();
   const auto send_data_ptr = mpi_gpu_aware ? send_data.Read() :
                              send_data.HostRead();
   auto x_shared_ptr = mpi_gpu_aware ? x_shared.Write() : x_shared.HostWrite();

   for (int i = 0; i < n_face_nbr; ++i)
   {
      const int nbr_rank = mesh->GetFaceNbrRank(i);
      const int tag = 0;

      MPI_Isend(send_data_ptr + send_offsets[i],
                send_offsets[i+1] - send_offsets[i],
                MPI_DOUBLE, nbr_rank, tag, comm, &send_reqs[i]);

      MPI_Irecv(x_shared_ptr + recv_offsets[i],
                recv_offsets[i+1] - recv_offsets[i],
                MPI_DOUBLE, nbr_rank, tag, comm, &recv_reqs[i]);
   }

   MPI_Waitall(n_face_nbr, send_reqs.data(), statuses.data());
   MPI_Waitall(n_face_nbr, recv_reqs.data(), statuses.data());
#endif
}

} // namespace mfem

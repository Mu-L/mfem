// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "../../general/forall.hpp"
#include "../bilininteg.hpp"
#include "../gridfunc.hpp"
#include "../qfunction.hpp"
#include "../ceed/integrators/mass/mass.hpp"
#include "bilininteg_mass_kernels.hpp"

using namespace std;

namespace mfem
{

// PA Mass Integrator

MassIntegrator::Kernels MassIntegrator::kernels;
MassIntegrator::Kernels::Kernels()
{
   // 2D
   MassIntegrator::AddSpecialization<2,2,2>();
   MassIntegrator::AddSpecialization<2,3,3>();
   MassIntegrator::AddSpecialization<2,4,4>();
   MassIntegrator::AddSpecialization<2,5,5>();
   MassIntegrator::AddSpecialization<2,6,6>();
   MassIntegrator::AddSpecialization<2,7,7>();
   MassIntegrator::AddSpecialization<2,8,8>();
   MassIntegrator::AddSpecialization<2,9,9>();
   // 3D
   MassIntegrator::AddSpecialization<3,2,2>();
   MassIntegrator::AddSpecialization<3,2,3>();
   MassIntegrator::AddSpecialization<3,3,4>();
   MassIntegrator::AddSpecialization<3,4,5>();
   MassIntegrator::AddSpecialization<3,4,6>();
   MassIntegrator::AddSpecialization<3,5,6>();
   MassIntegrator::AddSpecialization<3,5,8>();
   MassIntegrator::AddSpecialization<3,6,7>();
   MassIntegrator::AddSpecialization<3,7,8>();
   MassIntegrator::AddSpecialization<3,8,9>();
}

void MassIntegrator::AssemblePA(const FiniteElementSpace &fes)
{
   const MemoryType mt = (pa_mt == MemoryType::DEFAULT) ?
                         Device::GetDeviceMemoryType() : pa_mt;

   // Assuming the same element type
   fespace = &fes;
   Mesh *mesh = fes.GetMesh();
   if (mesh->GetNE() == 0) { return; }
   const FiniteElement &el = *fes.GetFE(0);
   ElementTransformation *T0 = mesh->GetElementTransformation(0);
   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, el, *T0);
   if (DeviceCanUseCeed())
   {
      delete ceedOp;
      const bool mixed = mesh->GetNumGeometries(mesh->Dimension()) > 1 ||
                         fes.IsVariableOrder();
      if (mixed)
      {
         ceedOp = new ceed::MixedPAMassIntegrator(*this, fes, Q);
      }
      else
      {
         ceedOp = new ceed::PAMassIntegrator(fes, *ir, Q);
      }
      return;
   }
   int map_type = el.GetMapType();
   dim = mesh->Dimension();
   ne = fes.GetMesh()->GetNE();
   nq = ir->GetNPoints();
   geom = mesh->GetGeometricFactors(*ir, GeometricFactors::DETERMINANTS, mt);
   maps = &el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   dofs1D = maps->ndof;
   quad1D = maps->nqpt;
   pa_data.SetSize(ne*nq, mt);

   QuadratureSpace qs(*mesh, *ir);
   CoefficientVector coeff(Q, qs, CoefficientStorage::COMPRESSED);

   if (dim==1) { MFEM_ABORT("Not supported yet... stay tuned!"); }
   if (dim==2)
   {
      const int NE = ne;
      const int Q1D = quad1D;
      const bool const_c = coeff.Size() == 1;
      const bool by_val = map_type == FiniteElement::VALUE;
      const auto W = Reshape(ir->GetWeights().Read(), Q1D,Q1D);
      const auto J = Reshape(geom->detJ.Read(), Q1D,Q1D,NE);
      const auto C = const_c ? Reshape(coeff.Read(), 1,1,1) :
                     Reshape(coeff.Read(), Q1D,Q1D,NE);
      auto v = Reshape(pa_data.Write(), Q1D,Q1D, NE);
      mfem::forall_2D(NE,Q1D,Q1D, [=] MFEM_HOST_DEVICE (int e)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               const double detJ = J(qx,qy,e);
               const double coeff = const_c ? C(0,0,0) : C(qx,qy,e);
               v(qx,qy,e) =  W(qx,qy) * coeff * (by_val ? detJ : 1.0/detJ);
            }
         }
      });
   }
   if (dim==3)
   {
      const int NE = ne;
      const int Q1D = quad1D;
      const bool const_c = coeff.Size() == 1;
      const bool by_val = map_type == FiniteElement::VALUE;
      const auto W = Reshape(ir->GetWeights().Read(), Q1D,Q1D,Q1D);
      const auto J = Reshape(geom->detJ.Read(), Q1D,Q1D,Q1D,NE);
      const auto C = const_c ? Reshape(coeff.Read(), 1,1,1,1) :
                     Reshape(coeff.Read(), Q1D,Q1D,Q1D,NE);
      auto v = Reshape(pa_data.Write(), Q1D,Q1D,Q1D,NE);
      mfem::forall_3D(NE, Q1D, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(qz,z,Q1D)
               {
                  const double detJ = J(qx,qy,qz,e);
                  const double coeff = const_c ? C(0,0,0,0) : C(qx,qy,qz,e);
                  v(qx,qy,qz,e) = W(qx,qy,qz) * coeff * (by_val ? detJ : 1.0/detJ);
               }
            }
         }
      });
   }
}

void MassIntegrator::AssemblePABoundary(const FiniteElementSpace &fes)
{
   const MemoryType mt = (pa_mt == MemoryType::DEFAULT) ?
                         Device::GetDeviceMemoryType() : pa_mt;

   // Assuming the same element type
   fespace = &fes;
   Mesh *mesh = fes.GetMesh();
   if (mesh->GetNBE() == 0) { return; }
   const FiniteElement &el = *fes.GetBE(0);
   ElementTransformation *T0 = mesh->GetBdrElementTransformation(0);
   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, el, *T0);

   int map_type = el.GetMapType();
   dim = el.GetDim(); // Dimension of the boundary element, *not* the mesh
   ne = fes.GetMesh()->GetNFbyType(FaceType::Boundary);
   nq = ir->GetNPoints();
   face_geom = mesh->GetFaceGeometricFactors(*ir, GeometricFactors::DETERMINANTS,
                                             FaceType::Boundary, mt);
   maps = &el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   dofs1D = maps->ndof;
   quad1D = maps->nqpt;
   pa_data.SetSize(ne*nq, mt);

   FaceQuadratureSpace qs(*mesh, *ir, FaceType::Boundary);
   CoefficientVector coeff(Q, qs, CoefficientStorage::COMPRESSED);

   const int NE = ne;
   const int Q1D = quad1D;
   const bool const_c = coeff.Size() == 1;
   const bool by_val = map_type == FiniteElement::VALUE;
   if (dim==1)
   {
      const auto W = Reshape(ir->GetWeights().Read(), Q1D);
      const auto J = Reshape(face_geom->detJ.Read(), Q1D, NE);
      const auto C = const_c ? Reshape(coeff.Read(), 1, 1) :
                     Reshape(coeff.Read(), Q1D, NE);
      auto v = Reshape(pa_data.Write(), Q1D, NE);
      mfem::forall_2D(NE, Q1D, 1, [=] MFEM_HOST_DEVICE (int e)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            const double detJ = J(qx,e);
            const double coeff = const_c ? C(0,0) : C(qx,e);
            v(qx,e) =  W(qx) * coeff * (by_val ? detJ : 1.0/detJ);
         }
      });
   }
   else if (dim==2)
   {
      const auto W = Reshape(ir->GetWeights().Read(), Q1D,Q1D);
      const auto J = Reshape(face_geom->detJ.Read(), Q1D,Q1D,NE);
      const auto C = const_c ? Reshape(coeff.Read(), 1,1,1) :
                     Reshape(coeff.Read(), Q1D,Q1D,NE);
      auto v = Reshape(pa_data.Write(), Q1D,Q1D, NE);
      mfem::forall_2D(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               const double detJ = J(qx,qy,e);
               const double coeff = const_c ? C(0,0,0) : C(qx,qy,e);
               v(qx,qy,e) =  W(qx,qy) * coeff * (by_val ? detJ : 1.0/detJ);
            }
         }
      });
   }
   else
   {
      MFEM_ABORT("Not supported.");
   }
}

void MassIntegrator::AssembleDiagonalPA(Vector &diag)
{
   if (DeviceCanUseCeed())
   {
      ceedOp->GetDiagonal(diag);
   }
   else
   {
      const int D1D = dofs1D, Q1D = quad1D;
      kernels.diag.Run(dim, D1D, Q1D, ne, maps->B, pa_data, diag);
   }
}


#ifdef MFEM_USE_OCCA
// OCCA PA Mass Apply 2D kernel
static void OccaPAMassApply2D(const int D1D,
                              const int Q1D,
                              const int NE,
                              const Array<double> &B,
                              const Array<double> &Bt,
                              const Vector &D,
                              const Vector &X,
                              Vector &Y)
{
   occa::properties props;
   props["defines/D1D"] = D1D;
   props["defines/Q1D"] = Q1D;
   const occa::memory o_B = OccaMemoryRead(B.GetMemory(), B.Size());
   const occa::memory o_Bt = OccaMemoryRead(Bt.GetMemory(), Bt.Size());
   const occa::memory o_D = OccaMemoryRead(D.GetMemory(), D.Size());
   const occa::memory o_X = OccaMemoryRead(X.GetMemory(), X.Size());
   occa::memory o_Y = OccaMemoryReadWrite(Y.GetMemory(), Y.Size());
   const occa_id_t id = std::make_pair(D1D,Q1D);
   if (!Device::Allows(Backend::OCCA_CUDA))
   {
      static occa_kernel_t OccaMassApply2D_cpu;
      if (OccaMassApply2D_cpu.find(id) == OccaMassApply2D_cpu.end())
      {
         const occa::kernel MassApply2D_CPU =
            mfem::OccaDev().buildKernel("occa://mfem/fem/occa.okl",
                                        "MassApply2D_CPU", props);
         OccaMassApply2D_cpu.emplace(id, MassApply2D_CPU);
      }
      OccaMassApply2D_cpu.at(id)(NE, o_B, o_Bt, o_D, o_X, o_Y);
   }
   else
   {
      static occa_kernel_t OccaMassApply2D_gpu;
      if (OccaMassApply2D_gpu.find(id) == OccaMassApply2D_gpu.end())
      {
         const occa::kernel MassApply2D_GPU =
            mfem::OccaDev().buildKernel("occa://mfem/fem/occa.okl",
                                        "MassApply2D_GPU", props);
         OccaMassApply2D_gpu.emplace(id, MassApply2D_GPU);
      }
      OccaMassApply2D_gpu.at(id)(NE, o_B, o_Bt, o_D, o_X, o_Y);
   }
}

// OCCA PA Mass Apply 3D kernel
static void OccaPAMassApply3D(const int D1D,
                              const int Q1D,
                              const int NE,
                              const Array<double> &B,
                              const Array<double> &Bt,
                              const Vector &D,
                              const Vector &X,
                              Vector &Y)
{
   occa::properties props;
   props["defines/D1D"] = D1D;
   props["defines/Q1D"] = Q1D;
   const occa::memory o_B = OccaMemoryRead(B.GetMemory(), B.Size());
   const occa::memory o_Bt = OccaMemoryRead(Bt.GetMemory(), Bt.Size());
   const occa::memory o_D = OccaMemoryRead(D.GetMemory(), D.Size());
   const occa::memory o_X = OccaMemoryRead(X.GetMemory(), X.Size());
   occa::memory o_Y = OccaMemoryReadWrite(Y.GetMemory(), Y.Size());
   const occa_id_t id = std::make_pair(D1D,Q1D);
   if (!Device::Allows(Backend::OCCA_CUDA))
   {
      static occa_kernel_t OccaMassApply3D_cpu;
      if (OccaMassApply3D_cpu.find(id) == OccaMassApply3D_cpu.end())
      {
         const occa::kernel MassApply3D_CPU =
            mfem::OccaDev().buildKernel("occa://mfem/fem/occa.okl",
                                        "MassApply3D_CPU", props);
         OccaMassApply3D_cpu.emplace(id, MassApply3D_CPU);
      }
      OccaMassApply3D_cpu.at(id)(NE, o_B, o_Bt, o_D, o_X, o_Y);
   }
   else
   {
      static occa_kernel_t OccaMassApply3D_gpu;
      if (OccaMassApply3D_gpu.find(id) == OccaMassApply3D_gpu.end())
      {
         const occa::kernel MassApply3D_GPU =
            mfem::OccaDev().buildKernel("occa://mfem/fem/occa.okl",
                                        "MassApply3D_GPU", props);
         OccaMassApply3D_gpu.emplace(id, MassApply3D_GPU);
      }
      OccaMassApply3D_gpu.at(id)(NE, o_B, o_Bt, o_D, o_X, o_Y);
   }
}
#endif // MFEM_USE_OCCA

void MassIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   if (DeviceCanUseCeed())
   {
      ceedOp->AddMult(x, y);
   }
   else
   {
      const int D1D = dofs1D;
      const int Q1D = quad1D;
      const Array<double> &B = maps->B;
      const Array<double> &Bt = maps->Bt;
      const Vector &D = pa_data;
#ifdef MFEM_USE_OCCA
      if (DeviceCanUseOcca())
      {
         if (dim == 2)
         {
            return OccaPAMassApply2D(D1D,Q1D,ne,B,Bt,D,x,y);
         }
         if (dim == 3)
         {
            return OccaPAMassApply3D(D1D,Q1D,ne,B,Bt,D,x,y);
         }
         MFEM_ABORT("OCCA PA Mass Apply unknown kernel!");
      }
#endif // MFEM_USE_OCCA
      kernels.apply.Run(dim, D1D, Q1D, ne, B, Bt, D, x, y);
   }
}

void MassIntegrator::AddMultTransposePA(const Vector &x, Vector &y) const
{
   // Mass integrator is symmetric
   AddMultPA(x, y);
}

} // namespace mfem

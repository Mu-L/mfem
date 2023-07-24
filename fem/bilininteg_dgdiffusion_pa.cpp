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

#include "../general/forall.hpp"
#include "../mesh/face_nbr_geom.hpp"
#include "gridfunc.hpp"
#include "qfunction.hpp"
#include "restriction.hpp"
#include "pfespace.hpp"
#include "fe/face_map_utils.hpp"

#include "../general/communication.hpp"

using namespace std;

namespace mfem
{

static void PADGDiffusionsetup2D(const int Q1D,
                                 const int NE,
                                 const int NF,
                                 const Array<double> &w,
                                 const GeometricFactors &el_geom,
                                 const FaceGeometricFactors &face_geom,
                                 const FaceNeighborGeometricFactors *nbr_geom,
                                 const Vector &q,
                                 const double sigma,
                                 const double kappa,
                                 Vector &pa_data,
                                 const Array<int> &iwork_)
{
   const auto J = Reshape(el_geom.J.Read(), Q1D, Q1D, 2, 2, NE);
   const auto detJe = Reshape(el_geom.detJ.Read(), Q1D, Q1D, NE);

   const int n_nbr = nbr_geom ? nbr_geom->num_neighbor_elems : 0;
   const auto J_shared = Reshape(nbr_geom ? nbr_geom->J.Read() : nullptr, Q1D, Q1D,
                                 2, 2, n_nbr);
   const auto detJ_shared = Reshape(nbr_geom ? nbr_geom->detJ.Read() : nullptr,
                                    Q1D, Q1D, n_nbr);

   const auto detJf = Reshape(face_geom.detJ.Read(), Q1D, NF);
   const auto n = Reshape(face_geom.normal.Read(), Q1D, 2, NF);

   const bool const_q = (q.Size() == 1);
   const auto Q = const_q ? Reshape(q.Read(), 1,1) : Reshape(q.Read(), Q1D,NF);

   const auto W = w.Read();

   // (normal0, normal1, e0, e1, fid0, fid1)
   const auto iwork = Reshape(iwork_.Read(), 6, NF);

   // (q, 1/h, J0_0, J0_1, J1_0, J1_1)
   auto pa = Reshape(pa_data.Write(), 6, Q1D, NF);

   mfem::forall(NF, [=] MFEM_HOST_DEVICE (int f) -> void
   {
      const int normal_dir[] = {iwork(0, f), iwork(1, f)};
      const int el[] = {iwork(2, f), iwork(3, f)};
      const int fid[] = {iwork(4, f), iwork(5, f)};

      const bool interior = el[1] >= 0;
      const int nsides = (interior) ? 2 : 1;
      const double factor = interior ? 0.5 : 1.0;

      const bool shared = el[1] >= NE;
      const int el_1 = shared ? el[1] - NE : el[1];

      const int sgn0 = (fid[0] == 0 || fid[0] == 1) ? 1 : -1;
      const int sgn1 = (fid[1] == 0 || fid[1] == 1) ? 1 : -1;

      for (int p = 0; p < Q1D; ++p)
      {
         const double Qp = const_q ? Q(0,0) : Q(p, f);
         pa(0, p, f) = kappa * Qp * W[p] * detJf(p, f);

         double hi = 0.0;
         for (int side = 0; side < nsides; ++side)
         {
            int i, j;
            internal::FaceQuad2Lex2D(p, Q1D, fid[0], fid[1], side, i, j);

            // Always opposite direction in "native" ordering
            // Need to multiply the native=>lex0 with native=>lex1 and negate
            const int sgn = (side == 1) ? -1*sgn0*sgn1 : 1;

            const int el_idx = (side == 0) ? el[0] : el_1;
            auto J_el = (side == 1 && shared) ? J_shared : J;
            auto detJ_el = (side == 1 && shared) ? detJ_shared : detJe;

            double nJi[2];
            nJi[0] = n(p,0,f)*J_el(i,j, 1,1, el_idx) - n(p,1,f)*J_el(i,j,0,1,el_idx);
            nJi[1] = -n(p,0,f)*J_el(i,j,1,0, el_idx) + n(p,1,f)*J_el(i,j,0,0,el_idx);

            const double dJe = detJ_el(i,j,el_idx);
            const double dJf = detJf(p, f);

            const double w = factor * Qp * W[p] * dJf / dJe;

            const int ni = normal_dir[side];
            const int ti = 1 - ni;

            // Normal
            pa(2 + 2*side + 0, p, f) = w * nJi[ni];
            // Tangential
            pa(2 + 2*side + 1, p, f) = sgn * w * nJi[ti];

            hi += factor * dJf / dJe;
         }

         if (nsides == 1)
         {
            pa(4, p, f) = 0.0;
            pa(5, p, f) = 0.0;
         }

         pa(1, p, f) = hi;
      }
   });
}

static void PADGDiffusionSetup3D(const int Q1D,
                                 const int NE,
                                 const int NF,
                                 const Array<double>& w,
                                 const GeometricFactors& el_geom,
                                 const FaceGeometricFactors& face_geom,
                                 const FaceNeighborGeometricFactors *nbr_geom,
                                 const Vector &q,
                                 const double sigma,
                                 const double kappa,
                                 Vector &pa_data,
                                 const Array<int>& iwork_)
{
   const auto J = Reshape(el_geom.J.Read(), Q1D, Q1D, Q1D, 3, 3, NE);
   const auto detJe = Reshape(el_geom.detJ.Read(), Q1D, Q1D, Q1D, NE);

   const auto detJf = Reshape(face_geom.detJ.Read(), Q1D, Q1D, NF);
   const auto n = Reshape(face_geom.normal.Read(), Q1D, Q1D, 3, NF);

   const bool const_q = (q.Size() == 1);
   const auto Q = const_q ? Reshape(q.Read(), 1, 1, 1) : Reshape(q.Read(), Q1D, Q1D, NF);

   const auto W = Reshape(w.Read(), Q1D, Q1D);

   const auto iwork = Reshape(iwork_.Read(), 6, 2, NF); // (perm[0], perm[1], perm[2], element_index, local_face_id, orientation)
   constexpr int _el_ = 3; // offset in iwork for element index
   constexpr int _fid_ = 4; // offset in iwork for local face id
   constexpr int _or_ = 5; // offset in iwork for orientation

   const auto pa = Reshape(pa_data.Write(), 8, Q1D, Q1D, NF); // (q, 1/h, J00, J01, J02, J10, J11, J12)

   for (int f = 0; f < NF; ++f)
   {
      const int perm[2][3] = {{iwork(0, 0, f), iwork(1, 0, f), iwork(2, 0, f)},
                              {iwork(0, 1, f), iwork(1, 1, f), iwork(2, 1, f)}};
      const int el[] = {iwork(_el_, 0, f), iwork(_el_, 1, f)};
      const int fid[] = {iwork(_fid_, 0, f), iwork(_fid_, 1, f)};
      const int ortn[] = {iwork(_or_, 0, f), iwork(_or_, 1, f)};

      const bool interior = el[1] >= 0;
      const int nsides = interior ? 2 : 1;
      const double factor = interior ? 0.5 : 1.0;

      for (int p1 = 0; p1 < Q1D; ++p1)
      {
         for (int p2 = 0; p2 < Q1D; ++p2)
         {
            const double Qp = const_q ? Q(0,0,0) : Q(p1, p2, f);
            pa(0, p1, p2, f) = kappa * Qp * W(p1, p2) * detJf(p1, p2, f);

            double hi = 0.0;
            const double dJf = detJf(p1,p2,f);

            for (int side = 0; side < nsides; ++side)
            {
               int i, j, k;
               internal::FaceQuad2Lex3D(p1 + Q1D*p2, Q1D, fid[0], fid[1], side, ortn[side], i, j, k);

               const int e = el[side];

               double nJi[3];

               nJi[0] = (-J(i,j,k, 1,2, e)*J(i,j,k, 2,1, e) + J(i,j,k, 1,1, e)*J(i,j,k, 2,2, e)) * n(p1, p2, 0, f)
                      + ( J(i,j,k, 1,2, e)*J(i,j,k, 2,0, e) - J(i,j,k, 1,0, e)*J(i,j,k, 2,2, e)) * n(p1, p2, 1, f)
                      + (-J(i,j,k, 1,1, e)*J(i,j,k, 2,0, e) + J(i,j,k, 1,0, e)*J(i,j,k, 2,1, e)) * n(p1, p2, 2, f);

               nJi[1] = ( J(i,j,k, 0,2, e)*J(i,j,k, 2,1, e) - J(i,j,k, 0,1, e)*J(i,j,k, 2,2, e)) * n(p1, p2, 0, f)
                      + (-J(i,j,k, 0,2, e)*J(i,j,k, 2,0, e) + J(i,j,k, 0,0, e)*J(i,j,k, 2,2, e)) * n(p1, p2, 1, f)
                      + ( J(i,j,k, 0,1, e)*J(i,j,k, 2,0, e) - J(i,j,k, 0,0, e)*J(i,j,k, 2,1, e)) * n(p1, p2, 2, f);

               nJi[2] = (-J(i,j,k, 0,2, e)*J(i,j,k, 1,1, e) + J(i,j,k, 0,1, e)*J(i,j,k, 1,2, e)) * n(p1, p2, 0, f)
                      + ( J(i,j,k, 0,2, e)*J(i,j,k, 1,0, e) - J(i,j,k, 0,0, e)*J(i,j,k, 1,2, e)) * n(p1, p2, 1, f)
                      + (-J(i,j,k, 0,1, e)*J(i,j,k, 1,0, e) + J(i,j,k, 0,0, e)*J(i,j,k, 1,1, e)) * n(p1, p2, 2, f);

               const double dJe = detJe(i,j,k,e);
               const double val = factor * Qp * W(p1, p2) * dJf / dJe;

               for (int d = 0; d < 3; ++d)
               {
                  const int idx = std::abs(perm[side][d]) - 1;
                  const int sgn = (perm[side][d] < 0) ? -1 : 1;
                  pa(2+3*side + d, p1, p2, f) = sgn * val * nJi[idx];
               }

               hi += factor * dJf / dJe;
            }

            if (nsides == 1)
            {
               pa(5, p1, p2, f) = 0.0;
               pa(6, p1, p2, f) = 0.0;
               pa(7, p1, p2, f) = 0.0;
            }

            pa(1, p1, p2, f) = hi;
         }
      }
   }
}

static void PADGDiffusionSetupIwork2D(const int nf, const Mesh& mesh,
                                      const FaceType type, Array<int>& iwork_)
{
   const int ne = mesh.GetNE();

   int fidx = 0;
   iwork_.SetSize(nf * 6);

   // normal0 and normal1 are the indices of the face normal direction relative
   // to the element in reference coordinates, i.e. if the face is normal to the
   // x-vector (left or right face), then it will be 0, otherwise 1.

   // 2d: (normal0, normal1, e0, e1, fid0, fid1)
   auto iwork = Reshape(iwork_.HostWrite(), 6, nf);
   for (int f = 0; f < mesh.GetNumFaces(); ++f)
   {
      auto face_info = mesh.GetFaceInformation(f);

      if (face_info.IsOfFaceType(type))
      {
         const int face_id_1 = face_info.element[0].local_face_id;
         iwork(0, fidx) = (face_id_1 == 1 || face_id_1 == 3) ? 0 : 1;
         iwork(2, fidx) = face_info.element[0].index;
         iwork(4, fidx) = face_id_1;

         if (face_info.IsInterior())
         {
            const int face_id_2 = face_info.element[1].local_face_id;
            iwork(1, fidx) = (face_id_2 == 1 || face_id_2 == 3) ? 0 : 1;
            if (face_info.IsShared())
            {
               iwork(3, fidx) = ne + face_info.element[1].index;
            }
            else
            {
               iwork(3, fidx) = face_info.element[1].index;
            }
            iwork(5, fidx) = face_id_2;
         }
         else
         {
            iwork(1, fidx) = -1;
            iwork(3, fidx) = -1;
            iwork(5, fidx) = -1;
         }

         fidx++;
      }
   }
}

// assigns to perm the permuation: perm[0] -> normal component, perm[1] -> first tangential, perm[2] -> second tangential according to the lexocographic ordering.
inline void FaceNormalPermutation(int perm[3], const int face_id)
{
   const bool xy_plane = (face_id == 0 || face_id == 5);
   const bool xz_plane = (face_id == 1 || face_id == 3);
   const bool yz_plane = (face_id == 2 || face_id == 4);

   perm[0] = (xy_plane) ? 3 : (xz_plane) ? 2 : 1;
   perm[1] = (xy_plane || xz_plane) ? 1 : 2;
   perm[2] = (xy_plane) ? 2 : 3;
}

// assigns to perm the permutation as in FaceNormalPermutation for the second element on the face but signed to indicate the sign of the normal derivative.
inline void SignedFaceNormalPermutation(int perm[3], const int face_id1, const int face_id2, const int orientation)
{
   // lex ordering
   FaceNormalPermutation(perm, face_id2);

   // convert from lex ordering to natural
   if (face_id1 == 3 || face_id1 == 4)
   {
      perm[1] *= -1;
   }
   else if (face_id1 == 0)
   {
      perm[2] *= -1;
   }

   // permute based on face orientation
   switch (orientation)
   {
   case 1:
      std::swap(perm[1], perm[2]);
      break;
   case 2:
      std::swap(perm[1], perm[2]);
      perm[2] *= -1;
      break;
   case 3:
      perm[1] *= -1;
      break;
   case 4:
      perm[1] *= -1;
      perm[2] *= -1;
      break;
   case 5:
      std::swap(perm[1], perm[2]);
      perm[1] *= -1;
      perm[2] *= -1;
      break;
   case 6:
      std::swap(perm[1], perm[2]);
      perm[1] *= -1;
      break;
   case 7:
      perm[2] *= -1;
      break;
   default:
      break;
   }

   // convert back to lex
   if (face_id2 == 3 || face_id2 == 4)
   {
      perm[1] *= -1;
   }
   else if (face_id2 == 0)
   {
      perm[2] *= -1;
   }
}

static void PADGDiffusionSetupIwork3D(const int nf, const Mesh& mesh, const FaceType type, Array<int>& iwork_)
{
   const int ne = mesh.GetNE();

   int fidx = 0;
   iwork_.SetSize(nf * 12);
   // (perm[0], perm[1], perm[2], element_index, local_face_id, orientation)
   constexpr int _e_ = 3; // offset for element index
   constexpr int _fid_ = 4; // offset for local face id
   constexpr int _or_ = 5; // offset for orientation

   auto iwork = Reshape(iwork_.HostWrite(), 6, 2, nf);
   for (int f = 0; f < mesh.GetNumFaces(); ++f)
   {
      auto face_info = mesh.GetFaceInformation(f);

      if (face_info.IsOfFaceType(type))
      {
         const int fid0 = face_info.element[0].local_face_id;
         const int or0 = face_info.element[0].orientation;

         iwork(  _e_, 0, fidx) = face_info.element[0].index;
         iwork(_fid_, 0, fidx) = fid0;
         iwork( _or_, 0, fidx) = or0;

         FaceNormalPermutation(&iwork(0, 0, fidx), fid0);

         if (face_info.IsInterior())
         {
            const int fid1 = face_info.element[1].local_face_id;
            const int or1 = face_info.element[1].orientation;

            iwork(  _e_, 1, fidx) = face_info.element[1].index;
            iwork(_fid_, 1, fidx) = fid1;
            iwork( _or_, 1, fidx) = or1;

            SignedFaceNormalPermutation(&iwork(0, 1, fidx), fid0, fid1, or1);
         }
         else
         {
            for (int i = 0; i < 6; ++i)
            {
               iwork(i, 1, fidx) = -1;
            }
         }

         fidx++;
      }
   }
}

void DGDiffusionIntegrator::SetupPA(const FiniteElementSpace &fes,
                                    FaceType type)
{
   const MemoryType mt = (pa_mt == MemoryType::DEFAULT) ?
                         Device::GetDeviceMemoryType() : pa_mt;

   const int ne = fes.GetNE();
   nf = fes.GetNFbyType(type);

   // if (nf == 0) { return; }

   // Assumes tensor-product elements
   Mesh &mesh = *fes.GetMesh();
   const FiniteElement &el =
      *fes.GetTraceElement(0, mesh.GetFaceGeometry(0));
   FaceElementTransformations &T0 =
      *fes.GetMesh()->GetFaceElementTransformations(0);
   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el.GetOrder(), T0);
   dim = mesh.Dimension();
   const int q1d = pow(double(ir->Size()), 1.0/(dim - 1));

   auto vol_ir = irs.Get(mesh.GetElementGeometry(0), 2*q1d - 3);
   auto el_geom = mesh.GetGeometricFactors(
                     vol_ir,
                     GeometricFactors::JACOBIANS | GeometricFactors::DETERMINANTS,
                     mt);

   std::unique_ptr<FaceNeighborGeometricFactors> nbr_geom;

   if (type == FaceType::Interior)
   {
      nbr_geom.reset(new FaceNeighborGeometricFactors(*el_geom));
   }

   auto face_geom = mesh.GetFaceGeometricFactors(
                       *ir,
                       FaceGeometricFactors::DETERMINANTS |
                       FaceGeometricFactors::NORMALS, type, mt);
   maps = &el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   dofs1D = maps->ndof;
   quad1D = maps->nqpt;

   const int pa_size = (dim == 2) ? (6 * q1d * nf) : (8 * q1d * q1d * nf);
   pa_data.SetSize(pa_size, Device::GetMemoryType());

   FaceQuadratureSpace fqs(mesh, *ir, type);
   CoefficientVector q(fqs, CoefficientStorage::COMPRESSED);
   if (Q)
   {
      q.Project(*Q);
   }
   else if (MQ)
   {
      MFEM_ABORT("Not yet implemented");
      // q.Project(*MQ);
   }
   else
   {
      q.SetConstant(1.0);
   }

   Array<int> iwork;
   if (dim == 2)
   {
      PADGDiffusionSetupIwork2D(nf, mesh, type, iwork);
   }
   else if (dim == 3)
   {
      PADGDiffusionSetupIwork3D(nf, mesh, type, iwork);
   }

   if (dim == 1)
   {
      MFEM_ABORT("dim==1 not supported in PADGTraceSetup");
   }
   else if (dim == 2)
   {
      PADGDiffusionsetup2D(quad1D, ne, nf, ir->GetWeights(), *el_geom, *face_geom,
                           nbr_geom.get(), q, sigma, kappa, pa_data, iwork);
   }
   else if (dim == 3)
   {
      PADGDiffusionSetup3D(quad1D, ne, nf, ir->GetWeights(), *el_geom, *face_geom, nbr_geom.get(), q, sigma, kappa, pa_data, iwork);
   }
}

void DGDiffusionIntegrator::AssemblePAInteriorFaces(const FiniteElementSpace&
                                                    fes)
{
   SetupPA(fes, FaceType::Interior);
}

void DGDiffusionIntegrator::AssemblePABoundaryFaces(const FiniteElementSpace&
                                                    fes)
{
   SetupPA(fes, FaceType::Boundary);
}

template<int T_D1D = 0, int T_Q1D = 0> static
void PADGDiffusionApply2D(const int NF,
                          const Array<double> &b,
                          const Array<double> &bt,
                          const Array<double>& g,
                          const Array<double>& gt,
                          const double sigma,
                          const double kappa,
                          const Vector &pa_data,
                          const Vector &x_,
                          const Vector &dxdn_,
                          Vector &y_,
                          Vector &dydn_,
                          const int d1d = 0,
                          const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");

   auto B_ = Reshape(b.Read(), Q1D, D1D);
   auto G_ = Reshape(g.Read(), Q1D, D1D);

   auto pa = Reshape(pa_data.Read(), 6, Q1D, NF); // (q, 1/h, J00, J01, J10, J11)

   auto x =    Reshape(x_.Read(),         D1D, 2, NF);
   auto y =    Reshape(y_.ReadWrite(),    D1D, 2, NF);
   auto dxdn = Reshape(dxdn_.Read(),      D1D, 2, NF);
   auto dydn = Reshape(dydn_.ReadWrite(), D1D, 2, NF);

   const int NBX = std::max(D1D, Q1D);

   mfem::forall_2D(NF, NBX, 2, [=] MFEM_HOST_DEVICE (int f) -> void
   {
      constexpr int max_D1D = T_D1D ? T_D1D : MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : MAX_Q1D;

      MFEM_SHARED double u0[max_D1D];
      MFEM_SHARED double u1[max_D1D];
      MFEM_SHARED double du0[max_D1D];
      MFEM_SHARED double du1[max_D1D];

      MFEM_SHARED double Bu0[max_Q1D];
      MFEM_SHARED double Bu1[max_Q1D];
      MFEM_SHARED double Bdu0[max_Q1D];
      MFEM_SHARED double Bdu1[max_Q1D];

      MFEM_SHARED double r[max_Q1D];

      MFEM_SHARED double BG[2*max_D1D*max_Q1D];
      DeviceMatrix B(BG, Q1D, D1D);
      DeviceMatrix G(BG + D1D*Q1D, Q1D, D1D);

      if (MFEM_THREAD_ID(y) == 0)
      {
         MFEM_FOREACH_THREAD(p,x,Q1D)
         {
            for (int d = 0; d < D1D; ++d)
            {
               B(p,d) = B_(p,d);
               G(p,d) = G_(p,d);
            }
         }
      }
      MFEM_SYNC_THREAD;

      // copy edge values to u0, u1 and copy edge normals to du0, du1
      MFEM_FOREACH_THREAD(side,y,2)
      {
         double *u = (side == 0) ? u0 : u1;
         double *du = (side == 0) ? du0 : du1;
         MFEM_FOREACH_THREAD(d,x,D1D)
         {
            u[d] = x(d, side, f);
            du[d] = dxdn(d, side, f);
         }
      }
      MFEM_SYNC_THREAD;

      // eval @ quad points
      MFEM_FOREACH_THREAD(side,y,2)
      {
         double *u = (side == 0) ? u0 : u1;
         double *du = (side == 0) ? du0 : du1;
         double *Bu = (side == 0) ? Bu0 : Bu1;
         double *Bdu = (side == 0) ? Bdu0 : Bdu1;

         MFEM_FOREACH_THREAD(p,x,Q1D)
         {
            const double Je_side[] = {pa(2 + 2*side, p, f), pa(2 + 2*side + 1, p, f)};

            Bu[p] = 0.0;
            Bdu[p] = 0.0;

            for (int d = 0; d < D1D; ++d)
            {
               const double b = B(p,d);
               const double g = G(p,d);

               Bu[p] += b*u[d];
               Bdu[p] += Je_side[0] * b * du[d] + Je_side[1] * g * u[d];
            }
         }
      }
      MFEM_SYNC_THREAD;

      // term - < {Q du/dn}, [v] > +  kappa * < {Q/h} [u], [v] >:
      if (MFEM_THREAD_ID(y) == 0)
      {
         MFEM_FOREACH_THREAD(p,x,Q1D)
         {
            const double q = pa(0, p, f);
            const double hi = pa(1, p, f);
            const double jump = Bu0[p] - Bu1[p];
            const double avg = Bdu0[p] + Bdu1[p]; // = {Q du/dn} * w * det(J)
            r[p] = -avg + hi * q * jump;
         }
      }
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD(d,x,D1D)
      {
         double Br = 0.0;

         for (int p = 0; p < Q1D; ++p)
         {
            Br += B(p, d) * r[p];
         }

         u0[d] =  Br; // overwrite u0, u1
         u1[d] = -Br;
      } // for d
      MFEM_SYNC_THREAD;


      MFEM_FOREACH_THREAD(side,y,2)
      {
         double *du = (side == 0) ? du0 : du1;
         MFEM_FOREACH_THREAD(d,x,D1D)
         {
            du[d] = 0.0;
         }
      }
      MFEM_SYNC_THREAD;

      // term sigma * < [u], {Q dv/dn} >
      MFEM_FOREACH_THREAD(side,y,2)
      {
         double * const du = (side == 0) ? du0 : du1;
         double * const u = (side == 0) ? u0 : u1;

         MFEM_FOREACH_THREAD(d,x,D1D)
         {
            for (int p = 0; p < Q1D; ++p)
            {
               const double Je[] = {pa(2 + 2*side, p, f), pa(2 + 2*side + 1, p, f)};
               const double jump = Bu0[p] - Bu1[p];
               const double r_p = Je[0] * jump; // normal
               const double w_p = Je[1] * jump; // tangential
               du[d] += sigma * B(p, d) * r_p;
               u[d] += sigma * G(p, d) * w_p;
            }
         }
      }
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD(side,y,2)
      {
         double *u = (side == 0) ? u0 : u1;
         double *du = (side == 0) ? du0 : du1;
         MFEM_FOREACH_THREAD(d,x,D1D)
         {
            y(d, side, f) += u[d];
            dydn(d, side, f) += du[d];
         }
      }
   }); // mfem::forall
}

template <int T_D1D = 0, int T_Q1D = 0>
static void PADGDiffusionApply3D(const int NF,
                                 const Array<double>& b,
                                 const Array<double>& bt,
                                 const Array<double>& g,
                                 const Array<double>& gt,
                                 const double sigma,
                                 const double kappa,
                                 const Vector& pa_data,
                                 const Vector& x_,
                                 const Vector& dxdn_,
                                 Vector& y_,
                                 Vector& dydn_,
                                 const int d1d = 0,
                                 const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");

   auto B = Reshape(b.Read(), Q1D, D1D);
   auto G = Reshape(g.Read(), Q1D, D1D);

   auto pa = Reshape(pa_data.Read(), 8, Q1D, Q1D, NF); // (q, 1/h, J0[0], J0[1], J0[2], J1[0], J1[1], J1[2])
   auto x =    Reshape(x_.Read(),         D1D, D1D, 2, NF);
   auto y =    Reshape(y_.ReadWrite(),    D1D, D1D, 2, NF);
   auto dxdn = Reshape(dxdn_.Read(),      D1D, D1D, 2, NF);
   auto dydn = Reshape(dydn_.ReadWrite(), D1D, D1D, 2, NF);

   for (int f = 0; f < NF; ++f)
   {
      constexpr int max_D1D = T_D1D ? T_D1D : MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : MAX_Q1D;

      double u0[max_D1D][max_D1D];
      double u1[max_D1D][max_D1D];
      double du0[max_D1D][max_D1D];
      double du1[max_D1D][max_D1D];

      double Bu0[max_Q1D][max_Q1D];
      double Bu1[max_Q1D][max_Q1D];
      double Bdu0[max_Q1D][max_Q1D];
      double Bdu1[max_Q1D][max_Q1D];

      double r[max_Q1D][max_Q1D];

      // copy edge values to u0, u1 and copy normals to du0, du1
      for (int side = 0; side < 2; ++side)
      {
         double (*u)[max_D1D] = (side == 0) ? u0 : u1;
         double (*du)[max_D1D] = (side == 0) ? du0 : du1;

         for (int d1 = 0; d1 < D1D; ++d1)
         {
            for (int d2 = 0; d2 < D1D; ++d2)
            {
               u[d1][d2] = x(d1, d2, side, f);
               du[d1][d2] = dxdn(d1, d2, side, f);
            }
         }
      }

      // eval u and physical normal deriv @ quad points
      for (int side = 0; side < 2; ++side)
      {
         double (*u)[max_D1D] = (side == 0) ? u0 : u1;
         double (*du)[max_D1D] = (side == 0) ? du0 : du1;
         double (*Bu)[max_Q1D] = (side == 0) ? Bu0 : Bu1;
         double (*Bdu)[max_Q1D] = (side == 0) ? Bdu0 : Bdu1;

         for (int p1 = 0; p1 < Q1D; ++p1)
         {
            for (int p2 = 0; p2 < Q1D; ++p2)
            {
               const double Je[] = {pa(2+3*side + 0, p1, p2, f), pa(2+3*side + 1, p1, p2, f), pa(2+3*side + 2, p1, p2, f)};

               Bu[p1][p2] = 0.0;
               Bdu[p1][p2] = 0.0;

               for (int d1 = 0; d1 < D1D; ++d1)
               {
                  for (int d2 = 0; d2 < D1D; ++d2)
                  {
                     const double b = B(p1, d1) * B(p2, d2);
                     const double g = Je[1] * G(p1, d1) * B(p2, d2) + Je[2] * B(p1, d1) * G(p2, d2);

                     Bu[p1][p2] += b * u[d1][d2];
                     Bdu[p1][p2] += Je[0] * b * du[d1][d2] + g * u[d1][d2];
                  }
               }
            }
         }
      }

      // term: - < {Q du/dn}, [v] > + kappa * < {Q/h} [u], [v] >
      for (int p1 = 0; p1 < Q1D; ++p1)
      {
         for (int p2 = 0; p2 < Q1D; ++p2)
         {
            const double q = pa(0, p1, p2, f);
            const double hi = pa(1, p1, p2, f);
            const double jump = Bu0[p1][p2] - Bu1[p1][p2];
            const double avg = Bdu0[p1][p2] + Bdu1[p1][p2]; // {Q du/dn} * w * det(J)
            r[p1][p2] = -avg + hi * q * jump;
         }
      }

      // u0, u1 <- B' * r
      for (int d1 = 0; d1 < D1D; ++d1)
      {
         for (int d2 = 0; d2 < D1D; ++d2)
         {
            double Br = 0.0;

            for (int p1 = 0; p1 < Q1D; ++p1)
            {
               for (int p2 = 0; p2 < Q1D; ++p2)
               {
                  Br += B(p1, d1) * B(p2, d2) * r[p1][p2];
               }
            }

            u0[d1][d2] =  Br; // overwrite u0, u1
            u1[d1][d2] = -Br;
         }
      }

      for (int side = 0; side < 2; ++side)
      {
         double (*du)[max_D1D] = (side == 0) ? du0 : du1;

         for (int d1 = 0; d1 < D1D; ++d1)
         {
            for (int d2 = 0; d2 < D1D; ++d2)
            {
               du[d1][d2] = 0.0;
            }
         }
      }

      // term: sigma * < [u], {Q dv/dn} >
      for (int side = 0; side < 2; ++side)
      {
         double (*du)[max_D1D] = (side == 0) ? du0 : du1;
         double (*u)[max_D1D] = (side == 0) ? u0 : u1;

         for (int d1 = 0; d1 < D1D; ++d1)
         {
            for (int d2 = 0; d2 < D1D; ++d2)
            {
               for (int p1 = 0; p1 < Q1D; ++p1)
               {
                  for (int p2 = 0; p2 < Q1D; ++p2)
                  {
                     const double Je[] = {pa(2+3*side + 0, p1, p2, f), pa(2+3*side + 1, p1, p2, f), pa(2+3*side + 2, p1, p2, f)};

                     const double jump = Bu0[p1][p2] - Bu1[p1][p2];

                     const double b = Je[0] * B(p1, d1) * B(p2, d2);
                     const double g = Je[1] * G(p1, d1) * B(p2, d2) + Je[2] * B(p1, d1) * G(p2, d2);

                     du[d1][d2] += sigma * b * jump;
                     u[d1][d2] += sigma * g * jump;
                  }
               }
            }
         }
      }

      // map back to y and dydn
      for (int side = 0; side < 2; ++side)
      {
         const double (*u)[max_D1D] = (side == 0) ? u0 : u1;
         const double (*du)[max_D1D] = (side == 0) ? du0 : du1;

         for (int d1 = 0; d1 < D1D; ++d1)
         {
            for (int d2 = 0; d2 < D1D; ++d2)
            {
               y(d1, d2, side, f) += u[d1][d2];
               dydn(d1, d2, side, f) += du[d1][d2];
            }
         }
      }
   }
}

static void PADGDiffusionApply(const int dim,
                               const int D1D,
                               const int Q1D,
                               const int NF,
                               const Array<double> &B,
                               const Array<double> &Bt,
                               const Array<double> &G,
                               const Array<double> &Gt,
                               const double sigma,
                               const double kappa,
                               const Vector &pa_data,
                               const Vector &x,
                               const Vector &dxdn,
                               Vector &y,
                               Vector &dydn)
{
   if (dim == 2)
   {
      switch ((D1D << 4 ) | Q1D)
      {
         case 0x23: return PADGDiffusionApply2D<2,3>(NF,B,Bt,G,Gt,sigma,kappa,pa_data,x,
                                                        dxdn,y,dydn);
         case 0x34: return PADGDiffusionApply2D<3,4>(NF,B,Bt,G,Gt,sigma,kappa,pa_data,x,
                                                        dxdn,y,dydn);
         case 0x45: return PADGDiffusionApply2D<4,5>(NF,B,Bt,G,Gt,sigma,kappa,pa_data,x,
                                                        dxdn,y,dydn);
         case 0x56: return PADGDiffusionApply2D<5,6>(NF,B,Bt,G,Gt,sigma,kappa,pa_data,x,
                                                        dxdn,y,dydn);
         case 0x67: return PADGDiffusionApply2D<6,7>(NF,B,Bt,G,Gt,sigma,kappa,pa_data,x,
                                                        dxdn,y,dydn);
         case 0x78: return PADGDiffusionApply2D<7,8>(NF,B,Bt,G,Gt,sigma,kappa,pa_data,x,
                                                        dxdn,y,dydn);
         case 0x89: return PADGDiffusionApply2D<8,9>(NF,B,Bt,G,Gt,sigma,kappa,pa_data,x,
                                                        dxdn,y,dydn);
         case 0x9A: return PADGDiffusionApply2D<9,10>(NF,B,Bt,G,Gt,sigma,kappa,pa_data,x,
                                                         dxdn,y,dydn);
         default:   return PADGDiffusionApply2D(NF,B,Bt,G,Gt,sigma,kappa,pa_data,x,dxdn,
                                                   y,dydn,D1D,Q1D);
      }
   }
   else if (dim == 3)
   {
      return PADGDiffusionApply3D(NF, B, Bt, G, Gt, sigma, kappa, pa_data, x, dxdn, y, dydn, D1D, Q1D);
   }
   MFEM_ABORT("Unknown kernel.");
}

void DGDiffusionIntegrator::AddMultPAFaceNormalDerivatives(const Vector &x,
                                                           const Vector &dxdn, Vector &y, Vector &dydn) const
{
   PADGDiffusionApply(dim, dofs1D, quad1D, nf,
                      maps->B, maps->Bt, maps->G, maps->Gt,
                      sigma, kappa, pa_data, x, dxdn, y, dydn);
}

const IntegrationRule &DGDiffusionIntegrator::GetRule(
   int order, FaceElementTransformations &T)
{
   int int_order = T.Elem1->OrderW() + 2*order;
   return irs.Get(T.GetGeometryType(), int_order);
}

} // namespace mfem

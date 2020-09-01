// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "quadinterpolator.hpp"
#include "../general/forall.hpp"
#include "../linalg/dtensor.hpp"
#include "../fem/kernels.hpp"
#include "../linalg/kernels.hpp"

using namespace mfem;

namespace mfem
{

template<int T_D1D = 0, int T_Q1D = 0, int MAX_D1D = 0, int MAX_Q1D = 0>
static void Det2D(const int NE,
                  const double *b,
                  const double *g,
                  const double *x,
                  double *y,
                  const int vdim = 1,
                  const int d1d = 0,
                  const int q1d = 0)
{
   constexpr int DIM = 2;
   constexpr int NBZ = 1;

   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto B = Reshape(b, Q1D, D1D);
   const auto G = Reshape(g, Q1D, D1D);
   const auto X = Reshape(x,  D1D, D1D, DIM, NE);
   auto Y = Reshape(y, Q1D, Q1D, NE);

   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   {
      constexpr int NBZ = 1;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;

      MFEM_SHARED double BG[2][MQ1*MD1];
      MFEM_SHARED double XY[2][NBZ][MD1*MD1];
      MFEM_SHARED double DQ[4][NBZ][MD1*MQ1];
      MFEM_SHARED double QQ[4][NBZ][MQ1*MQ1];

      kernels::LoadX<MD1,NBZ>(e,D1D,X,XY);
      kernels::LoadBG<MD1,MQ1>(D1D,Q1D,B,G,BG);

      kernels::GradX<MD1,MQ1,NBZ>(D1D,Q1D,BG,XY,DQ);
      kernels::GradY<MD1,MQ1,NBZ>(D1D,Q1D,BG,DQ,QQ);

      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double J[4];
            kernels::PullGrad<MQ1,NBZ>(qx,qy,QQ,J);
            Y(qx,qy,e) = kernels::Det<2>(J);
         }
      }
   });
}

template<int T_D1D = 0, int T_Q1D = 0, int MAX_D1D = 0, int MAX_Q1D = 0>
static void Det3D(const int NE,
                  const double *b,
                  const double *g,
                  const double *x,
                  double *y,
                  const int vdim = 1,
                  const int d1d = 0,
                  const int q1d = 0)
{
   constexpr int DIM = 3;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto B = Reshape(b, Q1D, D1D);
   const auto G = Reshape(g, Q1D, D1D);
   const auto X = Reshape(x, D1D, D1D, D1D, DIM, NE);
   auto Y = Reshape(y, Q1D, Q1D, Q1D, NE);

   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
      constexpr int MDQ = MQ1 > MD1 ? MQ1 : MD1;

      MFEM_SHARED double BG[2][MQ1*MD1];
      MFEM_SHARED double sm0[9][MDQ*MDQ*MDQ];
      MFEM_SHARED double sm1[9][MDQ*MDQ*MDQ];

      double (*DDD)[MD1*MD1*MD1] = (double (*)[MD1*MD1*MD1]) (sm0);
      double (*DDQ)[MD1*MD1*MQ1] = (double (*)[MD1*MD1*MQ1]) (sm1);
      double (*DQQ)[MD1*MQ1*MQ1] = (double (*)[MD1*MQ1*MQ1]) (sm0);
      double (*QQQ)[MQ1*MQ1*MQ1] = (double (*)[MQ1*MQ1*MQ1]) (sm1);

      kernels::LoadX<MD1>(e,D1D,X,DDD);
      kernels::LoadBG<MD1,MQ1>(D1D,Q1D,B,G,BG);

      kernels::GradX<MD1,MQ1>(D1D,Q1D,BG,DDD,DDQ);
      kernels::GradY<MD1,MQ1>(D1D,Q1D,BG,DDQ,DQQ);
      kernels::GradZ<MD1,MQ1>(D1D,Q1D,BG,DQQ,QQQ);

      MFEM_FOREACH_THREAD(qz,z,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               double J[9];
               kernels::PullGrad<MQ1>(qx,qy,qz, QQQ, J);
               Y(qx,qy,qz,e) = kernels::Det<3>(J);
            }
         }
      }
   });
}

void QuadratureInterpolator::Determinants(const Vector &e_vec,
                                          Vector &q_det) const
{
   if (use_tensor_products)
   {
      const int NE = fespace->GetNE();
      if (NE == 0) { return; }
      const int vdim = fespace->GetVDim();
      const int dim = fespace->GetMesh()->Dimension();
      const FiniteElement *fe = fespace->GetFE(0);
      const IntegrationRule *ir =
         IntRule ? IntRule : &qspace->GetElementIntRule(0);
      const DofToQuad &maps = fe->GetDofToQuad(*ir, DofToQuad::TENSOR);
      const int D1D = maps.ndof;
      const int Q1D = maps.nqpt;
      const double *B = maps.B.Read();
      const double *G = maps.G.Read();
      const double *X = e_vec.Read();
      double *Y = q_det.Write();

      const int id = (dim<<12) | (vdim<<8) | (D1D<<4) | Q1D;

      switch (id)
      {
         case 0x2222: return Det2D<2,2>(NE,B,G,X,Y);
         case 0x2223: return Det2D<2,3>(NE,B,G,X,Y);
         case 0x2224: return Det2D<2,4>(NE,B,G,X,Y);
         case 0x2226: return Det2D<2,6>(NE,B,G,X,Y);
         case 0x2234: return Det2D<3,4>(NE,B,G,X,Y);
         case 0x2236: return Det2D<3,6>(NE,B,G,X,Y);
         case 0x2244: return Det2D<4,4>(NE,B,G,X,Y);
         case 0x2246: return Det2D<4,6>(NE,B,G,X,Y);
         case 0x2256: return Det2D<5,6>(NE,B,G,X,Y);

         case 0x3324: return Det3D<2,4>(NE,B,G,X,Y);
         case 0x3333: return Det3D<3,3>(NE,B,G,X,Y);
         case 0x3335: return Det3D<3,5>(NE,B,G,X,Y);
         case 0x3336: return Det3D<3,6>(NE,B,G,X,Y);
         //case 0x3348: return Det3D<4,8>(NE,B,G,X,Y);

         default:
         {
            if (dim == 2)
            {
               constexpr int MD1 = 8;
               constexpr int MQ1 = 8;
               MFEM_VERIFY(D1D <= MD1, "Orders higher than " << MD1-1
                           << " are not supported!");
               MFEM_VERIFY(Q1D <= MQ1, "Quadrature rules with more than "
                           << MQ1 << " 1D points are not supported!");
               return Det2D<0,0,MD1,MQ1>(NE,B,G,X,Y,vdim,D1D,Q1D);
            }
            if (dim == 3)
            {
               constexpr int MD1 = 6;
               constexpr int MQ1 = 6;
               MFEM_VERIFY(D1D <= MD1, "Orders higher than " << MD1-1
                           << " are not supported!");
               MFEM_VERIFY(Q1D <= MQ1, "Quadrature rules with more than "
                           << MQ1 << " 1D points are not supported!");
               return Det3D<0,0,MD1,MQ1>(NE,B,G,X,Y,vdim,D1D,Q1D);
            }
         }

      }
      MFEM_ABORT("Kernel " << std::hex << id << std::dec << " not supported yet");
   }
   else
   {
      Vector empty;
      Mult(e_vec, DETERMINANTS, empty, empty, q_det);
   }
}

} // namespace mfem

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

#include "tmop.hpp"
#include "tmop_pa.hpp"
#include "linearform.hpp"
#include "pgridfunc.hpp"
#include "tmop_tools.hpp"
#include "../general/debug.hpp"
#include "../general/forall.hpp"
#include "../linalg/kernels.hpp"
#include "../linalg/dtensor.hpp"

namespace mfem
{

MFEM_REGISTER_TMOP_KERNELS(void, AddMultGradPA_Kernel_2D,
                           const int NE,
                           const Array<double> &b_,
                           const Array<double> &g_,
                           const DenseTensor &j_,
                           const Vector &p_,
                           const Vector &x_,
                           Vector &y_,
                           const int d1d,
                           const int q1d)
{
   dbg("");
   constexpr int DIM = 2;
   constexpr int NBZ = 1;

   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto b = Reshape(b_.Read(), Q1D, D1D);
   const auto g = Reshape(g_.Read(), Q1D, D1D);
   const auto J = Reshape(j_.Read(), DIM, DIM, Q1D, Q1D, NE);
   const auto X = Reshape(x_.Read(), D1D, D1D, DIM, NE);
   const auto dP = Reshape(p_.Read(), DIM, DIM, DIM, DIM, Q1D, Q1D, NE);
   auto Y = Reshape(y_.ReadWrite(), D1D, D1D, DIM, NE);

   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int NBZ = 1;
      constexpr int MQ1 = T_Q1D ? T_Q1D : T_MAX;
      constexpr int MD1 = T_D1D ? T_D1D : T_MAX;

      MFEM_SHARED double BG[2][MQ1*MD1];
      MFEM_SHARED double XY[2][NBZ][MD1*MD1];
      MFEM_SHARED double DQ[4][NBZ][MD1*MQ1];
      MFEM_SHARED double QQ[4][NBZ][MQ1*MQ1];

      kernels::LoadX<MD1,NBZ>(e,D1D,X,XY);
      kernels::LoadBG<MD1,MQ1>(D1D,Q1D,b,g,BG);

      kernels::GradX<MD1,MQ1,NBZ>(D1D,Q1D,BG,XY,DQ);
      kernels::GradY<MD1,MQ1,NBZ>(D1D,Q1D,BG,DQ,QQ);

      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double A[4], B[4], C[4];
            const double *Jtr = &J(0,0,qx,qy,e);

            // Jrt = Jtr^{-1}
            double Jrt[4];
            kernels::CalcInverse<2>(Jtr, Jrt);

            double hX[4];
            kernels::PullGradXY<MQ1,NBZ>(qx,qy,QQ,hX);

            // A = X^T . Jrt
            kernels::Mult(2,2,2, hX, Jrt, A);

            // B = A : dP
            for (int i = 0; i < DIM; i++)
            {
               for (int j = 0; j < DIM; j++)
               {
                  B[i+2*j] = 0.0;
                  for (int r = 0; r < DIM; r++)
                  {
                     for (int c = 0; c < DIM; c++)
                     {
                        B[i+2*j] += dP(r,c,i,j,qx,qy,e) * A[r+2*c];
                     }
                  }
               }
            }

            // C = Jrt . B
            kernels::MultABt(2,2,2, Jrt, B, C);
            kernels::PushGradXY<MQ1,NBZ>(qx,qy,C,QQ);
         }
      }
      MFEM_SYNC_THREAD;
      kernels::LoadBGt<MD1,MQ1>(D1D,Q1D,b,g,BG);
      kernels::GradYt<MD1,MQ1,NBZ>(D1D,Q1D,BG,QQ,DQ);
      kernels::GradXt<MD1,MQ1,NBZ>(D1D,Q1D,BG,DQ,Y,e);
   });
}

void TMOP_Integrator::AddMultGradPA_2D(const Vector &R, Vector &C) const
{
   const int N = PA.ne;
   const int D1D = PA.maps->ndof;
   const int Q1D = PA.maps->nqpt;
   const int id = (D1D << 4 ) | Q1D;
   const DenseTensor &J = PA.Jtr;
   const Array<double> &B = PA.maps->B;
   const Array<double> &G = PA.maps->G;
   const Vector &A = PA.A;

   if (KAddMultGradPA_Kernel_2D.Find(id))
   {
      return KAddMultGradPA_Kernel_2D.At(id)(N,B,G,J,A,R,C,0,0);
   }
   else
   {
      constexpr int T_MAX = 8;
      MFEM_VERIFY(D1D <= MAX_D1D && Q1D <= MAX_Q1D, "Max size error!");
      return AddMultGradPA_Kernel_2D<0,0,T_MAX>(N,B,G,J,A,R,C,D1D,Q1D);
   }
}

} // namespace mfem

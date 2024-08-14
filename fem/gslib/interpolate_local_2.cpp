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

#include "../gslib.hpp"
#include "findpts_2.hpp"
#include "../../general/forall.hpp"
#include "../../linalg/kernels.hpp"
#include "../../linalg/dinvariants.hpp"

#ifdef MFEM_USE_GSLIB

namespace mfem
{

#define CODE_INTERNAL 0
#define CODE_BORDER 1
#define CODE_NOT_FOUND 2
#define dlong int
#define dfloat double

static MFEM_HOST_DEVICE void lagrange_eval(dfloat *p0, dfloat x,
                                           dlong i, dlong p_Nq,
                                           dfloat *z, dfloat *lagrangeCoeff)
{
   dfloat p_i = (1 << (p_Nq - 1));
   for (dlong j = 0; j < p_Nq; ++j)
   {
      dfloat d_j = x - z[j];
      p_i *= j == i ? 1 : d_j;
   }
   p0[i] = lagrangeCoeff[i] * p_i;
}

template<int T_D1D = 0>
static void InterpolateLocal2D_Kernel(const dfloat *const gf_in,
                                      dlong *const el,
                                      dfloat *const r,
                                      dfloat *const int_out,
                                      const int npt,
                                      const int ncomp,
                                      const int nel,
                                      const int gf_offset,
                                      dfloat *gll1D,
                                      double *lagcoeff,
                                      const int pN = 0)
{
   const int Nfields = ncomp;
   const int fieldOffset = gf_offset;
   const int MD1 = T_D1D ? T_D1D : 14;
   const int D1D = T_D1D ? T_D1D : pN;
   const int p_Np = D1D*D1D;
   MFEM_VERIFY(MD1 <= 14,"Increase Max allowable polynomial order.");
   MFEM_VERIFY(D1D != 0, "Polynomial order not specified.");
   const int nThreads = 32;
   mfem::forall_2D(npt, nThreads, 1, [=] MFEM_HOST_DEVICE (int i)
   {
      MFEM_SHARED dfloat wtr[2*MD1];
      MFEM_SHARED dfloat sums[MD1];

      // Evaluate basis functions at the reference space coordinates
      MFEM_FOREACH_THREAD(j,x,nThreads)
      {
         if (j < 2*D1D)
         {
            const int qp = j % D1D;
            const int d = j / D1D;
            lagrange_eval(wtr + d*D1D, r[2 * i + d], qp, D1D, gll1D, lagcoeff);
         }
      }
      MFEM_SYNC_THREAD;

      for (int fld = 0; fld < Nfields; ++fld)
      {

         const dlong elemOffset = el[i] * p_Np + fld * fieldOffset;

         MFEM_FOREACH_THREAD(j,x,nThreads)
         {
            if (j < D1D)
            {
               dfloat sum_j = 0;
               for (dlong k = 0; k < D1D; ++k)
               {
                  sum_j += gf_in[elemOffset + j + k * D1D] * wtr[D1D+k];
               }
               sums[j] = wtr[j] * sum_j;
            }
         }
         MFEM_SYNC_THREAD;

         MFEM_FOREACH_THREAD(j,x,nThreads)
         {
            if (j == 0)
            {
               double sumv = 0.0;
               for (dlong jj = 0; jj < D1D; ++jj)
               {
                  sumv += sums[jj];
               }
               int_out[i + fld * npt] = sumv;
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

void FindPointsGSLIB::InterpolateLocal2(const Vector &field_in,
                                        Array<int> &gsl_elem_dev_l,
                                        Vector &gsl_ref_l,
                                        Vector &field_out,
                                        int npt, int ncomp,
                                        int nel, int dof1Dsol)
{
   if (npt == 0) { return; }
   const int gf_offset = field_in.Size()/ncomp;
   MFEM_VERIFY(dim == 2,"Kernel for 2D only.");
   auto pfin = field_in.Read();
   auto pgsl = gsl_elem_dev_l.ReadWrite();
   auto pgslr = gsl_ref_l.ReadWrite();
   auto pfout = field_out.Write();
   auto pgll = DEV.gll1d_sol.ReadWrite();
   auto plcf = DEV.lagcoeff_sol.ReadWrite();
   switch (dof1Dsol)
   {
      case 2: return InterpolateLocal2D_Kernel<2>(pfin, pgsl, pgslr, pfout,
                                                     npt, ncomp, nel, gf_offset,
                                                     pgll, plcf);
      case 3: return InterpolateLocal2D_Kernel<3>(pfin, pgsl, pgslr, pfout,
                                                     npt, ncomp, nel, gf_offset,
                                                     pgll, plcf);
      case 4: return InterpolateLocal2D_Kernel<4>(pfin, pgsl, pgslr, pfout,
                                                     npt, ncomp, nel, gf_offset,
                                                     pgll, plcf);
      case 5: return InterpolateLocal2D_Kernel<5>(pfin, pgsl, pgslr, pfout,
                                                     npt, ncomp, nel, gf_offset,
                                                     pgll, plcf);
      default: return InterpolateLocal2D_Kernel(pfin, pgsl, pgslr, pfout,
                                                   npt, ncomp, nel, gf_offset,
                                                   pgll, plcf, dof1Dsol);
   }
}


#undef CODE_INTERNAL
#undef CODE_BORDER
#undef CODE_NOT_FOUND
#undef dlong
#undef dfloat

} // namespace mfem

#endif //ifdef MFEM_USE_GSLIB

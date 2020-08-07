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
#include "quadinterpolator_grad.hpp"
#include "../general/forall.hpp"
#include "../linalg/dtensor.hpp"
#include "../linalg/kernels.hpp"

#define MFEM_DEBUG_COLOR 226
#include "../general/debug.hpp"

namespace mfem
{

template<>
void QuadratureInterpolator::Derivatives<QVectorLayout::byNODES>(
   const Vector &e_vec, Vector &q_der) const
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
   double *Y = q_der.Write();

   constexpr QVectorLayout L = QVectorLayout::byNODES;

   const int id = (dim<<12) | (vdim<<8) | (D1D<<4) | Q1D;

   switch (id)
   {
      case 0x2222: return Grad2D<L,2,2,2,16>(NE,B,G,X,Y);
      case 0x2223: return Grad2D<L,2,2,3,8>(NE,B,G,X,Y);
      case 0x2224: return Grad2D<L,2,2,4,4>(NE,B,G,X,Y);
      case 0x2225: return Grad2D<L,2,2,5,4>(NE,B,G,X,Y);
      case 0x2226: return Grad2D<L,2,2,6,2>(NE,B,G,X,Y);

      case 0x2233: return Grad2D<L,2,3,3,2>(NE,B,G,X,Y);
      case 0x2234: return Grad2D<L,2,3,4,4>(NE,B,G,X,Y);
      case 0x2236: return Grad2D<L,2,3,6,2>(NE,B,G,X,Y);

      case 0x2244: return Grad2D<L,2,4,4,2>(NE,B,G,X,Y);
      case 0x2245: return Grad2D<L,2,4,5,2>(NE,B,G,X,Y);
      case 0x2246: return Grad2D<L,2,4,6,2>(NE,B,G,X,Y);
      case 0x2247: return Grad2D<L,2,4,7,2>(NE,B,G,X,Y);

      case 0x2256: return Grad2D<L,2,5,6,2>(NE,B,G,X,Y);

      case 0x3124: return Grad3D<L,1,2,4>(NE,B,G,X,Y);
      case 0x3136: return Grad3D<L,1,3,6>(NE,B,G,X,Y);
      case 0x3148: return Grad3D<L,1,4,8>(NE,B,G,X,Y);

      case 0x3323: return Grad3D<L,3,2,3>(NE,B,G,X,Y);
      case 0x3324: return Grad3D<L,3,2,4>(NE,B,G,X,Y);
      case 0x3325: return Grad3D<L,3,2,5>(NE,B,G,X,Y);
      case 0x3326: return Grad3D<L,3,2,6>(NE,B,G,X,Y);

      case 0x3333: return Grad3D<L,3,3,3>(NE,B,G,X,Y);
      case 0x3334: return Grad3D<L,3,3,4>(NE,B,G,X,Y);
      case 0x3335: return Grad3D<L,3,3,5>(NE,B,G,X,Y);
      case 0x3336: return Grad3D<L,3,3,6>(NE,B,G,X,Y);
      case 0x3344: return Grad3D<L,3,4,4>(NE,B,G,X,Y);
      case 0x3346: return Grad3D<L,3,4,6>(NE,B,G,X,Y);
      case 0x3347: return Grad3D<L,3,4,7>(NE,B,G,X,Y);
      case 0x3348: return Grad3D<L,3,4,8>(NE,B,G,X,Y);
      default:
      {
         constexpr int MD1 = 8;
         constexpr int MQ1 = 8;
         dbg("Using standard kernel #id 0x%x", id);
         MFEM_VERIFY(D1D <= MD1, "Orders higher than " << MD1-1
                     << " are not supported!");
         MFEM_VERIFY(Q1D <= MQ1, "Quadrature rules with more than "
                     << MQ1 << " 1D points are not supported!");
         if (dim == 2)
         {
            return Grad2D<L,0,0,0,0,MD1,MQ1>(NE,B,G,X,Y,vdim,D1D,Q1D);
         }
         if (dim == 3)
         {
            return Grad3D<L,0,0,0,MD1,MQ1>(NE,B,G,X,Y,vdim,D1D,Q1D);
         }
      }
   }
   mfem::out << "Unknown kernel 0x" << std::hex << id << std::endl;
   MFEM_ABORT("Kernel not supported yet");
}

} // namespace mfem

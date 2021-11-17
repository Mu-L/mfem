// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "fem.hpp"
#include "lininteg_domain.hpp"
#include "lininteg_domain_grad.hpp"

namespace mfem
{

using namespace internal::linearform_extension;

void DomainLFGradIntegrator::AssembleFull(const FiniteElementSpace &fes,
                                          const Array<int> &markers,
                                          Vector &y)
{
   MFEM_VERIFY(fes.GetVDim()==1, "vdim != 1");
   GetOrder_f gof = [](const int el_order) { return 2.0 * el_order; };
   const IntegrationRule *ir = GetIntRuleFromOrder(fes, IntRule, gof);

   Vector coeff;
   const int NQ = ir->GetNPoints();
   const int NE = fes.GetMesh()->GetNE();

   if (VectorConstantCoefficient *vcQ =
          dynamic_cast<VectorConstantCoefficient*>(&Q))
   {
      coeff = vcQ->GetVec();
   }
   else if (VectorQuadratureFunctionCoefficient *vqfQ =
               dynamic_cast<VectorQuadratureFunctionCoefficient*>(&Q))
   {
      const QuadratureFunction &qfun = vqfQ->GetQuadFunction();
      MFEM_VERIFY(qfun.Size() == NE*NQ,
                  "Incompatible QuadratureFunction dimension \n");
      MFEM_VERIFY(ir == &qfun.GetSpace()->GetElementIntRule(0),
                  "IntegrationRule used within integrator and in"
                  " QuadratureFunction appear to be different.\n");
      qfun.Read();
      coeff.MakeRef(const_cast<QuadratureFunction&>(qfun),0);
   }
   else
   {
      const int qvdim = Q.GetVDim();
      Vector Qvec(qvdim);
      coeff.SetSize(qvdim * NQ * NE);
      auto C = Reshape(coeff.HostWrite(), qvdim, NQ, NE);
      for (int e = 0; e < NE; ++e)
      {
         ElementTransformation& T = *fes.GetElementTransformation(e);
         for (int q = 0; q < NQ; ++q)
         {
            Q.Eval(Qvec, T, ir->IntPoint(q));
            for (int c=0; c<qvdim; ++c)
            {
               C(c,q,e) = Qvec[c];
            }
         }
      }
   }

   const int id = GetKernelId(fes,ir);
   const int dim = fes.GetMesh()->Dimension();

   LinearFormExtensionKernel_f ker = nullptr;
   if (dim==2) { ker=VectorDomainLFGradIntegratorAssemble2D; }
   if (dim==3) { ker=VectorDomainLFGradIntegratorAssemble3D; }

   switch (id)
   {
      // 2D kernels, q=p+1
      case 0x222: ker=VectorDomainLFGradIntegratorAssemble2D<2,2>; break;
      case 0x233: ker=VectorDomainLFGradIntegratorAssemble2D<3,3>; break;
      case 0x244: ker=VectorDomainLFGradIntegratorAssemble2D<4,4>; break;
      case 0x255: ker=VectorDomainLFGradIntegratorAssemble2D<5,5>; break;

      // 2D kernels, q=p+2
      case 0x223: ker=VectorDomainLFGradIntegratorAssemble2D<2,3>; break;
      case 0x234: ker=VectorDomainLFGradIntegratorAssemble2D<3,4>; break;
      case 0x245: ker=VectorDomainLFGradIntegratorAssemble2D<4,5>; break;
      case 0x256: ker=VectorDomainLFGradIntegratorAssemble2D<5,6>; break;

      // 3D kernels, q=p+1
      case 0x322: ker=VectorDomainLFGradIntegratorAssemble3D<2,2>; break;
      case 0x333: ker=VectorDomainLFGradIntegratorAssemble3D<3,3>; break;
      case 0x344: ker=VectorDomainLFGradIntegratorAssemble3D<4,4>; break;
      case 0x355: ker=VectorDomainLFGradIntegratorAssemble3D<5,5>; break;

      // 3D kernels, q=p+2
      case 0x323: ker=VectorDomainLFGradIntegratorAssemble3D<2,3>; break;
      case 0x334: ker=VectorDomainLFGradIntegratorAssemble3D<3,4>; break;
      case 0x345: ker=VectorDomainLFGradIntegratorAssemble3D<4,5>; break;
      case 0x356: ker=VectorDomainLFGradIntegratorAssemble3D<5,6>; break;
   }
   MFEM_VERIFY(ker, "Unexpected kernel error!");
   Launch(ker,fes,ir,coeff,markers,y);
}

} // namespace mfem

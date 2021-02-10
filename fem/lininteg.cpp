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
#include <cmath>

namespace mfem
{

void LinearFormIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Tr, Vector &elvect)
{
   mfem_error("LinearFormIntegrator::AssembleRHSElementVect(...)");
}


void DomainLFIntegrator::AssembleRHSElementVect(const FiniteElement &el,
                                                ElementTransformation &Tr,
                                                Vector &elvect)
{
   int dof = el.GetDof();

   shape.SetSize(dof);       // vector of size dof
   elvect.SetSize(dof);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // ir = &IntRules.Get(el.GetGeomType(),
      //                    oa * el.GetOrder() + ob + Tr.OrderW());
      ir = &IntRules.Get(el.GetGeomType(), oa * el.GetOrder() + ob);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetIntPoint (&ip);
      double val = Tr.Weight() * Q.Eval(Tr, ip);

      el.CalcShape(ip, shape);

      add(elvect, ip.weight * val, shape, elvect);
   }
}

void DomainLFIntegrator::AssembleDeltaElementVect(
   const FiniteElement &fe, ElementTransformation &Trans, Vector &elvect)
{
   MFEM_ASSERT(delta != NULL, "coefficient must be DeltaCoefficient");
   elvect.SetSize(fe.GetDof());
   fe.CalcPhysShape(Trans, elvect);
   elvect *= delta->EvalDelta(Trans, Trans.GetIntPoint());
}

void DomainLFGradIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int dof = el.GetDof();
   int spaceDim = Tr.GetSpaceDim();

   dshape.SetSize(dof, spaceDim);

   elvect.SetSize(dof);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int intorder = 2 * el.GetOrder();
      ir = &IntRules.Get(el.GetGeomType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetIntPoint(&ip);
      el.CalcPhysDShape(Tr, dshape);

      Q.Eval(Qvec, Tr, ip);
      Qvec *= ip.weight * Tr.Weight();

      dshape.AddMult(Qvec, elvect);
   }
}

void DomainLFGradIntegrator::AssembleDeltaElementVect(
   const FiniteElement &fe, ElementTransformation &Trans, Vector &elvect)
{
   MFEM_ASSERT(vec_delta != NULL,"coefficient must be VectorDeltaCoefficient");
   int dof = fe.GetDof();
   int spaceDim = Trans.GetSpaceDim();

   dshape.SetSize(dof, spaceDim);
   fe.CalcPhysDShape(Trans, dshape);

   vec_delta->EvalDelta(Qvec, Trans, Trans.GetIntPoint());

   elvect.SetSize(dof);
   dshape.Mult(Qvec, elvect);
}

void BoundaryLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int dof = el.GetDof();

   shape.SetSize(dof);        // vector of size dof
   elvect.SetSize(dof);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int intorder = oa * el.GetOrder() + ob;  // <----------
      ir = &IntRules.Get(el.GetGeomType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetIntPoint (&ip);
      double val = Tr.Weight() * Q.Eval(Tr, ip);

      el.CalcShape(ip, shape);

      add(elvect, ip.weight * val, shape, elvect);
   }
}

void BoundaryLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Tr, Vector &elvect)
{
   int dof = el.GetDof();

   shape.SetSize(dof);        // vector of size dof
   elvect.SetSize(dof);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int intorder = oa * el.GetOrder() + ob;    // <------ user control
      ir = &IntRules.Get(Tr.FaceGeom, intorder); // of integration order
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      // Set the integration point in the face and the neighboring element
      Tr.SetAllIntPoints(&ip);

      // Access the neighboring element's integration point
      const IntegrationPoint &eip = Tr.GetElement1IntPoint();

      double val = Tr.Face->Weight() * ip.weight * Q.Eval(*Tr.Face, ip);

      el.CalcShape(eip, shape);

      add(elvect, val, shape, elvect);
   }
}

void BoundaryNormalLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int dim = el.GetDim()+1;
   int dof = el.GetDof();
   Vector nor(dim), Qvec;

   shape.SetSize(dof);
   elvect.SetSize(dof);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int intorder = oa * el.GetOrder() + ob;  // <----------
      ir = &IntRules.Get(el.GetGeomType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetIntPoint(&ip);
      CalcOrtho(Tr.Jacobian(), nor);
      Q.Eval(Qvec, Tr, ip);

      el.CalcShape(ip, shape);

      elvect.Add(ip.weight*(Qvec*nor), shape);
   }
}

void BoundaryTangentialLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int dim = el.GetDim()+1;
   int dof = el.GetDof();
   Vector tangent(dim), Qvec;

   shape.SetSize(dof);
   elvect.SetSize(dof);
   elvect = 0.0;

   if (dim != 2)
   {
      mfem_error("These methods make sense only in 2D problems.");
   }

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int intorder = oa * el.GetOrder() + ob;  // <----------
      ir = &IntRules.Get(el.GetGeomType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetIntPoint(&ip);
      const DenseMatrix &Jac = Tr.Jacobian();
      tangent(0) =  Jac(0,0);
      tangent(1) = Jac(1,0);

      Q.Eval(Qvec, Tr, ip);

      el.CalcShape(ip, shape);

      add(elvect, ip.weight*(Qvec*tangent), shape, elvect);
   }
}

void VectorDomainLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int vdim = Q.GetVDim();
   int dof  = el.GetDof();

   double val,cf;

   shape.SetSize(dof);       // vector of size dof

   elvect.SetSize(dof * vdim);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int intorder = 2*el.GetOrder();
      ir = &IntRules.Get(el.GetGeomType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetIntPoint (&ip);
      val = Tr.Weight();

      el.CalcShape(ip, shape);
      Q.Eval (Qvec, Tr, ip);

      for (int k = 0; k < vdim; k++)
      {
         cf = val * Qvec(k);

         for (int s = 0; s < dof; s++)
         {
            elvect(dof*k+s) += ip.weight * cf * shape(s);
         }
      }
   }
}

void VectorDomainLFIntegrator::AssembleDeltaElementVect(
   const FiniteElement &fe, ElementTransformation &Trans, Vector &elvect)
{
   MFEM_ASSERT(vec_delta != NULL, "coefficient must be VectorDeltaCoefficient");
   int vdim = Q.GetVDim();
   int dof  = fe.GetDof();

   shape.SetSize(dof);
   fe.CalcPhysShape(Trans, shape);

   vec_delta->EvalDelta(Qvec, Trans, Trans.GetIntPoint());

   elvect.SetSize(dof*vdim);
   DenseMatrix elvec_as_mat(elvect.GetData(), dof, vdim);
   MultVWt(shape, Qvec, elvec_as_mat);
}

void VectorBoundaryLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int vdim = Q.GetVDim();
   int dof  = el.GetDof();

   shape.SetSize(dof);
   vec.SetSize(vdim);

   elvect.SetSize(dof * vdim);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int intorder = 2*el.GetOrder();
      ir = &IntRules.Get(el.GetGeomType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Q.Eval(vec, Tr, ip);
      Tr.SetIntPoint (&ip);
      vec *= Tr.Weight() * ip.weight;
      el.CalcShape(ip, shape);
      for (int k = 0; k < vdim; k++)
         for (int s = 0; s < dof; s++)
         {
            elvect(dof*k+s) += vec(k) * shape(s);
         }
   }
}

void VectorBoundaryLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Tr, Vector &elvect)
{
   int vdim = Q.GetVDim();
   int dof  = el.GetDof();

   shape.SetSize(dof);
   vec.SetSize(vdim);

   elvect.SetSize(dof * vdim);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int intorder = 2*el.GetOrder();
      ir = &IntRules.Get(Tr.GetGeometryType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      // Set the integration point in the face and the neighboring element
      Tr.SetAllIntPoints(&ip);

      // Access the neighboring element's integration point
      const IntegrationPoint &eip = Tr.GetElement1IntPoint();

      // Use Tr transformation in case Q depends on boundary attribute
      Q.Eval(vec, Tr, ip);
      vec *= Tr.Weight() * ip.weight;
      el.CalcShape(eip, shape);
      for (int k = 0; k < vdim; k++)
      {
         for (int s = 0; s < dof; s++)
         {
            elvect(dof*k+s) += vec(k) * shape(s);
         }
      }
   }
}

void VectorFEDomainLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int dof = el.GetDof();
   int spaceDim = Tr.GetSpaceDim();

   vshape.SetSize(dof,spaceDim);
   vec.SetSize(spaceDim);

   elvect.SetSize(dof);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // int intorder = 2*el.GetOrder() - 1; // ok for O(h^{k+1}) conv. in L2
      int intorder = 2*el.GetOrder();
      ir = &IntRules.Get(el.GetGeomType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetIntPoint (&ip);
      el.CalcVShape(Tr, vshape);

      QF.Eval (vec, Tr, ip);
      vec *= ip.weight * Tr.Weight();
      vshape.AddMult (vec, elvect);
   }
}

void VectorFEDomainLFIntegrator::AssembleDeltaElementVect(
   const FiniteElement &fe, ElementTransformation &Trans, Vector &elvect)
{
   MFEM_ASSERT(vec_delta != NULL, "coefficient must be VectorDeltaCoefficient");
   int dof = fe.GetDof();
   int spaceDim = Trans.GetSpaceDim();

   vshape.SetSize(dof, spaceDim);
   fe.CalcPhysVShape(Trans, vshape);

   vec_delta->EvalDelta(vec, Trans, Trans.GetIntPoint());

   elvect.SetSize(dof);
   vshape.Mult(vec, elvect);
}

void VectorFEDomainLFCurlIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int dof = el.GetDof();
   int spaceDim = Tr.GetSpaceDim();
   int n=(spaceDim == 3)? spaceDim : 1;
   curlshape.SetSize(dof,n);
   vec.SetSize(n);

   elvect.SetSize(dof);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int intorder = 2*el.GetOrder();
      ir = &IntRules.Get(el.GetGeomType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetIntPoint (&ip);
      el.CalcPhysCurlShape(Tr, curlshape);
      QF->Eval(vec, Tr, ip);

      vec *= ip.weight * Tr.Weight();
      curlshape.AddMult (vec, elvect);
   }
}

void VectorFEDomainLFCurlIntegrator::AssembleDeltaElementVect(
   const FiniteElement &fe, ElementTransformation &Trans, Vector &elvect)
{
   int spaceDim = Trans.GetSpaceDim();
   MFEM_ASSERT(vec_delta != NULL,
               "coefficient must be VectorDeltaCoefficient");
   int dof = fe.GetDof();
   int n=(spaceDim == 3)? spaceDim : 1;
   vec.SetSize(n);
   curlshape.SetSize(dof, n);
   elvect.SetSize(dof);
   fe.CalcPhysCurlShape(Trans, curlshape);

   vec_delta->EvalDelta(vec, Trans, Trans.GetIntPoint());
   curlshape.Mult(vec, elvect);
}

void VectorFEDomainLFDivIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int dof = el.GetDof();

   divshape.SetSize(dof);       // vector of size dof
   elvect.SetSize(dof);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int intorder = 2 * el.GetOrder();
      ir = &IntRules.Get(el.GetGeomType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetIntPoint (&ip);
      double val = Tr.Weight() * Q.Eval(Tr, ip);
      el.CalcPhysDivShape(Tr, divshape);

      add(elvect, ip.weight * val, divshape, elvect);
   }
}

void VectorFEDomainLFDivIntegrator::AssembleDeltaElementVect(
   const FiniteElement &fe, ElementTransformation &Trans, Vector &elvect)
{
   MFEM_ASSERT(delta != NULL, "coefficient must be DeltaCoefficient");
   elvect.SetSize(fe.GetDof());
   fe.CalcPhysDivShape(Trans, elvect);
   elvect *= delta->EvalDelta(Trans, Trans.GetIntPoint());
}

void VectorBoundaryFluxLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int dim = el.GetDim()+1;
   int dof = el.GetDof();

   shape.SetSize (dof);
   nor.SetSize (dim);
   elvect.SetSize (dim*dof);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      ir = &IntRules.Get(el.GetGeomType(), el.GetOrder() + 1);
   }

   elvect = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Tr.SetIntPoint (&ip);
      CalcOrtho(Tr.Jacobian(), nor);
      el.CalcShape (ip, shape);
      nor *= Sign * ip.weight * F -> Eval (Tr, ip);
      for (int j = 0; j < dof; j++)
         for (int k = 0; k < dim; k++)
         {
            elvect(dof*k+j) += nor(k) * shape(j);
         }
   }
}


void VectorFEBoundaryFluxLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int dof = el.GetDof();

   shape.SetSize(dof);
   elvect.SetSize(dof);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int intorder = oa * el.GetOrder() + ob;  // <----------
      ir = &IntRules.Get(el.GetGeomType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      el.CalcShape(ip, shape);

      double val = ip.weight;
      if (F)
      {
         Tr.SetIntPoint (&ip);
         val *= F->Eval(Tr, ip);
      }

      elvect.Add(val, shape);
   }
}

void VectorFEBoundaryTangentLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int dof = el.GetDof();
   DenseMatrix vshape(dof, 2);
   Vector f_loc(3);
   Vector f_hat(2);

   elvect.SetSize(dof);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int intorder = oa * el.GetOrder() + ob;  // <----------
      ir = &IntRules.Get(el.GetGeomType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetIntPoint(&ip);
      f.Eval(f_loc, Tr, ip);
      Tr.Jacobian().MultTranspose(f_loc, f_hat);
      el.CalcVShape(ip, vshape);

      Swap<double>(f_hat(0), f_hat(1));
      f_hat(0) = -f_hat(0);
      f_hat *= ip.weight;
      vshape.AddMult(f_hat, elvect);
   }
}

void BoundaryFlowIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   mfem_error("BoundaryFlowIntegrator::AssembleRHSElementVect\n"
              "  is not implemented as boundary integrator!\n"
              "  Use LinearForm::AddBdrFaceIntegrator instead of\n"
              "  LinearForm::AddBoundaryIntegrator.");
}

void BoundaryFlowIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Tr, Vector &elvect)
{
   int dim, ndof, order;
   double un, w, vu_data[3], nor_data[3];

   dim  = el.GetDim();
   ndof = el.GetDof();
   Vector vu(vu_data, dim), nor(nor_data, dim);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // Assuming order(u)==order(mesh)
      order = Tr.Elem1->OrderW() + 2*el.GetOrder();
      if (el.Space() == FunctionSpace::Pk)
      {
         order++;
      }
      ir = &IntRules.Get(Tr.GetGeometryType(), order);
   }

   shape.SetSize(ndof);
   elvect.SetSize(ndof);
   elvect = 0.0;

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring element
      Tr.SetAllIntPoints(&ip);

      // Access the neighboring element's integration point
      const IntegrationPoint &eip = Tr.GetElement1IntPoint();
      el.CalcShape(eip, shape);

      // Use Tr.Elem1 transformation for u so that it matches the coefficient
      // used with the ConvectionIntegrator and/or the DGTraceIntegrator.
      u->Eval(vu, *Tr.Elem1, eip);

      if (dim == 1)
      {
         nor(0) = 2*eip.x - 1.0;
      }
      else
      {
         CalcOrtho(Tr.Jacobian(), nor);
      }

      un = vu * nor;
      w = 0.5*alpha*un - beta*fabs(un);
      w *= ip.weight*f->Eval(Tr, ip);
      elvect.Add(w, shape);
   }
}

void DGDirichletLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   mfem_error("DGDirichletLFIntegrator::AssembleRHSElementVect");
}

void DGDirichletLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Tr, Vector &elvect)
{
   int dim, ndof;
   bool kappa_is_nonzero = (kappa != 0.);
   double w;

   dim = el.GetDim();
   ndof = el.GetDof();

   nor.SetSize(dim);
   nh.SetSize(dim);
   ni.SetSize(dim);
   adjJ.SetSize(dim);
   if (MQ)
   {
      mq.SetSize(dim);
   }

   shape.SetSize(ndof);
   dshape.SetSize(ndof, dim);
   dshape_dn.SetSize(ndof);

   elvect.SetSize(ndof);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // a simple choice for the integration order; is this OK?
      int order = 2*el.GetOrder();
      ir = &IntRules.Get(Tr.GetGeometryType(), order);
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring element
      Tr.SetAllIntPoints(&ip);

      // Access the neighboring element's integration point
      const IntegrationPoint &eip = Tr.GetElement1IntPoint();

      if (dim == 1)
      {
         nor(0) = 2*eip.x - 1.0;
      }
      else
      {
         CalcOrtho(Tr.Jacobian(), nor);
      }

      el.CalcShape(eip, shape);
      el.CalcDShape(eip, dshape);

      // compute uD through the face transformation
      w = ip.weight * uD->Eval(Tr, ip) / Tr.Elem1->Weight();
      if (!MQ)
      {
         if (Q)
         {
            w *= Q->Eval(*Tr.Elem1, eip);
         }
         ni.Set(w, nor);
      }
      else
      {
         nh.Set(w, nor);
         MQ->Eval(mq, *Tr.Elem1, eip);
         mq.MultTranspose(nh, ni);
      }
      CalcAdjugate(Tr.Elem1->Jacobian(), adjJ);
      adjJ.Mult(ni, nh);

      dshape.Mult(nh, dshape_dn);
      elvect.Add(sigma, dshape_dn);

      if (kappa_is_nonzero)
      {
         elvect.Add(kappa*(ni*nor), shape);
      }
   }
}

void SBM2DirichletLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   mfem_error("SBM2DirichletLFIntegrator::AssembleRHSElementVect");
}

void SBM2DirichletLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Tr, Vector &elvect)
{
   int dim, ndof;
   double w;

   dim = el.GetDim();
   ndof = el.GetDof();

   nor.SetSize(dim);
   nh.SetSize(dim);
   ni.SetSize(dim);
   adjJ.SetSize(dim);


   shape.SetSize(ndof);
   dshape.SetSize(ndof, dim);
   dshape_dd.SetSize(ndof);
   dshape_dn.SetSize(ndof);

   elvect.SetSize(ndof);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // a simple choice for the integration order; is this OK?
      int order = 4*el.GetOrder();
      ir = &IntRules.Get(Tr.GetGeometryType(), order);
   }

   Array<DenseMatrix *> hess_ptr;
   DenseMatrix grad_phys;
   Vector Factorial;
   Array<DenseMatrix *> grad_phys_dir;

   if (nterms > 0)
   {
      if (elem1f)
      {
         el.ProjectGrad(el, *Tr.Elem1, grad_phys);
      }
      else
      {
         el.ProjectGrad(el, *Tr.Elem2, grad_phys);
      }

      DenseMatrix grad_work;
      grad_phys_dir.SetSize(dim); //NxN matrix for derivative in each direction
      for (int i = 0; i < dim; i++)
      {
         grad_phys_dir[i] = new DenseMatrix(ndof, ndof);
         grad_phys_dir[i]->CopyRows(grad_phys, i*ndof, (i+1)*ndof-1);
      }


      DenseMatrix grad_phys_work = grad_phys;
      grad_phys_work.SetSize(ndof, ndof*dim);

      hess_ptr.SetSize(nterms);

      for (int i = 0; i < nterms; i++)
      {
         int sz1 = pow(dim, i+1);
         hess_ptr[i] = new DenseMatrix(ndof, ndof*sz1*dim);
         int loc_col_per_dof = sz1;
         for (int k = 0; k < dim; k++)
         {
            grad_work.SetSize(ndof, ndof*sz1);
            // grad_work[k] has derivative in kth direction for each DOF.
            // grad_work[0] has d^2phi/dx^2 and d^2phi/dxdy terms and
            // grad_work[1] has d^2phi/dydx and d^2phi/dy2 terms for each dof
            if (i == 0)
            {
               Mult(*grad_phys_dir[k], grad_phys_work, grad_work);
            }
            else
            {
               Mult(*grad_phys_dir[k], *hess_ptr[i-1], grad_work);
            }
            // Now we must place columns for each dof together so that they are
            // in order: d^2phi/dx^2, d^2phi/dxdy, d^2phi/dydx, d^2phi/dy2.
            for (int j = 0; j < ndof; j++)
            {
               for (int d = 0; d < loc_col_per_dof; d++)
               {
                  Vector col;
                  int tot_col_per_dof = loc_col_per_dof*dim;
                  grad_work.GetColumn(j*loc_col_per_dof+d, col);
                  hess_ptr[i]->SetCol(j*tot_col_per_dof+k*loc_col_per_dof+d, col);
               }
            }
         }
      }

      for (int i = 0; i < grad_phys_dir.Size(); i++)
      {
         delete grad_phys_dir[i];
      }

      Factorial.SetSize(nterms);
      Factorial(0) = 2;
      for (int i = 1; i < nterms; i++)
      {
         Factorial(i) = Factorial(i-1)*(i+2);
      }
   }
   DenseMatrix q_hess_dn(dim, ndof);
   Vector q_hess_dn_work(q_hess_dn.GetData(), ndof*dim);
   Vector q_hess_dot_d(ndof);

   Vector D(vD->GetVDim());
   Vector wrk = shape;
   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring element
      Tr.SetAllIntPoints(&ip);

      // Access the neighboring element's integration point
      const IntegrationPoint &eip = Tr.GetElement1IntPoint();
      const IntegrationPoint &eip1 = Tr.GetElement1IntPoint();
      const IntegrationPoint &eip2 = Tr.GetElement2IntPoint();

      if (dim == 1)
      {
         nor(0) = 2*eip.x - 1.0;
      }
      else
      {
         CalcOrtho(Tr.Jacobian(), nor);
      }
      vD->Eval(D, Tr, ip);

      double nor_dot_d = nor*D;
      if (nor_dot_d < 0) { nor *= -1; }
      // note here that if we are clipping outside the domain, we will have to
      // flip the sign if nor_dot_d is +ve.

      double hinvdx;

      if (elem1f)
      {
         el.CalcShape(eip1, shape);
         el.CalcDShape(eip1, dshape);
         hinvdx =nor*nor/Tr.Elem1->Weight();
         w = ip.weight * uD->Eval(Tr, ip, D) / Tr.Elem1->Weight();
         CalcAdjugate(Tr.Elem1->Jacobian(), adjJ);
      }
      else
      {
         el.CalcShape(eip2, shape);
         el.CalcDShape(eip2, dshape);
         hinvdx = nor*nor/Tr.Elem2->Weight();
         w = ip.weight * uD->Eval(Tr, ip, D) / Tr.Elem2->Weight();
         CalcAdjugate(Tr.Elem2->Jacobian(), adjJ);
      }


      ni.Set(w, nor);
      adjJ.Mult(ni, nh);

      dshape.Mult(nh, dshape_dn);
      elvect.Add(-1., dshape_dn); //T2

      double jinv;
      if (elem1f)
      {
         w = ip.weight * uD->Eval(Tr, ip, D) * alpha * hinvdx;
         jinv = 1./Tr.Elem1->Weight();
      }
      else
      {
         w = ip.weight * uD->Eval(Tr, ip, D) * alpha * hinvdx;
         jinv = 1./Tr.Elem2->Weight();
      }
      adjJ.Mult(D, nh);
      nh *= jinv;
      dshape.Mult(nh, dshape_dd);

      q_hess_dot_d = 0.;
      for (int i = 0; i < nterms; i++)
      {
         int sz1 = pow(dim, i+1);
         DenseMatrix T1(dim, ndof*sz1);
         Vector T1_wrk(T1.GetData(), dim*ndof*sz1);
         hess_ptr[i]->MultTranspose(shape, T1_wrk);

         DenseMatrix T2;
         Vector T2_wrk;
         for (int j = 0; j < i+1; j++)
         {
            int sz2 = pow(dim, i-j);
            T2.SetSize(dim, ndof*sz2);
            T2_wrk.SetDataAndSize(T2.GetData(), dim*ndof*sz2);
            T1.MultTranspose(D, T2_wrk);
            T1 = T2;
         }
         Vector q_hess_dot_d_work(ndof);
         T1.MultTranspose(D, q_hess_dot_d_work);
         q_hess_dot_d_work *= 1./Factorial(i);
         q_hess_dot_d += q_hess_dot_d_work;
      }

      wrk = shape;
      wrk += dshape_dd; //\grad w .d
      wrk += q_hess_dot_d;
      elvect.Add(w, wrk); //<u, gradw.d>
   }
   for (int i = 0; i < hess_ptr.Size(); i++)
   {
      delete hess_ptr[i];
   }
}

void DGElasticityDirichletLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   mfem_error("DGElasticityDirichletLFIntegrator::AssembleRHSElementVect");
}

void DGElasticityDirichletLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Tr, Vector &elvect)
{
   MFEM_ASSERT(Tr.Elem2No < 0, "interior boundary is not supported");

#ifdef MFEM_THREAD_SAFE
   Vector shape;
   DenseMatrix dshape;
   DenseMatrix adjJ;
   DenseMatrix dshape_ps;
   Vector nor;
   Vector dshape_dn;
   Vector dshape_du;
   Vector u_dir;
#endif

   const int dim = el.GetDim();
   const int ndofs = el.GetDof();
   const int nvdofs = dim*ndofs;

   elvect.SetSize(nvdofs);
   elvect = 0.0;

   adjJ.SetSize(dim);
   shape.SetSize(ndofs);
   dshape.SetSize(ndofs, dim);
   dshape_ps.SetSize(ndofs, dim);
   nor.SetSize(dim);
   dshape_dn.SetSize(ndofs);
   dshape_du.SetSize(ndofs);
   u_dir.SetSize(dim);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      const int order = 2*el.GetOrder(); // <-----
      ir = &IntRules.Get(Tr.GetGeometryType(), order);
   }

   for (int pi = 0; pi < ir->GetNPoints(); ++pi)
   {
      const IntegrationPoint &ip = ir->IntPoint(pi);

      // Set the integration point in the face and the neighboring element
      Tr.SetAllIntPoints(&ip);

      // Access the neighboring element's integration point
      const IntegrationPoint &eip = Tr.GetElement1IntPoint();

      // Evaluate the Dirichlet b.c. using the face transformation.
      uD.Eval(u_dir, Tr, ip);

      el.CalcShape(eip, shape);
      el.CalcDShape(eip, dshape);

      CalcAdjugate(Tr.Elem1->Jacobian(), adjJ);
      Mult(dshape, adjJ, dshape_ps);

      if (dim == 1)
      {
         nor(0) = 2*eip.x - 1.0;
      }
      else
      {
         CalcOrtho(Tr.Jacobian(), nor);
      }

      double wL, wM, jcoef;
      {
         const double w = ip.weight / Tr.Elem1->Weight();
         wL = w * lambda->Eval(*Tr.Elem1, eip);
         wM = w * mu->Eval(*Tr.Elem1, eip);
         jcoef = kappa * (wL + 2.0*wM) * (nor*nor);
         dshape_ps.Mult(nor, dshape_dn);
         dshape_ps.Mult(u_dir, dshape_du);
      }

      // alpha < uD, (lambda div(v) I + mu (grad(v) + grad(v)^T)) . n > +
      //   + kappa < h^{-1} (lambda + 2 mu) uD, v >

      // i = idof + ndofs * im
      // v_phi(i,d) = delta(im,d) phi(idof)
      // div(v_phi(i)) = dphi(idof,im)
      // (grad(v_phi(i)))(k,l) = delta(im,k) dphi(idof,l)
      //
      // term 1:
      //   alpha < uD, lambda div(v_phi(i)) n >
      //   alpha lambda div(v_phi(i)) (uD.n) =
      //   alpha lambda dphi(idof,im) (uD.n) --> quadrature -->
      //   ip.weight/det(J1) alpha lambda (uD.nor) dshape_ps(idof,im) =
      //   alpha * wL * (u_dir*nor) * dshape_ps(idof,im)
      // term 2:
      //   < alpha uD, mu grad(v_phi(i)).n > =
      //   alpha mu uD^T grad(v_phi(i)) n =
      //   alpha mu uD(k) delta(im,k) dphi(idof,l) n(l) =
      //   alpha mu uD(im) dphi(idof,l) n(l) --> quadrature -->
      //   ip.weight/det(J1) alpha mu uD(im) dshape_ps(idof,l) nor(l) =
      //   alpha * wM * u_dir(im) * dshape_dn(idof)
      // term 3:
      //   < alpha uD, mu (grad(v_phi(i)))^T n > =
      //   alpha mu n^T grad(v_phi(i)) uD =
      //   alpha mu n(k) delta(im,k) dphi(idof,l) uD(l) =
      //   alpha mu n(im) dphi(idof,l) uD(l) --> quadrature -->
      //   ip.weight/det(J1) alpha mu nor(im) dshape_ps(idof,l) uD(l) =
      //   alpha * wM * nor(im) * dshape_du(idof)
      // term j:
      //   < kappa h^{-1} (lambda + 2 mu) uD, v_phi(i) > =
      //   kappa/h (lambda + 2 mu) uD(k) v_phi(i,k) =
      //   kappa/h (lambda + 2 mu) uD(k) delta(im,k) phi(idof) =
      //   kappa/h (lambda + 2 mu) uD(im) phi(idof) --> quadrature -->
      //      [ 1/h = |nor|/det(J1) ]
      //   ip.weight/det(J1) |nor|^2 kappa (lambda + 2 mu) uD(im) phi(idof) =
      //   jcoef * u_dir(im) * shape(idof)

      wM *= alpha;
      const double t1 = alpha * wL * (u_dir*nor);
      for (int im = 0, i = 0; im < dim; ++im)
      {
         const double t2 = wM * u_dir(im);
         const double t3 = wM * nor(im);
         const double tj = jcoef * u_dir(im);
         for (int idof = 0; idof < ndofs; ++idof, ++i)
         {
            elvect(i) += (t1*dshape_ps(idof,im) + t2*dshape_dn(idof) +
                          t3*dshape_du(idof) + tj*shape(idof));
         }
      }
   }
}

void VectorQuadratureLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &fe, ElementTransformation &Tr, Vector &elvect)
{
   const IntegrationRule *ir =
      &vqfc.GetQuadFunction().GetSpace()->GetElementIntRule(Tr.ElementNo);

   const int nqp = ir->GetNPoints();
   const int vdim = vqfc.GetVDim();
   const int ndofs = fe.GetDof();
   Vector shape(ndofs);
   Vector temp(vdim);
   elvect.SetSize(vdim * ndofs);
   elvect = 0.0;
   for (int q = 0; q < nqp; q++)
   {
      const IntegrationPoint &ip = ir->IntPoint(q);
      Tr.SetIntPoint(&ip);
      const double w = Tr.Weight() * ip.weight;
      vqfc.Eval(temp, Tr, ip);
      fe.CalcShape(ip, shape);
      for (int ind = 0; ind < vdim; ind++)
      {
         for (int nd = 0; nd < ndofs; nd++)
         {
            elvect(nd + ind * ndofs) += w * shape(nd) * temp(ind);
         }
      }
   }
}

void QuadratureLFIntegrator::AssembleRHSElementVect(const FiniteElement &fe,
                                                    ElementTransformation &Tr,
                                                    Vector &elvect)
{
   const IntegrationRule *ir =
      &qfc.GetQuadFunction().GetSpace()->GetElementIntRule(Tr.ElementNo);

   const int nqp = ir->GetNPoints();
   const int ndofs = fe.GetDof();
   Vector shape(ndofs);
   elvect.SetSize(ndofs);
   elvect = 0.0;
   for (int q = 0; q < nqp; q++)
   {
      const IntegrationPoint &ip = ir->IntPoint(q);
      Tr.SetIntPoint (&ip);
      const double w = Tr.Weight() * ip.weight;
      double temp = qfc.Eval(Tr, ip);
      fe.CalcShape(ip, shape);
      shape *= (w * temp);
      elvect += shape;
   }
}

}

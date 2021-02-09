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

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "fem.hpp"

namespace mfem
{

void ParLinearForm::Update(ParFiniteElementSpace *pf)
{
   if (pf) { pfes = pf; }
   LinearForm::Update(pfes);
}

void ParLinearForm::Update(ParFiniteElementSpace *pf, Vector &v, int v_offset)
{
   pfes = pf;
   LinearForm::Update(pf,v,v_offset);
}

void ParLinearForm::MakeRef(FiniteElementSpace *f, Vector &v, int v_offset)
{
   LinearForm::MakeRef(f, v, v_offset);
   pfes = dynamic_cast<ParFiniteElementSpace*>(f);
   MFEM_ASSERT(pfes != NULL, "not a ParFiniteElementSpace");
}

void ParLinearForm::MakeRef(ParFiniteElementSpace *pf, Vector &v, int v_offset)
{
   LinearForm::MakeRef(pf, v, v_offset);
   pfes = pf;
}

void ParLinearForm::Assemble()
{
   LinearForm::Assemble();

   if (sflfi.Size())
   {
      pfes->ExchangeFaceNbrData();
      AssembleSharedFaces();
   }
}

void ParLinearForm::AssembleSharedFaces()
{
   Array<int> vdofs;
   Vector elemvect;

   if (sflfi.Size())
   {
      ParMesh *pmesh = pfes->GetParMesh();
      for (int k = 0; k < sflfi.Size(); k++)
      {
         for (int i = 0; i < pmesh->GetNSharedFaces(); i++)
         {
            FaceElementTransformations *tr = NULL;
            tr = pmesh->GetSharedFaceTransformations(i);

            if (tr != NULL)
            {
               int ne1 = tr->Elem1No;
               int te1 = (*(sflfi_elem_marker[k]))[ne1];
               int te2 = (*(sflfi_elem_marker[k]))[i+pmesh->GetNE()];
               if (te2 == 2 && te1 == 0)
               {
                  fes -> GetElementVDofs (tr -> Elem1No, vdofs);
                  dynamic_cast<SBM2LFIntegrator *>(sflfi[k])->SetElem1Flag(true);
                  sflfi[0] -> AssembleRHSElementVect (*fes->GetFE(tr -> Elem1No),
                                                      *tr, elemvect);
                  AddElementVector (vdofs, elemvect);
               }
            }
         }
      }
   }
}

void ParLinearForm::ParallelAssemble(Vector &tv)
{
   const Operator* prolong = pfes->GetProlongationMatrix();
   prolong->MultTranspose(*this, tv);
}

HypreParVector *ParLinearForm::ParallelAssemble()
{
   HypreParVector *tv = pfes->NewTrueDofVector();
   const Operator* prolong = pfes->GetProlongationMatrix();
   prolong->MultTranspose(*this, *tv);
   return tv;
}

}

#endif

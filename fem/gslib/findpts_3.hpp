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

#ifndef MFEM_GSLIB_PA_HPP
#define MFEM_GSLIB_PA_HPP

#include "../../config/config.hpp"
#include "../../linalg/dtensor.hpp"

#include "../kernels.hpp"

#include <unordered_map>

namespace mfem
{

#define pDIM 3

struct findptsPt
{
   double x[pDIM], r[pDIM], oldr[pDIM], dist2, dist2p, tr;
   int flags;
};

struct findptsElemFace
{
   double *x[pDIM], *dxdn[pDIM];
};

struct findptsElemEdge
{
   double *x[pDIM], *dxdn1[pDIM], *dxdn2[pDIM], *d2xdn1[pDIM], *d2xdn2[pDIM];
};

struct findptsElemPt
{
   double x[pDIM], jac[pDIM * pDIM], hes[18];
};

struct dbl_range_t
{
   double min, max;
};

struct obbox_t
{
   double c0[pDIM], A[pDIM * pDIM];
   dbl_range_t x[pDIM];
};

struct findptsLocalHashData_t
{
   int hash_n;
   dbl_range_t bnd[pDIM];
   double fac[pDIM];
   unsigned int *offset;
   // int max;
};
#undef pdim

} // namespace mfem

#endif // MFEM_GSLIB_PA_HPP

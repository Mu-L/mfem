// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_TEMPLATE_CONFIG_QPX_HPP
#define MFEM_TEMPLATE_CONFIG_QPX_HPP

#include "builtins.h"

#define __ATTRS_ai __attribute__((__always_inline__))

template <typename,int,int=1> struct AutoSIMD;

#include "qpx64.hpp"

#include "qpx256.hpp"

#endif // MFEM_TEMPLATE_CONFIG_QPX_HPP

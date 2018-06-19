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

#ifndef MFEM_TEMPLATE_CONFIG_SIMD_M64
#define MFEM_TEMPLATE_CONFIG_SIMD_M64

// ****************************************************************************
template <typename scalar_t> struct AutoSIMD<scalar_t,1,1>
{
   typedef scalar_t scalar_type;
   static const int size = 1;
   static const int align_size = 8;

   scalar_t vec[size];

   scalar_t &operator[](int i) { return vec[0]; }
   const scalar_t &operator[](int i) const { return vec[0]; }

   AutoSIMD &operator=(const AutoSIMD &v)
   {
      vec[0] = v[0];
      return *this;
   }
   AutoSIMD &operator=(const scalar_t &e)
   {
      vec[0] = e;
      return *this;
   }
   AutoSIMD &operator+=(const AutoSIMD &v)
   {
      vec[0] += v[0];
      return *this;
   }
   AutoSIMD &operator+=(const scalar_t &e)
   {
      vec[0] += e;
      return *this;
   }
   AutoSIMD &operator-=(const AutoSIMD &v)
   {
      vec[0] -= v[0];
      return *this;
   }
   AutoSIMD &operator-=(const scalar_t &e)
   {
      vec[0] -= e;
      return *this;
   }
   AutoSIMD &operator*=(const AutoSIMD &v)
   {
      vec[0] *= v[0];
      return *this;
   }
   AutoSIMD &operator*=(const scalar_t &e)
   {
      vec[0] *= e;
      return *this;
   }
   AutoSIMD &operator/=(const AutoSIMD &v)
   {
      vec[0] /= v[0];
      return *this;
   }
   AutoSIMD &operator/=(const scalar_t &e)
   {
      vec[0] /= e;
      return *this;
   }

   AutoSIMD operator-() const
   {
      AutoSIMD r;
      r[0] = -vec[0];
      return r;
   }

   AutoSIMD operator+(const AutoSIMD &v) const
   {
      AutoSIMD r;
      r[0] = vec[0] + v[0];
      return r;
   }
   AutoSIMD operator+(const scalar_t &e) const
   {
      AutoSIMD r;
      r[0] = vec[0] + e;
      return r;
   }
   AutoSIMD operator-(const AutoSIMD &v) const
   {
      AutoSIMD r;
      r[0] = vec[0] - v[0];
      return r;
   }
   AutoSIMD operator-(const scalar_t &e) const
   {
      AutoSIMD r;
      r[0] = vec[0] - e;
      return r;
   }
   AutoSIMD operator*(const AutoSIMD &v) const
   {
      AutoSIMD r;
      r[0] = vec[0] * v[0];
      return r;
   }
   AutoSIMD operator*(const scalar_t &e) const
   {
      AutoSIMD r;
      r[0] = vec[0] * e;
      return r;
   }
   AutoSIMD operator/(const AutoSIMD &v) const
   {
      AutoSIMD r;
      r[0] = vec[0] / v[0];
      return r;
   }
   AutoSIMD operator/(const scalar_t &e) const
   {
      AutoSIMD r;
      r[0] = vec[0] / e;
      return r;
   }

   AutoSIMD &fma(const AutoSIMD &v, const AutoSIMD &w)
   {
      vec[0] += v[0] * w[0];
      return *this;
   }
   AutoSIMD &fma(const AutoSIMD &v, const scalar_t &e)
   {
      vec[0] += v[0] * e;
      return *this;
   }
   AutoSIMD &fma(const scalar_t &e, const AutoSIMD &v)
   {
      vec[0] += e * v[0];
      return *this;
   }

   AutoSIMD &mul(const AutoSIMD &v, const AutoSIMD &w)
   {
      vec[0] = v[0] * w[0];
      return *this;
   }
   AutoSIMD &mul(const AutoSIMD &v, const scalar_t &e)
   {
      vec[0] = v[0] * e;
      return *this;
   }
   AutoSIMD &mul(const scalar_t &e, const AutoSIMD &v)
   {
      vec[0] = e * v[0];
      return *this;
   }
};

// *****************************************************************************
template <typename scalar_t>
AutoSIMD<scalar_t,1,1> operator+(const scalar_t &e,
                                 const AutoSIMD<scalar_t,1,1> &v)
{
   AutoSIMD<scalar_t,1,1> r;
   MFEM_VECTORIZE_LOOP
   r[0] = e + v[0];
   return r;
}

// *****************************************************************************
template <typename scalar_t>
AutoSIMD<scalar_t,1,1> operator-(const scalar_t &e,
                                 const AutoSIMD<scalar_t,1,1> &v)
{
   AutoSIMD<scalar_t,1,1> r;
   r[0] = e - v[0];
   return r;
}

// *****************************************************************************
template <typename scalar_t>
AutoSIMD<scalar_t,1,1> operator*(const scalar_t &e,
                                 const AutoSIMD<scalar_t,1,1> &v)
{
   AutoSIMD<scalar_t,1,1> r;
   r[0] = e * v[0];
   return r;
}

// *****************************************************************************
template <typename scalar_t>
AutoSIMD<scalar_t,1,1> operator/(const scalar_t &e,
                                 const AutoSIMD<scalar_t,1,1> &v)
{
   AutoSIMD<scalar_t,1,1> r;
   r[0] = e / v[0];
   return r;
}

#endif // MFEM_TEMPLATE_CONFIG_SIMD_M64

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

#ifndef MFEM_KERNELDISPATCH_HPP
#define MFEM_KERNELDISPATCH_HPP

#include "../general/error.hpp"

#include <functional>
#include <unordered_map>
#include <tuple>
#include <cmath>

namespace mfem
{
namespace internal
{
template<typename... Types>
struct KernelTypeList {};
}

constexpr int ipow(int x, int p) { return p == 0 ? 1 : x*ipow(x, p-1); }

template<typename... T>
class ApplyPAKernelsClassTemplate {};

template<typename... T>
class DiagonalPAKernelsClassTemplate {};

template<typename Signature, typename... UserParams, typename... KernelParams>
class ApplyPAKernelsClassTemplate<Signature, internal::KernelTypeList<UserParams...>, internal::KernelTypeList<KernelParams...>>
{
private:
   constexpr static int D(int D1D) { return (11 - D1D) / 2; }
public:
   using KernelSignature = Signature;

   constexpr static int NBZ(int D1D, int Q1D)
   {
      return ipow(2, D(D1D) >= 0 ? D(D1D) : 0);
   }

   static KernelSignature Kernel1D();

   template<KernelParams... params>
   static KernelSignature Kernel2D();

   template<KernelParams... params>
   static KernelSignature Kernel3D();

   static KernelSignature Fallback2D();

   static KernelSignature Fallback3D();
};

template<typename Signature, typename... UserParams, typename... KernelParams>
class DiagonalPAKernelsClassTemplate<Signature, internal::KernelTypeList<UserParams...>, internal::KernelTypeList<KernelParams...>>
{
public:
   using KernelSignature = Signature;
   using KernelArgTypes2D = internal::KernelTypeList<UserParams...>;
   using KernelArgTypes3D = internal::KernelTypeList<KernelParams...>;

   static KernelSignature Kernel1D();

   template<KernelParams... params>
   static KernelSignature Kernel2D();

   template<KernelParams... params>
   static KernelSignature Kernel3D();

   static KernelSignature Fallback2D();

   static KernelSignature Fallback3D();
};

template <typename KeyType, typename KernelType, typename HashFunction>
class DispatchTable
{
protected:
   std::unordered_map<KeyType, KernelType, HashFunction> table;
};


template<typename ...KernelParameters>
struct KernelDispatchKeyHash
{
   using Tuple = std::tuple<KernelParameters...>;

private:
   template<int N>
   size_t operator()(Tuple value) const { return 0; }

   template<std::size_t N, typename THead, typename... TTail>
   size_t operator()(Tuple value) const
   {
      constexpr int Index = N - sizeof...(TTail) - 1;
      auto lhs_hash = std::hash<THead>()(std::get<Index>(value));
      auto rhs_hash = operator()<N, TTail...>(value);
      return lhs_hash ^(rhs_hash + 0x9e3779b9 + (lhs_hash << 6) + (lhs_hash >> 2));
   }

public:
   size_t operator()(Tuple value) const
   {
      return operator()<sizeof...(KernelParameters), KernelParameters...>(value);
   }
};

template<typename... T>
class KernelDispatchTable
{

};

template <typename ApplyKernelsHelperClass, typename... UserParams, typename... KernelParams>
class KernelDispatchTable<ApplyKernelsHelperClass, internal::KernelTypeList<UserParams...>, internal::KernelTypeList<KernelParams...>> :
         DispatchTable<std::tuple<int, UserParams...>, typename ApplyKernelsHelperClass::KernelSignature, KernelDispatchKeyHash<int, UserParams...>>
{

   // These typedefs prevent AddSpecialization from compiling unless the provided
   // kernel parameters match the kernel parameters specified to ApplyKernelsHelperClass.
   using Signature = typename ApplyKernelsHelperClass::KernelSignature;

private:
   // If the type U has member U::NBZ, this overload will be selected, and will
   // return U::NBZ(d1d, q1d).
   template <typename U>
   static constexpr int GetNBZ_(int d1d, int q1d, decltype(U::NBZ(0,0),nullptr))
   {
      return U::NBZ(d1d, q1d);
   }

   // If the type U does not have member U::NBZ, this "fallback" overload will
   // be selected
   template <typename U> static constexpr int GetNBZ_(int d1d, int q1d, ...)
   {
      return 0;
   }

   // Return T::NBZ(d1d, q1d) if T::NBZ is defined, 0 otherwise.
   static constexpr int GetNBZ(int d1d, int q1d)
   { return GetNBZ_<ApplyKernelsHelperClass>(d1d, q1d, nullptr); }

public:

   // TODO(bowen) Force this to use the same signature as the Signature typedef
   // above.
   template<typename... KernelArgs>
   void Run1D(UserParams... params, KernelArgs&... args)
   {
      std::tuple<int, UserParams...> key;
      std::get<0>(key) = 1;
      const auto it = this->table.find(key);
      if (it != this->table.end())
      {
         printf("Specialized.\n");
         it->second(args...);
      }
      else
      {
         MFEM_ABORT("1 dimensional kernel not registered.  This is an internal MFEM error.")
      }
   }

   // TODO(bowen) Force this to use the same signature as the Signature typedef
   // above.
   template<typename... KernelArgs>
   void Run2D(UserParams... params, KernelArgs&... args)
   {
      auto key = std::make_tuple(2, params...);
      const auto it = this->table.find(key);
      if (it != this->table.end())
      {
         printf("Specialized.\n");
         it->second(args...);
      }
      else
      {
         printf("falling back.\n");
         ApplyKernelsHelperClass::Fallback2D()(args...);
      }
   }

   template<typename... KernelArgs>
   void Run3D(UserParams... params, KernelArgs&... args)
   {
      auto key = std::make_tuple(3, params...);
      const auto it = this->table.find(key);
      if (it != this->table.end())
      {
         printf("Specialized.\n");
         it->second(args...);
      }
      else
      {
         printf("falling back.\n");
         ApplyKernelsHelperClass::Fallback3D()(args...);
      }
   }

   template<typename... KernelArgs>
   void Run(int dim, UserParams... params, KernelArgs&... args)
   {
      if (dim == 1)
      {
         Run1D(params..., args...);
      }
      else if (dim == 2)
      {
         Run2D(params..., args...);
      }
      else if (dim == 3)
      {
         Run3D(params..., args...);
      }
      else
      {
         MFEM_ABORT("Only 2 and 3 dimensional kernels exist");
      }

   }
   using SpecializedTableType =
      KernelDispatchTable<ApplyKernelsHelperClass, internal::KernelTypeList<UserParams...>, internal::KernelTypeList<KernelParams...>>;

   template<UserParams... params>
   struct AddSpecialization1D
   {
      void operator()(SpecializedTableType* table_ptr)
      {
         constexpr int DIM = 1;
         std::tuple<int, UserParams...> param_tuple  (DIM, params...);
         table_ptr->table[param_tuple] = ApplyKernelsHelperClass::Kernel1D();
      }
   };

   template<typename ... args>
   static constexpr int getD1D(int D1D, args...)
   {
      return D1D;
   }
   template<typename ... args>
   static constexpr int getQ1D(int /*D1D*/, int Q1D, args...)
   {
      return Q1D;
   }
   /// Functors are needed here instead of functions because of a bug in GCC where a variadic
   /// type template cannot be used to define a parameter pack.
   template<UserParams... params>
   struct AddSpecialization2D
   {
      void operator()(SpecializedTableType* table_ptr)
      {
         constexpr int DIM = 2;
         std::tuple<int, UserParams...> param_tuple (DIM, params...);

         // All kernels require at least D1D and Q1D, which are listed first in a
         // parameter pack.
         constexpr int D1D = getD1D(params...);
         constexpr int Q1D = getQ1D(params...);
         constexpr int NBZ = GetNBZ(D1D, Q1D);

         table_ptr->table[param_tuple] = ApplyKernelsHelperClass::template
                                         Kernel2D<params..., NBZ>();
      }
   };

   template<UserParams... params>
   struct AddSpecialization3D
   {
      void operator()(SpecializedTableType* table_ptr)
      {
         constexpr int DIM = 3;
         std::tuple<int, UserParams...> param_tuple (DIM, params...);
         table_ptr->table[param_tuple] = ApplyKernelsHelperClass::template
                                         Kernel3D<params...>();
      }
   };

};

}

#endif

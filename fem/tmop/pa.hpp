// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_TMOP_PA_HPP
#define MFEM_TMOP_PA_HPP

#include "../kernel_dispatch.hpp"
#include "../../linalg/dinvariants.hpp"

namespace mfem
{

/// Abstract base class for the 2D metric TMOP PA kernels.
struct TMOP_PA_Metric_2D
{
   static constexpr int DIM = 2;
   using Args = kernels::InvariantsEvaluator2D::Buffers;

   virtual MFEM_HOST_DEVICE real_t EvalW(const real_t (&Jpt)[DIM * DIM],
                                         const real_t *w) = 0;

   virtual MFEM_HOST_DEVICE void EvalP(const real_t (&Jpt)[DIM * DIM],
                                       const real_t *w,
                                       real_t (&P)[DIM * DIM]) = 0;

   virtual MFEM_HOST_DEVICE void AssembleH(const int qx, const int qy,
                                           const int e, const real_t weight,
                                           const real_t (&Jpt)[DIM * DIM],
                                           const real_t *w,
                                           const DeviceTensor<7> &H) = 0;
};

/// Abstract base class for the 3D metric TMOP PA kernels.
struct TMOP_PA_Metric_3D
{
   static constexpr int DIM = 3;
   using Args = kernels::InvariantsEvaluator3D::Buffers;

   virtual MFEM_HOST_DEVICE real_t EvalW(const real_t (&Jpt)[DIM * DIM],
                                         const real_t *w) = 0;

   virtual MFEM_HOST_DEVICE void EvalP(const real_t (&Jpt)[DIM * DIM],
                                       const real_t *w,
                                       real_t (&P)[DIM * DIM]) = 0;

   virtual MFEM_HOST_DEVICE void AssembleH(const int qx, const int qy,
                                           const int qz, const int e, const real_t weight,
                                           real_t *Jrt, real_t *Jpr,
                                           const real_t (&Jpt)[DIM * DIM],
                                           const real_t *w,
                                           const DeviceTensor<5 + DIM> &H) const = 0;
};

namespace tmop
{

template <typename T>
using TMOPFunction = void (*)(T &);

template <int Metric, typename Ker>
void Kernel(Ker &);

// Max & two templated arguments: MD1, MQ1, T_D1D, T_Q1D
#define MFEM_TMOP_MDQ_REGISTER(Name, Ker)             \
   using Ker##_t = decltype(&Ker<1,1,1,1>);           \
   MFEM_REGISTER_KERNELS(Name, Ker##_t, (int, int));  \
   template <int D, int Q>                            \
   Ker##_t Name::Kernel() { return Ker<D,Q, D,Q>; }   \
   Ker##_t Name::Fallback(int, int) {                 \
      return Ker<DofQuadLimits::MAX_D1D,              \
                 DofQuadLimits::MAX_Q1D>; }

template <typename Kernel>
int KernelSpecializationsMDQ()
{
   Kernel::template Specialization<2, 2>::Add();
   Kernel::template Specialization<2, 3>::Add();
   Kernel::template Specialization<2, 4>::Add();
   Kernel::template Specialization<2, 5>::Add();
   Kernel::template Specialization<2, 6>::Add();

   Kernel::template Specialization<3, 3>::Add();
   Kernel::template Specialization<3, 4>::Add();
   Kernel::template Specialization<3, 5>::Add();
   Kernel::template Specialization<3, 6>::Add();

   Kernel::template Specialization<4, 4>::Add();
   Kernel::template Specialization<4, 5>::Add();
   Kernel::template Specialization<4, 6>::Add();

   Kernel::template Specialization<5, 5>::Add();
   Kernel::template Specialization<5, 6>::Add();
   Kernel::template Specialization<6, 6>::Add();
   return 0;
}
#define MFEM_TMOP_MDQ_SPECIALIZE(Name)                                    \
   namespace                                                              \
   {                                                                      \
   static bool k##Name{ (tmop::KernelSpecializationsMDQ<Name>(), true) }; \
   }

// Two templated arguments: T_D1D, T_Q1D
#define MFEM_TMOP_REGISTER_KERNELS(Name, Ker)         \
   using Ker##_t = decltype(&Ker<>);                  \
   MFEM_REGISTER_KERNELS(Name, Ker##_t, (int, int));  \
   template <int D, int Q>                            \
   Ker##_t Name::Kernel()                             \
   {                                                  \
      return Ker<D, Q>;                               \
   }                                                  \
   Ker##_t Name::Fallback(int, int) { return Ker<>; }

template <typename Kernel>
int KernelSpecializations()
{
   Kernel::template Specialization<2, 2>::Add();
   Kernel::template Specialization<2, 3>::Add();
   Kernel::template Specialization<2, 4>::Add();
   Kernel::template Specialization<2, 5>::Add();
   Kernel::template Specialization<2, 6>::Add();

   Kernel::template Specialization<3, 3>::Add();
   Kernel::template Specialization<3, 4>::Add();
   Kernel::template Specialization<3, 5>::Add();
   Kernel::template Specialization<3, 6>::Add();

   Kernel::template Specialization<4, 4>::Add();
   Kernel::template Specialization<4, 5>::Add();
   Kernel::template Specialization<4, 6>::Add();

   Kernel::template Specialization<5, 5>::Add();
   Kernel::template Specialization<5, 6>::Add();
   Kernel::template Specialization<6, 6>::Add();
   return 0;
}

#define MFEM_TMOP_ADD_SPECIALIZED_KERNELS(Name)                        \
   namespace                                                           \
   {                                                                   \
   static bool k##Name{ (tmop::KernelSpecializations<Name>(), true) }; \
   }

// A single templated argument: T_Q1D
#define MFEM_TMOP_REGISTER_KERNELS_1(Name, Ker)  \
   using Ker##_t = decltype(&Ker<>);             \
   MFEM_REGISTER_KERNELS(Name, Ker##_t, (int));  \
   template <int Q>                              \
   Ker##_t Name::Kernel()                        \
   {                                             \
      return Ker<Q>;                             \
   }                                             \
   Ker##_t Name::Fallback(int) { return Ker<>; }

template <typename Kernel>
int KernelSpecializations1()
{
   Kernel::template Specialization<2>::Add();
   Kernel::template Specialization<3>::Add();
   Kernel::template Specialization<4>::Add();
   Kernel::template Specialization<5>::Add();
   Kernel::template Specialization<6>::Add();
   return 0;
}

#define MFEM_TMOP_ADD_SPECIALIZED_KERNELS_1(Name)                       \
   namespace                                                            \
   {                                                                    \
   static bool k##Name{ (tmop::KernelSpecializations1<Name>(), true) }; \
   }

// Register kernel instances for a given metric, assemble, energy, and mult
#define MFEM_TMOP_REGISTER_METRIC_INSTANCE(i, Metric, Name)               \
   using Name##_t = tmop::TMOPFunction<Name>;                             \
   MFEM_REGISTER_KERNELS(Name##_##i, Name##_t, (int, int));               \
   MFEM_TMOP_MDQ_SPECIALIZE(Name##_##i);                                  \
   template <int D, int Q>                                                \
   Name##_t Name##_##i::Kernel()                                          \
   {                                                                      \
      return Name::Mult<D, Q, Metric, D, Q>;                              \
   }                                                                      \
   Name##_t Name##_##i::Fallback(int, int) {                              \
      return Name::Mult<DofQuadLimits::MAX_D1D,                           \
                        DofQuadLimits::MAX_Q1D, Metric>; }                \
   template <>                                                            \
   void tmop::Kernel<i>(Name & ker)                                       \
   {                                                                      \
      Name##_##i::Run(ker.Ndof(), ker.Nqpt(), ker);                       \
   }

#define MFEM_TMOP_REGISTER_METRIC(metric, assemble, energy, mult, i) \
   MFEM_TMOP_REGISTER_METRIC_INSTANCE(i, metric, assemble)           \
   MFEM_TMOP_REGISTER_METRIC_INSTANCE(i, metric, energy)             \
   MFEM_TMOP_REGISTER_METRIC_INSTANCE(i, metric, mult)

template <typename F> inline MFEM_HOST_DEVICE
void foreach_x_thread(const int N, F&& func)
{
#if ((defined(MFEM_USE_CUDA) && defined(__CUDA_ARCH__)) || \
     (defined(MFEM_USE_HIP)  && defined(__HIP_DEVICE_COMPILE__)))
   if (MFEM_THREAD_ID(x) < N) { func(MFEM_THREAD_ID(x)); }
#else
   for (int i = 0; i < N; ++i) { func(i); }
#endif
}

template <typename F> inline MFEM_HOST_DEVICE
void foreach_y_thread(const int N, F&& func)
{
#if ((defined(MFEM_USE_CUDA) && defined(__CUDA_ARCH__)) || \
     (defined(MFEM_USE_HIP)  && defined(__HIP_DEVICE_COMPILE__)))
   if (MFEM_THREAD_ID(y) < N) { func(MFEM_THREAD_ID(y)); }
#else
   for (int i = 0; i < N; ++i) { func(i); }
#endif
}

} // namespace tmop

} // namespace mfem

#endif // MFEM_TMOP_PA_HPP

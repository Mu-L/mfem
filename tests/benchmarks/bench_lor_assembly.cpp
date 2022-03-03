// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include <cmath>
#include <cassert>

#include "bench.hpp"

#ifdef MFEM_USE_BENCHMARK

#include "fem/lor/lor.hpp"

#define MFEM_DEBUG_COLOR 119
#include "general/debug.hpp"


struct LORBench
{
   static constexpr double EPS = 1e-14;
   static constexpr int RANDOM_SEED = 0x100001b3;
   GeometricFactors::FactorFlags DETERMINANTS = GeometricFactors::DETERMINANTS;
   const int p, c, q, n, nx, ny, nz, dim = 3;
   const bool check_x, check_y, check_z, checked;
   IntegrationRules irs;
   const IntegrationRule *ir;
   Mesh mesh;
   H1_FECollection fec;
   FiniteElementSpace fes_mesh;
   GridFunction mesh_coords;
   const bool postfix_mesh_coords;
   FiniteElementSpace fes_ho;
   Array<int> ess_tdofs;
   const bool prefix_lor_setup;
   BilinearForm a;
   ConstantCoefficient diff_coeff, mass_coeff;
   const bool a_ho_setup;
   LORDiscretization lor;
   FiniteElementSpace &fes_lor;
   BilinearForm a_lor_legacy, a_lor_full;
   OperatorHandle A_lor_legacy, A_lor_full;
   const int nvdofs;
   double mdof;
   Vector x, y;

   LORBench(int p, int side):
      p(p),
      c(side),
      q(2*p + 2),
      n((assert(c>=p),c/p)),
      nx(n + (p*(n+1)*p*n*p*n < c*c*c ?1:0)),
      ny(n + (p*(n+1)*p*(n+1)*p*n < c*c*c ?1:0)),
      nz(n),
      check_x(p*nx * p*ny * p*nz <= c*c*c),
      check_y(p*(nx+1) * p*(ny+1) * p*nz > c*c*c),
      check_z(p*(nx+1) * p*(ny+1) * p*(nz+1) > c*c*c),
      checked((assert(check_x && check_y && check_z), true)),
      irs(0, Quadrature1D::GaussLobatto),
      ir(&irs.Get(Geometry::CUBE, 1)),
      mesh(Mesh::MakeCartesian3D(nx,ny,nz, Element::HEXAHEDRON)),
      fec(p, dim),
      fes_mesh(&mesh, &fec, dim),
      mesh_coords(&fes_mesh),
      postfix_mesh_coords((SetupRandomMesh(), true)),
      fes_ho(&mesh, &fec),
      prefix_lor_setup((fes_ho.GetBoundaryTrueDofs(ess_tdofs), true)),
      a(&fes_ho),
      diff_coeff(M_PI),
      mass_coeff(1.0/M_PI),
      a_ho_setup((a.AddDomainIntegrator(new DiffusionIntegrator(diff_coeff)),
                  a.AddDomainIntegrator(new MassIntegrator(mass_coeff)),
                  true)),
      lor(a, ess_tdofs),
      fes_lor(lor.GetFESpace()),
      a_lor_legacy(&fes_lor),
      a_lor_full(&fes_lor),
      A_lor_legacy(),
      A_lor_full(),
      nvdofs(fes_ho.GetVSize()),
      mdof(0.0),
      x(nvdofs),
      y(nvdofs)
   {
      a_lor_legacy.AddDomainIntegrator(new DiffusionIntegrator(diff_coeff, ir));
      a_lor_legacy.AddDomainIntegrator(new MassIntegrator(mass_coeff, ir));
      a_lor_legacy.SetAssemblyLevel(AssemblyLevel::LEGACY);

      a_lor_full.AddDomainIntegrator(new DiffusionIntegrator(diff_coeff, ir));
      a_lor_full.AddDomainIntegrator(new MassIntegrator(mass_coeff, ir));
      a_lor_full.SetAssemblyLevel(AssemblyLevel::FULL);

      assert(fes_lor.GetMesh()); // will be built on request
      Mesh *lmm = fes_lor.GetMesh();
      auto *lmf = lmm->GetGeometricFactors(*ir, DETERMINANTS);
      const double lor_mesh_detJ_min = lmf->detJ.Min();
      assert(lor_mesh_detJ_min > 0.0);

      x.Randomize(RANDOM_SEED);
      y.Randomize(RANDOM_SEED);

      tic_toc.Clear();
   }

   void SetupRandomMesh() noexcept
   {
      mesh.SetNodalFESpace(&fes_mesh);
      mesh.SetNodalGridFunction(&mesh_coords);
      const double jitter = 1e-2/(M_PI*M_PI);
      const double h0 = mesh.GetElementSize(0);
      GridFunction rdm(&fes_mesh);
      rdm.Randomize(RANDOM_SEED);
      rdm -= 0.5; // Shift to random values in [-0.5,0.5]
      rdm *= jitter * h0; // Scale the random values to be of same order
      mesh_coords -= rdm;
      assert(mesh.GetGeometricFactors(*ir, DETERMINANTS)->detJ.Min() > 0.0);
   }

   void SanityChecks()
   {
      a_lor_legacy = 0.0;
      MFEM_DEVICE_SYNC;
      tic();
      a_lor_legacy.Assemble();
      a_lor_legacy.EliminateVDofs(ess_tdofs, Operator::DIAG_KEEP);
      MFEM_DEVICE_SYNC;
      SparseMatrix &A_lor_legacy_sp = a_lor_legacy.SpMat();
      const double dot_legacy = A_lor_legacy_sp.InnerProduct(x,y);

      MFEM_DEVICE_SYNC;
      tic();
      a_lor_full.Assemble();
      a_lor_full.EliminateVDofs(ess_tdofs, Operator::DIAG_KEEP);
      MFEM_DEVICE_SYNC;
      SparseMatrix &A_lor_full_sp = a_lor_full.SpMat();
      const double dot_full = A_lor_full_sp.InnerProduct(x,y);
      MFEM_VERIFY(almost_equal(dot_legacy, dot_full), "dot_full error!");
      A_lor_full_sp.Add(-1.0, A_lor_legacy_sp);
      const double max_norm_full = a_lor_full.SpMat().MaxNorm();
      MFEM_VERIFY(max_norm_full < EPS, "max_norm_full error!");

      MFEM_DEVICE_SYNC;
      tic();
      lor.AssembleSystem(a, ess_tdofs);
      MFEM_DEVICE_SYNC;
      SparseMatrix &A_lor_batched_sp = lor.GetAssembledMatrix();
      const double dot_batched = A_lor_batched_sp.InnerProduct(x,y);
      MFEM_VERIFY(almost_equal(dot_legacy, dot_batched), "dot_batched error!");
      A_lor_batched_sp.Add(-1.0, A_lor_legacy_sp);
      const double max_norm_batched = A_lor_batched_sp.MaxNorm();
      MFEM_VERIFY(max_norm_batched < EPS, "max_norm_batched");
   }

   void LORLegacy()
   {
      MFEM_DEVICE_SYNC;
      tic_toc.Start();
      a_lor_legacy.Assemble();
      a_lor_legacy.EliminateVDofs(ess_tdofs, Operator::DIAG_KEEP);
      MFEM_DEVICE_SYNC;
      tic_toc.Stop();
      mdof += 1e-6 * nvdofs;
   }

   void LORFull()
   {
      MFEM_DEVICE_SYNC;
      tic_toc.Start();
      a_lor_full.Assemble();
      a_lor_full.EliminateVDofs(ess_tdofs, Operator::DIAG_KEEP);
      MFEM_DEVICE_SYNC;
      tic_toc.Stop();
      mdof += 1e-6 * nvdofs;
   }

   void LORBatched()
   {
      MFEM_DEVICE_SYNC;
      tic_toc.Start();
      lor.AssembleSystem(a, ess_tdofs);
      MFEM_DEVICE_SYNC;
      tic_toc.Stop();
      mdof += 1e-6 * nvdofs;
   }

   double Mdofs() const { return mdof / tic_toc.RealTime(); }
};

// The different orders the tests can run
#define P_ORDERS bm::CreateDenseRange(1,5,2)

// The different sides of the mesh
#define N_SIDES bm::CreateDenseRange(5,120,5)
#define MAX_NDOFS 2*1024*1024

/// Kernels definitions and registrations
#define Benchmark(Name)\
static void Name(bm::State &state){\
   const int p = state.range(0);\
   const int side = state.range(1);\
   LORBench lor(p, side);\
   if (lor.nvdofs > MAX_NDOFS) { state.SkipWithError("MAX_NDOFS"); }\
   while (state.KeepRunning()) { lor.Name(); }\
   bm::Counter::Flags flags = bm::Counter::kIsIterationInvariantRate;\
   state.counters["Dofs/s"] = bm::Counter(lor.nvdofs, flags);\
   state.counters["dofs"] = bm::Counter(lor.nvdofs);\
   state.counters["p"] = bm::Counter(p);\
}\
BENCHMARK(Name)\
            -> ArgsProduct({P_ORDERS,N_SIDES})\
            -> Unit(bm::kMillisecond)\
            ->Iterations(10);

Benchmark(SanityChecks)

Benchmark(LORLegacy)
Benchmark(LORFull)
Benchmark(LORBatched)

/**
 * @brief main entry point
 * --benchmark_filter=SanityChecks/3/16
 * --benchmark_filter=\(Batched\|Deviced\|Full\)/4/16
 * --benchmark_context=device=cuda
 */
int main(int argc, char *argv[])
{
   bm::ConsoleReporter CR;
   bm::Initialize(&argc, argv);

   // Device setup, cpu by default
   std::string device_config = "cpu";
   if (bmi::global_context != nullptr)
   {
      const auto device = bmi::global_context->find("device");
      if (device != bmi::global_context->end())
      {
         mfem::out << device->first << " : " << device->second << std::endl;
         device_config = device->second;
      }
   }
   Device device(device_config.c_str());
   device.Print();

   if (bm::ReportUnrecognizedArguments(argc, argv)) { return 1; }
   bm::RunSpecifiedBenchmarks(&CR);
   return 0;
}

#endif // MFEM_USE_BENCHMARK

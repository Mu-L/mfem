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
//
// Description:  This miniapp compares various linear solvers for the saddle
//               point system obtained from mixed finite element discretization
//               of the simple mixed Darcy problem in ex5p
//                                 k*u + grad p = f
//                                 - div u      = g
//               with natural boundary condition -p = <given pressure>.
//               Here, we use a given exact solution (u,p) and compute the
//               corresponding r.h.s. (f,g).  We discretize with Raviart-Thomas
//               finite elements (velocity u) and piecewise discontinuous
//               polynomials (pressure p).
//
//               The solvers being compared include:
//                 1. MINRES preconditioned by a block diagonal preconditioner
//                 2. The divergence free solver
//
//               We recommend viewing example 5 before viewing this miniapp.

#include "mfem.hpp"
#include "div_free_solver.hpp"
#include <fstream>
#include <iostream>
#include <assert.h>
#include <memory>

using namespace std;
using namespace mfem;

// Exact solution, u and p, and r.h.s., f and g.
void u_exact(const Vector & x, Vector & u);
double p_exact(const Vector & x);
void f_exact(const Vector & x, Vector & f);
double g_exact(const Vector & x);
double natural_bc(const Vector & x);

/// Wrapper for assembling the discrete Darcy problem (ex5p)
/**
 *     Assemble the finite element matrices for the Darcy problem
 *                            D = [ M  B^T ]
 *                                [ B   0  ]
 *     where:
 *
 *     M = \int_\Omega u_h \cdot v_h d\Omega   u_h, v_h \in R_h
 *     B   = -\int_\Omega \div u_h q_h d\Omega   u_h \in R_h, q_h \in W_h
 */
class DarcyProblem
{
   OperatorPtr M_;
   OperatorPtr B_;
   Vector rhs_;
   Vector ess_data_;
   ParGridFunction u_;
   ParGridFunction p_;
   ParMesh mesh_;
   VectorFunctionCoefficient ucoeff_;
   FunctionCoefficient pcoeff_;
   DFSDataCollector collector_;
   const IntegrationRule *irs_[Geometry::NumGeom];
public:
   DarcyProblem(Mesh& mesh, int num_refines, int order,
                Array<int>& ess_bdr, DFSParameters param);

   HypreParMatrix& GetM() { return *M_.As<HypreParMatrix>(); }
   HypreParMatrix& GetB() { return *B_.As<HypreParMatrix>(); }
   const Vector& GetRHS() { return rhs_; }
   const Vector& GetBC() { return ess_data_; }
   const DFSDataCollector& GetDFSDataCollector() const { return collector_; }
   void ShowError(const Vector& sol, bool verbose);
   void VisualizeSolution(const Vector& sol, string tag);
};

DarcyProblem::DarcyProblem(Mesh& mesh, int num_refines, int order,
                           Array<int>& ess_bdr, DFSParameters dfs_param)
   : mesh_(MPI_COMM_WORLD, mesh), ucoeff_(mesh.Dimension(), u_exact),
     pcoeff_(p_exact), collector_(order, num_refines, &mesh_, ess_bdr, dfs_param)
{
   for (int l = 0; l < num_refines; l++)
   {
      mesh_.UniformRefinement();
      collector_.CollectData();
   }

   VectorFunctionCoefficient fcoeff(mesh_.Dimension(), f_exact);
   FunctionCoefficient natcoeff(natural_bc);
   FunctionCoefficient gcoeff(g_exact);

   u_.SetSpace(collector_.hdiv_fes_.get());
   p_.SetSpace(collector_.l2_fes_.get());
   u_ = 0.0;
   u_.ProjectBdrCoefficientNormal(ucoeff_, ess_bdr);

   ParLinearForm fform(collector_.hdiv_fes_.get());
   fform.AddDomainIntegrator(new VectorFEDomainLFIntegrator(fcoeff));
   fform.AddBoundaryIntegrator(new VectorFEBoundaryFluxLFIntegrator(natcoeff));
   fform.Assemble();

   ParLinearForm gform(collector_.l2_fes_.get());
   gform.AddDomainIntegrator(new DomainLFIntegrator(gcoeff));
   gform.Assemble();

   ParBilinearForm mVarf(collector_.hdiv_fes_.get());
   ParMixedBilinearForm bVarf(&(*collector_.hdiv_fes_), &(*collector_.l2_fes_));

   mVarf.AddDomainIntegrator(new VectorFEMassIntegrator);
   mVarf.Assemble();
   mVarf.EliminateEssentialBC(ess_bdr, u_, fform);
   mVarf.Finalize();
   M_.Reset(mVarf.ParallelAssemble());

   bVarf.AddDomainIntegrator(new VectorFEDivergenceIntegrator);
   bVarf.Assemble();
   bVarf.SpMat() *= -1.0;
   bVarf.EliminateTrialDofs(ess_bdr, u_, gform);
   bVarf.Finalize();
   B_.Reset(bVarf.ParallelAssemble());

   rhs_.SetSize(M_->NumRows() + B_->NumRows());
   Vector rhs_block0(rhs_.GetData(), M_->NumRows());
   Vector rhs_block1(rhs_.GetData()+M_->NumRows(), B_->NumRows());
   fform.ParallelAssemble(rhs_block0);
   gform.ParallelAssemble(rhs_block1);

   ess_data_.SetSize(M_->NumRows() + B_->NumRows());
   ess_data_ = 0.0;
   Vector ess_data_block0(ess_data_.GetData(), M_->NumRows());
   u_.ParallelProject(ess_data_block0);

   int order_quad = max(2, 2*order+1);
   for (int i=0; i < Geometry::NumGeom; ++i)
   {
      irs_[i] = &(IntRules.Get(i, order_quad));
   }
}

void DarcyProblem::ShowError(const Vector& sol, bool verbose)
{
   u_.Distribute(Vector(sol.GetData(), M_->NumRows()));
   p_.Distribute(Vector(sol.GetData()+M_->NumRows(), B_->NumRows()));

   double err_u  = u_.ComputeL2Error(ucoeff_, irs_);
   double norm_u = ComputeGlobalLpNorm(2, ucoeff_, mesh_, irs_);
   double err_p  = p_.ComputeL2Error(pcoeff_, irs_);
   double norm_p = ComputeGlobalLpNorm(2, pcoeff_, mesh_, irs_);

   if (!verbose) { return; }
   cout << "\n|| u_h - u_ex || / || u_ex || = " << err_u / norm_u << "\n";
   cout << "|| p_h - p_ex || / || p_ex || = " << err_p / norm_p << "\n";
}

void DarcyProblem::VisualizeSolution(const Vector& sol, string tag)
{
   int num_procs, myid;
   MPI_Comm_size(mesh_.GetComm(), &num_procs);
   MPI_Comm_rank(mesh_.GetComm(), &myid);

   u_.Distribute(Vector(sol.GetData(), M_->NumRows()));
   p_.Distribute(Vector(sol.GetData()+M_->NumRows(), B_->NumRows()));

   const char vishost[] = "localhost";
   const int  visport   = 19916;
   socketstream u_sock(vishost, visport);
   u_sock << "parallel " << num_procs << " " << myid << "\n";
   u_sock.precision(8);
   u_sock << "solution\n" << mesh_ << u_ << "window_title 'Velocity ("
          << tag << " solver)'" << endl;
   MPI_Barrier(mesh_.GetComm());
   socketstream p_sock(vishost, visport);
   p_sock << "parallel " << num_procs << " " << myid << "\n";
   p_sock.precision(8);
   p_sock << "solution\n" << mesh_ << p_ << "window_title 'Pressure ("
          << tag << " solver)'" << endl;
}

int main(int argc, char *argv[])
{
   StopWatch chrono;
   auto ResetTimer = [&chrono]() { chrono.Clear(); chrono.Start(); };

   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   bool verbose = (myid == 0);

   // 2. Parse command-line options.
   int order = 0;
   int num_refines = 2;
   bool use_tet_mesh = false;
   bool coupled_solve = true;
   bool show_error = false;
   bool visualization = false;
   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&num_refines, "-r", "--ref",
                  "Number of parallel refinement steps.");
   args.AddOption(&use_tet_mesh, "-tet", "--tet-mesh", "-hex", "--hex-mesh",
                  "Use a tetrahedral or hexahedral mesh (on unit cube).");
   args.AddOption(&coupled_solve, "-cs", "--coupled-solve", "-ss",
                  "--separate-solve",
                  "Whether to solve all unknowns together in div free solver.");
   args.AddOption(&show_error, "-se", "--show-error", "-no-se",
                  "--no-show-error",
                  "Show or not show approximation error.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      if (verbose) { args.PrintUsage(cout); }
      MPI_Finalize();
      return 1;
   }
   if (verbose) { args.PrintOptions(cout); }

   // Initialize the mesh, boundary attributes, and solver parameters
   auto elem_type = use_tet_mesh ? Element::TETRAHEDRON : Element::HEXAHEDRON;
   Mesh mesh(2, 2, 2, elem_type, true);
   for (int i = 0; i < (int)(log(num_procs)/log(8)); ++i)
   {
      mesh.UniformRefinement();
   }

   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 0;
   ess_bdr[1] = 1;

   DFSParameters param;
   param.B_has_nullity_one = (ess_bdr.Sum() == ess_bdr.Size());
   param.coupled_solve = coupled_solve;

   string line = "\n*******************************************************\n";
   {
      ResetTimer();

      // Generate components of the saddle point problem
      DarcyProblem darcy(mesh, num_refines, order, ess_bdr, param);
      HypreParMatrix& M = darcy.GetM();
      HypreParMatrix& B = darcy.GetB();
      const DFSDataCollector& collector = darcy.GetDFSDataCollector();

      if (verbose)
      {
         cout << line << "dim(R) = " << M.M() << ", dim(W) = " << B.M() << ", ";
         cout << "dim(N) = " << collector.hcurl_fes_->GlobalTrueVSize() << "\n";
         cout << "System assembled in " << chrono.RealTime() << "s.\n";
      }

      // Setup various solvers for the discrete problem
      std::map<const DarcySolver*, double> setup_time;
      ResetTimer();
      DivFreeSolver dfs(M, B, collector.GetData());
      setup_time[&dfs] = chrono.RealTime();

      ResetTimer();
      BDPMinresSolver bdp(M, B, false, param);
      setup_time[&bdp] = chrono.RealTime();

      std::map<const DarcySolver*, std::string> solver_to_name;
      solver_to_name[&dfs] = "Divergence free";
      solver_to_name[&bdp] = "Block-diagonal-preconditioned MINRES";

      // Solve the problem using all solvers
      for (const auto& solver_pair : solver_to_name)
      {
         auto& solver = solver_pair.first;
         auto& name = solver_pair.second;

         const Vector& rhs = darcy.GetRHS();
         Vector sol = darcy.GetBC();
         ResetTimer();
         solver->Mult(rhs, sol);
         chrono.Stop();

         if (verbose)
         {
            cout << line << name << " solver:\n  Setup time: "
                 << setup_time[solver] << "s.\n  Solve time: "
                 << chrono.RealTime() << "s.\n  Total time: "
                 << setup_time[solver] + chrono.RealTime() << "s.\n"
                 << "  Iteration count: " << solver->GetNumIterations() <<"\n";
         }
         if (show_error) { darcy.ShowError(sol, verbose); }
         if (visualization) { darcy.VisualizeSolution(sol, name); }
      }
   }

   MPI_Finalize();
   return 0;
}


void u_exact(const Vector & x, Vector & u)
{
   double xi(x(0));
   double yi(x(1));
   double zi(0.0);
   if (x.Size() == 3)
   {
      zi = x(2);
   }

   u(0) = - exp(xi)*sin(yi)*cos(zi);
   u(1) = - exp(xi)*cos(yi)*cos(zi);

   if (x.Size() == 3)
   {
      u(2) = exp(xi)*sin(yi)*sin(zi);
   }
}

// Change if needed
double p_exact(const Vector & x)
{
   double xi(x(0));
   double yi(x(1));
   double zi(0.0);

   if (x.Size() == 3)
   {
      zi = x(2);
   }

   return exp(xi)*sin(yi)*cos(zi);
}

void f_exact(const Vector & x, Vector & f)
{
   f = 0.0;
}

double g_exact(const Vector & x)
{
   if (x.Size() == 3) { return -p_exact(x); }
   return 0;
}

double natural_bc(const Vector & x)
{
   return (-p_exact(x));
}

// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
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
//          ----------------------------------------------------------
//          Block Solvers Miniapp: Compare Saddle Point System Solvers
//          ----------------------------------------------------------
//
// This miniapp compares various linear solvers for the saddle point system
// obtained from mixed finite element discretization of the simple mixed Darcy
// problem in ex5p
//
//                            k*u + grad p = f
//                           - div u      = g
//
// with natural boundary condition -p = <given pressure>. We use a given exact
// solution (u,p) and compute the corresponding r.h.s. (f,g). We discretize
// with Raviart-Thomas finite elements (velocity u) and piecewise discontinuous
// polynomials (pressure p).
//
// The solvers being compared include:
//    1. MINRES preconditioned by a block diagonal preconditioner
//    2. The divergence free solver (couple and decoupled modes)
//    3. A block hybridization solver
//    4. The Bramble-Pasciak solver (using BPCG or regular PCG)
//
// We recommend viewing example 5 before viewing this miniapp.
//
// Sample runs:
//
//    mpirun -np 8 block-solvers -pr 2 -o 0
//    mpirun -np 8 block-solvers -m anisotropic.mesh -c anisotropic.coeff -eb anisotropic.bdr
//
//
// NOTE:  The coefficient file (provided through -c) defines a piecewise constant
//        scalar coefficient k. The number of entries in this file must equal the
//        number of "element attributes" in the mesh file. The value of the
//        coefficient in elements with the i-th attribute is given by the i-th
//        entry of the coefficient file.
//
//
// NOTE:  The essential boundary attribute file (provided through -eb) defines
//        which attributes to impose an essential boundary condition (on u).
//        The number of entries in this file must equal the number of "boundary
//        attributes" in the mesh file. If the i-th entry of the file is nonzero
//        (respectively zero), then the essential (respectively natural) boundary
//        condition will be imposed on boundary elements with the i-th attribute.

#include "mfem.hpp"
#include "bramble_pasciak.hpp"
#include "div_free_solver.hpp"
#include "block_hybridization.hpp"
#include <fstream>
#include <iostream>
#include <memory>

using namespace std;
using namespace mfem;
using namespace blocksolvers;

// Exact solution, u and p, and r.h.s., f and g.
void u_exact(const Vector &x, Vector &u);
double p_exact(const Vector &x);
void f_exact(const Vector &x, Vector &f);
double g_exact(const Vector &x);
double natural_bc(const Vector &x);
double pi = M_PI;

/** Wrapper for assembling the discrete Darcy problem (ex5p)
                     [ M  B^T ] [u] = [f]
                     [ B   0  ] [p] = [g]
    where:
       M = \int_\Omega (k u_h) \cdot v_h dx,
       B = -\int_\Omega (div_h u_h) q_h dx,
       f = \int_\Omega f_exact v_h dx + \int_D natural_bc v_h dS,
       g = \int_\Omega g_exact q_h dx,
       u_h, v_h \in R_h (Raviart-Thomas finite element space),
       q_h \in W_h (piecewise discontinuous polynomials),
       D: subset of the boundary where natural boundary condition is imposed. */
class DarcyProblem
{
   Array<int> offsets_;
   Vector rhs_;
   Vector ess_data_;
   ParGridFunction u_;
   ParGridFunction p_;
   ParMesh mesh_;
   shared_ptr<ParBilinearForm> Mform_;
   shared_ptr<ParMixedBilinearForm> Bform_;
   unique_ptr<PWConstCoefficient> mass_coeff_;
   VectorFunctionCoefficient ucoeff_;
   FunctionCoefficient pcoeff_;
   const Array<int> &ess_bdr_;
   DFSSpaces dfs_spaces_;
   const IntegrationRule *irs_[Geometry::NumGeom];

   void Distribute(const Vector &x);
public:
   DarcyProblem(MPI_Comm comm, Mesh &mesh, int num_refines, int order,
                const char *coef_file, Array<int> &ess_bdr, DFSParameters param);

   void GetParallelSystems(shared_ptr<HypreParMatrix> &M,
                           shared_ptr<HypreParMatrix> &B,
                           shared_ptr<HypreParMatrix> &M_e,
                           shared_ptr<HypreParMatrix> &B_e,
                           mfem::Array<int> &ess_tdof_list) const;

   void ShowError(const Vector &sol, bool verbose);
   void VisualizeSolution(const Vector &sol, string tag);

   const Vector &GetRHS() const { return rhs_; }
   const Vector &GetEssentialBC() const { return ess_data_; }
   shared_ptr<ParBilinearForm> GetMform() const { return Mform_; }
   shared_ptr<ParMixedBilinearForm> GetBform() const { return Bform_; }
   const DFSData &GetDFSData() const { return dfs_spaces_.GetDFSData(); }
};

DarcyProblem::DarcyProblem(MPI_Comm comm, Mesh &mesh, int num_refs, int order,
                           const char *coef_file, Array<int> &ess_bdr,
                           DFSParameters dfs_param)
   : mesh_(comm, mesh), ucoeff_(mesh.Dimension(), u_exact), pcoeff_(p_exact),
     ess_bdr_(ess_bdr), dfs_spaces_(order, num_refs, &mesh_, ess_bdr, dfs_param)
{
   for (int l = 0; l < num_refs; l++)
   {
      mesh_.UniformRefinement();
      dfs_spaces_.CollectDFSData();
   }

   Vector coef_vector(mesh.GetNE());
   coef_vector = 1.0;
   if (std::strcmp(coef_file, ""))
   {
      ifstream coef_str(coef_file);
      coef_vector.Load(coef_str, mesh.GetNE());
   }
   mass_coeff_.reset(new PWConstCoefficient(coef_vector));

   VectorFunctionCoefficient fcoeff(mesh_.Dimension(), f_exact);
   FunctionCoefficient natcoeff(natural_bc);
   FunctionCoefficient gcoeff(g_exact);

   ParFiniteElementSpace* u_fes = dfs_spaces_.GetHdivFES();
   ParFiniteElementSpace* p_fes = dfs_spaces_.GetL2FES();
   offsets_.SetSize(3, 0);
   offsets_[1] = u_fes->GetTrueVSize();
   offsets_[2] = offsets_[1] + p_fes->GetTrueVSize();

   u_.SetSpace(u_fes);
   p_.SetSpace(p_fes);
   p_ = 0.0;
   u_ = 0.0;
   u_.ProjectBdrCoefficientNormal(ucoeff_, ess_bdr);

   ParLinearForm fform(u_fes);
   fform.AddDomainIntegrator(new VectorFEDomainLFIntegrator(fcoeff));
   fform.AddBoundaryIntegrator(new VectorFEBoundaryFluxLFIntegrator(natcoeff));
   fform.Assemble();

   ParLinearForm gform(p_fes);
   gform.AddDomainIntegrator(new DomainLFIntegrator(gcoeff));
   gform.Assemble();

   rhs_.SetSize(offsets_[2]);
   BlockVector blk_rhs(rhs_, offsets_);
   fform.ParallelAssemble(blk_rhs.GetBlock(0));
   gform.ParallelAssemble(blk_rhs.GetBlock(1));

   Mform_ = make_shared<ParBilinearForm>(u_fes);
   Mform_->AddDomainIntegrator(new VectorFEMassIntegrator(*mass_coeff_));
   // Mform_->ComputeElementMatrices();
   Mform_->Assemble();
   Mform_->Finalize();

   Bform_ = make_shared<ParMixedBilinearForm>(u_fes, p_fes);
   Bform_->AddDomainIntegrator(new VectorFEDivergenceIntegrator());
   // Bform_->ComputeElementMatrices();
   Bform_->Assemble();
   Bform_->SpMat() *= -1.0;
   Bform_->Finalize();

   ess_data_.SetSize(rhs_.Size());
   ess_data_ = 0.0;
   Vector ess_data_block0(ess_data_.GetData(), offsets_[1]);
   u_.ParallelProject(ess_data_block0);

   int order_quad = max(2, 2*order+1);
   for (int i=0; i < Geometry::NumGeom; ++i)
   {
      irs_[i] = &(IntRules.Get(i, order_quad));
   }
}

void DarcyProblem::Distribute(const Vector &x)
{
   BlockVector blk_x(x.GetData(), offsets_);
   u_.Distribute(blk_x.GetBlock(0));
   p_.Distribute(blk_x.GetBlock(1));
}

void DarcyProblem::GetParallelSystems(shared_ptr<HypreParMatrix> &M,
                                      shared_ptr<HypreParMatrix> &B,
                                      shared_ptr<HypreParMatrix> &M_e,
                                      shared_ptr<HypreParMatrix> &B_e,
                                      mfem::Array<int> &ess_tdof_list) const
{
   dfs_spaces_.GetHdivFES()->GetEssentialTrueDofs(ess_bdr_, ess_tdof_list);
   M.reset(Mform_->ParallelAssemble());
   M_e.reset(M->EliminateRowsCols(ess_tdof_list));
   B.reset(Bform_->ParallelAssemble());
   B_e.reset(B->EliminateCols(ess_tdof_list));
}

void DarcyProblem::ShowError(const Vector& sol, bool verbose)
{
   Distribute(sol);
   double err_u  = u_.ComputeL2Error(ucoeff_, irs_);
   double norm_u = ComputeGlobalLpNorm(2, ucoeff_, mesh_, irs_);
   double err_p  = p_.ComputeL2Error(pcoeff_, irs_);
   double norm_p = ComputeGlobalLpNorm(2, pcoeff_, mesh_, irs_);

   if (!verbose) { return; }
   cout << "|| u_h - u_ex || / || u_ex || = " << err_u / norm_u << "\n";
   cout << "|| p_h - p_ex || / || p_ex || = " << err_p / norm_p << "\n";
}

void DarcyProblem::VisualizeSolution(const Vector& sol, string tag)
{
   int num_procs, myid;
   MPI_Comm_size(mesh_.GetComm(), &num_procs);
   MPI_Comm_rank(mesh_.GetComm(), &myid);

   Distribute(sol);
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

bool IsAllNeumannBoundary(const Array<int>& ess_bdr_attr)
{
   for (int attr : ess_bdr_attr) { if (attr == 0) { return false; } }
   return true;
}

int main(int argc, char *argv[])
{
#ifdef HYPRE_USING_GPU
   cout << "\nAs of mfem-4.3 and hypre-2.22.0 (July 2021) this miniapp\n"
        << "is NOT supported with the GPU version of hypre.\n\n";
   return 242;
#endif

   // Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   Hypre::Init();

   StopWatch chrono;
   auto ResetTimer = [&chrono]() { chrono.Clear(); chrono.Start(); };

   // Parse command-line options.
   const char *mesh_file = "../../data/beam-hex.mesh";
   const char *coef_file = "";
   const char *ess_bdr_attr_file = "";
   int order = 0;
   int ser_ref_levels = 1;
   int par_ref_levels = 1;
   bool show_error = false;
   bool visualization = false;

   DFSParameters param;
#ifdef MFEM_USE_LAPACK
   BPSParameters bps_param;
#endif

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&ser_ref_levels, "-sr", "--serial-ref",
                  "Number of serial refinement steps.");
   args.AddOption(&par_ref_levels, "-pr", "--parallel-ref",
                  "Number of parallel refinement steps.");
   args.AddOption(&coef_file, "-c", "--coef",
                  "Coefficient file to use.");
   args.AddOption(&ess_bdr_attr_file, "-eb", "--ess-bdr",
                  "Essential boundary attribute file to use.");
   args.AddOption(&show_error, "-se", "--show-error", "-no-se",
                  "--no-show-error",
                  "Show or not show approximation error.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      if (Mpi::Root()) { args.PrintUsage(cout); }
      return 1;
   }
   if (Mpi::Root()) { args.PrintOptions(cout); }

   if (Mpi::Root() && par_ref_levels == 0)
   {
      std::cout << "WARNING: DivFree solver is equivalent to BDPMinresSolver "
                << "when par_ref_levels == 0.\n";
   }

   // Initialize the mesh, boundary attributes, and solver parameters
   // Mesh *mesh = new Mesh(mesh_file, 1, 1);
   Mesh *mesh = new Mesh(Mesh::MakeCartesian2D(8, 8, Element::TRIANGLE, 1));
   int dim = mesh->Dimension();

   for (int i = 0; i < ser_ref_levels; ++i)
   {
      mesh->UniformRefinement();
   }

   if (Mpi::Root())
   {
      MFEM_ASSERT(Mpi::WorldSize() < mesh->GetNE(),
                  "Not enough elements in the mesh to be distributed:\n"
                  << "Number of processors: " << Mpi::WorldSize() << "\n"
                  << "Number of elements:   " << mesh->GetNE());
   }

   Array<int> ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 1;
   ess_bdr[0] = 0;
   if (std::strcmp(ess_bdr_attr_file, ""))
   {
      ifstream ess_bdr_attr_str(ess_bdr_attr_file);
      ess_bdr.Load(mesh->bdr_attributes.Max(), ess_bdr_attr_str);
   }
   if (IsAllNeumannBoundary(ess_bdr))
   {
      if (Mpi::Root())
      {
         cout << "\nSolution is not unique when Neumann boundary condition is "
              << "imposed on the entire boundary. \nPlease provide a different "
              << "boundary condition.\n";
      }
      delete mesh;
      return 0;
   }

   string line = "**********************************************************\n";

   ResetTimer();

   // Generate components of the saddle point problem
   mfem::Array<int> ess_tdof_list;
   shared_ptr<HypreParMatrix> M, B, M_e, B_e;
   DarcyProblem darcy(MPI_COMM_WORLD, *mesh, par_ref_levels, order,
                      coef_file, ess_bdr, param);
   darcy.GetParallelSystems(M, B, M_e, B_e, ess_tdof_list);
   const DFSData& DFS_data = darcy.GetDFSData();
   delete mesh;

   if (Mpi::Root())
   {
      cout << line << "System assembled in " << chrono.RealTime() << "s.\n";
      cout << "Dimension of the physical space: " << dim << "\n";
      cout << "Size of the discrete Darcy system: " << M->M() + B->M() << "\n";
      if (par_ref_levels > 0)
      {
         cout << "Dimension of the divergence free subspace: "
              << DFS_data.C.back().Ptr()->NumCols() << "\n\n";
      }
   }

   // Setup various solvers for the discrete problem
   std::map<const DarcySolver*, double> setup_time;

   ResetTimer();
   BlockHybridizationSolver bh(darcy.GetMform(), darcy.GetBform(), param, ess_bdr);
   bh.SetEliminatedSystems(M_e, B_e, ess_tdof_list);
   setup_time[&bh] = chrono.RealTime();
   /*
   ResetTimer();
   BDPMinresSolver bdp(*M, *B, param);
   bdp.SetEliminatedSystems(M_e, B_e, ess_tdof_list);
   setup_time[&bdp] = chrono.RealTime();

   ResetTimer();
   DivFreeSolver dfs_dm(*M, *B, DFS_data);
   dfs_dm.SetEliminatedSystems(M_e, B_e, ess_tdof_list);
   setup_time[&dfs_dm] = chrono.RealTime();

   ResetTimer();
   const_cast<bool&>(DFS_data.param.coupled_solve) = true;
   DivFreeSolver dfs_cm(*M, *B, DFS_data);
   dfs_cm.SetEliminatedSystems(M_e, B_e, ess_tdof_list);
   setup_time[&dfs_cm] = chrono.RealTime();

#ifdef MFEM_USE_LAPACK
   ResetTimer();
   BramblePasciakSolver bp_bpcg(darcy.GetMform(), darcy.GetBform(), ess_tdof_list,
                                bps_param);
   bp_bpcg.SetEliminatedSystems(M_e, B_e, ess_tdof_list);
   setup_time[&bp_bpcg] = chrono.RealTime();

   ResetTimer();
   bps_param.use_bpcg = false;
   BramblePasciakSolver bp_pcg(darcy.GetMform(), darcy.GetBform(), ess_tdof_list,
                               bps_param);
   bp_pcg.SetEliminatedSystems(M_e, B_e, ess_tdof_list);
   setup_time[&bp_pcg] = chrono.RealTime();
#else
   MFEM_WARNING("BramblePasciakSolver class unavailable: Compiled without LAPACK");
#endif
   */

   std::map<const DarcySolver*, std::string> solver_to_name;
   solver_to_name[&bh] = "Block hybridization";
   /*
   solver_to_name[&bdp] = "Block-diagonal-preconditioned MINRES";
   solver_to_name[&dfs_dm] = "Divergence free (decoupled mode)";
   solver_to_name[&dfs_cm] = "Divergence free (coupled mode)";
#ifdef MFEM_USE_LAPACK
   solver_to_name[&bp_bpcg] = "Bramble Pasciak CG (using BPCG)";
   solver_to_name[&bp_pcg] = "Bramble Pasciak CG (using regular PCG)";
#endif
   */

   // Solve the problem using all solvers
   for (const auto& solver_pair : solver_to_name)
   {
      auto& solver = solver_pair.first;
      auto& name = solver_pair.second;

      Vector sol = darcy.GetEssentialBC();

      ResetTimer();
      solver->Mult(darcy.GetRHS(), sol);
      chrono.Stop();

      if (Mpi::Root())
      {
         cout << line << name << " solver:\n   Setup time: "
              << setup_time[solver] << "s.\n   Solve time: "
              << chrono.RealTime() << "s.\n   Total time: "
              << setup_time[solver] + chrono.RealTime() << "s.\n"
              << "   Iteration count: " << solver->GetNumIterations() <<"\n\n";
      }
      if (show_error && std::strcmp(coef_file, "") == 0)
      {
         darcy.ShowError(sol, Mpi::Root());
      }
      else if (show_error && Mpi::Root())
      {
         cout << "Exact solution is unknown for coefficient '" << coef_file
              << "'.\nApproximation error is not computed in this case!\n\n";
      }

      if (visualization) { darcy.VisualizeSolution(sol, name); }

   }

   return 0;
}

void u_exact(const Vector &x, Vector &u)
{
   double xi(x(0));
   double xj(x(1));
   u(0) = -pi * pow(sin(pi*xi), 2) * sin(2*pi*xj);
   u(1) = pi * sin(2*pi*xi) * pow(sin(pi*xj), 2);
}

double p_exact(const Vector &x)
{
   double xi(x(0));
   return -sin(pi*xi) + 2/pi;
}

void f_exact(const Vector &x, Vector &f)
{
   u_exact(x, f);
   f(0) -= pi * cos(pi*x(0));
}

double g_exact(const Vector &x)
{
   return 0.0;
}

double natural_bc(const Vector &x)
{
   return -p_exact(x);
}

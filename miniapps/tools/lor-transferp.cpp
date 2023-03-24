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
//
//   -----------------------------------------------------------------------
//   Parallel LOR Transfer Miniapp:  Map functions between HO and LOR spaces
//   -----------------------------------------------------------------------
//
// This miniapp visualizes the maps between a high-order (HO) finite element
// space, typically using high-order functions on a high-order mesh, and a
// low-order refined (LOR) finite element space, typically defined by 0th or 1st
// order functions on a low-order refinement of the HO mesh.
//
// The grid transfer operators are represented using either
// InterpolationGridTransfer or L2ProjectionGridTransfer (depending on the
// options requested by the user). The two transfer operators are then:
//
//  1. R: HO -> LOR, defined by GridTransfer::ForwardOperator
//  2. P: LOR -> HO, defined by GridTransfer::BackwardOperator
//
// While defined generally, these operators have some nice properties for
// particular finite element spaces. For example they satisfy PR=I, plus mass
// conservation in both directions for L2 fields.
//
// Compile with: make lor-transferp
//
// Sample runs:  lor-transferp
//               lor-transferp -h1
//               lor-transferp -t
//               lor-transferp -m ../../data/star-q2.mesh -lref 5 -p 4
//               lor-transferp -m ../../data/star-mixed.mesh -lref 3 -p 2
//               lor-transferp -lref 4 -o 4 -lo 0 -p 1
//               lor-transferp -lref 5 -o 4 -lo 0 -p 1
//               lor-transferp -lref 5 -o 4 -lo 3 -p 2
//               lor-transferp -lref 5 -o 4 -lo 0 -p 3

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int problem = 1; // problem type

int Wx = 0, Wy = 0; // window position
int Ww = 350, Wh = 350; // window size
int offx = Ww+5, offy = Wh+25; // window offsets

string space;
string direction;

// Exact functions to project
double RHO_exact(const Vector &x);

// Helper functions
void visualize(VisItDataCollection &, string, int, int);
double compute_mass(ParFiniteElementSpace *, double, VisItDataCollection &,
                    string);

int main(int argc, char *argv[])
{
   // Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   Hypre::Init();

   // Parse command-line options.
   const char *mesh_file = "../../data/star.mesh";
   int order = 3;
   int lref = order+1;
   int lorder = 0;
   bool vis = true;
   bool useH1 = false;
   bool use_pointwise_transfer = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem type (see the RHO_exact function).");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&lref, "-lref", "--lor-ref-level", "LOR refinement level.");
   args.AddOption(&lorder, "-lo", "--lor-order",
                  "LOR space order (polynomial degree, zero by default).");
   args.AddOption(&vis, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&useH1, "-h1", "--use-h1", "-l2", "--use-l2",
                  "Use H1 spaces instead of L2.");
   args.AddOption(&use_pointwise_transfer, "-t", "--use-pointwise-transfer",
                  "-no-t", "--dont-use-pointwise-transfer",
                  "Use pointwise transfer operators instead of L2 projection.");
   args.Parse();
   if (!args.Good())
   {
      if (Mpi::Root())
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (Mpi::Root())
   {
      args.PrintOptions(cout);
   }

   // Read the mesh from the given mesh file.
   Mesh serial_mesh(mesh_file, 1, 1);
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   serial_mesh.Clear();
   int dim = mesh.Dimension();

   // Create the low-order refined mesh
   int basis_lor = BasisType::GaussLobatto; // BasisType::ClosedUniform;
   ParMesh mesh_lor = ParMesh::MakeRefined(mesh, lref, basis_lor);

   // Create spaces
   FiniteElementCollection *fec, *fec_lor;
   if (useH1)
   {
      space = "H1";
      if (lorder == 0)
      {
         lorder = 1;
         cerr << "Switching the H1 LOR space order from 0 to 1\n";
      }
      fec = new H1_FECollection(order, dim);
      fec_lor = new H1_FECollection(lorder, dim);
   }
   else
   {
      space = "L2";
      fec = new L2_FECollection(order, dim);
      fec_lor = new L2_FECollection(lorder, dim);
   }

   ParFiniteElementSpace fespace(&mesh, fec);
   ParFiniteElementSpace fespace_lor(&mesh_lor, fec_lor);

   ParGridFunction rho(&fespace);
   ParGridFunction rho_lor(&fespace_lor);

   // Data collections for vis/analysis
   VisItDataCollection HO_dc(MPI_COMM_WORLD, "HO", &mesh);
   HO_dc.RegisterField("density", &rho);
   VisItDataCollection LOR_dc(MPI_COMM_WORLD, "LOR", &mesh_lor);
   LOR_dc.RegisterField("density", &rho_lor);

   ParBilinearForm M_ho(&fespace);
   M_ho.AddDomainIntegrator(new MassIntegrator);
   M_ho.Assemble();
   M_ho.Finalize();
   HypreParMatrix* M_ho_tdof = M_ho.ParallelAssemble();

   ParBilinearForm M_lor(&fespace_lor);
   M_lor.AddDomainIntegrator(new MassIntegrator);
   M_lor.Assemble();
   M_lor.Finalize();
   HypreParMatrix* M_lor_tdof = M_lor.ParallelAssemble();

   // HO projections
   direction = "HO -> LOR @ HO";
   FunctionCoefficient RHO(RHO_exact);
   rho.ProjectCoefficient(RHO);
   double ho_mass = compute_mass(&fespace, -1.0, HO_dc, "HO       ");
   if (vis) { visualize(HO_dc, "HO", Wx, Wy); Wx += offx; }

   GridTransfer *gt;
   if (use_pointwise_transfer)
   {
      gt = new InterpolationGridTransfer(fespace, fespace_lor);
   }
   else
   {
      gt = new L2ProjectionGridTransfer(fespace, fespace_lor);
   }
   const Operator &R = gt->ForwardOperator();

   // HO->LOR restriction
   direction = "HO -> LOR @ LOR";
   R.Mult(rho, rho_lor);
   compute_mass(&fespace_lor, ho_mass, LOR_dc, "R(HO)    ");
   if (vis) { visualize(LOR_dc, "R(HO)", Wx, Wy); Wx += offx; }

   if (gt->SupportsBackwardsOperator())
   {
      const Operator &P = gt->BackwardOperator();
      // LOR->HO prolongation
      direction = "HO -> LOR @ HO";
      ParGridFunction rho_prev = rho;
      P.Mult(rho_lor, rho);
      compute_mass(&fespace, ho_mass, HO_dc, "P(R(HO)) ");
      if (vis) { visualize(HO_dc, "P(R(HO))", Wx, Wy); Wx = 0; Wy += offy; }

      rho_prev -= rho;
      Vector rho_prev_true(fespace.GetTrueVSize());
      rho_prev.GetTrueDofs(rho_prev_true);
      double l_inf_local = rho_prev_true.Normlinf();
      double l_inf;
      MPI_Allreduce(&l_inf_local, &l_inf, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      if (Mpi::Root())
      {
         cout.precision(12);
         cout << "|HO - P(R(HO))|_∞   = " << l_inf << endl;
      }
   }

   // HO* to LOR* dual fields
   ParGridFunction ones(&fespace), ones_lor(&fespace_lor);
   ones = 1.0;
   ones_lor = 1.0;
   ParLinearForm M_rho(&fespace), M_rho_lor(&fespace_lor);
   if (!use_pointwise_transfer && gt->SupportsBackwardsOperator())
   {
      const Operator &P = gt->BackwardOperator();
      Vector rho_true(rho.ParFESpace()->GetTrueVSize());
      rho.GetTrueDofs(rho_true);
      Vector M_rho_true(M_rho.ParFESpace()->GetTrueVSize());
      M_ho_tdof->Mult(rho_true, M_rho_true);
      M_rho.ParFESpace()->GetProlongationMatrix()->Mult(M_rho_true, M_rho);
      P.MultTranspose(M_rho, M_rho_lor);
      Vector M_rho_lor_true(M_rho_lor.ParFESpace()->GetTrueVSize());
      M_rho_lor.ParFESpace()->GetRestrictionOperator()->Mult(M_rho_lor,
                                                             M_rho_lor_true);
      double local_ho_mass = M_rho_true.Sum();
      double local_lor_mass = M_rho_lor_true.Sum();
      double ho_mass;
      double lor_mass;
      MPI_Allreduce(&local_ho_mass, &ho_mass, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(&local_lor_mass, &lor_mass, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
      if (Mpi::Root())
      {
         cout << "HO -> LOR dual field: " << fabs(ho_mass - lor_mass) << endl << endl;
      }
   }

   // LOR projections
   direction = "LOR -> HO @ LOR";
   rho_lor.ProjectCoefficient(RHO);
   ParGridFunction rho_lor_prev = rho_lor;
   double lor_mass = compute_mass(&fespace_lor, -1.0, LOR_dc, "LOR      ");
   if (vis) { visualize(LOR_dc, "LOR", Wx, Wy); Wx += offx; }

   if (gt->SupportsBackwardsOperator())
   {
      const Operator &P = gt->BackwardOperator();
      // Prolongate to HO space
      direction = "LOR -> HO @ HO";
      P.Mult(rho_lor, rho);
      compute_mass(&fespace, lor_mass, HO_dc, "P(LOR)   ");
      if (vis) { visualize(HO_dc, "P(LOR)", Wx, Wy); Wx += offx; }

      // Restrict back to LOR space. This won't give the original function because
      // the rho_lor doesn't necessarily live in the range of R.
      direction = "LOR -> HO @ LOR";
      R.Mult(rho, rho_lor);
      compute_mass(&fespace_lor, lor_mass, LOR_dc, "R(P(LOR))");
      if (vis) { visualize(LOR_dc, "R(P(LOR))", Wx, Wy); }

      rho_lor_prev -= rho_lor;
      Vector rho_lor_prev_true(fespace_lor.GetTrueVSize());
      rho_lor_prev.GetTrueDofs(rho_lor_prev_true);
      double l_inf_local = rho_lor_prev_true.Normlinf();
      double l_inf;
      MPI_Allreduce(&l_inf_local, &l_inf, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      if (Mpi::Root())
      {
         cout.precision(12);
         cout << "|LOR - R(P(LOR))|_∞ = " << l_inf << endl;
      }
   }

   // LOR* to HO* dual fields
   if (!use_pointwise_transfer)
   {
      Vector rho_lor_true(rho_lor.ParFESpace()->GetTrueVSize());
      rho_lor.GetTrueDofs(rho_lor_true);
      Vector M_rho_lor_true(M_rho_lor.ParFESpace()->GetTrueVSize());
      M_lor_tdof->Mult(rho_lor_true, M_rho_lor_true);
      M_rho_lor.ParFESpace()->GetProlongationMatrix()->Mult(M_rho_lor_true,
                                                            M_rho_lor);
      R.MultTranspose(M_rho_lor, M_rho);
      Vector M_rho_true(M_rho.ParFESpace()->GetTrueVSize());
      M_rho.ParFESpace()->GetRestrictionOperator()->Mult(M_rho, M_rho_true);
      double local_ho_mass = M_rho_true.Sum();
      double local_lor_mass = M_rho_lor_true.Sum();
      double ho_mass;
      double lor_mass;
      MPI_Allreduce(&local_ho_mass, &ho_mass, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(&local_lor_mass, &lor_mass, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
      if (Mpi::Root())
      {
         cout << "LOR -> HO dual field: " << fabs(ho_mass - lor_mass) << '\n';
      }
   }

   Mpi::Finalize();

   delete fec;
   delete fec_lor;
   delete M_ho_tdof;
   delete M_lor_tdof;
   delete gt;

   return 0;
}


double RHO_exact(const Vector &x)
{
   switch (problem)
   {
      case 1: // smooth field
         return x(1)+0.25*cos(2*M_PI*x.Norml2());
      case 2: // cubic function
         return x(1)*x(1)*x(1) + 2*x(0)*x(1) + x(0);
      case 3: // sharp gradient
         return M_PI/2-atan(5*(2*x.Norml2()-1));
      case 4: // basis function
         return (x.Norml2() < 0.1) ? 1 : 0;
      default:
         return 1.0;
   }
}


void visualize(VisItDataCollection &dc, string prefix, int x, int y)
{
   int w = Ww, h = Wh;

   char vishost[] = "localhost";
   int  visport   = 19916;

   socketstream sol_sockL2(vishost, visport);
   sol_sockL2 << "parallel " << Mpi::WorldSize() << " " << Mpi::WorldRank() <<
              "\n";
   sol_sockL2.precision(8);
   sol_sockL2 << "solution\n" << *dc.GetMesh() << *dc.GetField("density")
              << "window_geometry " << x << " " << y << " " << w << " " << h
              << "plot_caption '" << space << " " << prefix << " Density'"
              << "window_title '" << direction << "'" << flush;
}


double compute_mass(ParFiniteElementSpace *L2, double massL2,
                    VisItDataCollection &dc, string prefix)
{
   ConstantCoefficient one(1.0);
   ParBilinearForm ML2(L2);
   ML2.AddDomainIntegrator(new MassIntegrator(one));
   ML2.Assemble();
   ML2.Finalize();
   HypreParMatrix* pML2 = ML2.ParallelAssemble();

   Vector rhoone(L2->GetTrueVSize());
   rhoone = 1.0;

   Vector Mdiag(L2->GetTrueVSize());
   pML2->Mult(rhoone, Mdiag);
   delete pML2;
   HypreParVector* rho = dc.GetParField("density")->GetTrueDofs();
   double newmass = InnerProduct(MPI_COMM_WORLD, *rho, Mdiag);
   delete rho;
   if (Mpi::Root())
   {
      cout.precision(18);
      cout << space << " " << prefix << " mass   = " << newmass;
      if (massL2 >= 0)
      {
         cout.precision(4);
         cout << " ("  << fabs(newmass-massL2)*100/massL2 << "%)";
      }
      cout << endl;
   }
   return newmass;
}

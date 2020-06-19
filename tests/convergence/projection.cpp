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

// Compile with: make projection
//
// Sample runs:  mpirun -np 4 projection -m ../../data/inline-segment.mesh -sr 1 -pr 4 -prob 0 -o 1
//               mpirun -np 4 projection -m ../../data/inline-quad.mesh -sr 1 -pr 3 -prob 0 -o 2
//               mpirun -np 4 projection -m ../../data/inline-quad.mesh -sr 1 -pr 3 -prob 1 -o 2
//               mpirun -np 4 projection -m ../../data/inline-quad.mesh -sr 1 -pr 3 -prob 2 -o 2
//               mpirun -np 4 projection -m ../../data/inline-tri.mesh -sr 1 -pr 3 -prob 2 -o 3
//               mpirun -np 4 projection -m ../../data/star.mesh -sr 1 -pr 2 -prob 1 -o 4
//               mpirun -np 4 projection -m ../../data/fichera.mesh -sr 1 -pr 2 -prob 2 -o 2
//               mpirun -np 4 projection -m ../../data/inline-wedge.mesh -sr 0 -pr 2 -prob 0 -o 2
//               mpirun -np 4 projection -m ../../data/inline-hex.mesh -sr 0 -pr 1 -prob 1 -o 3
//               mpirun -np 4 projection -m ../../data/square-disc.mesh -sr 1 -pr 2 -prob 1 -o 2
//
// Description:  This example code is used for testing the LF-integrators
//               (Q,grad v), (Q,curl V), (Q, div v)
//               by solving the appropriate energy projection problems
//
//               prob 0: (grad u, grad v) + (u,v) = (grad u_exact, grad v) + (u_exact, v)
//               prob 1: (curl u, curl v) + (u,v) = (curl u_exact, curl v) + (u_exact, v)
//               prob 2: (div  u, div  v) + (u,v) = (div  u_exact, div  v) + (u_exact, v)

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// H1
double u_exact(const Vector &x);
void gradu_exact(const Vector &x, Vector &gradu);

// Vector FE
void U_exact(const Vector &x, Vector & U);
// H(curl)
void curlU_exact(const Vector &x, Vector &curlU);
double curlU2D_exact(const Vector &x);
// H(div)
double divU_exact(const Vector &x);

int dim;
int prob=0;
Vector alpha;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   const char *mesh_file = "../../data/inline-quad.mesh";
   int order = 1;
   bool visualization = 1;
   int sr = 1;
   int pr = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree)");
   args.AddOption(&prob, "-prob", "--problem",
                  "Problem kind: 0: H1, 1: H(curl), 2: H(div)");
   args.AddOption(&sr, "-sr", "--serial_ref",
                  "Number of serial refinements.");
   args.AddOption(&pr, "-pr", "--parallel_ref",
                  "Number of parallel refinements.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // 3. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   dim = mesh->Dimension();  if (dim == 1 ) prob = 0;

   // 4. Set up parameters for exact solution
   alpha.SetSize(dim); // x,y,z coefficients of the solution
   for (int i=0; i<dim; i++) { alpha(i) = M_PI*(double)(i+1);}

   // 5. Refine the serial mesh on all processors to increase the resolution.
   for (int i = 0; i < sr; i++ )
   {
      mesh->UniformRefinement();
   }

   // 6. Define a parallel mesh by a partitioning of the serial mesh. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   // 7. Define a parallel finite element space on the parallel mesh.
   FiniteElementCollection *fec=nullptr;
   switch (prob)
   {
      case 0: fec = new H1_FECollection(order,dim);   break;
      case 1: fec = new ND_FECollection(order,dim);   break;
      case 2: fec = new RT_FECollection(order-1,dim); break;
      default: break;
   }
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);

   // 8. Define the solution vector u_gf as a parallel finite element grid function
   //     corresponding to fespace.
   ParGridFunction u_gf(fespace);

   // 9. Set up the parallel linear form b(.) and the parallel bilinear form
   //    a(.,.).
   FunctionCoefficient *u=nullptr;
   FunctionCoefficient *divU=nullptr;
   FunctionCoefficient *curlU2D=nullptr;
   VectorFunctionCoefficient *U=nullptr;
   VectorFunctionCoefficient *gradu=nullptr;
   VectorFunctionCoefficient *curlU=nullptr;

   ConstantCoefficient one(1.0);
   ParLinearForm b(fespace);
   ParBilinearForm a(fespace);

   switch (prob)
   {
      case 0:
         //(grad u_ex, grad v) + (u_ex,v)
         u = new FunctionCoefficient(u_exact);
         gradu = new VectorFunctionCoefficient(dim,gradu_exact);
         b.AddDomainIntegrator(new DomainLFGradIntegrator(*gradu));
         b.AddDomainIntegrator(new DomainLFIntegrator(*u));

         // (grad u, grad v) + (u,v)
         a.AddDomainIntegrator(new DiffusionIntegrator(one));
         a.AddDomainIntegrator(new MassIntegrator(one));

         break;
      case 1:
         //(curl u_ex, curl v) + (u_ex,v)
         U = new VectorFunctionCoefficient(dim,U_exact);
         if (dim == 3)
         {
            curlU = new VectorFunctionCoefficient(dim,curlU_exact);
            b.AddDomainIntegrator(new VectorFEDomainLFCurlIntegrator(*curlU));
         }
         else if (dim == 2)
         {
            curlU2D = new FunctionCoefficient(curlU2D_exact);
            b.AddDomainIntegrator(new VectorFEDomainLFCurlIntegrator(*curlU2D));
         }
         b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(*U));

         // (curl u, curl v) + (u,v)
         a.AddDomainIntegrator(new CurlCurlIntegrator(one));
         a.AddDomainIntegrator(new VectorFEMassIntegrator(one));
         break;

      case 2:
         //(div u_ex, div v) + (u_ex,v)
         U = new VectorFunctionCoefficient(dim,U_exact);
         divU = new FunctionCoefficient(divU_exact);
         b.AddDomainIntegrator(new VectorFEDomainLFDivIntegrator(*divU));
         b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(*U));

         // (div u, div v) + (u,v)
         a.AddDomainIntegrator(new DivDivIntegrator(one));
         a.AddDomainIntegrator(new VectorFEMassIntegrator(one));
         break;

      default:
         break;
   }

   // 10. Perform successive parallel refinements, compute the L2 error and the
   //     corresponding rate of convergence
   double L2err0 = 0.0;
   for (int l = 0; l <= pr; l++)
   {
      b.Assemble();
      a.Assemble();
      Array<int> ess_tdof_list;
      if (pmesh->bdr_attributes.Size())
      {
         Array<int> ess_bdr(pmesh->bdr_attributes.Max());
         ess_bdr = 0;
         fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      }
      OperatorPtr A;
      Vector X, B;
      a.FormLinearSystem(ess_tdof_list, u_gf, b, A, X,B);

      Solver *prec = NULL;
      switch (prob)
      {
         case 0:
            prec = new HypreBoomerAMG(*A.As<HypreParMatrix>());
            dynamic_cast<HypreBoomerAMG *>(prec)->SetPrintLevel(0);
            break;
         case 1:
            prec = new HypreAMS(*A.As<HypreParMatrix>(), fespace);
            dynamic_cast<HypreAMS *>(prec)->SetPrintLevel(0);
            break;
         case 2:
            if (dim == 2)
            {
               prec = new HypreAMS(*A.As<HypreParMatrix>(), fespace);
               dynamic_cast<HypreAMS *>(prec)->SetPrintLevel(0);
            }
            else
            {
               prec = new HypreADS(*A.As<HypreParMatrix>(), fespace);
               dynamic_cast<HypreADS *>(prec)->SetPrintLevel(0);
            }
            break;
         default:
            break;
      }

      CGSolver cg(MPI_COMM_WORLD);
      cg.SetRelTol(1e-12);
      cg.SetMaxIter(2000);
      cg.SetPrintLevel(0);
      if (prec) { cg.SetPreconditioner(*prec); }
      cg.SetOperator(*A);
      cg.Mult(B, X);
      delete prec;

      a.RecoverFEMSolution(X,B,u_gf);

      double L2err = 0.0;
      switch (prob)
      {
         case 0:
            L2err = u_gf.ComputeL2Error(*u);
            break;
         case 1:
         case 2:
            L2err = u_gf.ComputeL2Error(*U);
            break;
         default:
            break;
      }
      if (myid == 0)
      {
         double rate=0.0;
         if (l>0)
         {
            rate = log(L2err0/L2err)/log(2.0);
         }
         cout << setprecision(3);

         cout << "|| u_h - u ||_{L^2} = " << scientific
              << L2err << ",  rate: " << fixed << rate << endl;
         L2err0 = L2err;
      }

      if (l==pr) break;

      pmesh->UniformRefinement();
      fespace->Update();
      a.Update();
      b.Update();
      u_gf.Update();
   }

   // 11. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      string keys;
      if (dim ==2 )
      {
         keys = "keys UUmrRljc\n";
      }
      else
      {
         keys = "keys mc\n";
      }
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *pmesh << u_gf <<
               "window_title 'Numerical Pressure (real part)' "
               << keys << flush;
   }

   // 12. Free the used memory.
   delete u;
   delete divU;
   delete curlU2D;
   delete U;
   delete gradu;
   delete curlU;
   delete fespace;
   delete fec;
   delete pmesh;

   MPI_Finalize();

   return 0;
}

double u_exact(const Vector &x)
{
   double u;
   double y=0;
   for (int i=0; i<dim; i++)
   {
      y+= alpha(i) * x(i);
   }
   u = cos(y);
   return u;
}

void gradu_exact(const Vector &x, Vector &du)
{
   double s=0.0;
   for (int i=0; i<dim; i++)
   {
      s+= alpha(i) * x(i);
   }
   for (int i=0; i<dim; i++)
   {
      du[i] = -alpha(i) * sin(s);
   }
}

void U_exact(const Vector &x, Vector & U)
{
   double s = x.Sum();
   for (int i=0; i<dim; i++)
   {
      U[i] = cos(alpha(i) * s);
   }
}
// H(curl)
void curlU_exact(const Vector &x, Vector &curlU)
{
   MFEM_VERIFY(dim == 3, "This should be called only for 3D cases");

   double s = x.Sum();
   curlU[0] = -alpha(2)*sin(alpha(2) * s) + alpha(1)*sin(alpha(1) * s);
   curlU[1] = -alpha(0)*sin(alpha(0) * s) + alpha(2)*sin(alpha(2) * s);
   curlU[2] = -alpha(1)*sin(alpha(1) * s) + alpha(0)*sin(alpha(0) * s);
}

double curlU2D_exact(const Vector &x)
{
   MFEM_VERIFY(dim == 2, "This should be called only for 2D cases");
   double s = x(0) + x(1);
   return -alpha(1)*sin(alpha(1) * s) + alpha(0)*sin(alpha(0) * s);
}

// H(div)
double divU_exact(const Vector &x)
{
   double divu = 0.0;
   double s = x.Sum();

   for (int i = 0; i<dim; i++)
   {
      divu += -alpha(i) * sin(alpha(i) * s);
   }
   return divu;
}

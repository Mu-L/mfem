//                      MFEM Example 31
//
// Compile with: make ex42
//
// Sample runs:  mpirun -np 4 ex31 -m ../data/square-disc.mesh -alpha 0.33 -o 2
//               mpirun -np 4 ex31 -m ../data/star.mesh -alpha 0.99 -o 3
//               mpirun -np 4 ex31 -m ../data/inline-quad.mesh -alpha 0.2 -o 3
//               mpirun -np 4 ex31 -m ../data/disc-nurbs.mesh -alpha 0.33 -o 3
//
//
// Description:
//
//  In this example we solve the following fractional PDE with MFEM:
//
//    ( - Δ )^α u = f  in Ω,      u = 0  on ∂Ω,      0 < α < 1,
//
//  To solve this FPDE, we rely on a rational approximation [2] of the normal
//  linear operator A^{-α}, where A = - Δ (with associated homogenous
//  boundary conditions). Namely, we first approximate the operator
//
//    A^{-α} ≈ Σ_{i=0}^N c_i (A + d_i I)^{-1},      d_0 = 0,   d_i > 0,
//
//  where I is the L2-identity operator and the coefficients c_i and d_i
//  are generated offline to a prescribed accuracy in a pre-processing step.
//  We use the triple-A algorithm [1] to generate the rational approximation
//  that this partial fractional expansion derives from. We then solve N+1
//  independent integer-order PDEs,
//
//    A u_i + d_i u_i = c_i f  in Ω,      u_i = 0  on ∂Ω,      i=0,...,N,
//
//  using MFEM and sum u_i to arrive at an approximate solution of the FPDE
//
//    u ≈ Σ_{i=0}^N u_i.
//
//
// References:
//
// [1] Nakatsukasa, Y., Sète, O., & Trefethen, L. N. (2018). The AAA algorithm
//     for rational approximation. SIAM Journal on Scientific Computing, 40(3),
//     A1494-A1522.
//
// [2] Harizanov, S., Lazarov, R., Margenov, S., Marinov, P., & Pasciak, J.
//     (2020). Analysis of numerical methods for spectral fractional elliptic
//     equations based on the best uniform rational approximation. Journal of
//     Computational Physics, 408, 109285.
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "ex31.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 0. Initialize MPI.
   MPI_Session mpi;
   int num_procs = mpi.WorldSize();
   int myid = mpi.WorldRank();

   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   bool visualization = true;
   double alpha = 0.2;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&alpha, "-alpha", "--alpha",
                  "Fractional exponent");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   Array<double> coeffs, poles;

   // 2. Compute the coefficients that define the integer-order PDEs.
   ComputePartialFractionApproximation(alpha,coeffs,poles);

   // 3. Read the mesh from the given mesh file.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   // 4. Refine the mesh to increase the resolution.
   mesh.UniformRefinement();
   mesh.UniformRefinement();
   mesh.UniformRefinement();

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   // 5. Define a finite element space on the mesh.
   FiniteElementCollection *fec = new H1_FECollection(order, dim);
   ParFiniteElementSpace fespace(&pmesh, fec);
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: "
           << fespace.GetTrueVSize() << endl;
   }

   // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
   Array<int> ess_tdof_list;
   if (pmesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh.bdr_attributes.Max());
      ess_bdr = 1;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 7. Define diffusion coefficient, load, and solution GridFunction.
   ConstantCoefficient f(1.0);
   ConstantCoefficient one(1.0);
   ParGridFunction u(&fespace);
   u = 0.;

   // 8. Prepare for visualization.
   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream xout;
   socketstream uout;
   if (visualization)
   {
      xout.open(vishost, visport);
      xout.precision(8);
      uout.open(vishost, visport);
      uout.precision(8);
   }

   for (int i = 0; i<coeffs.Size(); i++)
   {
      // 9. Set up the linear form b(.) for integer-order PDE solve.
      ParLinearForm b(&fespace);
      ProductCoefficient cf(coeffs[i], f);
      b.AddDomainIntegrator(new DomainLFIntegrator(cf));
      b.Assemble();

      // 10. Define GridFunction for integer-order PDE solve.
      ParGridFunction x(&fespace);
      x = 0.0;

      // 11. Set up the bilinear form a(.,.) for integer-order PDE solve.
      ParBilinearForm a(&fespace);
      a.AddDomainIntegrator(new DiffusionIntegrator(one));
      ConstantCoefficient c2(-poles[i]);
      a.AddDomainIntegrator(new MassIntegrator(c2));
      a.Assemble();

      // 12. Assemble the bilinear form and the corresponding linear system.
      OperatorPtr A;
      Vector B, X;
      a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

      // 13. Solve the linear system A X = B.
      Solver *M = new OperatorJacobiSmoother(a, ess_tdof_list);;
      CGSolver cg(MPI_COMM_WORLD);
      cg.SetRelTol(1e-12);
      cg.SetMaxIter(2000);
      cg.SetPrintLevel(3);
      cg.SetPreconditioner(*M);
      cg.SetOperator(*A);
      cg.Mult(B, X);
      delete M;

      // 14. Recover the solution as a finite element grid function.
      a.RecoverFEMSolution(X, b, x);

      // 15. Accumulate integer-order PDE solutions.
      u+=x;

      // 16. Send the solutions by socket to a GLVis server.
      if (visualization)
      {
         xout << "parallel " << num_procs << " " << myid << "\n";
         xout << "solution\n" << pmesh << x << flush;
         uout << "parallel " << num_procs << " " << myid << "\n";
         uout << "solution\n" << pmesh << u << flush;
      }
   }

   delete fec;
   return 0;
}

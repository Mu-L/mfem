#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "vec_coeffs.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   const char *mesh_file = "mesh/2d_mesh.mesh";
   bool visualization = true;
   bool mixed_bilinear_form = false;

   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   ifstream temp_log("input/psi.gf");
   GridFunction psi(&mesh, temp_log);

   cout << "Mesh loaded" << endl;

   const char *new_mesh_file = "mesh/2d_mesh.mesh";
   Mesh *new_mesh = new Mesh(new_mesh_file, 1, 1);

   ND_FECollection fec(1, dim);
   FiniteElementSpace fespace(new_mesh, &fec);

   
   GridFunction B_pol(&fespace);
   cout << B_pol.FESpace()->GetTrueVSize() << endl;
   B_pol = 0.0;
   LinearForm b(&fespace);
   if (!mixed_bilinear_form)
   {
      cout << "Using linear form" << endl;
      // project the grid function onto the new space
      

      // 1. make the linear form
      PsiVectorGridFunctionCoefficient psi_coef(&psi, false);
      b.AddDomainIntegrator(new VectorFEDomainLFCurlIntegrator(psi_coef));

      PsiVectorGridFunctionCoefficient neg_psi_coef(&psi, true);
      b.AddBoundaryIntegrator(new VectorFEDomainLFIntegrator(neg_psi_coef));
      b.Assemble();
   }
   else
   {
      cout << "Using bilinear form" << endl;
      // project the grid function onto the new space
      

      // 1.a make the RHS bilinear form
      // Assert that the two spaces are on the same mesh
      MFEM_ASSERT(psi.FESpace()->GetMesh()->GetNE() == fespace.GetMesh()->GetNE(), "The two spaces are not on the same mesh");
      MixedBilinearForm b_bi(psi.FESpace(), &fespace);
      ConstantCoefficient one(1.0);
      b_bi.AddDomainIntegrator(new MixedScalarWeakCurlIntegrator(one));
      b_bi.Assemble();

      // 1.b form linear form from bilinear form
      LinearForm b_li(&fespace);
      b_bi.Mult(psi, b_li);
      PsiVectorGridFunctionCoefficient neg_psi_coef(&psi, true);
      b.AddBoundaryIntegrator(new VectorFEDomainLFIntegrator(neg_psi_coef));
      b.Assemble();
      b += b_li;
   }
   // 2. make the bilinear form
   BilinearForm a(&fespace);
   RGridFunctionCoefficient r_coef;
   a.AddDomainIntegrator(new VectorFEMassIntegrator(r_coef));
   a.Assemble();
   a.Finalize();

   // 3. solve the system
   CGSolver M_solver;
   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-24);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(1e5);
   M_solver.SetPrintLevel(1);
   M_solver.SetOperator(a.SpMat());

   Vector X(B_pol.Size());
   X = 0.0;
   M_solver.Mult(b, X);

   B_pol.SetFromTrueDofs(X);

   if (visualization)
   {
      char vishost[] = "localhost";
      int visport = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n"
               << *new_mesh << B_pol << flush;
   }

   // paraview
   {
      ParaViewDataCollection paraview_dc("B_pol_Hcurl", new_mesh);
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(1);
      paraview_dc.SetCycle(0);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetTime(0.0); // set the time
      paraview_dc.RegisterField("B_pol_Hcurl", &B_pol);
      paraview_dc.Save();
   }

   ofstream sol_ofs("output/B_pol_Hcurl.gf");
   sol_ofs.precision(8);
   B_pol.Save(sol_ofs);

   delete new_mesh;

   return 0;
}

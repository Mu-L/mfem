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
   bool project_mesh = true;

   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   ifstream temp_log("input/psi.gf");
   GridFunction temp_psi(&mesh, temp_log);

   cout << "Mesh loaded" << endl;

   const char *out_mesh_file = "mesh/2d_mesh.mesh";
   Mesh *new_mesh = new Mesh(out_mesh_file, 1, 1);

   RT_FECollection fec(0, dim);
   FiniteElementSpace fespace(new_mesh, &fec);
   H1_FECollection scalar_fec(1, dim);
   FiniteElementSpace scalar_fespace(new_mesh, &scalar_fec);
   GridFunction psi(&scalar_fespace);

   if (project_mesh)
   {
      GridFunction psi_projected(&scalar_fespace);

      // 1. make the linear form
      LinearForm b(&scalar_fespace);
      PsiRGridFunctionCoefficient psi_r_coef(&temp_psi, false);
      b.AddDomainIntegrator(new DomainLFIntegrator(psi_r_coef));
      b.Assemble();

      // 2. make the bilinear form
      BilinearForm a(&scalar_fespace);
      RGridFunctionCoefficient r_coef;
      a.AddDomainIntegrator(new MassIntegrator(r_coef));
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

      Vector X(psi_projected.Size());
      X = 0.0;
      M_solver.Mult(b, X);

      psi_projected.SetFromTrueDofs(X);

      psi = psi_projected;
   }
   else
   {
      MFEM_ASSERT(temp_psi.FESpace()->GetMesh()->GetNE() == scalar_fespace.GetMesh()->GetNE(), "The two spaces are not on the same mesh");
      psi = temp_psi;
   }

   
   GridFunction B_pol(&fespace);
   cout << B_pol.FESpace()->GetTrueVSize() << endl;
   B_pol = 0.0;

   LinearForm b(&fespace);

   // project the grid function onto the new space
   

   // 1.a make the RHS bilinear form
   MixedBilinearForm b_bi(psi.FESpace(), &fespace);
   DenseMatrix perp_rotation(dim);
   perp_rotation(0, 0) = 0.0;
   perp_rotation(0, 1) = -1.0;
   perp_rotation(1, 0) = 1.0;
   perp_rotation(1, 1) = 0.0;
   MatrixConstantCoefficient perp_rot_coef(perp_rotation);
   b_bi.AddDomainIntegrator(new MixedVectorGradientIntegrator(perp_rot_coef));
   b_bi.Assemble();

   // 1.b form linear form from bilinear form
   LinearForm b_li(&fespace);
   b_bi.Mult(psi, b_li);
   b.Assemble();
   b += b_li;

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
      ParaViewDataCollection paraview_dc("B_pol_Hdiv", new_mesh);
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(1);
      paraview_dc.SetCycle(0);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetTime(0.0); // set the time
      paraview_dc.RegisterField("B_pol_Hdiv", &B_pol);
      paraview_dc.Save();
   }

   ofstream sol_ofs("output/B_pol_Hdiv.gf");
   sol_ofs.precision(8);
   B_pol.Save(sol_ofs);

   delete new_mesh;

   return 0;
}
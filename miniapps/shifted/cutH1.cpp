//            MFEM Example for cutting an H1 space along select faces.
//
// Compile with: make cutH1
//
// Sample runs:
//   ./cutH1 -m ../../data/star.mesh -rs 1
//

#include "mfem.hpp"
#include "../common/mfem-common.hpp"

using namespace std;
using namespace mfem;
using namespace common;

// Used for debugging the elem-to-dof tables when the elements' attributes
// are associated with materials. Options for lvl:
// 0 - only duplicated materials per DOF.
// 1 - all materials per DOF .
// 2 - full element/material output per DOF.
void PrintDofElemTable(const Table &elem_dof, const ParMesh &pmesh,
                       int lvl, bool boundary)
{
   Table dof_elem;
   Transpose(elem_dof, dof_elem);

   const int nrows = dof_elem.Size();
   if (boundary == false)
   {
      std::cout << "--- Dof-to-Elem. Total elem DOFs: " << nrows << std::endl;
   }
   else
   {
      std::cout << "--- Dof-to-Bdr. Total bndry DOFs: " << nrows << std::endl;
   }

   Array<int> dof_elements;
   for (int dof = 0; dof < nrows; dof++)
   {
      // Find the materials that share the current dof.
      std::set<int> dof_materials;
      dof_elem.GetRow(dof, dof_elements);
      if (lvl == 2) { std::cout << "Elements for DOF " << dof << ": \n"; }
      for (int e = 0; e < dof_elements.Size(); e++)
      {
         int mat_id;
         if (boundary == false)
         {
            mat_id = pmesh.GetAttribute(dof_elements[e]);
         }
         else
         {
            int face_id = pmesh.GetBdrFace(dof_elements[e]);
            int elem_id, tmp;
            pmesh.GetFaceElements(face_id, &elem_id, &tmp);
            mat_id = pmesh.GetAttribute(elem_id);
         }

         if (lvl == 2) { cout << dof_elements[e] << "(" << mat_id << ") "; }

         dof_materials.insert(mat_id);
      }
      if (lvl == 2) { std::cout << std::endl; }

      if (lvl == 2) { continue; }
      if (lvl == 0 && dof_materials.size() < 2) { continue; }

      std::cout << "Materials for DOF " << dof << ": " << std::endl;
      for (auto it = dof_materials.cbegin(); it != dof_materials.cend(); it++)
      { std::cout << *it << ' '; }
      std::cout << std::endl;
   }
   std::cout << "--- End of Table" << std::endl;
}

void VisualizeL2(ParGridFunction &gf, int size, int x, int y)
{
   int myid = Mpi::WorldRank();
   int num_procs = Mpi::WorldSize();

   ParMesh *pmesh = gf.ParFESpace()->GetParMesh();
   const int order = gf.ParFESpace()->GetOrder(0);
   L2_FECollection fec(order, pmesh->Dimension());
   ParFiniteElementSpace pfes(pmesh, &fec);
   ParGridFunction gf_l2(&pfes);
   gf_l2.ProjectGridFunction(gf);

   char vishost[] = "localhost";
   int  visport   = 19916;

   socketstream sol_sock(vishost, visport);
   sol_sock << "parallel " << num_procs << " " << myid << "\n";
   sol_sock.precision(8);
   sol_sock << "solution\n" << *pmesh << gf_l2;
   sol_sock << "window_geometry " << x << " " << y << " "
            << size << " " << size << "\n"
            << "window_title '" << "Y" << "'\n"
            << "keys mRjlc\n" << flush;
}

void cutH1Space(ParFiniteElementSpace &pfes, bool vis, bool print)
{
   ParMesh &pmesh = *pfes.GetParMesh();
   ParGridFunction x_vis(&pfes);

   // Duplicate DOFs on the material interface.
   // That is, the DOF touches different element attributes.
   const Table &elem_dof = pfes.GetElementToDofTable(),
                &bdre_dof = pfes.GetBdrElementToDofTable();
   Table dof_elem, dof_bdre;
   Table new_elem_dof(elem_dof), new_bdre_dof(bdre_dof);
   Transpose(elem_dof, dof_elem);
   Transpose(bdre_dof, dof_bdre);
   const int nrows = dof_elem.Size(), n_bdr_dofs = dof_bdre.Size();
   int ndofs = nrows;
   Array<int> dof_elements, dof_boundaries;
   if (print)
   {
      PrintDofElemTable(elem_dof, pmesh, 0, false);
      PrintDofElemTable(bdre_dof, pmesh, 2, true);
   }
   for (int dof = 0; dof < nrows; dof++)
   {
      // Check which materials share the current dof.
      std::set<int> dof_materials;
      dof_elem.GetRow(dof, dof_elements);
      for (int e = 0; e < dof_elements.Size(); e++)
      {
         const int mat_id = pmesh.GetAttribute(dof_elements[e]);
         dof_materials.insert(mat_id);
      }
      // Count the materials for the current DOF.
      const int dof_mat_cnt = dof_materials.size();

      // Duplicate the dof if it is shared between materials.
      if (dof_mat_cnt > 1)
      {
         // The material with the lowest index keeps the old DOF id.
         // All other materials duplicate the dof.
         auto mat = dof_materials.cbegin();
         mat++;
         while (mat != dof_materials.cend())
         {
            // Replace in all elements with material mat.
            const int new_dof_id = ndofs;
            for (int e = 0; e < dof_elements.Size(); e++)
            {
               if (pmesh.GetAttribute(dof_elements[e]) == *mat)
               {
                  if (print)
                  {
                     std::cout << "Replacing DOF (for element) : "
                               << dof << " -> " << new_dof_id
                               << " in EL " << dof_elements[e] << std::endl;
                  }
                  new_elem_dof.ReplaceConnection(dof_elements[e],
                                                 dof, new_dof_id);
               }
            }

            // Replace in all boundary elements with material mat.
            int dof_bdr_cnt = 0;
            if (dof < n_bdr_dofs)
            {
               dof_bdre.GetRow(dof, dof_boundaries);
               dof_bdr_cnt = dof_boundaries.Size();
            }
            for (int b = 0; b < dof_bdr_cnt; b++)
            {
               int face_id = pmesh.GetBdrFace(dof_boundaries[b]);
               int elem_id, tmp;
               pmesh.GetFaceElements(face_id, &elem_id, &tmp);
               if (pmesh.GetAttribute(elem_id) == *mat)
               {
                  std::cout << "Replacing DOF (for boundary): "
                            << dof << " -> " << new_dof_id
                            << " in BE " << dof_boundaries[b] << std::endl;
                  new_bdre_dof.ReplaceConnection(dof_boundaries[b],
                                                 dof, new_dof_id);
               }
            }

            // TODO go over faces (in face_dof) that have the replaced dof (the
            // old id), and check if they have the higher el-attributes on
            // noth sides. For such faces, the face_dof table should be updated
            // with the  new_dof_id.
            // These are faces that touch the interface at a point or an edge.

            ndofs++;
            mat++;
         }
      }

      // Used only for visualization.
      // Must be visualized before the space update.
      x_vis(dof) = dof_mat_cnt;
   }

   // Send the solution by socket to a GLVis server.
   if (vis)
   {
      int size = 500;
      char vishost[] = "localhost";
      int  visport   = 19916;
      const int myid = pfes.GetMyRank(), num_procs = pfes.GetNRanks();

      socketstream sol_sock_x(vishost, visport);
      sol_sock_x << "parallel " << num_procs << " " << myid << "\n";
      sol_sock_x.precision(8);
      sol_sock_x << "solution\n" << pmesh << x_vis;
      sol_sock_x << "window_geometry " << 0 << " " << 0 << " "
                 << size << " " << size << "\n"
                 << "window_title '" << "X" << "'\n"
                 << "keys mRjlc\n" << flush;
   }

   if (print)
   {
      PrintDofElemTable(elem_dof, pmesh, 0, false);
      PrintDofElemTable(new_elem_dof, pmesh, 0, false);
   }

   // Remove face dofs for cut faces.
   const Table &face_dof = pfes.GetFaceToDofTable();
   Table new_face_dof(face_dof);
   for (int f = 0; f < pmesh.GetNumFaces(); f++)
   {
      auto *ftr = pmesh.GetFaceElementTransformations(f, 3);
      if (ftr->Elem2No > 0 &&
          pmesh.GetAttribute(ftr->Elem1No) != pmesh.GetAttribute(ftr->Elem2No))
      {
         if (print)
         {
            std::cout << ftr->Elem1No << " " << ftr->Elem2No << std::endl;
            std::cout << pmesh.GetAttribute(ftr->Elem1No) << " "
                      << pmesh.GetAttribute(ftr->Elem2No) << std::endl;
            std::cout << "Removing face dofs for face " << f << std::endl;
         }
         new_face_dof.RemoveRow(f);
      }
   }
   new_face_dof.Finalize();

   // Cut the space.
   pfes.ReplaceElemDofTable(new_elem_dof, ndofs);
   pfes.ReplaceBdrElemDofTable(new_bdre_dof);
   pfes.ReplaceFaceDofTable(new_face_dof);
}

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   Mpi::Init(argc, argv);
   int myid = Mpi::WorldRank();
   int num_procs = Mpi::WorldSize();
   Hypre::Init();

   // 2. Parse command-line options.
   const char *mesh_file = "../../data/inline-quad.mesh";
   int rs_levels = 0;
   int order = 2;
   const char *device_config = "cpu";
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
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
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }

   // 3. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (myid == 0) { device.Print(); }

   // Refine the mesh.
   Mesh mesh(mesh_file, 1, 1);
   const int dim = mesh.Dimension();
   for (int lev = 0; lev < rs_levels; lev++) { mesh.UniformRefinement(); }

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   H1_FECollection fec(order, dim);
   ParFiniteElementSpace pfes(&pmesh, &fec);

   // Assign material indices to the element attributes.
   const int NE = pmesh.GetNE();
   for (int i = 0; i < NE; i++)
   {
      Vector center;
      pmesh.GetElementCenter(i, center);
      Element *el = pmesh.GetElement(i);
      if (center(0) <= 0.5 || center(1) >= 0.5)
      {
         el->SetAttribute(0);
      }
      else { el->SetAttribute(1); }
   }

   cutH1Space(pfes, true, true);

   // Set face_attribute = 77 to faces that are on the material interface.
   // Remove face dofs for cut faces.
   for (int f = 0; f < pmesh.GetNumFaces(); f++)
   {
      auto *ftr = pmesh.GetFaceElementTransformations(f, 3);
      if (ftr->Elem2No > 0 &&
          pmesh.GetAttribute(ftr->Elem1No) != pmesh.GetAttribute(ftr->Elem2No))
      {
         pmesh.SetFaceAttribute(f, 77);
      }
   }

   // Simple Dirichlet BC.
   Array<int> ess_tdof_list;
   if (pmesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh.bdr_attributes.Max());
      ess_bdr = 1;
      pfes.FiniteElementSpace::GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // RHS.
   ParLinearForm b(&pfes);
   ConstantCoefficient one(1.0);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();

   // LHS.
   BilinearForm a(&pfes);
   a.AddDomainIntegrator(new DiffusionIntegrator(one));
   Array<int> cut_face_attributes(1);
   cut_face_attributes[0] = 77;
   const double sigma = -1.0, kappa = -1.0;
   a.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa),
                               &cut_face_attributes);
   a.Assemble();

   // Form the system.
   ParGridFunction u(&pfes);
   u = 0.0;
   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, u, b, A, X, B);

   // Solve.
   GSSmoother M((SparseMatrix&)(*A));
   //PCG(*A, M, B, X, 1, 200, 1e-12, 0.0);
   CG(*A, B, X, 1, 1000, 1e-12, 0.0);
   a.RecoverFEMSolution(X, b, u);

   VisualizeL2(u, 500, 500, 0);

   const double norm = u.Norml2();
   std::cout << "Norm: " << norm << std::endl;
   return 0;
}

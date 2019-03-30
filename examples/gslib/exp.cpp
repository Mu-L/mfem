﻿// mpirun -np 2 ./exp -m RT2D.mesh -qo 8 -o 3

#include "mfem.hpp"
#include "fem/gslib.hpp" // TODO move to mfem.hpp (double declaration bug ??)

#include <fstream>
#include <ctime>

using namespace mfem;
using namespace std;

IntegrationRules IntRulesLo(0, Quadrature1D::GaussLobatto);

int main (int argc, char *argv[])
{
   // Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // Set the method's default parameters.
   const char *mesh_file = "icf.mesh";
   int mesh_poly_deg     = 1;
   int rs_levels         = 0;
   int rp_levels         = 0;
   int quad_type         = 1;
   int quad_order        = 8;

   // Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&mesh_poly_deg, "-o", "--order",
                  "Polynomial degree of mesh finite element space.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&rp_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&quad_order, "-qo", "--quad_order",
                  "Order of the quadrature rule.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }

   // Initialize and refine the starting mesh.
   Mesh *mesh = new Mesh(mesh_file, 1, 1, false);
   for (int lev = 0; lev < rs_levels; lev++) { mesh->UniformRefinement(); }
   const int dim = mesh->Dimension();
   if (myid == 0)
   {
      cout << "Mesh curvature: ";
      if (mesh->GetNodes()) { cout << mesh->GetNodes()->OwnFEC()->Name(); }
      else { cout << "(NONE)"; }
      cout << endl;
   }
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < rp_levels; lev++) { pmesh->UniformRefinement(); }

   // 4. Define a finite element space on the mesh. Here we use vector finite
   //    elements which are tensor products of quadratic finite elements. The
   //    number of components in the vector finite element space is specified by
   //    the last parameter of the FiniteElementSpace constructor.
   FiniteElementCollection *fec;
   if (mesh_poly_deg <= 0)
   {
      fec = new QuadraticPosFECollection;
      mesh_poly_deg = 2;
   }
   else { fec = new H1_FECollection(mesh_poly_deg, dim); }
   ParFiniteElementSpace *pfespace = new ParFiniteElementSpace(pmesh, fec, dim);

   // 5. Make the mesh curved based on the above finite element space. This
   //    means that we define the mesh elements through a fespace-based
   //    transformation of the reference element.
   pmesh->SetNodalFESpace(pfespace);

   // 7. Get the mesh nodes (vertices and other degrees of freedom in the finite
   //    element space) as a finite element grid function in fespace. Note that
   //    changing x automatically changes the shapes of the mesh elements.
   ParGridFunction x(pfespace);
   pmesh->SetNodalGridFunction(&x);

   //x.SetTrueVector();
   //x.SetFromTrueVector();

   // 8. Store the starting (prior to the optimization) positions.
   ParGridFunction x0(pfespace);
   x0 = x;

   // 9. Setup the quadrature rule for the non-linear form integrator.
   const IntegrationRule *ir = NULL;
   const int geom_type = pfespace->GetFE(0)->GetGeomType();
   if (quad_order > 4)
   {
      if (quad_order % 2 == 0) {quad_order = 2*quad_order - 4;}
      else {quad_order = 2*quad_order - 3;}
   }
   switch (quad_type)
   {
      case 1: ir = &IntRulesLo.Get(geom_type, quad_order);
   }
   if (myid==0) {cout << "Quadrature points per cell: " << ir->GetNPoints() << endl;}

   const int NE = pfespace->GetMesh()->GetNE(), nsp = ir->GetNPoints();
   
   ParGridFunction nodes(pfespace);
   pmesh->GetNodes(nodes);

   int NR = sqrt(nsp);
   if (dim==3) {NR = cbrt(nsp);}

   int sz1 = pow(NR,dim);
   Vector fx(dim*NE*sz1);
   Vector dumfield(NE*sz1);
   int np;

   np = 0;
   int tnp = NE*nsp;
   for (int i = 0; i < NE; i++)
   {
      for (int j = 0; j < nsp; j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         fx[np] = nodes.GetValue(i, ip, 1);
         fx[tnp+np] = nodes.GetValue(i, ip, 2);
         dumfield[np] = pow(fx[np],2)+pow(fx[tnp+np],2);
         if (dim==3)
         {
            fx[2*tnp+np] =nodes.GetValue(i, ip, 3);
            dumfield[np] += pow(fx[2*tnp+np],2);
         }
         np = np+1;
      }
   }

   findpts_gslib *gsfl = NULL;
   gsfl = new findpts_gslib(MPI_COMM_WORLD);
   MPI_Barrier(MPI_COMM_WORLD);
   int start_s = clock();
   gsfl->gslib_findpts_setup(pfespace,pmesh,quad_order);
   MPI_Barrier(MPI_COMM_WORLD);
   int stop_s=clock();
   if (myid == 0)
   {
      cout << "findpts order: " << NR << " \n";
      cout << "findpts setup time (sec): " << (stop_s-start_s)/1000000. << endl;
   }

   // generate random points by r,s,t
   int llim = 10;
   int nlim = NE*llim;
   Vector rrxa(nlim),rrya(nlim),rrza(nlim);
   Vector vxyz(nlim*dim);

   np = 0;
   IntegrationPoint ipt;
   for (int i = 0; i < NE; i++)
   {
      for (int j = 0; j < llim; j++)
      {
         Geometries.GetRandomPoint(pfespace->GetFE(i)->GetGeomType(), ipt);

         rrxa[np] = ipt.x;
         rrya[np] = ipt.y;
         if (dim == 3) { rrza[np] = ipt.z; }

         vxyz[np] = nodes.GetValue(i, ipt, 1);
         vxyz[np+nlim] = nodes.GetValue(i, ipt, 2);
         if ( dim==3 ) { vxyz[np+2*nlim] = nodes.GetValue(i, ipt, 3); }

         np++;
      }
   }

   int nxyz = nlim;

   if (myid == 0)
   {
      cout << "Num procs: " << num_procs << " \n";
      cout << "Points per proc: " << nxyz << " \n";
      cout << "Points per elem: " << llim << " \n";
      cout << "Total Points to be found: " << nxyz*num_procs << " \n";
   }

   Array<uint> pel(nxyz), pcode(nxyz), pproc(nxyz);
   Vector pr(nxyz*dim), pd(nxyz);
   MPI_Barrier(MPI_COMM_WORLD);

   start_s = clock();
   gsfl->gslib_findpts(&pcode, &pproc ,&pel, &pr, &pd, &vxyz, nxyz);

   MPI_Barrier(MPI_COMM_WORLD);
   stop_s = clock();

   if (myid == 0)
   {
      cout << "findpts time (sec): " << (stop_s-start_s)/1000000. << endl;
   }

   // FINDPTS_EVAL
   Vector fout(nxyz);
   MPI_Barrier(MPI_COMM_WORLD);
   start_s=clock();
   gsfl->gslib_findpts_eval(&fout, &pcode, &pproc, &pel, &pr, &dumfield, nxyz);
   stop_s=clock();
   if (myid == 0)
   {
      cout << "findpts_eval time (sec): " << (stop_s-start_s)/1000000. << endl;
   }
   gsfl->gslib_findpts_free();

   int nbp = 0, nnpt = 0, nerrh = 0;
   double maxv = -100.0 ,maxvr = -100.0;
   for (int it = 0; it < nxyz; it++)
   {
      if (pcode[it] < 2)
      {
         double val = pow(vxyz[it],2)+pow(vxyz[it+nlim],2);
         if (dim==3) { val += pow(vxyz[it+2*nlim],2); }
         double delv = abs(val-fout[it]);
         double rxe = abs(rrxa[it] - 0.5*pr[it*dim+0]-0.5);
         double rye = abs(rrya[it] - 0.5*pr[it*dim+1]-0.5);
         double rze = abs(rrza[it] - 0.5*pr[it*dim+2]-0.5);
         double delvr =  ( rxe < rye ) ? rye : rxe;
         if (dim==3) { delvr = ( ( delvr < rze ) ? rze : delvr ); }
         if (delv > maxv) {maxv = delv;}
         if (delvr > maxvr) {maxvr = delvr;}
         if (pcode[it] == 1) {nbp += 1;}
         if (delvr > 1.e-10) {nerrh += 1;}
      }
      else { nnpt++; }
   }

   MPI_Barrier(MPI_COMM_WORLD);
   double glob_maxerr;
   MPI_Allreduce(&maxv, &glob_maxerr, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   double glob_maxrerr;
   MPI_Allreduce(&maxvr, &glob_maxrerr, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   int glob_nnpt;
   MPI_Allreduce(&nnpt, &glob_nnpt, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   int glob_nbp;
   MPI_Allreduce(&nbp, &glob_nbp, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   int glob_nerrh;
   MPI_Allreduce(&nerrh, &glob_nerrh, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   if (myid == 0)
   {
      cout << setprecision(16);
      cout << "maximum error: " << glob_maxerr << " \n";
      cout << "maximum rst error: " << glob_maxrerr << " \n";
      cout << "points not found: " << glob_nnpt << " \n";
      cout << "points on element border: " << glob_nbp << " \n";
      cout << "points with error > 1.e-10: " << glob_nerrh << " \n";
   }

   delete pfespace;
   delete fec;
   delete pmesh;
   MPI_Finalize();

   return 0;
}

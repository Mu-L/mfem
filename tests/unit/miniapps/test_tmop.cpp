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

#ifdef _WIN32
#define _USE_MATH_DEFINES
#include <cmath>
#endif

#include <list>
#include <fstream>
#include <iostream>

#include "catch.hpp"

#include "mfem.hpp"
#include "miniapps/meshing/mesh-optimizer.hpp"

#if defined(MFEM_USE_MPI) && defined(MFEM_TMOP_MPI)
extern mfem::MPI_Session *GlobalMPISession;
#define PFesGetParMeshGetComm(pfes) pfes.GetParMesh()->GetComm()
#define SetDiscreteTargetSize SetParDiscreteTargetSize
#define SetDiscreteTargetAspectRatio SetParDiscreteTargetAspectRatio
#define GradientClass HypreParMatrix
#else
typedef int MPI_Session;
#define ParMesh Mesh
#define ParGridFunction GridFunction
#define ParNonlinearForm NonlinearForm
#define ParFiniteElementSpace FiniteElementSpace
#define GetParGridFunctionEnergy GetGridFunctionEnergy
#define PFesGetParMeshGetComm(...)
#define MPI_Allreduce(src,dst,...) *dst = *src
#define SetDiscreteTargetSize SetSerialDiscreteTargetSize
#define SetDiscreteTargetAspectRatio SetSerialDiscreteTargetAspectRatio
#define GradientClass SparseMatrix
#endif

using namespace std;
using namespace mfem;

namespace mfem
{

struct Req
{
   double init_energy;
   double tauval;
   double dot;
   double final_energy;
   double diag;
};

int tmop(int myid, Req &res, int argc, char *argv[])
{
   bool pa               = false;
   const char *mesh_file = nullptr;
   int order             = 1;
   int rs_levels         = 0;
   int metric_id         = 1;
   int target_id         = 1;
   int quad_type         = 1;
   int quad_order        = 2;
   int newton_iter       = 100;
   double newton_rtol    = 1e-8;
   int lin_solver        = 2;
   int max_lin_iter      = 100;
   double lim_const      = 0.0;
   int normalization     = 0;
   double jitter         = 0.0;
   bool diag             = true;
   int newton_loop       = 1;

   constexpr int combomet  = 0;
   constexpr int verbosity_level = 0;
   constexpr int seed = 0x100001b3;
   constexpr bool move_bnd = false;
   constexpr bool fdscheme = false;

   REQUIRE_FALSE(fdscheme);
   REQUIRE(combomet == 0);
   REQUIRE_FALSE(move_bnd);

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "");
   args.AddOption(&order, "-o", "--order", "");
   args.AddOption(&rs_levels, "-rs", "--refine-serial", "");
   args.AddOption(&metric_id, "-mid", "--metric-id", "");
   args.AddOption(&target_id, "-tid", "--target-id", "");
   args.AddOption(&quad_type, "-qt", "--quad-type", "");
   args.AddOption(&quad_order, "-qo", "--quad_order", "");
   args.AddOption(&newton_iter, "-ni", "--newton-iters","");
   args.AddOption(&newton_loop, "-nl", "--newton-loops","");
   args.AddOption(&newton_rtol, "-rtol", "--newton-rel-tolerance", "");
   args.AddOption(&lin_solver, "-ls", "--lin-solver", "");
   args.AddOption(&max_lin_iter, "-li", "--lin-iter", "");
   args.AddOption(&lim_const, "-lc", "--limit-const", "");
   args.AddOption(&normalization, "-nor", "--normalization", "");
   args.AddOption(&pa, "-pa", "--pa", "-no-pa", "--no-pa", "");
   args.AddOption(&jitter, "-ji", "--jitter", "");
   args.AddOption(&diag, "-diag", "--diag", "-no-diag", "--no-diag", "");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }
   if (verbosity_level > 0) { if (myid == 0) {args.PrintOptions(cout); } }

   REQUIRE(mesh_file);
   Mesh smesh(mesh_file, 1, 1, false);
   for (int lev = 0; lev < rs_levels; lev++) { smesh.UniformRefinement(); }
   const int dim = smesh.Dimension();
   ParMesh *pmesh = nullptr;
#if defined(MFEM_USE_MPI) && defined(MFEM_TMOP_MPI)
   pmesh = new ParMesh(MPI_COMM_WORLD, smesh);
#else
   pmesh = new Mesh(smesh);
#endif

   REQUIRE(order > 0);
   H1_FECollection fec(order, dim);
   ParFiniteElementSpace fes(pmesh, &fec, dim);
   pmesh->SetNodalFESpace(&fes);
   ParGridFunction x0(&fes), x(&fes);
   pmesh->SetNodalGridFunction(&x);

   Vector h0(fes.GetNDofs());
   h0 = infinity();
   double volume = 0.0;
   {
      Array<int> dofs;
      for (int i = 0; i < pmesh->GetNE(); i++)
      {
         fes.GetElementDofs(i, dofs);
         const double hi = pmesh->GetElementSize(i);
         for (int j = 0; j < dofs.Size(); j++)
         {
            h0(dofs[j]) = min(h0(dofs[j]), hi);
         }
         volume += pmesh->GetElementVolume(i);
      }
   }
   const double small_phys_size = pow(volume, 1.0 / dim) / 100.0;

   ParGridFunction rdm(&fes);
   rdm.Randomize(seed);
   rdm -= 0.5;
   rdm *= jitter;
   rdm.HostReadWrite();
   // Scale the random values to be of order of the local mesh size.
   for (int i = 0; i < fes.GetNDofs(); i++)
   {
      for (int d = 0; d < dim; d++)
      {
         rdm(fes.DofToVDof(i,d)) *= h0(i);
      }
   }
   Array<int> vdofs;
   for (int i = 0; i < fes.GetNBE(); i++)
   {
      fes.GetBdrElementVDofs(i, vdofs);
      for (int j = 0; j < vdofs.Size(); j++) { rdm(vdofs[j]) = 0.0; }
   }
   x -= rdm;
   x.SetTrueVector();
   x.SetFromTrueVector();
   x0 = x;

   TMOP_QualityMetric *metric = nullptr;
   switch (metric_id)
   {
      case   1: metric = new TMOP_Metric_001; break;
      case   2: metric = new TMOP_Metric_002; break;
      case   7: metric = new TMOP_Metric_007; break;
      case 302: metric = new TMOP_Metric_302; break;
      case 303: metric = new TMOP_Metric_303; break;
      case 321: metric = new TMOP_Metric_321; break;
      default:
      {
         if (myid == 0) { cout << "Unknown metric_id: " << metric_id << endl; }
         return 2;
      }
   }

   TargetConstructor::TargetType target_t;
   TargetConstructor *target_c = nullptr;
   HessianCoefficient *adapt_coeff = nullptr;
   constexpr int mesh_poly_deg = 1;
   H1_FECollection ind_fec(mesh_poly_deg, dim);
   ParFiniteElementSpace ind_fes(pmesh, &ind_fec);
   ParGridFunction size(&ind_fes);
   ParFiniteElementSpace ind_fesv(pmesh, &ind_fec, dim);
   ParGridFunction aspr3d(&ind_fesv);
   const AssemblyLevel al =
      pa ? AssemblyLevel::PARTIAL : AssemblyLevel::LEGACYFULL;

   switch (target_id)
   {
      case 1: target_t = TargetConstructor::IDEAL_SHAPE_UNIT_SIZE; break;
      case 2: target_t = TargetConstructor::IDEAL_SHAPE_EQUAL_SIZE; break;
      case 3: target_t = TargetConstructor::IDEAL_SHAPE_GIVEN_SIZE; break;
      case 4: // Analytic
      {
         target_t = TargetConstructor::GIVEN_FULL;
         AnalyticAdaptTC *tc = new AnalyticAdaptTC(target_t);
         adapt_coeff = new HessianCoefficient(dim, metric_id);
         tc->SetAnalyticTargetSpec(NULL, NULL, adapt_coeff);
         target_c = tc;
         break;
      }
      case 5: // Discrete size 2D or 3D
      {
         target_t = TargetConstructor::IDEAL_SHAPE_GIVEN_SIZE;
         DiscreteAdaptTC *tc = new DiscreteAdaptTC(target_t);
         tc->SetAdaptivityEvaluator(new AdvectorCG(al));
         if (dim == 2)
         {
            FunctionCoefficient ind_coeff(discrete_size_2d);
            size.ProjectCoefficient(ind_coeff);
         }
         else if (dim == 3)
         {
            FunctionCoefficient ind_coeff(discrete_size_3d);
            size.ProjectCoefficient(ind_coeff);
         }
         tc->SetDiscreteTargetSize(size);
         target_c = tc;
         break;
      }
      case 7: // aspect-ratio 3D
      {
         target_t = TargetConstructor::GIVEN_SHAPE_AND_SIZE;
         DiscreteAdaptTC *tc = new DiscreteAdaptTC(target_t);
         tc->SetAdaptivityEvaluator(new AdvectorCG(al));
         VectorFunctionCoefficient fd_aspr3d(dim, discrete_aspr_3d);
         aspr3d.ProjectCoefficient(fd_aspr3d);
         tc->SetDiscreteTargetAspectRatio(aspr3d);
         target_c = tc;
         break;
      }
      default:
      {
         if (myid == 0) { cout << "Unknown target_id: " << target_id << endl; }
         return 3;
      }
   }
#if defined(MFEM_USE_MPI) && defined(MFEM_TMOP_MPI)
   if (target_c == NULL)
   {
      target_c = new TargetConstructor(target_t, MPI_COMM_WORLD);
   }
#else
   if (target_c == nullptr)
   {
      target_c = new TargetConstructor(target_t);
   }
#endif
   target_c->SetNodes(x0);

   // Setup the quadrature rule for the non-linear form integrator.
   const IntegrationRule *ir = nullptr;
   IntegrationRules IntRulesLo(0, Quadrature1D::GaussLobatto);
   IntegrationRules IntRulesCU(0, Quadrature1D::ClosedUniform);
   const int geom_type = fes.GetFE(0)->GetGeomType();
   switch (quad_type)
   {
      case 1: ir = &IntRulesLo.Get(geom_type, quad_order); break;
      case 2: ir = &IntRules.Get(geom_type, quad_order); break;
      case 3: ir = &IntRulesCU.Get(geom_type, quad_order); break;
      default:
      {
         if (myid == 0) { cout << "Unknown quad_type: " << quad_type << endl; }
         return 4;
      }
   }

   TMOP_Integrator *he_nlf_integ = new TMOP_Integrator(metric, target_c);
   he_nlf_integ->SetIntegrationRule(*ir);

   if (normalization == 1) { he_nlf_integ->EnableNormalization(x0); }

   ParGridFunction dist(&fes);
   dist = 1.0;
   if (normalization == 1) { dist = small_phys_size; }
   ConstantCoefficient lim_coeff(lim_const);
   if (lim_const != 0.0) { he_nlf_integ->EnableLimiting(x0, dist, lim_coeff); }

   ParNonlinearForm nlf(&fes);
   nlf.SetAssemblyLevel(pa ? AssemblyLevel::PARTIAL : AssemblyLevel::NONE);
   nlf.AddDomainIntegrator(he_nlf_integ);
   nlf.Setup();

   const double init_energy = nlf.GetParGridFunctionEnergy(x);
   res.init_energy = init_energy;

   // Fix all boundary nodes (-fix-bnd)
   Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 1;
   nlf.SetEssentialBC(ess_bdr);

   // Diagonal test
   res.diag = 0.0;
   if (diag)
   {
      Vector d(fes.GetTrueVSize());
      Vector &xt(x.GetTrueVector());
      d.UseDevice(true);
      if (pa)
      {
         nlf.GetGradient(xt);
         nlf.AssembleGradientDiagonal(d);
      }
      else
      {
         ParNonlinearForm nlf_fa(&fes);
         TMOP_Integrator *nlfi_fa = new TMOP_Integrator(metric, target_c);
         nlfi_fa->SetIntegrationRule(*ir);
         if (normalization == 1) { nlfi_fa->EnableNormalization(x0); }
         if (lim_const != 0.0) { nlfi_fa->EnableLimiting(x0, dist, lim_coeff); }
         nlf_fa.AddDomainIntegrator(nlfi_fa);
         // We don't set the EssentialBC in order to get the same diagonal
         // nlf_fa.SetEssentialBC(ess_bdr);
         dynamic_cast<GradientClass&>(nlf_fa.GetGradient(xt)).GetDiag(d);
      }
      res.diag = d*d;
   }

   // Linear solver for the system's Jacobian
   Solver *S = nullptr, *S_prec = nullptr;
   constexpr double linsol_rtol = 1e-12;
   if (lin_solver == 0)
   {
      S = new DSmoother(1, 1.0, max_lin_iter);
   }
   else if (lin_solver == 1)
   {
      CGSolver *cg = new CGSolver(PFesGetParMeshGetComm(fes));
      cg->SetMaxIter(max_lin_iter);
      cg->SetRelTol(linsol_rtol);
      cg->SetAbsTol(0.0);
      cg->SetPrintLevel(verbosity_level >= 2 ? 3 : -1);
      S = cg;
   }
   else
   {
      MINRESSolver *minres = new MINRESSolver(PFesGetParMeshGetComm(fes));
      minres->SetMaxIter(max_lin_iter);
      minres->SetRelTol(linsol_rtol);
      minres->SetAbsTol(0.0);
      minres->SetPrintLevel(verbosity_level >= 2 ? 3 : -1);
      if (lin_solver == 3 || lin_solver == 4)
      {
         if (pa)
         {
            MFEM_VERIFY(lin_solver != 4, "PA l1-Jacobi is not implemented");
            S_prec = new OperatorJacobiSmoother(nlf, nlf.GetEssentialTrueDofs());
         }
#if defined(MFEM_USE_MPI) && defined(MFEM_TMOP_MPI)
         else
         {
            HypreSmoother *hs = new HypreSmoother;
            hs->SetType((lin_solver == 3) ? HypreSmoother::Jacobi
                        : HypreSmoother::l1Jacobi, 1);
            S_prec = hs;
         }
#else
         else { S_prec = new DSmoother((lin_solver == 3) ? 0 : 1, 1.0, 1); }
#endif
         minres->SetPreconditioner(*S_prec);
      }
      S = minres;
   }

   // Compute the minimum det(J) of the starting mesh
   double tauval = infinity();
   const int NE = pmesh->GetNE();
   for (int i = 0; i < NE; i++)
   {
      ElementTransformation *transf = pmesh->GetElementTransformation(i);
      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         transf->SetIntPoint(&ir->IntPoint(j));
         tauval = min(tauval, transf->Jacobian().Det());
      }
   }
   double minJ0;
   MPI_Allreduce(&tauval, &minJ0, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
   tauval = minJ0;
   //if (myid == 0) { cout << "Min det(J) of the mesh is " << tauval << endl; }
   REQUIRE(tauval > 0.0);
   double h0min = h0.Min(), h0min_all;
   MPI_Allreduce(&h0min, &h0min_all, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
   tauval -= 0.01 * h0min_all; // Slightly below minJ0 to avoid div by 0.
   res.tauval = tauval;

   // Perform the nonlinear optimization
   Vector b(0);
   b.UseDevice(true);
#if defined(MFEM_USE_MPI) && defined(MFEM_TMOP_MPI)
   NewtonSolver *newton = new TMOPNewtonSolver(PFesGetParMeshGetComm(fes),*ir);
#else
   NewtonSolver *newton = new TMOPNewtonSolver(*ir);
#endif
   newton->SetPreconditioner(*S);
   newton->SetMaxIter(newton_iter);
   newton->SetRelTol(newton_rtol);
   newton->SetAbsTol(0.0);
   newton->SetPrintLevel(verbosity_level >= 1 ? 1 : -1);
   newton->SetOperator(nlf);

   Vector x_init(x);
   for (int i = 0; i < newton_loop; i++)
   {
      x = x_init;
      x.SetTrueVector();

      DiscreteAdaptTC *datc = dynamic_cast<DiscreteAdaptTC*>(target_c);
      if (datc && target_id == 5) { datc->SetDiscreteTargetSize(size); }
      if (datc && target_id == 7) { datc->SetDiscreteTargetAspectRatio(aspr3d); }

      dist *= 0.93;
      if (normalization == 1) { dist = small_phys_size; }

      ConstantCoefficient lim_coeff(lim_const);
      if (lim_const != 0.0) { he_nlf_integ->EnableLimiting(x0, dist, lim_coeff); }

      if (normalization == 1) { he_nlf_integ->EnableNormalization(x); }

      newton->Mult(b, x.GetTrueVector());
      x.SetFromTrueVector();

      REQUIRE(newton->GetConverged());

      const double final_energy = nlf.GetParGridFunctionEnergy(x);
      res.final_energy = final_energy;
   }

   Vector &x_t(x.GetTrueVector());
   double x_t_dot = x_t*x_t, dot;
   MPI_Allreduce(&x_t_dot, &dot, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());
   res.dot = dot;

   delete S;
   delete S_prec;
   delete pmesh;
   delete metric;
   delete newton;
   delete target_c;
   delete adapt_coeff;

   return 0;
}

} // namespace mfem

static int argn(const char *argv[], int argc =0)
{
   while (argv[argc]) { argc+=1; }
   return argc;
}

static void req_tmop(int myid, const char *args[], Req &res)
{ REQUIRE(tmop(myid, res, argn(args), const_cast<char**>(args))==0); }

#define DEFAULT_ARGS const char *args[] = { \
    "tmop_tests", "-pa", "-m", "mesh", "-o", "0", "-rs", "0", \
    "-mid", "0", "-tid", "0", "-qt", "1", "-qo", "0", \
    "-ni", "10", "-rtol", "1e-8", "-ls", "2", "-li", "100", \
    "-lc", "0", "-nor", "0", "-ji", "0", "-nl", "1", nullptr }
constexpr int ALV = 1;
constexpr int MSH = 3;
constexpr int POR = 5;
constexpr int RS = 7;
constexpr int MID = 9;
constexpr int TID = 11;
constexpr int QTY = 13;
constexpr int QOR = 15;
constexpr int NI  = 17;
constexpr int LS  = 21;
constexpr int LI  = 23;
constexpr int LC  = 25;
constexpr int NOR = 27;
constexpr int JI  = 29;
constexpr int NL  = 31;

static void dump_args(const char *args[])
{
   const char *format =
      "tmop -m %s -o %s -qo %s -mid %s -tid %s -ls %s"
      "%s%s%s%s"     // Optional args: RS, QTY
      "%s%s%s%s%s%s" // Optional args:  LC, NOR, JI & NL
      " %s\n";       // Assembly level
   printf(format,
          args[MSH], args[POR], args[QOR], args[MID], args[TID], args[LS],
          // Optional args: RS, QTY
          args[RS][0] == '0' ? "" : " -rs ",
          args[RS][0] == '0' ? "" : args[RS],
          args[QTY][0] == '1' ? "" : " -qt ",
          args[QTY][0] == '1' ? "" : args[QTY],
          // Optional args:  LC, NOR, JI & NL
          args[LC][0] == '0' ? "" : " -lc ",
          args[LC][0] == '0' ? "" : args[LC],
          args[NOR][0] == '0' ? "" : " -nor",
          args[JI][0] == '0' ? "" : " -jitter",
          args[NL][0] == '1' ? "" : " -nl ",
          args[NL][0] == '1' ? "" : args[NL],
          // Assembly level
          args[ALV]);
   fflush(0);
}

static void tmop_require(int myid, const char *args[])
{
   Req res[2];
   (args[ALV] = "-pa", dump_args(args), req_tmop(myid, args, res[0]));
   (args[ALV] = "-no-pa", dump_args(args), req_tmop(myid, args, res[1]));
   REQUIRE(res[0].dot == Approx(res[1].dot));
   REQUIRE(res[0].tauval == Approx(res[1].tauval));
   REQUIRE(res[0].init_energy == Approx(res[1].init_energy));
   REQUIRE(res[0].final_energy == Approx(res[1].final_energy));
   REQUIRE(res[0].diag == Approx(res[1].diag));
}

static inline const char *itoa(const int i, char *buf)
{
   std::sprintf(buf, "%d", i);
   return buf;
}

static inline const char *dtoa(const double d, char *buf)
{
   std::sprintf(buf, "%.4f", d);
   return buf;
}

class Launch
{
   typedef std::list<int> set;
public:
   class Args
   {
      friend class Launch;
   private:
      const char *name = nullptr;
      const char *mesh = "star.mesh";
      int newton_iter = 100;
      int rs_levels = 0;
      int max_lin_iter  = 100;
      bool normalization = false;
      double lim_const = 0.0;
      double jitter = 0.0;
      set order = {1,2,3,4};
      set target_id = {1,2,3};
      set metric_id = {1,2};
      set quad_order = {2,4,8};
      set lin_solver = {3,2,1};
      set newton_loop = {1,3};

   public:
      Args(const char *name =nullptr): name(name) {}
      Args &MESH(const char *arg) { mesh = arg; return *this; }
      Args &NEWTON_ITERATIONS(const int arg) { newton_iter = arg; return *this; }
      Args &REFINE(const int arg) { rs_levels = arg; return *this; }
      Args &LINEAR_ITERATIONS(const int arg) { max_lin_iter = arg; return *this; }
      Args &NORMALIZATION(const bool arg) { normalization = arg; return *this; }
      Args &LIMITING(const double arg) { lim_const = arg; return *this; }
      Args &JI(const double arg) { jitter = arg; return *this; }

      Args &POR(set arg) { order = arg; return *this; }
      Args &TID(set arg) { target_id = arg; return *this; }
      Args &MID(set arg) { metric_id = arg; return *this; }
      Args &QOR(set arg) { quad_order = arg; return *this; }
      Args &LS(set arg) { lin_solver = arg; return *this; }
      Args &NL(set arg) { newton_loop = arg; return *this; }
   };
   const char *name, *mesh;
   int NEWTON_ITERATIONS, REFINE, LINEAR_ITERATIONS;
   bool NORMALIZATION;
   double LIMITING, JITTER;
   set P_ORDERS, TARGET_IDS, METRIC_IDS, Q_ORDERS, LINEAR_SOLVERS, NEWTON_LOOPS;
public:
   Launch(Args a = Args()):
      name(a.name), mesh(a.mesh),
      NEWTON_ITERATIONS(a.newton_iter), REFINE(a.rs_levels),
      LINEAR_ITERATIONS(a.max_lin_iter),
      NORMALIZATION(a.normalization), LIMITING(a.lim_const), JITTER(a.jitter),
      P_ORDERS(a.order), TARGET_IDS(a.target_id), METRIC_IDS(a.metric_id),
      Q_ORDERS(a.quad_order), LINEAR_SOLVERS(a.lin_solver),
      NEWTON_LOOPS(a.newton_loop)
   { }

   void Run(const int myid =0) const
   {
      static bool all = getenv("MFEM_TESTS_UNIT_TMOP_ALL");
      if (name) { printf("[%s]\n", name); fflush(0); }
      DEFAULT_ARGS;
      char ni[8] {}, rs[8] {}, li[8] {}, lc[16] {}, ji[16] {};
      args[MSH] = mesh;
      args[RS] = itoa(REFINE,rs);
      args[NI] = itoa(NEWTON_ITERATIONS,ni);
      args[LI] = itoa(LINEAR_ITERATIONS,li);
      args[LC] = dtoa(LIMITING,lc);
      args[JI] = dtoa(JITTER,ji);
      args[NOR] = NORMALIZATION ? "1" : "0";
      for (int p : P_ORDERS)
      {
         char por[2] {};
         args[POR] = itoa(p, por);
         for (int t : TARGET_IDS)
         {
            char tid[2] {};
            args[TID] = itoa(t, tid);
            for (int m : METRIC_IDS)
            {
               char mid[4] {};
               args[MID] = itoa(m, mid);
               for (int q : Q_ORDERS)
               {
                  if (q <= p) { continue; }
                  char qor[2] {};
                  args[QOR] = itoa(q, qor);
                  for (int ls : LINEAR_SOLVERS)
                  {
                     char lsb[2] {};
                     args[LS] = itoa(ls, lsb);
                     for (int n : NEWTON_LOOPS)
                     {
                        char nl[2] {};
                        args[NL] = itoa(n, nl);
                        tmop_require(myid, args);
                        if (!all) { break; }
                     }
                     if (!all) { break; }
                  }
                  if (!all) { break; }
               }
               if (!all) { break; }
            }
            if (!all) { break; }
         }
         if (!all) { break; }
      }
   }
};

static void tmop_tests(int id)
{
   const double jitter = 1./(M_PI*M_PI);

   // NURBS
   Launch(Launch::Args("2D Nurbs").
          MESH("square-disc-nurbs.mesh").REFINE(1).JI(jitter).
          POR({1,2}).QOR({2,4}).
          TID({1,2,3}).MID({1,2})).Run(id);

   Launch(Launch::Args("3D Nurbs").
          MESH("beam-hex-nurbs.mesh").REFINE(1).JI(jitter).
          POR({1,2}).QOR({2,4}).
          TID({1,2,3}).MID({302,321})).Run(id);

   // -m cube.mesh -rs 1 -tid 5 -mid 321 -ni 5 -ls 3 -li 100 -lc 1.0 -nor
   Launch(Launch::Args("Cube + Blast options").
          MESH("cube.mesh").REFINE(1).
          TID({5}).MID({321}).LS({3}).LINEAR_ITERATIONS(100).
          LIMITING(M_PI).NORMALIZATION(true).
          POR({1,2,3}).QOR({2,4}).NL({1,2})).Run(id);

   Launch(Launch::Args("Star").
          MESH("star.mesh").
          POR({1,2,3,4}).QOR({2,4,8}).
          TID({1,2,3}).MID({1,2})).Run(id);

   Launch(Launch::Args("Square01 + Adapted analytic Hessian").
          MESH("square01.mesh").REFINE(1).
          POR({1,2}).QOR({2,4}).
          TID({4}).MID({1,2})).Run(id);

   Launch(Launch::Args("Blade").
          MESH("blade.mesh").
          POR({1,2}).QOR({2,4}).
          TID({1,2,3}).MID({2}).LS({2})).Run(id);

   Launch(Launch::Args("Blade + normalization").
          MESH("blade.mesh").
          NORMALIZATION(true).
          POR({1,2}).QOR({2,4}).
          TID({1,2,3}).MID({2})).Run(id);

   Launch(Launch::Args("Blade + limiting + normalization").
          MESH("blade.mesh").
          NORMALIZATION(true).LIMITING(M_PI).
          POR({1,2}).QOR({2,4}).
          TID({1,2,3}).MID({2})).Run(id);

   Launch(Launch::Args("Blade + Discrete size + normalization").
          MESH("blade.mesh").
          LINEAR_ITERATIONS(300).NORMALIZATION(true).
          POR({1}).QOR({2}).
          TID({5}).MID({7}).LS({2}).NL({2})).Run(id);

   Launch(Launch::Args("Blade + Discrete size + normalization").
          MESH("blade.mesh").
          LINEAR_ITERATIONS(200).NORMALIZATION(true).
          POR({1}).QOR({2}).
          TID({5}).MID({2})).Run(id);

   Launch(Launch::Args("Cube").
          MESH("cube.mesh").REFINE(1).JI(jitter).
          POR({1,2}).QOR({2,4}).
          TID({2,3}).MID({302,303})).Run(id);

   Launch(Launch::Args("Cube + Discrete size & aspect + normalization + limiting").
          MESH("cube.mesh").
          NORMALIZATION(true).LIMITING(M_PI).
          POR({1,2}).QOR({4,2}).
          TID({7}).MID({302,321})).Run(id);

   Launch(Launch::Args("Toroid-Hex").
          MESH("toroid-hex.mesh").
          POR({1,2}).QOR({2,4,8}).
          TID({1,2,3}).MID({302,303,321})).Run(id);

   Launch(Launch::Args("Toroid-Hex + limiting").
          MESH("toroid-hex.mesh").
          LIMITING(M_PI).
          POR({1,2}).QOR({2,4}).
          TID({1,2}).MID({321})).Run(id);

   Launch(Launch::Args("Toroid-Hex + limiting + norm.").
          MESH("toroid-hex.mesh").
          LIMITING(M_PI).NORMALIZATION(true).
          POR({1,2}).QOR({2,4}).
          TID({1,2}).MID({321})).Run(id);
}

#if defined(MFEM_TMOP_MPI)
#ifndef MFEM_TMOP_TESTS
TEST_CASE("TMOP", "[TMOP], [Parallel]")
{
   tmop_tests(GlobalMPISession->WorldRank());
}
#else
TEST_CASE("TMOP", "[TMOP], [Parallel]")
{
   Device device;
   device.Configure(MFEM_TMOP_DEVICE);
   device.Print();
   tmop_tests(GlobalMPISession->WorldRank());
}
#endif
#else
#ifndef MFEM_TMOP_TESTS
TEST_CASE("TMOP", "[TMOP]")
{
   tmop_tests(0);
}
#else
TEST_CASE("TMOP", "[TMOP]")
{
   Device device;
   device.Configure(MFEM_TMOP_DEVICE);
   device.Print();
   tmop_tests(0);
}
#endif
#endif

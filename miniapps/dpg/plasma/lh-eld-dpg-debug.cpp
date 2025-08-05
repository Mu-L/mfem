// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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
//                   MFEM Ultraweak DPG Maxwell parallel example
//
// Compile with: make lh-eld-dpg-debug
//
// mpirun -np 8 ./lh-eld-dpg-debug -o 3 -paraview -pr 0
// mpirun -np 8 ./lh-eld-dpg-debug -o 3 -paraview -pr 1 -sc

// Electron Landau Damping
// Strong formulation:
//     ∇×(1/μ₀∇×E) - ω² ϵ₀ ϵ E + i ω²ϵ₀(J₁ + J₂) = 0,   in Ω
//                   - Δ∥ J₁ + c₁ J₁ - c₁ P(r) E∥ = 0,   in Ω     
//                   - Δ∥ J₂ + c₂ J₂ + c₂ P(r) E∥ = 0,   in Ω 
//                                            E×n = E₀,  on ∂Ω
//                                            J₁  = 0,  on ∂Ω
//                                            J₂  = 0,  on ∂Ω
// The DPG UW deals with the First Order System
//  ωμ₀  H + ∇ × E                = 0,   in Ω
//  ωϵ₀ϵ E + ∇ × H - iωϵ₀ (J₁+J₂) = 0,   in Ω
//     -Δ∥ J₁ + c₁ J₁ - c₁ P(r) E∥ = 0,   in Ω     
//     -Δ∥ J₂ + c₂ J₂ + c₂ P(r) E∥ = 0,   in Ω 
//                            E×n = E₀,  on ∂Ω
//                            J₁  = 0,  on ∂Ω
//                            J₂  = 0,  on ∂Ω


// in 2D
// E is vector valued and H is scalar.
//      (∇ × E, δE) = (E, ∇ × δE ) + < n × E , δE >
// or (∇ ⋅ AE , δE) = (AE, ∇ δE ) + < AE ⋅ n, δE >
// where A = [0 1; -1 0];

// E ∈ (L²(Ω))² , H ∈ L²(Ω), J ∈ (H¹(Ω))²
// Ê ∈ H^-1/2(Γₕ), Ĥ ∈ H^1/2(Γₕ), Ĵ₁, Ĵ₂ ∈ (H^-1/2(Γₕ))²
//  ωμ₀  (H,δE) + (E,∇×δE) + < AÊ, δE > = 0,                    ∀ δE ∈ H¹(Ω)
//  ωϵ₀ϵ (E,δH) + (H,∇×δH) + < Ĥ, δH×n > - iωϵ₀(J₁+J₂,δH) = 0,  ∀ δH ∈ H(curl,Ω)
// ( (b⋅∇)J₁,(b⋅∇) δJ₁ ) + <Ĵ₁, δJ₁> + c₁ (J₁,δJ₁) - c₁ (P(r) b⊗b E, δJ₁) = 0,  ∀ δJ₁ ∈ (H¹(Ω))²
// ( (b⋅∇)J₂,(b⋅∇) δJ₂ ) + <Ĵ₂, δJ₂> + c₂ (J₂,δJ₂) + c₂ (P(r) b⊗b E, δJ₂) = 0,  ∀ δJ₁ ∈ (H¹(Ω))²
//                                                                  Ê = E₀, on ∂Ω
//                                                            Ĵ₁ = Ĵ₂ = 0,  on ∂Ω
// ----------------------------------------------------------------------------------------------------------------
// |   |      E       |     H    |   Ê   |   Ĥ    |        J₁        |        J₂        |   Ĵ₁   |   Ĵ₂   |  RHS  |
// ----------------------------------------------------------------------------------------------------------------
// |δE |  (E,∇ × δE)  | ωμ₀(H,δE)| <Ê,δE>|        |                  |                  |        |        |   0   |  
// |   |              |          |       |        |                  |                  |        |        |       |  
// |δH |  ωϵ₀ϵ(E,δH)  | (H,∇×δH) |       |<Ĥ,δH×n>|  -iωϵ₀ (J₁,δH)   | -iωϵ₀ (J₂,δH)    |        |        |   0   |  
// |   |              |          |       |        |                  |                  |        |        |       |  
// |δJ₁|-c₁(P(r)E,δJ₁)|          |       |        |((b⋅∇)J₁,(b⋅∇)δJ₁)|                  |<Ĵ₁,δJ₁>|        |   0   |  
// |   |              |          |       |        |     + c₁ (J₁,δJ₁)|                  |        |        |       |    
// |δJ₂| c₂(P(r)E,δJ₂)|          |       |        |                  |((b⋅∇)J₂,(b⋅∇)δJ₂)|        |<Ĵ₂,δJ₂>|   0   |  
// |   |              |          |       |        |                  |     + c₂ (J₂,δJ₂)|        |        |       |    
// where (δE,δH,δJ₁,δJ₂) ∈  H¹(Ω) × H(curl,Ω) × (H¹(Ω))² × (H¹(Ω))² 



#include "mfem.hpp"
#include "../util/pcomplexweakform.hpp"
#include "../../common/mfem-common.hpp"
#include "../util/maxwell_utils.hpp"
#include "../util/utils.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mfem::common;

real_t delta = 0.01; 
real_t a0 = -1.0;    
real_t a1 = 5.0;  

real_t pfunc_r(const Vector &x)
{
   real_t r = std::sqrt(x(0) * x(0) + x(1) * x(1));
   return a0 + a1 *(r-0.9);
}

real_t pfunc_i(const Vector &x)
{
   return delta;   
}

real_t sfunc_r(const Vector &x)
{
   return 1.0;
}

real_t sfunc_i(const Vector &x)
{
   return delta;
}

void bfunc(const Vector &x, Vector &b)
{
   real_t r = std::sqrt(x(0) * x(0) + x(1) * x(1));
   int dim = x.Size();
   b.SetSize(dim); b = 0.0;
   b(0) = -x(1) / r;
   b(1) =  x(0) / r;
   if (dim == 3) b(2) = 0.0;
}

void bcrossb(const Vector &x, DenseMatrix &bb)
{
   Vector b;
   bfunc(x, b);
   bb.SetSize(b.Size());
   MultVVt(b, bb);
}

int main(int argc, char *argv[])
{
   Mpi::Init();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   const char *mesh_file = "data/LH_hot.msh";
   int order = 2;
   int delta_order = 1;
   int par_ref_levels = 0;
   int ser_ref_levels = 0;

   // real_t rnum=1.5e9;
   // real_t mu = 1.257e-6;
   // real_t eps0 = 8.8541878128e-12;

   real_t rnum=1.5;
   real_t mu = 1.257;
   real_t eps0 = 8.8541878128;
   real_t cfactor = 1e-6;

   bool eld = false; // enable/disable electron Landau damping 

   bool static_cond = false;
   bool visualization = false;
   bool paraview = false;
   bool debug = false;
   bool mumps_solver = false;
   real_t norm_scale = 1.0;
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree)");
   args.AddOption(&ser_ref_levels, "-sr", "--serial-refinement_levels",
                  "Number of serial refinement levels.");                  
   args.AddOption(&par_ref_levels, "-pr", "--parallel-refinement_levels",
                  "Number of parallel refinement levels.");
   args.AddOption(&rnum, "-rnum", "--number_of_wavelenths",
                  "Number of wavelengths");
   args.AddOption(&mu, "-mu", "--permeability",
                  "Permeability of free space (or 1/(spring constant)).");
   args.AddOption(&a0, "-a0", "--a0", "P(r) first parameter.");
   args.AddOption(&a1, "-a1", "--a1", "P(r) second parameter.");
   args.AddOption(&delta, "-delta", "--delta", "stability parameter.");
   args.AddOption(&eld, "-eld", "--eld", "-no-eld",
                  "--no-eld",
                  "Enable or disable electron Landau damping.");
   args.AddOption(&mumps_solver, "-mumps", "--mumps", "-no-mumps",
                  "--no-mumps",
                  "Enable or disable MUMPS solver.");
   args.AddOption(&paraview, "-paraview", "--paraview", "-no-paraview",
                  "--no-paraview",
                  "Enable or disable ParaView visualization.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&debug, "-debug", "--debug", "-no-debug",
                  "--no-debug",
                  "Enable or disable debug mode (delta = 0.01 and no coupling).");                  
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

   // number of diffusion equations
   int ndiffusionequations = (eld) ?  2 : 0; 

   Vector cvals(ndiffusionequations);
   Vector csigns(ndiffusionequations);
   if (eld)
   {
      cvals(0)  = 25e6;  cvals(1)  = 1e6;
      csigns(0) = -1.0;  csigns(1) = 1.0;
   }
   cvals *= cfactor; // scale the coefficients

   real_t omega = 2.*M_PI*rnum;
   int test_order = order+delta_order;

   if (eld && !debug) 
   {
      delta = 0.0; // disable delta if electron Landau damping is enabled
      if (Mpi::Root())
      {
         cout << "Electron Landau damping enabled, delta set to 0.0." << endl;
      }
   }    

   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();
   MFEM_VERIFY(dim == 2, "Dimension != 2 is not supported in this example");

   for (int i = 0; i < ser_ref_levels; i++)
   {
      mesh.UniformRefinement();
   }

   mesh.RemoveInternalBoundaries();
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   for (int i = 0; i < par_ref_levels; i++)
   {
      pmesh.UniformRefinement();
   }  

   int nattr = (pmesh.attributes.Size()) ? pmesh.attributes.Max() : 0;
   Array<int> attr(nattr);
   for (int i = 0; i<nattr; i++) { attr[i] = i+1; }

   // Define coefficients
   ConstantCoefficient muinv(1./mu);
   ConstantCoefficient one_cf(1.0);
   //  ωμ₀
   ConstantCoefficient omegamu_cf(omega*mu);
   // -ω μ₀  
   ConstantCoefficient negomegamu_cf(-omega*mu);
   // -ωϵ₀
   real_t scale = (debug) ? 0.0 : 1.0;
   ConstantCoefficient negomegeps0_cf(-omega*eps0 * scale);
   // μ₀² ω²
   ConstantCoefficient mu2omeg2_cf((mu*mu*omega*omega));

   Vector zero(dim); zero = 0.0;
   Vector one_x(dim); one_x = 0.0; one_x(0) = 1.0;
   Vector negone_x(dim); negone_x = 0.0; negone_x(0) = -1.0;
   VectorConstantCoefficient zero_vcf(zero);
   VectorConstantCoefficient one_x_cf(one_x);
   VectorConstantCoefficient negone_x_cf(negone_x);

   DenseMatrix Mone(dim); 
   Mone = 0.0; Mone(0,0) = Mone(1,1) = 1.0;
   MatrixConstantCoefficient Mone_cf(Mone);
   DenseMatrix Mzero(dim); Mzero = 0.0;
   MatrixConstantCoefficient Mzero_cf(Mzero);

   Array<MatrixCoefficient*> coefs_r(nattr);
   Array<MatrixCoefficient*> coefs_i(nattr);
   for (int i = 0; i < nattr-1; ++i)
   {
      coefs_r[i] = &Mone_cf;
      coefs_i[i] = &Mzero_cf;
   }

   // S(r) 
   FunctionCoefficient S_cf_r(sfunc_r), S_cf_i(sfunc_i);
   // P(r) 
   FunctionCoefficient P_cf_r(pfunc_r), P_cf_i(pfunc_i); 

   VectorFunctionCoefficient b_cf(dim,bfunc);// b
   ScalarVectorProductCoefficient scaled_b_cf(sqrt(cfactor), b_cf);

   MatrixFunctionCoefficient bb_cf(dim,bcrossb); // b⊗b
   MatrixSumCoefficient oneminusbb(Mone_cf, bb_cf, 1.0, -1.0); // 1 - b⊗b

   // S(r) (I - b⊗b)
   ScalarMatrixProductCoefficient Soneminusbb_r(S_cf_r, oneminusbb), Soneminusbb_i(S_cf_i, oneminusbb); 

   // P(r) b⊗b 
   ScalarMatrixProductCoefficient P_cf_bb_r(P_cf_r, bb_cf), P_cf_bb_i(P_cf_i, bb_cf); 

   // ε = S(r) (I - b⊗b) + P(r) b⊗b 
   MatrixSumCoefficient eps_r(Soneminusbb_r, P_cf_bb_r, 1.0, 1.0); 
   MatrixSumCoefficient eps_i(Soneminusbb_i, P_cf_bb_i, 1.0, 1.0); 

   coefs_r[nattr-1] = &eps_r;
   coefs_i[nattr-1] = &eps_i;

   PWMatrixCoefficient eps_cf_r(dim, attr, coefs_r);
   PWMatrixCoefficient eps_cf_i(dim, attr, coefs_i);

   ConstantCoefficient eps0omeg(omega * eps0);
   ConstantCoefficient negeps0omeg(-omega * eps0);

   // ω ϵ₀ ϵᵣ 
   ScalarMatrixProductCoefficient eps0omeg_eps_r(eps0omeg, eps_cf_r);
   // ω ϵ₀ ϵᵢ
   ScalarMatrixProductCoefficient eps0omeg_eps_i(eps0omeg, eps_cf_i);
   // -ω ϵ₀ ϵᵣ 
   ScalarMatrixProductCoefficient negeps0omeg_eps_r(negeps0omeg, eps_cf_r);
   // -ω ϵ₀ ϵᵢ
   ScalarMatrixProductCoefficient negeps0omeg_eps_i(eps0omeg, eps_cf_i);

   // A = [0 1; -1 0]
   DenseMatrix rot_mat(2);
   rot_mat(0,0) = 0.; rot_mat(0,1) = 1.;
   rot_mat(1,0) = -1.; rot_mat(1,1) = 0.;
   MatrixConstantCoefficient rot(rot_mat);

   // ω ϵ₀ ϵᵣ A
   MatrixProductCoefficient eps0omeg_eps_r_rot(eps0omeg_eps_r, rot);
   // ω ϵ₀ ϵᵢ A
   MatrixProductCoefficient eps0omeg_eps_i_rot(eps0omeg_eps_i, rot);
   // -ω ϵ₀ ϵᵣ A
   MatrixProductCoefficient negeps0omeg_eps_r_rot(negeps0omeg_eps_r, rot);
   // -ω ϵ₀ ϵᵢ A
   MatrixProductCoefficient negeps0omeg_eps_i_rot(negeps0omeg_eps_i, rot);

   // (ωϵ₀ϵ)(ωϵ₀ϵ)^*  (δH, δH)
   TransposeMatrixCoefficient eps0omeg_eps_r_t(eps0omeg_eps_r);
   TransposeMatrixCoefficient eps0omeg_eps_i_t(eps0omeg_eps_i);

   MatrixProductCoefficient eps0omeg_eps_r_t_rot(eps0omeg_eps_r, rot);
   MatrixProductCoefficient eps0omeg_eps_i_t_rot(eps0omeg_eps_i, rot);

   MatrixProductCoefficient MrMrt_cf(eps0omeg_eps_r, eps0omeg_eps_r_t);
   MatrixProductCoefficient MiMit_cf(eps0omeg_eps_i, eps0omeg_eps_i_t);
   MatrixProductCoefficient MiMrt_cf(eps0omeg_eps_i, eps0omeg_eps_r_t);
   MatrixProductCoefficient MrMit_cf(eps0omeg_eps_r, eps0omeg_eps_i_t);

   // (MᵣMᵣᵗ + MᵢMᵢᵗ) + i (MᵢMᵣᵗ - MᵣMᵢᵗ)
   MatrixSumCoefficient Mreal_cf(MrMrt_cf,MiMit_cf);
   MatrixSumCoefficient Mimag_cf(MiMrt_cf,MrMit_cf,1.0,-1.0);

   // if ELD
   Array<Vector *> c_arrays(ndiffusionequations);
   Array<PWConstCoefficient *> pw_c_coeffs(ndiffusionequations);
   Array<MatrixCoefficient *> cPrbb_cf(ndiffusionequations);
   Array<MatrixCoefficient *> cPibb_cf(ndiffusionequations);
   Array<MatrixCoefficient *> signedcPrbb_cf(ndiffusionequations);
   Array<MatrixCoefficient *> signedcPibb_cf(ndiffusionequations);
   Vector temp(nattr); temp=0.0;
   Array<ConstantCoefficient *> c_coeffs(ndiffusionequations);
   for (int i = 0; i<ndiffusionequations; i++)
   {
      temp[nattr-1] = cvals(i);
      pw_c_coeffs[i] = new PWConstCoefficient(temp);
      c_coeffs[i] = new ConstantCoefficient(cvals(i));
      cPrbb_cf[i] = new ScalarMatrixProductCoefficient(*pw_c_coeffs[i], P_cf_bb_r);
      cPibb_cf[i] = new ScalarMatrixProductCoefficient(*pw_c_coeffs[i], P_cf_bb_i);
      signedcPrbb_cf[i] = new ScalarMatrixProductCoefficient(csigns[i], *cPrbb_cf[i]);
      signedcPibb_cf[i] = new ScalarMatrixProductCoefficient(csigns[i], *cPibb_cf[i]);
   }   

   ConstantCoefficient norm_scale_cf(norm_scale);


   // Define the spaces
   Array<FiniteElementCollection *> trial_fecols;
   Array<FiniteElementCollection *> test_fecols;
   Array<ParFiniteElementSpace *> pfes;

   // Vector L2 space for E
   trial_fecols.Append(new L2_FECollection(order-1, dim));
   pfes.Append(new ParFiniteElementSpace(&pmesh, trial_fecols.Last(), dim));
   // Scalar L2 space for H
   trial_fecols.Append(new L2_FECollection(order-1, dim));
   pfes.Append(new ParFiniteElementSpace(&pmesh, trial_fecols.Last()));

   // Trial trace space for Ê 
   trial_fecols.Append(new RT_Trace_FECollection(order-1, dim));
   pfes.Append(new ParFiniteElementSpace(&pmesh, trial_fecols.Last()));

   // Trial trace space for Ĥ 
   trial_fecols.Append(new H1_Trace_FECollection(order, dim));
   pfes.Append(new ParFiniteElementSpace(&pmesh, trial_fecols.Last()));


   // Vector H1 space for Js
   for (int i = 0; i < ndiffusionequations; i++)
   {
      trial_fecols.Append(new H1_FECollection(order, dim));
      pfes.Append(new ParFiniteElementSpace(&pmesh, trial_fecols.Last(), dim));
   }

   // Vector Trace spaces for Js   
   for (int i = 0; i < ndiffusionequations; i++)
   {
      trial_fecols.Append(new RT_Trace_FECollection(order-1,dim));
      pfes.Append(new ParFiniteElementSpace(&pmesh, trial_fecols.Last(),dim));
   }

   Array<HYPRE_BigInt> tdofs(pfes.Size());
   for (int i = 0; i < pfes.Size(); ++i)
   {
      tdofs[i] = pfes[i]->GlobalTrueVSize();
      if (Mpi::Root())
      {
         cout << "ParFiniteElementSpace " << i << " has " << tdofs[i]
              << " true dofs." << endl;
      }
   }
   if (Mpi::Root())
   {
      cout << "Total number of true dofs: " << tdofs.Sum() << endl;
   }

   // test spaces for E and H
   test_fecols.Append(new H1_FECollection(test_order, dim));
   test_fecols.Append(new ND_FECollection(test_order, dim));
   // Test spaces δJs 
   for (int i = 0; i < ndiffusionequations; i++)
   {
      test_fecols.Append(new H1_FECollection(test_order, dim));
   }

   ParComplexDPGWeakForm * a = new ParComplexDPGWeakForm(pfes,test_fecols);
   for (int i = 0; i < ndiffusionequations; i++)
   {
      a->SetTestFECollVdim(i+2,dim);
   }

   // (E,∇ × δE)
   a->AddTrialIntegrator(new TransposeIntegrator(new MixedCurlIntegrator(one_cf)),
                         nullptr,0, 0);
   //  ωϵ₀(ϵE,δH) = ω ϵ₀(ϵᵣ + i ϵᵢ E, δH)
   //             = (ω ϵ₀ ϵᵣ E, δH) + i (ω ϵ₀ϵᵢ E, δH)
   a->AddTrialIntegrator(
      new TransposeIntegrator(new VectorFEMassIntegrator(eps0omeg_eps_r)), 
      new TransposeIntegrator(new VectorFEMassIntegrator(eps0omeg_eps_i)),
                              0,1);
   // ωμ₀(H,δE) 
   a->AddTrialIntegrator(new MixedScalarMassIntegrator(omegamu_cf),
                         nullptr,1, 0);
   // (H,∇ × δH)                         
   a->AddTrialIntegrator(
      new TransposeIntegrator(new MixedCurlIntegrator(one_cf)), nullptr,1, 1);

   // Trace integrators               
   //  <Ê,δE>
   a->AddTrialIntegrator(new TraceIntegrator,nullptr, 2, 0);
   // <Ĥ,δH × n>
   a->AddTrialIntegrator(new TangentTraceIntegrator,nullptr, 3, 1);
   if (eld)
   {
      for (int i = 0; i < ndiffusionequations; i++)
      {
         // ±cᵢ(P(r) (b ⊗ b) E, δJᵢ)
         a->AddTrialIntegrator(new VectorMassIntegrator(*signedcPrbb_cf[i]),
                               new VectorMassIntegrator(*signedcPibb_cf[i]),
                               0, i+2);
         // -iωϵ₀ (Jᵢ ,δH)
         a->AddTrialIntegrator(nullptr,
         new TransposeIntegrator(new VectorFEMassIntegrator(negomegeps0_cf)),i+4, 1);
         
         // ((b⋅∇)Jᵢ, (b⋅∇) δJᵢ)
         a->AddTrialIntegrator(new DirectionalDiffusionIntegrator(scaled_b_cf), nullptr,
         // a->AddTrialIntegrator(new DirectionalDiffusionIntegrator(b_cf), nullptr,
                               i+4, i+2);
         // cᵢ(Jᵢ, δJᵢ)
         a->AddTrialIntegrator(new VectorMassIntegrator(*pw_c_coeffs[i]), nullptr,
                               i+4, i+2);

         // <Ĵᵢ,δJᵢ>
         a->AddTrialIntegrator(new VectorTraceIntegrator,nullptr,
                               i + ndiffusionequations + 4, i+2);                      
      }
   }      


   // test integrators
   // (∇δE,∇δE)
   a->AddTestIntegrator(new DiffusionIntegrator(one_cf),nullptr, 0, 0);
   // (δE,δE)
   a->AddTestIntegrator(new MassIntegrator(norm_scale_cf),nullptr, 0, 0);
   // μ₀² ω² (δE,δE)
   a->AddTestIntegrator(new MassIntegrator(mu2omeg2_cf),nullptr,0, 0);
   // ω μ₀ (δE,∇ × δH) 
   a->AddTestIntegrator(
         new TransposeIntegrator(new MixedCurlIntegrator(omegamu_cf)),
         nullptr,0, 1);
   //  ωϵ₀ϵ(∇ × δE, δH) = ωϵ₀(ϵᵣ+iϵᵢ) A ∇ δE,δE), A = [0 1; -1 0]
   //                   = (ω ϵ₀ ϵᵣ A ∇ δE,δE) + i (ω ϵ₀ ϵᵢ A ∇ δE,δE)
   a->AddTestIntegrator(new MixedVectorGradientIntegrator(eps0omeg_eps_r_rot),
                        new MixedVectorGradientIntegrator(eps0omeg_eps_i_rot),0, 1);
   // ωμ₀(∇ × δH ,δE)  
   a->AddTestIntegrator(new MixedCurlIntegrator(omegamu_cf),nullptr,
                        1, 0);
   // ω ϵ₀ϵ (δH, ∇ × δE ) = ω (ϵ₀ϵᵣ + i ϵᵢ) δH, A ∇ δE) 
   //                     = (δH, ωϵ₀ ϵᵣᵀ A ∇ δE) + i (δH, ω ϵ₀ ϵᵢᵀ A ∇ δE)
   a->AddTestIntegrator(
      new TransposeIntegrator(new MixedVectorGradientIntegrator(eps0omeg_eps_r_t_rot)),
      new TransposeIntegrator(new MixedVectorGradientIntegrator(eps0omeg_eps_i_t_rot)),1, 0);
   // (ωϵ₀ϵ)(ωϵ₀ϵ)^*  (δH, δH)
   // (MᵣMᵣᵗ + MᵢMᵢᵗ) + i (MᵢMᵣᵗ - MᵣMᵢᵗ)
   a->AddTestIntegrator(new VectorFEMassIntegrator(Mreal_cf),
                        new VectorFEMassIntegrator(Mimag_cf),1, 1);
   // (∇×δH ,∇×δH)
   a->AddTestIntegrator(new CurlCurlIntegrator(one_cf),nullptr,1,1);
   // (δH,δH)
   a->AddTestIntegrator(new VectorFEMassIntegrator(norm_scale_cf),nullptr,1,1);

   for (int i = 0; i < ndiffusionequations; i++)
   {
      // (∇δJ,∇δJ) 
      a->AddTestIntegrator(new VectorDiffusionIntegrator(one_cf),nullptr,
                           i+2,i+2);
      // (b⋅∇δJ, b⋅∇δJ)
      // a->AddTestIntegrator(new DirectionalDiffusionIntegrator(scaled_b_cf),nullptr,
      // a->AddTestIntegrator(new DirectionalDiffusionIntegrator(b_cf),nullptr,
                           // i+2,i+2);                     
      // (δJ,δJ)
      a->AddTestIntegrator(new VectorMassIntegrator(norm_scale_cf),nullptr, 
      // a->AddTestIntegrator(new VectorMassIntegrator(*c_coeffs[i]),nullptr, 
                           i+2,i+2);
   }

   if (static_cond) { a->EnableStaticCondensation(); }
   a->Assemble();

   for (int i = 0; i<ndiffusionequations; i++)
   {
      delete pw_c_coeffs[i];
      delete c_coeffs[i];
      delete cPrbb_cf[i]; 
      delete cPibb_cf[i]; 
      delete signedcPrbb_cf[i]; 
      delete signedcPibb_cf[i]; 
   }

   socketstream E_out_r;

   int npfes = pfes.Size();
   Array<int> offsets(npfes+1);  offsets[0] = 0;
   Array<int> toffsets(npfes+1); toffsets[0] = 0;
   for (int i = 0; i<npfes; i++)
   {
      offsets[i+1] = pfes[i]->GetVSize();
      toffsets[i+1] = pfes[i]->TrueVSize();
   }
   offsets.PartialSum();
   toffsets.PartialSum();

   Vector x(2*offsets.Last());
   x = 0.;
   
   Array<ParGridFunction *> pgf_r(npfes);
   Array<ParGridFunction *> pgf_i(npfes);

   for (int i = 0; i < npfes; ++i)
   {
      pgf_r[i] = new ParGridFunction(pfes[i], x, offsets[i]);
      pgf_i[i] = new ParGridFunction(pfes[i], x, offsets.Last() + offsets[i]);
   }

   L2_FECollection L2fec(order, dim);
   ParFiniteElementSpace L2_fes(&pmesh, &L2fec);
   ParGridFunction E_par_r(&L2_fes);
   ParGridFunction E_par_i(&L2_fes);

   ParaViewDataCollection * paraview_dc = nullptr;

   std::string output_dir = "ParaView/UW/" + GetTimestamp();

   if (paraview)
   {
      if (Mpi::Root()) { WriteParametersToFile(args, output_dir); }
      std::ostringstream paraview_file_name;
      std::string filename = GetFilename(mesh_file);
      paraview_file_name << filename
                         << "_par_ref_" << par_ref_levels
                         << "_order_" << order
                         << "_eld_" << eld;
      paraview_dc = new ParaViewDataCollection(paraview_file_name.str(), &pmesh);
      paraview_dc->SetPrefixPath(output_dir);
      paraview_dc->SetLevelsOfDetail(order);
      paraview_dc->SetCycle(0);
      paraview_dc->SetDataFormat(VTKFormat::BINARY);
      paraview_dc->SetHighOrderOutput(true);
      paraview_dc->SetTime(0.0); // set the time
      paraview_dc->RegisterField("E_r",pgf_r[0]);
      paraview_dc->RegisterField("E_i",pgf_i[0]);
      paraview_dc->RegisterField("E_par_r",&E_par_r);
      paraview_dc->RegisterField("E_par_i",&E_par_i);
      paraview_dc->RegisterField("H_r",pgf_r[1]);
      paraview_dc->RegisterField("H_i",pgf_i[1]);      
      if (eld)
      {
         paraview_dc->RegisterField("Jh_1_r",pgf_r[4]);
         paraview_dc->RegisterField("Jh_1_i",pgf_i[4]);
         paraview_dc->RegisterField("Jh_2_r",pgf_r[5]);
         paraview_dc->RegisterField("Jh_2_i",pgf_i[5]);
      }
   }

   Array<int> ess_tdof_list;
   Array<int> ess_tdof_listJ;
   Array<int> ess_bdr;
   Array<int> one_r_bdr;
   Array<int> one_i_bdr;
   Array<int> negone_r_bdr;
   Array<int> negone_i_bdr;
   
   if (pmesh.bdr_attributes.Size())
   {
      ess_bdr.SetSize(pmesh.bdr_attributes.Max());
      one_r_bdr.SetSize(pmesh.bdr_attributes.Max());
      one_i_bdr.SetSize(pmesh.bdr_attributes.Max());
      negone_r_bdr.SetSize(pmesh.bdr_attributes.Max());
      negone_i_bdr.SetSize(pmesh.bdr_attributes.Max());
      ess_bdr = 1;
      pfes[2]->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      for (int j = 0; j < ess_tdof_list.Size(); j++)
      {
         ess_tdof_list[j] += toffsets[2];
      }
      for (int i = 0; i<ndiffusionequations;i++)
      {
         ess_tdof_listJ.SetSize(0);
         pfes[i+4]->GetEssentialTrueDofs(ess_bdr, ess_tdof_listJ);
         for (int j = 0; j < ess_tdof_listJ.Size(); j++)
         {
            ess_tdof_listJ[j] += toffsets[i+4];
         }
         ess_tdof_list.Append(ess_tdof_listJ);
      }
   
      one_r_bdr = 0;  one_i_bdr = 0;
      negone_r_bdr = 0;  negone_i_bdr = 0;
      // attr = 30,2 (real)
      one_r_bdr[30-1] = 1;  one_r_bdr[2-1] = 1;
      // attr = 26,6 (imag)
      one_i_bdr[26-1] = 1;  one_i_bdr[6-1] = 1;
      // attr = 22,10 (real)
      negone_r_bdr[22-1] = 1; negone_r_bdr[10-1] = 1;
      // attr = 18,14 (imag)
      negone_i_bdr[18-1] = 1; negone_i_bdr[14-1] = 1;
   }
   

   // rotate the vector
   // (x,y) -> (y,-x)
   Vector rot_one_x(dim); rot_one_x = 0.0; rot_one_x(1) = -1.0;
   Vector rot_negone_x(dim); rot_negone_x = 0.0; rot_negone_x(1) = 1.0;
   VectorConstantCoefficient rot_one_x_cf(rot_one_x);
   VectorConstantCoefficient rot_negone_x_cf(rot_negone_x);

   pgf_r[2]->ProjectBdrCoefficientNormal(rot_one_x_cf, one_r_bdr);
   pgf_r[2]->ProjectBdrCoefficientNormal(rot_negone_x_cf, negone_r_bdr);
   pgf_i[2]->ProjectBdrCoefficientNormal(rot_one_x_cf, one_i_bdr);
   pgf_i[2]->ProjectBdrCoefficientNormal(rot_negone_x_cf, negone_i_bdr);

   OperatorPtr Ah;
   Vector X,B;
   a->FormLinearSystem(ess_tdof_list,x,Ah, X,B);
   
   ComplexOperator * Ahc = Ah.As<ComplexOperator>();

   BlockOperator * BlockA_r = dynamic_cast<BlockOperator *>(&Ahc->real());
   BlockOperator * BlockA_i = dynamic_cast<BlockOperator *>(&Ahc->imag());

   int nblocks = BlockA_r->NumRowBlocks();
   
   Array2D<const HypreParMatrix*> A_r_matrices(nblocks, nblocks);
   Array2D<const HypreParMatrix*> A_i_matrices(nblocks, nblocks);
   for (int i = 0; i < nblocks; i++)
   {
      for (int j = 0; j < nblocks; j++)
      {
         A_r_matrices(i,j) = dynamic_cast<HypreParMatrix*>(&BlockA_r->GetBlock(i,j));
         A_i_matrices(i,j) = dynamic_cast<HypreParMatrix*>(&BlockA_i->GetBlock(i,j));

         // std::ostringstream fname_real, fname_imag;
         // fname_real << "Areal_" << i << "_" << j << ".dat";
         // fname_imag << "Aimag_" << i << "_" << j << ".dat";

         // // Open file streams
         // std::ofstream out_real(fname_real.str());
         // std::ofstream out_imag(fname_imag.str());

         // // Print to files
         // A_r_matrices(i,j)->PrintMatlab(out_real);
         // A_i_matrices(i,j)->PrintMatlab(out_imag);

         // mfem::out << "Wrote matrices to files: " << fname_real.str() << ", " << fname_imag.str() << endl;
         // cin.get();
      }
   }







   HypreParMatrix * Ahr = HypreParMatrixFromBlocks(A_r_matrices);
   HypreParMatrix * Ahi = HypreParMatrixFromBlocks(A_i_matrices);

   ComplexHypreParMatrix * Ahc_hypre =
      new ComplexHypreParMatrix(Ahr, Ahi,false, false);

   if (Mpi::Root())
   {
      mfem::out << "Assembly finished successfully." << endl;
   }
      HypreParMatrix *A = Ahc_hypre->GetSystemMatrix();

#ifdef MFEM_USE_MUMPS
   if (mumps_solver)
   {
      auto solver = new MUMPSSolver(MPI_COMM_WORLD);
      solver->SetMatrixSymType(MUMPSSolver::MatType::UNSYMMETRIC);
      solver->SetPrintLevel(1);
      solver->SetOperator(*A);
      solver->Mult(B,X);
      delete A;
      delete solver;
   }
#else
   if (mumps_solver)
   {
      MFEM_WARNING("MFEM compiled without mumps. Switching to an iterative solver");
   }
   mumps_solver = false;
#endif
   int num_iter = -1;

   Array<int> tdof_offsets(2*nblocks+1);
   int skip = (static_cond) ? 0 : 2;
   tdof_offsets[0] = 0;
   int k = (static_cond) ? 2 : 0;
   for (int i=0; i<nblocks; i++)
   {
      tdof_offsets[i+1] = A_r_matrices(i,i)->Height();
      tdof_offsets[nblocks+i+1] = tdof_offsets[i+1];
   }
   tdof_offsets.PartialSum();

   if (!mumps_solver)
   {

      BlockDiagonalPreconditioner M(tdof_offsets);

      if (!static_cond)
      {
         HypreBoomerAMG * solver_E = new HypreBoomerAMG((HypreParMatrix &)
                                                   BlockA_r->GetBlock(0,0));
         solver_E->SetPrintLevel(0);
         solver_E->SetSystemsOptions(dim);
         HypreBoomerAMG * solver_H = new HypreBoomerAMG((HypreParMatrix &)
                                                   BlockA_r->GetBlock(1,1));
         solver_H->SetPrintLevel(0);
         // solver_H->SetSystemsOptions(dim);
         M.SetDiagonalBlock(0,solver_E);
         M.SetDiagonalBlock(1,solver_H);
         M.SetDiagonalBlock(nblocks,solver_E);
         M.SetDiagonalBlock(nblocks+1,solver_H);
      }
      HypreAMS * solver_hatE = 
      new HypreAMS((HypreParMatrix &)BlockA_r->GetBlock(skip,
                                    skip), pfes[2]);
      HypreBoomerAMG * solver_hatH = new HypreBoomerAMG((HypreParMatrix &)
                                     BlockA_r->GetBlock(skip+1,skip+1));
      solver_hatE->SetPrintLevel(0);
      solver_hatH->SetPrintLevel(0);
      solver_hatH->SetRelaxType(88);

      M.SetDiagonalBlock(skip,solver_hatE);
      M.SetDiagonalBlock(skip+1,solver_hatH);
      M.SetDiagonalBlock(skip+nblocks,solver_hatE);
      M.SetDiagonalBlock(skip+nblocks+1,solver_hatH);

      if (eld)
      {
         HypreBoomerAMG * solver_J1 = new HypreBoomerAMG((HypreParMatrix &)
                               BlockA_r->GetBlock(skip+2,skip+2));
         solver_J1->SetPrintLevel(0);
         solver_J1->SetSystemsOptions(dim);
         M.SetDiagonalBlock(skip+2,solver_J1);
         M.SetDiagonalBlock(skip+nblocks+2,solver_J1);
         HypreBoomerAMG * solver_J2 = new HypreBoomerAMG((HypreParMatrix &)
                               BlockA_r->GetBlock(skip+3,skip+3));
         solver_J2->SetPrintLevel(0);
         solver_J2->SetSystemsOptions(dim);
         M.SetDiagonalBlock(skip+3,solver_J2);
         M.SetDiagonalBlock(skip+nblocks+3,solver_J2);
         HypreBoomerAMG * solver_hatJ1 = new HypreBoomerAMG((HypreParMatrix &)
                                            BlockA_r->GetBlock(skip+4,skip+4));
         solver_hatJ1->SetPrintLevel(0);
         solver_hatJ1->SetSystemsOptions(dim);
         solver_hatJ1->SetRelaxType(88);
         M.SetDiagonalBlock(skip+4,solver_hatJ1);
         M.SetDiagonalBlock(skip+nblocks+4,solver_hatJ1);
         HypreBoomerAMG * solver_hatJ2 = new HypreBoomerAMG((HypreParMatrix &)
                                            BlockA_r->GetBlock(skip+5,skip+5));
         solver_hatJ2->SetPrintLevel(0);
         solver_hatJ2->SetSystemsOptions(dim);
         solver_hatJ2->SetRelaxType(88);
         M.SetDiagonalBlock(skip+5,solver_hatJ2);
         M.SetDiagonalBlock(skip+nblocks+5,solver_hatJ2);
      }
      CGSolver cg(MPI_COMM_WORLD);
      cg.SetRelTol(1e-10);
      cg.SetMaxIter(1000);
      cg.SetPrintLevel(1);
      cg.SetPreconditioner(M);
      cg.SetOperator(*A);
      cg.Mult(B, X);
   }

   a->RecoverFEMSolution(X, x);

   for (int i = 0; i < npfes; ++i)
   {
      pgf_r[i]->MakeRef(pfes[i], x, offsets[i]);
      pgf_i[i]->MakeRef(pfes[i], x, offsets.Last() + offsets[i]);
   }
   
   ParallelECoefficient par_e_r(pgf_r[0]);
   ParallelECoefficient par_e_i(pgf_i[0]);
   E_par_r.ProjectCoefficient(par_e_r);
   E_par_i.ProjectCoefficient(par_e_i);
   
   if (visualization)
   {
      const char * keys = nullptr;
      char vishost[] = "localhost";
      int  visport   = 19916;
      common::VisualizeField(E_out_r,vishost, visport, *pgf_r[0],
                             "Numerical Electric field (real part)", 0, 0, 500, 500, keys);
   }

   if (paraview)
   {
      paraview_dc->SetCycle(0);
      paraview_dc->SetTime((real_t)0);
      paraview_dc->Save();
      delete paraview_dc;
   }


   delete a;
   for (int i = 0; i < trial_fecols.Size(); ++i)
   {
      delete trial_fecols[i];
      delete pfes[i];
   }
   for (int i = 0; i< test_fecols.Size(); ++i)
   {
      delete test_fecols[i];
   }

   return 0;
}

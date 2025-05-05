// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_COLD_PLASMA_DIELECTRIC_SOLVER
#define MFEM_COLD_PLASMA_DIELECTRIC_SOLVER

#include "../common/pfem_extras.hpp"
#include "plasma.hpp"

#ifdef MFEM_USE_MPI

#include <string>
#include <map>

namespace mfem
{

using common::H1_ParFESpace;
using common::ND_ParFESpace;
using common::RT_ParFESpace;
using common::L2_ParFESpace;
using common::ParDiscreteGradOperator;
using common::ParDiscreteCurlOperator;

namespace plasma
{

// Solver options
struct SolverOptions
{
   int maxIter;
   int kDim;
   int printLvl;
   double relTol;
   double absTol;

   // Euclid Options
   int euLvl;
};

struct AttributeArrays
{
   Array<int> attr;
   Array<int> attr_marker;

};

struct ComplexCoefficientByAttr : public AttributeArrays
{
   Coefficient * real;
   Coefficient * imag;
};

struct ComplexVectorCoefficientByAttr : public AttributeArrays
{
   VectorCoefficient * real;
   VectorCoefficient * imag;
};

// Used for combining scalar coefficients
double prodFunc(double a, double b);

class ElectricEnergyDensityCoef : public Coefficient
{
public:
   ElectricEnergyDensityCoef(VectorCoefficient &Er, VectorCoefficient &Ei,
                             MatrixCoefficient &epsr, MatrixCoefficient &epsi);

   double Eval(ElementTransformation &T,
               const IntegrationPoint &ip);

private:
   VectorCoefficient &ErCoef_;
   VectorCoefficient &EiCoef_;
   MatrixCoefficient &epsrCoef_;
   MatrixCoefficient &epsiCoef_;

   mutable Vector Er_;
   mutable Vector Ei_;
   mutable Vector Dr_;
   mutable Vector Di_;
   mutable DenseMatrix eps_r_;
   mutable DenseMatrix eps_i_;
};

class MagneticEnergyDensityCoef : public Coefficient
{
public:
   MagneticEnergyDensityCoef(double omega,
                             VectorCoefficient &dEr, VectorCoefficient &dEi,
                             Coefficient &muInv);

   double Eval(ElementTransformation &T,
               const IntegrationPoint &ip);

private:
   double omega_;

   VectorCoefficient &dErCoef_;
   VectorCoefficient &dEiCoef_;
   Coefficient &muInvCoef_;

   mutable Vector Br_;
   mutable Vector Bi_;
};

class EnergyDensityCoef : public Coefficient
{
public:
   EnergyDensityCoef(double omega,
                     VectorCoefficient &Er, VectorCoefficient &Ei,
                     VectorCoefficient &dEr, VectorCoefficient &dEi,
                     MatrixCoefficient &epsr, MatrixCoefficient &epsi,
                     Coefficient &muInv);

   double Eval(ElementTransformation &T,
               const IntegrationPoint &ip);

private:
   double omega_;

   VectorCoefficient &ErCoef_;
   VectorCoefficient &EiCoef_;
   VectorCoefficient &dErCoef_;
   VectorCoefficient &dEiCoef_;
   MatrixCoefficient &epsrCoef_;
   MatrixCoefficient &epsiCoef_;
   Coefficient &muInvCoef_;

   mutable Vector Er_;
   mutable Vector Ei_;
   mutable Vector Dr_;
   mutable Vector Di_;
   mutable Vector Br_;
   mutable Vector Bi_;
   mutable DenseMatrix eps_r_;
   mutable DenseMatrix eps_i_;
};

class PoyntingVectorReCoef : public VectorCoefficient
{
public:
   PoyntingVectorReCoef(double omega,
                        VectorCoefficient &Er, VectorCoefficient &Ei,
                        VectorCoefficient &dEr, VectorCoefficient &dEi,
                        Coefficient &muInv);

   void Eval(Vector &S, ElementTransformation &T,
             const IntegrationPoint &ip);

private:
   double omega_;

   VectorCoefficient &ErCoef_;
   VectorCoefficient &EiCoef_;
   VectorCoefficient &dErCoef_;
   VectorCoefficient &dEiCoef_;
   Coefficient &muInvCoef_;

   mutable Vector Er_;
   mutable Vector Ei_;
   mutable Vector Hr_;
   mutable Vector Hi_;
};

class PoyntingVectorImCoef : public VectorCoefficient
{
public:
   PoyntingVectorImCoef(double omega,
                        VectorCoefficient &Er, VectorCoefficient &Ei,
                        VectorCoefficient &dEr, VectorCoefficient &dEi,
                        Coefficient &muInv);

   void Eval(Vector &S, ElementTransformation &T,
             const IntegrationPoint &ip);

private:
   double omega_;

   VectorCoefficient &ErCoef_;
   VectorCoefficient &EiCoef_;
   VectorCoefficient &dErCoef_;
   VectorCoefficient &dEiCoef_;
   Coefficient &muInvCoef_;

   mutable Vector Er_;
   mutable Vector Ei_;
   mutable Vector Hr_;
   mutable Vector Hi_;
};

/// Cold Plasma Dielectric Solver
class CPDSolver
{
public:

   enum PrecondType
   {
      INVALID_PC  = -1,
      DIAG_SCALE  =  1,
      PARASAILS   =  2,
      EUCLID      =  3,
      AMS         =  4
   };

   enum SolverType
   {
      INVALID_SOL = -1,
      GMRES       =  1,
      FGMRES      =  2,
      MINRES      =  3,
      SUPERLU     =  4,
      STRUMPACK   =  5,
      DMUMPS      =  6,
      ZMUMPS      =  7
   };

   CPDSolver(ParMesh & pmesh, int order, double omega,
             CPDSolver::SolverType s, SolverOptions & sOpts,
             CPDSolver::PrecondType p,
             ComplexOperator::Convention conv,
             VectorCoefficient & BCoef,
             MatrixCoefficient & epsReCoef,
             MatrixCoefficient & epsImCoef,
             MatrixCoefficient & epsAbsCoef,
             MatrixCoefficient & susceptReCoef,
             MatrixCoefficient & susceptImCoef,
             Coefficient & muInvCoef,
             Coefficient * etaInvCoef,
             VectorCoefficient * kReCoef,
             VectorCoefficient * kImCoef,
             Array<int> & abcs,
             Array<ComplexVectorCoefficientByAttr*> & dbcs,
             Array<ComplexVectorCoefficientByAttr*> & nbcs,
             Array<ComplexCoefficientByAttr*> & sbcs,
             void (*j_r_src)(const Vector&, Vector&),
             void (*j_i_src)(const Vector&, Vector&),
             bool vis_u = false,
             bool pa = false);
   ~CPDSolver();

   HYPRE_Int GetProblemSize();

   void PrintSizes();

   void Assemble();

   void Update();

   void Solve();

   double GetError(const VectorCoefficient & EReCoef,
                   const VectorCoefficient & EImCoef) const;

   void GetErrorEstimates(Vector & errors);

   double GetGlobalDissipation() const;

   //double GetCoreDissipation() const;

   //double GetSOLDissipation() const;

   void RegisterVisItFields(VisItDataCollection & visit_dc);

   void WriteVisItFields(int it = 0);

   void InitializeGLVis();

   void DisplayToGLVis();

   void DisplayAnimationToGLVis();

   // const ParGridFunction & GetVectorPotential() { return *a_; }

private:

   class kmkCoefficient : public MatrixCoefficient
   {
   private:
      VectorCoefficient * krCoef_;
      VectorCoefficient * kiCoef_;
      Coefficient       * mCoef_;

      bool realPart_;
      double a_;

      mutable Vector kr;
      mutable Vector ki;

      void kmk(double a,
               const Vector & kl, double m, const Vector &kr,
               DenseMatrix & M)
      {
         double kk = kl * kr;
         for (int i=0; i<3; i++)
         {
            for (int j=0; j<3; j++)
            {
               M(i,j) += a * m * kl(j) * kr(i);
            }
            M(i,i) -= a * m * kk;
         }
      }

   public:
      kmkCoefficient(VectorCoefficient *krCoef, VectorCoefficient *kiCoef,
                     Coefficient *mCoef,
                     bool realPart, double a = 1.0)
         : MatrixCoefficient(3),
           krCoef_(krCoef), kiCoef_(kiCoef),
           mCoef_(mCoef),
           realPart_(realPart),
           a_(a), kr(3), ki(3)
      { kr = 0.0; ki = 0.0; }

      void Eval(DenseMatrix &M, ElementTransformation &T,
                const IntegrationPoint &ip)
      {
         M.SetSize(3);
         M = 0.0;
         if ((krCoef_ == NULL && kiCoef_ == NULL) ||
             mCoef_ == NULL)
         {
            return;
         }
         double m = 0.0;
         if (krCoef_) { krCoef_->Eval(kr, T, ip); }
         if (kiCoef_) { kiCoef_->Eval(ki, T, ip); }
         if (mCoef_) { m = mCoef_->Eval(T, ip); }

         if (realPart_)
         {
            if (krCoef_) { kmk(1.0, kr, m, kr, M); }
            if (kiCoef_) { kmk(-1.0, ki, m, ki, M); }
         }
         else
         {
            if (krCoef_ && kiCoef_) { kmk(1.0, kr, m, ki, M); }
            if (kiCoef_ && krCoef_) { kmk(1.0, ki, m, kr, M); }
         }
         if (a_ != 1.0) { M *= a_; }
      }
   };

   class CrossCoefficient : public MatrixCoefficient
   {
   private:
      VectorCoefficient * kCoef_;
      Coefficient * mCoef_;

      double a_;

      mutable Vector k;

   public:
      CrossCoefficient(VectorCoefficient *kCoef,
                       Coefficient *mCoef,
                       double a = 1.0)
         : MatrixCoefficient(3),
           kCoef_(kCoef),
           mCoef_(mCoef),
           a_(a), k(3)
      { k = 0.0; }

      void Eval(DenseMatrix &M, ElementTransformation &T,
                const IntegrationPoint &ip)
      {
         M.SetSize(3);
         M = 0.0;

         double m = 0.0;
         if (kCoef_) { kCoef_->Eval(k, T, ip); }
         if (mCoef_) { m = mCoef_->Eval(T, ip); }

         M(2,1) = a_ * m * k(0);
         M(0,2) = a_ * m * k(1);
         M(1,0) = a_ * m * k(2);

         M(1,2) = -M(2,1);
         M(2,0) = -M(0,2);
         M(0,1) = -M(1,0);
      }
   };

   void computeB(const ParComplexGridFunction & e,
                 ParComplexGridFunction & b);

   void computeD(const ParComplexGridFunction & e,
                 ParComplexGridFunction & d);

   int myid_;
   int num_procs_;
   int order_;
   int logging_;

   SolverType sol_;
   SolverOptions & solOpts_;
   PrecondType prec_;

   ComplexOperator::Convention conv_;

   bool ownsEtaInv_;
   bool vis_u_;
   bool pa_;

   double omega_;

   // double solNorm_;

   ParMesh * pmesh_;

   L2_ParFESpace * L2FESpace_;
   L2_ParFESpace * L2FESpace2p_;
   L2_ParFESpace * L2VFESpace_;
   H1_ParFESpace * H1FESpace_;
   ND_ParFESpace * HCurlFESpace_;
   RT_ParFESpace * HDivFESpace_;
   RT_ParFESpace * HDivFESpace2p_;

   Array<HYPRE_Int> blockTrueOffsets_;

   // ParSesquilinearForm * a0_;
   ParSesquilinearForm * a1_;
   ParBilinearForm * b1_;

   ParBilinearForm * m2_;
   ParMixedBilinearForm * m12EpsRe_;
   ParMixedBilinearForm * m12EpsIm_;

   ParDiscreteCurlOperator * curl_; // For Computing D from H
   ParDiscreteLinearOperator * kReCross_;
   ParDiscreteLinearOperator * kImCross_;

   ParBilinearForm * m0_;
   ParMixedBilinearForm * n20ZRe_;
   ParMixedBilinearForm * n20ZIm_;
   ParBilinearForm * m4r_;
   ParBilinearForm * m4i_;
   /*
   ParBilinearForm * m4cr_;
   ParBilinearForm * m4ci_;
   ParBilinearForm * m4solr_;
   ParBilinearForm * m4soli_;
   */

   ParComplexGridFunction * e_;   // Complex electric field (HCurl)
   ParComplexGridFunction * e_tmp_; // Temporary complex electric field (HCurl)
   ParComplexGridFunction * d_;   // Complex electric flux (HDiv)
   ParComplexGridFunction * b_;   // Complex magnetic flux (HDiv)

   ParGridFunction * temp_; // Temporary grid function (HCurl)
   ParDiscreteGradOperator * grad_; // For Computing E = Grad phi
   ParDiscreteLinearOperator * kOpr_; // E += i k phi
   ParDiscreteLinearOperator * kOpi_; // E += i (ik) phi
   ParComplexGridFunction * phi_; // Complex sheath potential (H1)
   ParComplexGridFunction * prev_phi_; // Complex sheath potential temporary (H1)
   ParComplexGridFunction * next_phi_; // Complex sheath potential temporary (H1)
   ParComplexGridFunction * z_; // Complex sheath potential (H1)
   ParGridFunction        * power_absorp_; // Real valued power absorption (H1)

   ParGridFunction * rectPot_; // Real valued rectified potential (H1)
   ParComplexGridFunction * j_;   // Complex current density (HCurl)
   ParComplexLinearForm   * rhs_; // Dual of complex current density (HCurl)
   ParGridFunction        * e_t_; // Time dependent Electric field
   ParComplexGridFunction * e_b_; // Complex parallel electric field (L2)
   //ParComplexGridFunction * e_perp_; // Complex perpendicular electric field (L2)
   ParComplexGridFunction * e_plus_; // Complex + polarized electric field (L2)
   ParComplexGridFunction * e_min_; // Complex - polarized electric field (L2)
   ParComplexGridFunction * e_v_; // Complex electric field (L2^d)
   ParComplexGridFunction * d_v_; // Complex electric flux (L2^d)
   ParComplexGridFunction * phi_v_; // Complex sheath potential (L2)
   ParComplexGridFunction * j_v_; // Complex current density (L2^d)
   ParGridFunction        * b_hat_; // Unit vector along B (HDiv)
   ParGridFunction        * u_;   // Energy density (L2)
   ParGridFunction        * uE_;  // Electric Energy density (L2)
   ParGridFunction        * uB_;  // Magnetic Energy density (L2)
   ParComplexGridFunction * S_;  // Poynting Vector (HDiv)
   ParComplexGridFunction * StixS_; // Stix S Coefficient (L2)
   ParComplexGridFunction * StixD_; // Stix D Coefficient (L2)
   ParComplexGridFunction * StixP_; // Stix P Coefficient (L2)
   //ParComplexGridFunction * EpsPara_; // B^T eps B / |B|^2 Coefficient (L2)

   HypreParMatrix * M4r_;
   HypreParMatrix * M4i_;
   /*
   HypreParMatrix * M4cr_;
   HypreParMatrix * M4ci_;
   HypreParMatrix * M4solr_;
   HypreParMatrix * M4soli_;
   */
   mutable HypreParVector * RHSr1_;
   mutable HypreParVector * RHSi1_;
   mutable HypreParVector * RHSr2_;
   mutable HypreParVector * RHSi2_;
   mutable HypreParVector * RHSr3_;
   mutable HypreParVector * RHSi3_;
   mutable HypreParVector * RHSr4_;
   mutable HypreParVector * RHSi4_;
   mutable HypreParVector * TMPr2_;
   mutable HypreParVector * TMPi2_;
   mutable HypreParVector * TMPr3_;
   mutable HypreParVector * TMPi3_;
   mutable HypreParVector * TMPr4_;
   mutable HypreParVector * TMPi4_;
   HypreParVector * Er_; 
   HypreParVector * Ei_;  

   VectorCoefficient * BCoef_;        // B Field Unit Vector
   MatrixCoefficient * epsReCoef_;    // Dielectric Material Coefficient
   MatrixCoefficient * epsImCoef_;    // Dielectric Material Coefficient
   MatrixCoefficient * epsAbsCoef_;   // Dielectric Material Coefficient
   MatrixCoefficient * susceptReCoef_;    // Real Susceptibility Coefficient
   MatrixCoefficient * susceptImCoef_;    // Imag Susceptibility Coefficient
   Coefficient       * muInvCoef_;    // Dia/Paramagnetic Material Coefficient
   Coefficient       * etaInvCoef_;   // Admittance Coefficient
   VectorCoefficient * kReCoef_;        // Wave Vector
   VectorCoefficient * kImCoef_;        // Wave Vector

   Coefficient * SReCoef_; // Stix S Coefficient
   Coefficient * SImCoef_; // Stix S Coefficient
   Coefficient * DReCoef_; // Stix D Coefficient
   Coefficient * DImCoef_; // Stix D Coefficient
   Coefficient * PReCoef_; // Stix P Coefficient
   Coefficient * PImCoef_; // Stix P Coefficient

   Coefficient * omegaCoef_;     // omega expressed as a Coefficient
   Coefficient * negOmegaCoef_;  // -omega expressed as a Coefficient
   Coefficient * omega2Coef_;    // omega^2 expressed as a Coefficient
   Coefficient * negOmega2Coef_; // -omega^2 expressed as a Coefficient
   Coefficient * abcCoef_;       // -omega eta^{-1}
   // Coefficient * sbcReCoef_;     //  omega Im(eta^{-1})
   // Coefficient * sbcImCoef_;     // -omega Re(eta^{-1})
   Coefficient * sinkx_;         // sin(ky * y + kz * z)
   Coefficient * coskx_;         // cos(ky * y + kz * z)
   Coefficient * negsinkx_;      // -sin(ky * y + kz * z)
   // Coefficient * negMuInvCoef_;  // -1.0 / mu

   MatrixCoefficient * massReCoef_;  // -omega^2 Re(epsilon)
   MatrixCoefficient * massImCoef_;  // omega^2 Im(epsilon)
   MatrixCoefficient * posMassCoef_; // omega^2 Abs(epsilon)
   // MatrixCoefficient * negMuInvkxkxCoef_; // -\vec{k}\times\vec{k}\times/mu

   kmkCoefficient kmkReCoef_;
   kmkCoefficient kmkImCoef_;
   CrossCoefficient kmReCoef_;
   CrossCoefficient kmImCoef_;

   VectorCoefficient * jrCoef_;     // Volume Current Density Function
   VectorCoefficient * jiCoef_;     // Volume Current Density Function
   VectorCoefficient * rhsrCoef_;     // Volume Current Density Function
   VectorCoefficient * rhsiCoef_;     // Volume Current Density Function

   VectorGridFunctionCoefficient erCoef_;
   VectorGridFunctionCoefficient eiCoef_;

   CurlGridFunctionCoefficient derCoef_;
   CurlGridFunctionCoefficient deiCoef_;

   EnergyDensityCoef     uCoef_;
   ElectricEnergyDensityCoef uECoef_;
   MagneticEnergyDensityCoef uBCoef_;
   PoyntingVectorReCoef SrCoef_;
   PoyntingVectorImCoef SiCoef_;

   // const VectorCoefficient & erCoef_;     // Electric Field Boundary Condition
   // const VectorCoefficient & eiCoef_;     // Electric Field Boundary Condition

   void   (*j_r_src_)(const Vector&, Vector&);
   void   (*j_i_src_)(const Vector&, Vector&);

   // Array of 0's and 1's marking the location of absorbing surfaces
   Array<int> abc_bdr_marker_;

   // Array of 0's and 1's marking the location of sheath surfaces
   // Array<int> sbc_marker_;

   // Array of 0's and 1's marking the location of Dirichlet boundaries
   Array<int> dbc_bdr_marker_;
   // void   (*e_r_bc_)(const Vector&, Vector&);
   // void   (*e_i_bc_)(const Vector&, Vector&);

   // Array<int> * dbcs_;
   Array<ComplexVectorCoefficientByAttr*> * dbcs_;
   Array<int> ess_bdr_;
   Array<int> ess_bdr_tdofs_;
   Array<int> non_k_bdr_;
   Array<int> core_attr_marker_;
   Array<int> sol_attr_marker_;

   Array<ComplexVectorCoefficientByAttr*> * nbcs_; // Surface current BCs
   Array<ComplexVectorCoefficientByAttr*> * nkbcs_; // Neumann BCs (-i*omega*K)

   Array<ComplexCoefficientByAttr*> * sbcs_; // Sheath BCs

   VisItDataCollection * visit_dc_;

   std::map<std::string,socketstream*> socks_;
};

} // namespace plasma

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_COLD_PLASMA_DIELECTRIC_SOLVER
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
//    ---------------------------------------------------------------------
//    Mesh Optimizer NLP Miniapp: Optimize high-order meshes - Parallel Version
//    ---------------------------------------------------------------------
//
// This miniapp performs mesh optimization using the Target-Matrix Optimization
// Paradigm (TMOP) by P.Knupp et al., it used non-linear programming techniques
// to solve the proble,
//
// Compile with: make pmesh-optimizer_NLP
// Solid block
// make pmesh-optimizer_NLP -j4 && mpirun -np 10 pmesh-optimizer_NLP -met 1 -ch 3e-3 -ni 200 -w1 -1e4 -w2 1e-1 -rs 2 -o 2 -lsn 2.0 -lse 1.01 -alpha 20 -bndrfree -qt 7 -ft 9 -vis -filter -frad 0.1 -ph 1

// Square hole
// make pmesh-optimizer_NLP -j4 && mpirun -np 10  pmesh-optimizer_NLP -met 1 -ch 3e-3 -ni 300 -w1 -1e5 -w2 1e-1 -rs 3 -o 2 -lsn 2.0 -lse 1.01 -alpha 20 -bndrfree -qt 7 -ft 9 -vis -filter -frad 0.1 -ph 1 -m SquareFrame.mesh -jid 111

// beam
// make pmesh-optimizer_NLP -j4 && mpirun -np 1 pmesh-optimizer_NLP -met 1 -ch 3e-3 -ni 300 -w1 -1e6 -w2 5e-1 -rs 0 -o 2 -lsn 2.0 -lse 1.01 -alpha 20 -bndrfree -qt 7 -ft 10 -vis -filter -frad 0.001 -ph 1 -beam -jid 102
// make pmesh-optimizer_NLP -j4 && mpirun -np 1 pmesh-optimizer_NLP -met 1 -ch 3e-3 -ni 300 -w1 -1e6 -w2 5e-1 -rs 0 -o 1 -lsn 2.0 -lse 1.01 -alpha 20 -bndrfree -qt 7 -ft 10 -vis -filter -frad 0.001 -ph 1 -beam -jid 101
/*******************************/
///// Convergence results - Poisson - 2nd order - shock wave - zz
// make pmesh-optimizer_NLP -j4 && mpirun -np 10 pmesh-optimizer_NLP -met 0 -ch 2e-3 -ni 300  -w1 1e4 -w2 1e-2  -rs 1 -o 2 -lsn 10.1 -lse 10.1 -alpha 20 -bndrfree -qt 5 -ft 2 -vis -weakbc -filter -frad 0.05 -jid 11
// make pmesh-optimizer_NLP -j4 && mpirun -np 10 pmesh-optimizer_NLP -met 0 -ch 2e-3 -ni 300 -w1 1e5 -w2 1e-2 -rs 2 -o 2 -lsn 10.1 -lse 10.1 -alpha 20 -bndrfree -qt 5 -ft 2 -vis -weakbc -filter -frad 0.05 -jid 12
// make pmesh-optimizer_NLP -j4 && mpirun -np 10 pmesh-optimizer_NLP -met 0 -ch 5e-4 -ni 300 -w1 1e6 -w2 1e-2 -rs 3 -o 2 -lsn 10.1 -lse 10.1 -alpha 20 -bndrfree -qt 5 -ft 2 -vis -weakbc -filter -frad 0.005 -jid 13
// make pmesh-optimizer_NLP -j4 && mpirun -np 10 pmesh-optimizer_NLP -met 0 -ch 1e-3 -ni 200 -w1 1e7 -w2 1e-3 -rs 4 -o 2 -lsn 10.1 -lse 10.1 -alpha 20 -bndrfree -qt 5 -ft 2 -vis -weakbc -filter -frad 0.02 -jid 14
///// Convergence results - Poisson - 1st order - shock wave - zz
// make pmesh-optimizer_NLP -j4 && mpirun -np 10 pmesh-optimizer_NLP -met 0 -ch 1e-3 -ni 1000 -w1 1e2 -w2 10e-1 -rs 1 -o 1 -lsn 10.1 -lse 10.1 -alpha 20 -bndrfree -qt 5 -ft 2 -vis -weakbc -filter -frad -0.1 -jid 21
// make pmesh-optimizer_NLP -j4 && mpirun -np 10 pmesh-optimizer_NLP -met 0 -ch 1e-3 -ni 1000 -w1 1e3 -w2 10e-1 -rs 2 -o 1 -lsn 10.1 -lse 10.1 -alpha 20 -bndrfree -qt 5 -ft 2 -vis -weakbc -filter -frad -0.1 -jid 22
// make pmesh-optimizer_NLP -j4 && mpirun -np 10 pmesh-optimizer_NLP -met 0 -ch 1e-3 -ni 1000 -w1 1e4 -w2 10e-1 -rs 3 -o 1 -lsn 10.1 -lse 10.1 -alpha 20 -bndrfree -qt 5 -ft 2 -vis -weakbc -filter -frad -0.1 -jid 23
// make pmesh-optimizer_NLP -j4 && mpirun -np 10 pmesh-optimizer_NLP -met 0 -ch 1e-3 -ni 1000 -w1 1e5 -w2 10e-1 -rs 4 -o 1 -lsn 10.1 -lse 10.1 -alpha 20 -bndrfree -qt 5 -ft 2 -vis -weakbc -filter -frad -0.1 -jid 24
////// simplices
// make pmesh-optimizer_NLP -j4 && mpirun -np 10 pmesh-optimizer_NLP -met 0 -ch 2e-3 -ni 300 -w1 1e5 -w2 1e-2 -rs 1 -o 2 -lsn 10.1 -lse 10.1 -alpha 20 -bndrfree -qt 5 -ft 2 -vis -weakbc -filter -frad 0.05 -m square01-tri.mesh -jid 25


// average error - 2nd order - shock wave around corner - convergence
// make pmesh-optimizer_NLP -j4 && mpirun -np 10 pmesh-optimizer_NLP -met 0 -ch 2e-3 -ni 300 -w1 1e2 -w2 1e-2 -rs 1 -o 2 -lsn 2.01 -lse 1.01 -alpha 20 -bndrfree -qt 3 -ft 2 -vis -weakbc -filter -frad -0.01 -jid 31
// make pmesh-optimizer_NLP -j4 && mpirun -np 10 pmesh-optimizer_NLP -met 0 -ch 2e-3 -ni 300 -w1 1e3 -w2 1e-2 -rs 2 -o 2 -lsn 2.01 -lse 1.01 -alpha 20 -bndrfree -qt 3 -ft 2 -vis -weakbc -filter -frad -0.01 -jid 32
// make pmesh-optimizer_NLP -j4 && mpirun -np 10 pmesh-optimizer_NLP -met 0 -ch 2e-3 -ni 300 -w1 1e4 -w2 1e-2 -rs 3 -o 2 -lsn 2.01 -lse 1.01 -alpha 20 -bndrfree -qt 3 -ft 2 -vis -weakbc -filter -frad -0.01 -jid 33
// make pmesh-optimizer_NLP -j4 && mpirun -np 10 pmesh-optimizer_NLP -met 0 -ch 2e-3 -ni 300 -w1 1e5 -w2 1e-2 -rs 4 -o 2 -lsn 2.01 -lse 1.01 -alpha 20 -bndrfree -qt 3 -ft 2 -vis -weakbc -filter -frad -0.01 -jid 34
// 1st order
// make pmesh-optimizer_NLP -j4 && mpirun -np 10 pmesh-optimizer_NLP -met 0 -ch 2e-3 -ni 300 -w2 1 -o 1 -lsn 2.01 -lse 1.01 -alpha 20 -bndrfree -qt 3 -ft 2 -vis -weakbc -filter -frad -0.01 -w1 2e3 -rs 1 -jid 41
// make pmesh-optimizer_NLP -j4 && mpirun -np 10 pmesh-optimizer_NLP -met 0 -ch 2e-3 -ni 300 -w2 1 -o 1 -lsn 2.01 -lse 1.01 -alpha 20 -bndrfree -qt 3 -ft 2 -vis -weakbc -filter -frad -0.01 -w1 4e4 -rs 2 -jid 42
// make pmesh-optimizer_NLP -j4 && mpirun -np 10 pmesh-optimizer_NLP -met 0 -ch 2e-3 -ni 300 -w2 1 -o 1 -lsn 2.01 -lse 1.01 -alpha 20 -bndrfree -qt 3 -ft 2 -vis -weakbc -filter -frad -0.01 -w1 8e5 -rs 3 -jid 43
// make pmesh-optimizer_NLP -j4 && mpirun -np 10 pmesh-optimizer_NLP -met 0 -ch 2e-3 -ni 300 -w2 1 -o 1 -lsn 2.01 -lse 1.01 -alpha 20 -bndrfree -qt 3 -ft 2 -vis -weakbc -filter -frad -0.01 -w1 16e6 -rs 4 -jid 44

// solution for vis only
// make pmesh-optimizer_NLP -j4 && mpirun -np 10 pmesh-optimizer_NLP -met 0 -ch 2e-3 -ni 0 -w2 1 -o 1 -lsn 2.01 -lse 1.01 -alpha 20 -bndrfree -qt 3 -ft 2 -vis -weakbc -filter -frad -0.01 -w1 2e3 -rs 5 -jid 49

// same but with simplices
// make pmesh-optimizer_NLP -j4 && mpirun -np 10 pmesh-optimizer_NLP -met 0 -ch 2e-3 -ni 300 -w1 1e5 -w2 1 -rs 1 -o 2 -lsn 2.01 -lse 1.01 -alpha 20 -bndrfree -qt 3 -ft 2 -vis -weakbc -filter -frad 0.005 -m square01-tri.mesh -jid 51
// 3D with avg error - o1
// make pmesh-optimizer_NLP -j4 && mpirun -np 10 pmesh-optimizer_NLP -met 0 -ch 1e-3 -ni 300 -w1 1e4 -w2 1e-2 -o 1 -lsn 10.1 -lse 10.1 -alpha 20 -bndrfree -qt 3 -ft 2 -vis -weakbc -filter -frad 0.01 -rs 0 -m cube-tet.mesh -mid 303 -jid 52
// make pmesh-optimizer_NLP -j4 && mpirun -np 10 pmesh-optimizer_NLP -met 0 -ch 1e-3 -ni 300 -w1 1e4 -w2 1e-2 -o 1 -lsn 10.1 -lse 10.1 -alpha 20 -bndrfree -qt 3 -ft 2 -vis -weakbc -filter -frad 0.01 -rs 2 -m cube.mesh -mid 303 -jid 53
// 3D with avg error - o2
// make pmesh-optimizer_NLP -j4 && mpirun -np 10 pmesh-optimizer_NLP -met 0 -ch 1e-3 -ni 300 -w1 1e7 -w2 1 -o 2 -lsn 10.1 -lse 10.1 -alpha 20 -bndrfree -qt 3 -ft 2 -vis -weakbc -filter -frad 0.05 -rs 0 -m cube-tet.mesh -mid 303 -jid 54
// make pmesh-optimizer_NLP -j4 && mpirun -np 10 pmesh-optimizer_NLP -met 0 -ch 1e-3 -ni 250 -w1 1e6 -w2 1 -o 2 -lsn 10.1 -lse 10.1 -alpha 20 -bndrfree -qt 3 -ft 2 -vis -weakbc -filter -frad 0.005 -rs 2 -m cube.mesh -mid 303 -jid 55

// l2 with wave around center - linear
// make pmesh-optimizer_NLP -j && mpirun -np 10 pmesh-optimizer_NLP -met 0 -ch 1e-4 -ni 400 -ft 1 --qtype 0 -w1 2e4 -w2 1e-1 -m square01.mesh -rs 2 -o 1 -lsn 1.01 -lse 1.01 -alpha 10 -bndrfree
// h1 with wave around center - linear
// make pmesh-optimizer_NLP -j && mpirun -np 10 pmesh-optimizer_NLP -met 0 -ch 1e-3 -ni 200 -ft 1 --qtype 1 -w1 2e2 -w2 15e-1 -m uare01.mesh -rs 2 -o 1 -lsn 1.01 -lse 1.01 -alpha 10 -bndrfree

// inclined wave with avg error
//  make pmesh-optimizer_NLP -j4 && mpirun -np 10 pmesh-optimizer_NLP -met 0 -ch 2e-3 -ni 1000 -w1 1e3 -w2 1e-2 -rs 2 -o 2 -lsn 2.0 -lse 1.01 -alpha 20 -bndrfree -qt 3 -ft 3 -vis -weakbc -filter -frad 0.005

// avg error - inclined wave + analytic orientation
// make pmesh-optimizer_NLP -j4 && mpirun -np 10 pmesh-optimizer_NLP -met 0 -ch 2e-3 -ni 500 -w1 4e3 -w2 1 -rs 2 -o 2 -lsn 2.0 -lse 1.01 -alpha 20 -bndrfree -qt 3 -ft 3 -vis -weakbc -filter -frad 0.005 -mid 107 -tid 5 -jid 71
// make pmesh-optimizer_NLP -j4 && mpirun -np 10 pmesh-optimizer_NLP -met 0 -ch 2e-3 -ni 500 -w1 4e4 -w2 1 -rs 2 -o 2 -lsn 2.0 -lse 1.01 -alpha 20 -bndrfree -qt 3 -ft 3 -vis -weakbc -filter -frad 0.005 -mid 107 -tid 5 -jid 72

#include "mfem.hpp"
#include "../common/mfem-common.hpp"
#include "linalg/dual.hpp"
#include <iostream>
#include <fstream>
#include "mesh-optimizer_using_NLP.hpp"
#include "MMA.hpp"
#include "mesh-optimizer.hpp"

using namespace mfem;
using namespace std;

/// MFEM native AD-type for first derivatives
typedef internal::dual<real_t, real_t> ADFType;
/// MFEM native AD-type for second derivatives
typedef internal::dual<ADFType, ADFType> ADSType;
/// MFEM native AD-type for third derivatives
typedef internal::dual<ADSType, ADSType> ADTType;

real_t ADVal_func( const Vector &x, std::function<ADFType( std::vector<ADFType>&)> func)
{
  int dim = x.Size();
  int matsize = dim;
   std::vector<ADFType> adinp(matsize);
   for (int i=0; i<matsize; i++) { adinp[i] = ADFType{x[i], 0.0}; }

   return func(adinp).value;
}


void ADGrad_func( const Vector &x, std::function<ADFType( std::vector<ADFType>&)> func, Vector &grad)
{
   int dim = x.Size();

   std::vector<ADFType> adinp(dim);

   for (int i=0; i<dim; i++) { adinp[i] = ADFType{x[i], 0.0}; }

   for (int i=0; i<dim; i++)
   {
      adinp[i] = ADFType{x[i], 1.0};
      ADFType rez = func(adinp);
      grad[i] = rez.gradient;
      adinp[i] = ADFType{x[i], 0.0};
   }
}

void ADHessian_func(const Vector &x, std::function<ADSType( std::vector<ADSType>&)> func, DenseMatrix &H)
{
   int dim = x.Size();

   //use forward-forward mode
   std::vector<ADSType> aduu(dim);
   for (int ii = 0; ii < dim; ii++)
   {
      aduu[ii].value = ADFType{x[ii], 0.0};
      aduu[ii].gradient = ADFType{0.0, 0.0};
   }

   for (int ii = 0; ii < dim; ii++)
   {
      aduu[ii].value = ADFType{x[ii], 1.0};
      for (int jj = 0; jj < (ii + 1); jj++)
      {
         aduu[jj].gradient = ADFType{1.0, 0.0};
         ADSType rez = func(aduu);
         H(ii,jj) = rez.gradient.gradient;
         H(jj,ii) = rez.gradient.gradient;
         aduu[jj].gradient = ADFType{0.0, 0.0};
      }
      aduu[ii].value = ADFType{x[ii], 0.0};
   }
   return;
}

void AD3rdDeric_func(const Vector &x, std::function<ADTType( std::vector<ADTType>&)> func, std::vector<DenseMatrix> &TRD)
{
   int dim = x.Size();

   //use forward-forward mode
   std::vector<ADTType> aduu(dim);
   for (int ii = 0; ii < dim; ii++)
   {
      aduu[ii].value.value = ADFType{x[ii], 0.0};
      aduu[ii].value.gradient = ADFType{0.0, 0.0};
      aduu[ii].gradient.value = ADFType{0.0, 0.0};
      aduu[ii].gradient.gradient = ADFType{0.0, 0.0};
   }

   for (int ii = 0; ii < dim; ii++)
   {
      aduu[ii].value.value = ADFType{x[ii], 1.0};
      for (int jj = 0; jj < dim; jj++)
      {
         aduu[jj].value.gradient = ADFType{1.0, 0.0};
         for (int kk = 0; kk < dim; kk++)                    // FIXME is ymmetric, only loop over half the possibilites
         {
            aduu[kk].gradient.value = ADFType{1.0, 0.0};
            //aduu[kk].gradient.gradient = ADFType{1.0, 0.0};
            ADTType rez = func(aduu);
            TRD[ii](jj,kk) = rez.gradient.gradient.gradient;
            aduu[kk].gradient.value = ADFType{0.0, 0.0};
            //aduu[kk].gradient.gradient = ADFType{0.0, 0.0};
         }
         aduu[jj].value.gradient = ADFType{0.0, 0.0};
      }
      aduu[ii].value.value = ADFType{x[ii], 0.0};
   }
   return;
}

int ftype = 1;
double kw = 10.0;
double alphaw = 50;

template <typename type>
auto func_0( std::vector<type>& x ) -> type
{
   return sin( 1.0*M_PI *x[0] )*sin(2.0*M_PI*x[1]);;
};

template <typename type>
auto func_1( std::vector<type>& x ) -> type
{
    double k_w = kw;
    double k_t = 0.5;
    double T_ref = 1.0;

    return 0.5+0.5*tanh(k_w*  ((sin( M_PI *x[0] )*sin(M_PI *x[1]))-k_t*T_ref)   );
};

template <typename type>
auto func_8( std::vector<type>& x ) -> type
{
   return sin( M_PI *x[0] );
};

template <typename type>
auto func_2( std::vector<type>& x ) -> type
{
    double theta = 0.0;
    auto xv = 1.0-x[0];
    auto yv = 1.0-x[1];
    auto zv = 1.0-x[1];
    if (x.size() == 3) {
      zv = 1.0-x[2];
    }
    double xc = -0.05,
           yc = -0.05,
           zc = -0.05,
           rc = 0.7,
           alpha = alphaw;
    auto dx = xv-xc,
         dy = yv-yc,
         dz = dy;
    if (x.size() == 3) {
      dz = zv-zc;
    }
    auto val = dx*dx + dy*dy;
    if (x.size() == 3) { val += dz*dz; }
    if (val > 0.0) { val = sqrt(val); }
    val -= rc;
    val = alpha*val;
    // return 5.0+atan(val);
    return atan(val);
};

template <typename type>
auto func_3( std::vector<type>& x ) -> type
{
    auto xv = x[0];
    auto yv = x[1];
    auto alpha = alphaw;
    auto dx = xv - 0.5-0.2*(yv-0.5);
    // auto dx = xv-0.5;
    // auto dx = xv - 0.5-0.2*(yv-0.5);
    return atan(alpha*dx);
}

double tanh_left_right_walls(const Vector &x)
{
  double xv = x(0);
  double yv = x(1);
  double beta = 20.0;
  double betay = 50.0;
  double yscale = 0.5*(std::tanh(betay*(yv-.2))-std::tanh(betay*(yv-0.8)));
  double xscale = 0.5*(std::tanh(beta*(xv-.2))-std::tanh(beta*(xv-0.8)));
  return xscale;
}

class OSCoefficient : public TMOPMatrixCoefficient
{
private:
   int metric, dd;

public:
   OSCoefficient(int dim, int metric_id)
      : TMOPMatrixCoefficient(dim), dd(dim), metric(metric_id) { }

   void Eval(DenseMatrix &K, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      Vector pos(dd);
      T.Transform(ip, pos);
      MFEM_VERIFY(dd == 2,"OSCoefficient does not support 3D\n");
      const real_t xc = pos(0), yc = pos(1);
      real_t theta = M_PI * yc * (1.0 - yc) * cos(2 * M_PI * xc);
      // real_t alpha_bar = 0.1;
      K(0, 0) =  cos(theta);
      K(1, 0) =  sin(theta);
      K(0, 1) = -sin(theta);
      K(1, 1) =  cos(theta);
      // K *= alpha_bar;
   }

    void EvalGrad(DenseMatrix &K, ElementTransformation &T,
                 const IntegrationPoint &ip, int comp) override
   {
      Vector pos(dd);
      T.Transform(ip, pos);
      K = 0.;
   }
};

double trueSolFunc(const Vector & x)
{
  if (ftype == 0)
  {
    return ADVal_func(x, func_0<ADFType>);
  }
  else if (ftype == 1) // circular wave centered in domain
  {
      return ADVal_func(x, func_1<ADFType>);
  }
  else if (ftype == 2) // circular shock wave front centered at origin
  {
    return ADVal_func(x, func_2<ADFType>);
  }
  else if (ftype == 3) // incline shock
  {
    return ADVal_func(x, func_3<ADFType>);
  }
  else if (ftype == 4)
  {
    double xv = x[0], yv = x[1];
    double yc = yv-0.5;
    double delta = 0.1;
    return std::atan(alphaw*(yv - 0.5 - delta*sin(2*M_PI*xv)));
  }
  else if (ftype == 5)
  {
    real_t xv = x[0];
    real_t yv = x[1];
    real_t r = sqrt(xv*xv + yv*yv);
    real_t alpha = 2./3.;
    real_t phi = atan2(yv,xv);
    if (phi < 0) { phi += 2*M_PI; }
    return pow(r,alpha) * sin(alpha * phi);
  }
  else if (ftype == 6)
  {
    double xv = x[0];
    double yv = x[1];
    double alpha = alphaw;
    double dx = xv - 0.48;
    return std::atan(alpha*dx);
  }
  else if (ftype == 7) // circular wave centered in domain
  {
    return x[0]*x[0];
  }
  else if (ftype == 8)
  {
    double val = std::sin( M_PI *x[0] );
    return val;
  }
  return 0.0;
};

void trueSolGradFunc(const Vector & x,Vector & grad)
{
  if (ftype == 0)
  {
    ADGrad_func(x, func_0<ADFType>, grad);
  }
  else if (ftype == 1) // circular wave centered in domain
  {
    ADGrad_func(x, func_1<ADFType>, grad);
  }
  else if (ftype == 2) // circular shock wave front centered at origin
  {
    ADGrad_func(x, func_2<ADFType>, grad);
  }
  else if (ftype == 3)
  {
    ADGrad_func(x, func_3<ADFType>, grad);
  }
  else if (ftype == 4)
  {
    double xv = x[0], yv = x[1];
    double delta = 0.1;
    double phi = alphaw*(yv-0.5-delta*std::sin(2*M_PI*xv));
    double den = 1.0 + phi*phi;
    grad[0] = -2.0*M_PI*alphaw*delta*std::cos(2*M_PI*xv)/den;
    grad[1] = alphaw/den;
  }
  else if (ftype == 5)
  {
    real_t xv = x[0];
    real_t yv = x[1];
    real_t r = sqrt(xv*xv + yv*yv);
    real_t alpha = 2./3.;
    real_t phi = atan2(yv,xv);
    if (phi < 0) { phi += 2*M_PI; }

    real_t r_x = xv/r;
    real_t r_y = yv/r;
    real_t phi_x = - yv / (r*r);
    real_t phi_y = xv / (r*r);
    real_t beta = alpha * pow(r,alpha - 1.);
    grad[0] = beta*(r_x * sin(alpha*phi) + r * phi_x * cos(alpha*phi));
    grad[1] = beta*(r_y * sin(alpha*phi) + r * phi_y * cos(alpha*phi));
  }
  else if (ftype == 6)
  {
    double xv = x[0];
    double yv = x[1];
    double alpha = alphaw;
    double dx = xv - 0.48;
    grad[0] = alpha/(1.0+std::pow(dx*alpha,2.0));
    grad[1] = 0.0;
  }
  else if (ftype == 7)
  {
    grad[0] = 2.0*x[0];
    grad[1] = 0.0;
  }
  else if (ftype == 8)
  {
    grad[0] = M_PI*std::cos( M_PI *x[0] );
    grad[1] = 0.0;
  }
};

double loadFunc(const Vector & x)
{
  if (ftype == 0)
  {
    DenseMatrix Hessian(x.Size());
    ADHessian_func(x, func_0<ADSType>, Hessian);
    double val = -1.0*(Hessian(0,0)+Hessian(1,1));
    return val;
  }
  else if (ftype == 1)
  {
    DenseMatrix Hessian(x.Size());
    ADHessian_func(x, func_1<ADSType>, Hessian);
    double val = -1.0*(Hessian(0,0)+Hessian(1,1));
    return val;
  }
  else if (ftype == 2)
  {
    DenseMatrix Hessian(x.Size());
    ADHessian_func(x, func_2<ADSType>, Hessian);
    double val = -1.0*(Hessian(0,0)+Hessian(1,1));
    return val;
  }
  else if (ftype == 3)
  {
    DenseMatrix Hessian(x.Size());
    ADHessian_func(x, func_3<ADSType>, Hessian);
    double val = -1.0*(Hessian(0,0)+Hessian(1,1));
    return val;
  }
  else if (ftype == 4)
  {
    double xv = x[0], yv = x[1];
    double delta = 0.1;
    double phi = alphaw*(yv-0.5-delta*std::sin(2*M_PI*xv));
    double den = 1.0 + phi*phi;
    double phi_x = -2.0*M_PI*alphaw*delta*std::cos(2*M_PI*xv);
    double term1 = (2*phi/(den*den))*(phi_x*phi_x+alphaw*alphaw);
    double term2 = 4*M_PI*M_PI*alphaw*delta*std::sin(2*M_PI*xv)/den;
    return term1-term2;
  }
  else if (ftype == 5)
  {
    return 0.0;
  }
  else if (ftype == 6)
  {
    double xv = x[0];
    double yv = x[1];
    double alpha = alphaw;
    double dx = xv - 0.48;
    double num1 = std::pow(alpha,3.0)*dx;
    double den1 = std::pow((1.0+std::pow(dx*alpha,2.0)),2.0);
    return 2.0*num1/den1;
  }
  else if (ftype == 7)
  {
    return -2.0;
  }
  else if (ftype == 8)
  {
    double val = M_PI*M_PI * std::sin( M_PI *x[0] );
    return val;
  }
  else if (ftype == 9)
  {
    return -1.0;
  }
  else if (ftype == 10)
  {
    return -1e-3;
  }
  return 0.0;
};

void trueLoadFuncGrad(const Vector & x,Vector & grad)
{
  if (ftype == 0)
  {
    std::vector<DenseMatrix> TRD(x.Size());
    for(int i=0; i<x.Size();i++){TRD[i].SetSize(x.Size());}
    AD3rdDeric_func(x, func_0<ADTType>, TRD);
    grad[0] = -1.0*(TRD[0](0,0)+TRD[0](1,1));
    grad[1] = -1.0*(TRD[1](0,0)+TRD[1](1,1));
  }
  else if (ftype == 1)
  {
    std::vector<DenseMatrix> TRD(x.Size());
    for(int i=0; i<x.Size();i++){TRD[i].SetSize(x.Size());}
    AD3rdDeric_func(x, func_1<ADTType>, TRD);
    grad[0] = -1.0*(TRD[0](0,0)+TRD[1](1,0));
    grad[1] = -1.0*(TRD[0](0,1)+TRD[1](1,1));
  }
  else if (ftype == 2)
  {
    std::vector<DenseMatrix> TRD(x.Size());
    for(int i=0; i<x.Size();i++){TRD[i].SetSize(x.Size());}
    AD3rdDeric_func(x, func_2<ADTType>, TRD);
    grad[0] = -1.0*(TRD[0](0,0)+TRD[0](1,1));
    grad[1] = -1.0*(TRD[1](0,0)+TRD[1](1,1));
  }
  else if (ftype == 3)
  {
    std::vector<DenseMatrix> TRD(x.Size());
    for(int i=0; i<x.Size();i++){TRD[i].SetSize(x.Size());}
    AD3rdDeric_func(x, func_3<ADTType>, TRD);
    grad[0] = -1.0*(TRD[0](0,0)+TRD[0](1,1));
    grad[1] = -1.0*(TRD[1](0,0)+TRD[1](1,1));
  }
  else if (ftype == 8)
  {
    std::vector<DenseMatrix> TRD(x.Size());
    for(int i=0; i<x.Size();i++){TRD[i].SetSize(x.Size());}
    AD3rdDeric_func(x, func_8<ADTType>, TRD);
    grad[0] = -1.0*(TRD[0](0,0)+TRD[1](1,0));
    grad[1] = -1.0*(TRD[0](0,1)+TRD[1](1,1));

    // grad[0] = M_PI*M_PI*M_PI*std::cos( M_PI *x[0] );
    // grad[1] = 0.0;
  }
}

void trueHessianFunc(const Vector & x,DenseMatrix & Hessian)
{
  Hessian.SetSize(x.Size());
  if (ftype == 0)
  {
    ADHessian_func(x, func_0<ADSType>, Hessian);
  }
  else if (ftype == 1)
  {
    ADHessian_func(x, func_1<ADSType>, Hessian);
  }
  else if (ftype == 2)
  {
    ADHessian_func(x, func_2<ADSType>, Hessian);
  }
  else if (ftype == 3)
  {
    ADHessian_func(x, func_3<ADSType>, Hessian);
  }
  else
  {
    MFEM_ABORT("Not implemented\n");
  }
}

void trueHessianFunc_v(const Vector & x,Vector & Hessian)
{
  DenseMatrix HessianM;
  trueHessianFunc(x,HessianM);
  Hessian.SetSize(HessianM.Height()*HessianM.Width());
  Hessian = HessianM.GetData();
}

void VisVectorField(OSCoefficient *adapt_coeff, ParMesh *pmesh, ParGridFunction *orifield)
{
  ParFiniteElementSpace *pfespace = orifield->ParFESpace();
  int dim = pfespace->GetMesh()->Dimension();

    DenseMatrix mat(dim);
    Vector vec(dim);
    Array<int> dofs;
  // Loop over the elements and project the adapt_coeff to vector field
  for (int e = 0; e < pmesh->GetNE(); e++)
  {
    const FiniteElement *fe = pfespace->GetFE(e);
    const IntegrationRule ir = fe->GetNodes();
    const int dof = fe->GetDof();
    ElementTransformation *trans = pmesh->GetElementTransformation(e);
    Vector nodevals(dof*dim);
    for (int q = 0; q < ir.GetNPoints(); q++)
    {
      const IntegrationPoint &ip = ir.IntPoint(q);
      trans->SetIntPoint(&ip);
      adapt_coeff->Eval(mat, *trans, ip);
      mat.GetColumn(0, vec);
      nodevals[q + dof*0] = vec[0];
      nodevals[q + dof*1] = vec[1];
    }
    pfespace->GetElementVDofs(e, dofs);
    orifield->SetSubVector(dofs, nodevals);
  }
}

void GetScalarDerivative(ParGridFunction *inp, ParGridFunction *outp, const int dim)
{
  for (int d = 0; d < dim; d++)
    {
      ParGridFunction outp_grad_temp(inp->ParFESpace(), outp->GetData() + d*inp->Size());
      inp->GetDerivative(1,d,outp_grad_temp);
    }
}

void GetMaxDisplacement(ParGridFunction *solu, Vector &maxdisp)
{
  ParMesh *PMesh = solu->ParFESpace()->GetParMesh();
  GridFunction *nodes = PMesh->GetNodes();
  const int dim = PMesh->SpaceDimension();
  // compare solution at xmax, ymin
  Vector xmin(dim), xmax(dim);
  PMesh->GetBoundingBox(xmin, xmax);
  int nnodes = nodes->Size()/dim;

  maxdisp.SetSize(3);
  maxdisp = -std::numeric_limits<double>::max();
  for (int i = 0; i < nnodes; i++)
  {
    if ((*nodes)(i) == xmax(0) && (*nodes)(i+nnodes) == xmin(1))
    {
      maxdisp(0) = (*solu)(i);
      maxdisp(1) = (*solu)(solu->Size()/dim+i);
      maxdisp(2) = sqrt(maxdisp(0)*maxdisp(0) + maxdisp(1)*maxdisp(1));
      break;
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, maxdisp.GetData(), 3, MPI_DOUBLE, MPI_MAX, PMesh->GetComm());
}

int main (int argc, char *argv[])
{
   // 0. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int myid = Mpi::WorldRank();
   int nranks = Mpi::WorldSize();
   Hypre::Init();

#ifdef MFEM_USE_PETSC
   const char *petscrc_file = "";
   MFEMInitializePetsc(NULL,NULL,petscrc_file,NULL);
#endif

  int qoitype = static_cast<int>(QoIType::H1S_ERROR);
  bool weakBC = false;
  bool perturbMesh = false;
  double epsilon_pert =  0.006;
  int ref_ser = 2;
  int mesh_node_ordering = 0;
  int max_it = 100;
  double max_ch=0.002; //max design change
  double weight_1 = 1e4; //1e7; // 5e2;
  double weight_tmop = 1e-2;
  int metric_id   = 2;
  int target_id   = 1;
  int quad_order  = 8;
  int quad_order2 = 8;
  srand(9898975);
  bool visualization = false;
  double filterRadius = 0.000;
  int method = 0;
  int mesh_poly_deg     = 1;
  int physics_deg       = -1;
  int nx                = 4;
  int ny                = 4;
  const char *mesh_file = "null.mesh";
  bool exactaction      = false;
  double ls_norm_fac    = 1.2;
  double ls_energy_fac  = 1.1;
  bool   bndr_fix       = true;
  bool   filter         = false;
  bool   beam_case      = false;
  int    physics = 0;
  int physicsdim = 1;
  int bcNeuman   = 1;
  double lx = 1.0;
  double ly = 1.0;
  int jobid = 0;

  OptionsParser args(argc, argv);
  args.AddOption(&ref_ser, "-rs", "--refine-serial",
                 "Number of times to refine the mesh uniformly in serial.");
  args.AddOption(&metric_id, "-mid", "--metric-id",
                "Mesh optimization metric:\n\t"
                "T-metrics\n\t"
                "1  : |T|^2                          -- 2D no type\n\t"
                "2  : 0.5|T|^2/tau-1                 -- 2D shape (condition number)\n\t"
                "7  : |T-T^-t|^2                     -- 2D shape+size\n\t"
                "9  : tau*|T-T^-t|^2                 -- 2D shape+size\n\t"
                );
  args.AddOption(&target_id, "-tid", "--target-id",
                "Target (ideal element) type:\n\t"
                "1: Ideal shape, unit size\n\t"
                "2: Ideal shape, equal size\n\t"
                "3: Ideal shape, initial size\n\t"
                "4: Given full analytic Jacobian (in physical space)\n\t"
                "5: Ideal shape, given size (in physical space)");
   args.AddOption(&quad_order, "-qo", "--quad_order",
                  "Order of the quadrature rule.");
  args.AddOption(&quad_order2, "-qo2", "--quad_order2",
                  "Order of the quadrature rule for sensitivities.");
   args.AddOption(&method, "-met", "--method",
                  "0(Defaults to TMOP_MMA), 1 - MS");
   args.AddOption(&max_ch, "-ch", "--max-ch",
                  "max node movement");
   args.AddOption(&max_it, "-ni", "--newton-oter",
                  "number of iters");
   args.AddOption(&ftype, "-ft", "--ftype",
                  "function type");
   args.AddOption(&alphaw, "-alpha", "--alpha",
                  "alpha weight for functions");
   args.AddOption(&qoitype, "-qt", "--qtype",
                  "Quantity of interest type");
   args.AddOption(&weight_1, "-w1", "--weight1",
                  "Quantity of interest weight");
   args.AddOption(&weight_tmop, "-w2", "--weight2",
                  "Mesh quality weight type");
   args.AddOption(&mesh_poly_deg, "-o", "--order",
                  "Polynomial degree of mesh finite element space.");
   args.AddOption(&physics_deg, "-so", "--physics-order",
                  "Polynomial degree of solution space.");
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
    args.AddOption(&exactaction, "-ex", "--exact_action",
                  "-no-ex", "--no-exact-action",
                  "Enable exact action of TMOP_Integrator.");
   args.AddOption(&ls_norm_fac, "-lsn", "--ls-norm-fac",
                  "line-search norm factor");
   args.AddOption(&ls_energy_fac, "-lse", "--ls-energy-fac",
                  "line-search energy factor");
    args.AddOption(&bndr_fix, "-bndr", "--bndrfix",
                  "-bndrfree", "--bndr-free",
                  "Enable exact action of TMOP_Integrator.");
    args.AddOption(&visualization, "-vis", "--vis",
                  "-no-vis", "--no-vis",
                  "Enable/disable visualization.");
    args.AddOption(&weakBC, "-weakbc", "--weakbc",
                  "-no-weakbc", "--no-weakbc",
                  "Enable/disable weak boundary condition.");
    args.AddOption(&filter, "-filter", "--filter",
                  "-no-filter", "--no-filter",
                  "Use vector helmholtz filter.");
    args.AddOption(&filterRadius, "-frad", "--frad",
                    "Filter radius");
    args.AddOption(&physics, "-ph", "--physics",
                    "Physics");
    args.AddOption(&beam_case, "-beam", "--beam",
                   "-no-beam", "--no-beam",
                   "Beam with dmensions 1x0.1");
    args.AddOption(&jobid, "-jid", "--jid",
                    "suffix for saved files");

   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }

   if(beam_case)
   {
     bcNeuman = 2;
     nx = 10;
     ny = 4;
     lx = 1.0;
     ly = 0.1;
   }
   if (physics_deg < 0) { physics_deg = mesh_poly_deg; }
   if (filterRadius < 0) {
    filterRadius = -filterRadius/std::pow(2.0, ref_ser-1);
    if (myid == 0)
    {
      std::cout << "FilterRadius scaled to: " << filterRadius << std::endl;
    }
   }

  enum QoIType qoiType  = static_cast<enum QoIType>(qoitype);
  bool dQduFD =false;
  bool dQdxFD =false;
  bool dQdxFD_global =false;
  bool BreakAfterFirstIt = false;

  // Create mesh
  Mesh *des_mesh = nullptr;
  if (strcmp(mesh_file, "null.mesh") == 0)
  {
     des_mesh = new Mesh(Mesh::MakeCartesian2D(nx, ny, Element::QUADRILATERAL,
                                        true, lx, ly));
  }
  else
  {
    des_mesh = new Mesh(mesh_file, 1, 1, false);
  }

  if(perturbMesh)
  {
     int tNumVertices  = des_mesh->GetNV();
     for (int i = 0; i < tNumVertices; ++i) {
        double * Coords = des_mesh->GetVertex(i);
        //if (Coords[ 0 ] != 0.0 && Coords[ 0 ] != 1.0 && Coords[ 1 ] != 0.0 && Coords[ 1 ] != 1.0) {
          //  Coords[ 0 ] = Coords[ 0 ] + ((rand() / double(RAND_MAX)* 2.0 - 1.0)* epsilon_pert);
          //  Coords[ 1 ] = Coords[ 1 ] + ((rand() / double(RAND_MAX)* 2.0 - 1.0)* epsilon_pert);

          Coords[ 0 ] = Coords[ 0 ] +0.5;
          //Coords[ 1 ] = Coords[ 1 ] + ((rand() / double(RAND_MAX)* 2.0 - 1.0)* epsilon_pert);
        //}
     }
  }

  ConstantCoefficient firstLameCoef(0.5769230769);
  ConstantCoefficient secondLameCoef(1.0/2.6);
  ConstantCoefficient zerocoeff(0.0);

  StrainEnergyDensityCoefficient tStraincoeff(&firstLameCoef, &secondLameCoef, nullptr);
  StrainEnergyDensityCoefficient tStraincoeff_ref(&firstLameCoef, &secondLameCoef, nullptr);

  // Refine mesh in serial
  for (int lev = 0; lev < ref_ser; lev++) { des_mesh->UniformRefinement(); }

  auto PMesh = new ParMesh(MPI_COMM_WORLD, *des_mesh);

  int dim = PMesh->SpaceDimension();
  delete des_mesh;

  ParMesh *PMesh_ref = nullptr;

  if( physics ==1)
  {
    physicsdim = dim;
    PMesh_ref = new ParMesh(*PMesh);
    int nrefs = 3;
    for (int i = 0; i < nrefs; i++)
    {
      PMesh_ref->UniformRefinement();
    }
    PMesh_ref->SetCurvature(mesh_poly_deg, false, -1, 0);
  }


  // -----------------------
  // Remaining mesh settings
  // -----------------------

  // Nodes are only active for higher order meshes, and share locations with
  // the vertices, plus all the higher- order control points within  the
  // element and along the edges and on the faces.
  if (nullptr == PMesh->GetNodes())
  {
    PMesh->SetCurvature(mesh_poly_deg, false, -1, 0);
  }


  // 4. Define a finite element space on the mesh. Here we use vector finite
  //    elements which are tensor products of quadratic finite elements. The
  //    number of components in the vector finite element space is specified by
  //    the last parameter of the FiniteElementSpace constructor.
  FiniteElementCollection *fec= new H1_FECollection(mesh_poly_deg, dim);
  ParFiniteElementSpace *pfespace = new ParFiniteElementSpace(PMesh, fec, dim,
                                                               mesh_node_ordering);
  auto fespace_scalar = new ParFiniteElementSpace(PMesh, fec, 1);
  ParFiniteElementSpace pfespace_gf(PMesh, fec);
  ParGridFunction x_gf(&pfespace_gf);

  // Strain energy density
  ParFiniteElementSpace tspace(PMesh,fec,1);
  ParGridFunction tGF(&tspace);
  ParFiniteElementSpace tspace_grad(PMesh, fec, dim);
  ParGridFunction tGF_grad(&tspace_grad);

  double init_mesh_size = PMesh->GetElementSize(0, 0);

  // 5. Make the mesh curved based on the above finite element space. This
  //    means that we define the mesh elements through a fespace-based
  //    transformation of the reference element.
  PMesh->SetNodalFESpace(pfespace);

  // 7. Get the mesh nodes (vertices and other degrees of freedom in the finite
  //    element space) as a finite element grid function in fespace. Note that
  //    changing x automatically changes the shapes of the mesh elements.
  ParGridFunction x(pfespace);
  PMesh->SetNodalGridFunction(&x);
  ParGridFunction x0(pfespace);
  x0 = x;
  ParGridFunction orifield(pfespace);
  int numOptVars = pfespace->GetTrueVSize();

  // TMOP Integrator setup
     TMOP_QualityMetric *metric = NULL;
   switch (metric_id)
   {
      // T-metrics
      case 1: metric = new TMOP_Metric_001; break;
      case 2: metric = new TMOP_Metric_002; break;
      case 4: metric = new TMOP_Metric_004; break;
      case 7: metric = new TMOP_Metric_007; break;
      case 9: metric = new TMOP_Metric_009; break;
      case 36: metric = new TMOP_AMetric_036; break;
      case 49: metric = new TMOP_AMetric_049(0.01); break;
      case 501: metric = new TMOP_AMetric_050; break;
      case 50: metric = new TMOP_Metric_050; break;
      case 80: metric = new TMOP_Metric_080(0.8); break;
      case 85: metric = new TMOP_Metric_085; break;
      case 98: metric = new TMOP_Metric_098; break;
      case 107: metric = new TMOP_AMetric_107; break;
      case 130: metric = new TMOP_AMetric_OQ; break;
      case 131: metric = new TMOP_AMetric_OQ2; break;
      case 303: metric = new TMOP_Metric_303; break;
      default:
         if (myid == 0) { cout << "Unknown metric_id: " << metric_id << endl; }
         return 3;
   }

   TargetConstructor::TargetType target_t;
   TargetConstructor *target_c = NULL;
   TargetConstructor *target_c2 = NULL;
   OSCoefficient *adapt_coeff = NULL;
    VisItDataCollection *visdcori = new VisItDataCollection("orientation"+std::to_string(jobid), PMesh);
   switch (target_id)
   {
      case 1: target_t = TargetConstructor::IDEAL_SHAPE_UNIT_SIZE; break;
      case 2: target_t = TargetConstructor::IDEAL_SHAPE_EQUAL_SIZE; break;
      case 3: target_t = TargetConstructor::IDEAL_SHAPE_GIVEN_SIZE; break;
      case 4: target_t = TargetConstructor::GIVEN_SHAPE_AND_SIZE; break;
      case 5:
      {
         target_t = TargetConstructor::GIVEN_FULL;
         AnalyticAdaptTC *tc = new AnalyticAdaptTC(target_t);
         adapt_coeff = new OSCoefficient(dim, metric_id);
         tc->SetAnalyticTargetSpec(NULL, NULL, adapt_coeff);
         target_c = tc;
         VisVectorField(adapt_coeff, PMesh, &orifield);
         socketstream vis;
         common::VisualizeField(vis, "localhost", 19916, orifield,
                               "Orientation", 400, 480, 400, 400, "jRmclAevvppp]]]]]]]]]]]]]]]");
          visdcori->RegisterField("solution", &orifield);
          visdcori->SetCycle(0);
          visdcori->SetTime(0.0);
          visdcori->Save();
         break;
      }
      default:
         if (myid == 0) { cout << "Unknown target_id: " << target_id << endl; }
         return 3;
   }
   if (target_c == NULL)
   {
    target_c = new TargetConstructor(target_t, MPI_COMM_WORLD);
   }
   if (target_c2 == NULL)
   {
    target_c2 = new TargetConstructor(target_t, MPI_COMM_WORLD);
   }
   target_c->SetNodes(x0);


   IntegrationRules *irules = &IntRulesLo;
   auto tmop_integ = new TMOP_Integrator(metric, target_c);
   tmop_integ->SetIntegrationRules(*irules, quad_order);

   ConstantCoefficient metric_w(weight_tmop);
   tmop_integ->SetCoefficient(metric_w);
   tmop_integ->SetExactActionFlag(exactaction);

  // set esing variable bounds
  Vector objgrad(numOptVars); objgrad=0.0;
  Vector volgrad(numOptVars); volgrad=1.0;
  Vector xxmax(numOptVars);   xxmax=  0.001;
  Vector xxmin(numOptVars);   xxmin= -0.001;

  ParGridFunction gridfuncOptVar(pfespace);
  gridfuncOptVar = 0.0;
  ParGridFunction gridfuncLSBoundIndicator(pfespace);
  gridfuncLSBoundIndicator = 0.0;
  Array<int> vdofs;

  // Identify coordinate dofs perpendicular to BE
  if (strcmp(mesh_file, "null.mesh") == 0 && physics ==0)
  {
    for (int i = 0; i < PMesh->GetNBE(); i++)
    {
      Element * tEle = PMesh->GetBdrElement(i);
      int attribute = tEle->GetAttribute();
      pfespace->GetBdrElementVDofs(i, vdofs);
      const int nd = pfespace->GetBE(i)->GetDof();

      if (attribute == 1 || attribute == 3) // zero out motion in y
      {
        for (int j = 0; j < nd; j++)
        {
          gridfuncLSBoundIndicator[ vdofs[j+nd] ] = 1.0;
          // gridfuncLSBoundIndicator[ vdofs[j] ] = 1.0;
        }
      }
      else if (attribute == 2 || attribute == 4) // zero out in x
      {
        for (int j = 0; j < nd; j++)
        {
          gridfuncLSBoundIndicator[ vdofs[j] ] = 1.0;
          // gridfuncLSBoundIndicator[ vdofs[j+nd] ] = 1.0;
        }
      }
    }
  }
  else if(physics ==1)
  {
    for (int i = 0; i < PMesh->GetNBE(); i++)
    {
      Element * tEle = PMesh->GetBdrElement(i);
      int attribute = tEle->GetAttribute();
      pfespace->GetBdrElementVDofs(i, vdofs);
      const int nd = pfespace->GetBE(i)->GetDof();

      if(strcmp(mesh_file, "null.mesh") == 0)
      {
        if (attribute == 1 ||
            attribute == 3 ||
            attribute == 5 ) // zero out motion in y
        {
          for (int j = 0; j < nd; j++)
          {
           gridfuncLSBoundIndicator[ vdofs[j+nd] ] = 1.0;
          }
        }
        else if (attribute == 2 ||
            attribute == 4 ||
            attribute == 6 ) // zero out in x
        {
          for (int j = 0; j < nd; j++)
          {
            gridfuncLSBoundIndicator[ vdofs[j] ] = 1.0;
          }
        }
      }
      else
      {
        if (attribute == 1 ||
            attribute == 3 ) // zero out motion in y
        {
          for (int j = 0; j < nd; j++)
          {
            gridfuncLSBoundIndicator[ vdofs[j+nd] ] = 1.0;
          }
        }
        else if (attribute == 2 ||
            attribute == 4 ) // zero out in x
        {
          for (int j = 0; j < nd; j++)
          {
            gridfuncLSBoundIndicator[ vdofs[j] ] = 1.0;
          }
        }
      }
    }
  }
  else
  {
    for (int i = 0; i < PMesh->GetNBE(); i++)
    {
      Element * tEle = PMesh->GetBdrElement(i);
      int attribute = tEle->GetAttribute();
      pfespace->GetBdrElementVDofs(i, vdofs);
      const int nd = pfespace->GetBE(i)->GetDof();

      if (attribute == 2) // zero out motion in y
      {
        for (int j = 0; j < nd; j++)
        {
          gridfuncLSBoundIndicator[ vdofs[j+nd] ] = 1.0;
          if (bndr_fix) {
            gridfuncLSBoundIndicator[ vdofs[j+0*nd] ] = 1.0; // stops all motion
          }
        }
      }
      else if (attribute == 1) // zero out in x
      {
        for (int j = 0; j < nd; j++)
        {
          gridfuncLSBoundIndicator[ vdofs[j] ] = 1.0;
          if (bndr_fix)
          {
            gridfuncLSBoundIndicator[ vdofs[j+nd] ] = 1.0; // stops all motion
          }
        }
      }
      else if (dim == 3 && attribute == 3) // zero out in z
      {
        for (int j = 0; j < nd; j++)
        {
          gridfuncLSBoundIndicator[ vdofs[j+2*nd] ] = 1.0;
        }
      }
    }
  }

  gridfuncOptVar.SetTrueVector();
  gridfuncLSBoundIndicator.SetTrueVector();

  Vector & trueOptvar = gridfuncOptVar.GetTrueVector();

  const int nbattr = PMesh->bdr_attributes.Max();
  std::vector<std::pair<int, double>> essentialBC(nbattr);
  std::vector<std::pair<int, int>> essentialBCfilter(nbattr);
  if( physics ==1 )
  {
    essentialBC.resize(1);
    essentialBC[0] = {4, 0};
  }
  else
  {
    for (int i = 0; i < nbattr; i++)
    {
      essentialBC[i] = {i+1, 0};
    }
  }

  if (strcmp(mesh_file, "null.mesh") == 0)
  {
    essentialBCfilter[0] = {1, 1};
    essentialBCfilter[1] = {2, 0};
    essentialBCfilter[2] = {3, 1};
    essentialBCfilter[3] = {4, 0};
  }
  else if( physics ==1 )
  {
    essentialBCfilter[0] = {1, 1};
    essentialBCfilter[1] = {2, 0};
    essentialBCfilter[2] = {3, 1};
    essentialBCfilter[3] = {4, 0};
    essentialBCfilter[4] = {5, 1};
    essentialBCfilter[5] = {6, 0};
  }
  else
  {
    essentialBCfilter[0] = {1, 0};
    essentialBCfilter[1] = {2, 1};
    if (dim == 3)
    {
      essentialBCfilter[2] = {3, 2};
    }
  }

  const IntegrationRule &ir =
      irules->Get(pfespace->GetFE(0)->GetGeomType(), quad_order);
  MMAOpt* mma = nullptr;
#ifdef MFEM_USE_PETSC
  NativeMMA* mmaPetsc = nullptr;
#endif
    // NativeMMA* mma = nullptr;
  TMOP_MMA *tmma = new TMOP_MMA(MPI_COMM_WORLD, trueOptvar.Size(), 0,
                                 trueOptvar, ir);
  {
#ifdef MFEM_USE_PETSC
    double a=0.0;
    double c=1000.0;
    double d=0.0;
    mmaPetsc=new NativeMMA(MPI_COMM_WORLD,1, objgrad,&a,&c,&d);
#else
    mma=new MMAOpt(MPI_COMM_WORLD, trueOptvar.Size(), 0, trueOptvar);
#endif
  }

if (myid == 0) {
  switch (qoiType) {
  case 0:
    std::cout<<" L2 Error"<<std::endl;
    break;
  case 1:
    std::cout<<" H1 semi-norm Error"<<std::endl;
    break;
  case 2:
    std::cout<<" ZZ Error"<<std::endl;
    break;
  case 3:
    std::cout<<" Avg Error"<<std::endl;;
    break;
  case 4:
    std::cout<<" Energy"<<std::endl;;
    break;
  case 5:
    std::cout<<" Global ZZ"<<std::endl;
    break;
  case 7:
    std::cout<<" Struct Compliance"<<std::endl;;
    break;
  default:
    std::cout << "Unknown Error Coeff: " << qoiType << std::endl;
  }
}

  Array<int> neumannBdr(PMesh->bdr_attributes.Max());
  neumannBdr = 0; neumannBdr[bcNeuman] = 1;

  VectorCoefficient *loadFuncGrad = new VectorFunctionCoefficient(dim,
                                                              trueLoadFuncGrad);
  VectorCoefficient *trueSolutionGrad =
                          new VectorFunctionCoefficient(dim, trueSolGradFunc);
  MatrixCoefficient *trueSolutionHess = new
                                MatrixFunctionCoefficient(dim,trueHessianFunc);
  VectorCoefficient *trueSolutionHessV =
                          new VectorFunctionCoefficient(dim*dim, trueHessianFunc_v);
  PhysicsSolverBase * solver = nullptr;
  PhysicsSolverBase * solver_ref = nullptr;
  PhysicsSolverBase * solver_strong = nullptr;
  QuantityOfInterest QoIEvaluator(PMesh, qoiType, mesh_poly_deg, physics_deg,neumannBdr, physicsdim);
  NodeAwareTMOPQuality MeshQualityEvaluator(PMesh, mesh_poly_deg, metric, target_c2);
  Coefficient *trueSolution = new FunctionCoefficient(trueSolFunc);
  Coefficient *QCoef = new FunctionCoefficient(loadFunc);


  VectorArrayCoefficient tractionLoad(PMesh->SpaceDimension());
  tractionLoad.Set(1, QCoef);
  tractionLoad.Set(0, new ConstantCoefficient(0.0));

  if( physics ==0)
  {
    Diffusion_Solver * diffsolver = new Diffusion_Solver(PMesh, essentialBC, mesh_poly_deg, physics_deg, trueSolution, weakBC, loadFuncGrad);
    diffsolver->SetManufacturedSolution(QCoef);
    diffsolver->setTrueSolGradCoeff(trueSolutionGrad);

    Diffusion_Solver * diffsolver_strong = new Diffusion_Solver(PMesh, essentialBC, mesh_poly_deg, physics_deg, trueSolution, false, loadFuncGrad);
    diffsolver_strong->SetManufacturedSolution(QCoef);
    diffsolver_strong->setTrueSolGradCoeff(trueSolutionGrad);

    solver = diffsolver;
    solver_strong = diffsolver_strong;

    QoIEvaluator.setTrueSolCoeff( trueSolution );
    if(qoiType == QoIType::ENERGY){QoIEvaluator.setTrueSolCoeff( QCoef );}
    QoIEvaluator.setTrueSolGradCoeff(trueSolutionGrad);
    QoIEvaluator.setTrueSolHessCoeff(trueSolutionHess);
    QoIEvaluator.setTrueSolHessCoeff(trueSolutionHessV);
    QoIEvaluator.SetManufacturedSolution(QCoef);
    QoIEvaluator.SetManufacturedSolutionGrad(loadFuncGrad);
  }
  else if(physics ==1)
  {
    //Elasticity_Solver * elasticitysolver = new Elasticity_Solver(PMesh, dirichletBC, tractionBC, mesh_poly_deg);
    Elasticity_Solver * elasticitysolver = new Elasticity_Solver(PMesh, essentialBC, neumannBdr, mesh_poly_deg);
    elasticitysolver->SetLoad(&tractionLoad);

    solver = elasticitysolver;

    QoIEvaluator.setTractionCoeff(&tractionLoad);

    Elasticity_Solver * elasticitysolver_ref = new Elasticity_Solver(PMesh_ref, essentialBC, neumannBdr, mesh_poly_deg);
    elasticitysolver_ref->SetLoad(&tractionLoad);

    solver_ref = elasticitysolver_ref;
  }

  //std::vector<std::pair<int, double>> essentialBC_filter(0);
  FunctionCoefficient leftrightwalls(&tanh_left_right_walls);
  ProductCoefficient filterRadiusCoeff(filterRadius, leftrightwalls);
  VectorHelmholtz *filterSolver;
  if (metric_id == 107)
  {
    // filterSolver = new VectorHelmholtz(PMesh, essentialBCfilter, &filterRadiusCoeff, mesh_poly_deg);
    filterSolver = new VectorHelmholtz(PMesh, essentialBCfilter, filterRadius, mesh_poly_deg, mesh_poly_deg);
  }
  else
  {
    filterSolver = new VectorHelmholtz(PMesh, essentialBCfilter, filterRadius, mesh_poly_deg, mesh_poly_deg);
  }

  QoIEvaluator.SetIntegrationRules(&IntRulesLo, quad_order2);
  x_gf.ProjectCoefficient(*trueSolution);

  Diffusion_Solver solver_FD1(PMesh, essentialBC, mesh_poly_deg,  physics_deg, trueSolution, weakBC);
  Diffusion_Solver solver_FD2(PMesh, essentialBC, mesh_poly_deg,  physics_deg, trueSolution, weakBC);
  solver_FD1.SetManufacturedSolution(QCoef);
  solver_FD2.SetManufacturedSolution(QCoef);

  QuantityOfInterest QoIEvaluator_FD1(PMesh, qoiType, 1, physics_deg);
  QuantityOfInterest QoIEvaluator_FD2(PMesh, qoiType, 1, physics_deg);

  ParaViewDataCollection paraview_dc("MeshOptimizer", PMesh);
  paraview_dc.SetLevelsOfDetail(1);
  paraview_dc.SetDataFormat(VTKFormat::BINARY);
  paraview_dc.SetHighOrderOutput(true);

  //
  ParGridFunction & discreteSol = solver->GetSolution();
  Vector maxdisp_ref, maxdisp_init, maxdisp_opt;
  discreteSol.ProjectCoefficient(*trueSolution);
  if (visualization)
  {
      socketstream vis;
      common::VisualizeField(vis, "localhost", 19916, discreteSol,
                            "Initial Projected Solution", 0, 0, 400, 400, "jRmclAppppppppppppp]]]]]]]]]]]]]]]");
  }
  {
    solver->SetDesignVarFromUpdatedLocations(x);
    solver->FSolve();
    discreteSol = solver->GetSolution();
    if (physics == 1)
    {
      GetMaxDisplacement(&discreteSol, maxdisp_init);
    }
    if (visualization)
    {
        socketstream vis;
        common::VisualizeField(vis, "localhost", 19916, discreteSol,
                              "Initial Solver Solution", 0, 480, 400, 400, "jRmclAppppppppppppp]]]]]]]]]]]]]]]");
    }
  }
  double strain_energy_ref = 0.0;
  if (physics == 1)
  {
    GridFunction *x_ref = PMesh_ref->GetNodes();
    solver_ref->SetDesignVarFromUpdatedLocations(*x_ref);
    solver_ref->FSolve();
    ParGridFunction &discreteSol_ref = solver_ref->GetSolution();
    ParFiniteElementSpace tspace_ref(PMesh_ref, x_ref->FESpace()->FEColl(), 1);
    ParGridFunction tGF_ref(&tspace_ref);
    ParFiniteElementSpace tspace_grad_ref(PMesh_ref, x_ref->FESpace()->FEColl(), dim);
    ParGridFunction tGF_grad_ref(&tspace_grad_ref);

    if (visualization)
    {
        socketstream vis;
        common::VisualizeField(vis, "localhost", 19916, discreteSol_ref,
                              "Initial Solver Solution", 0, 960, 400, 400, "jRmclAppppppppppppp]]]]]]]]]]]]]]]");
    }
    GetMaxDisplacement(&discreteSol_ref, maxdisp_ref);

    tStraincoeff_ref.SetU(&discreteSol_ref);
    tGF_ref.ProjectCoefficient(tStraincoeff_ref);
    strain_energy_ref = tGF_ref.ComputeIntegral();
    GetScalarDerivative(&tGF_ref, &tGF_grad_ref, dim);

    VisItDataCollection *visdc = new VisItDataCollection("tmop-pde-ref"+std::to_string(jobid), PMesh_ref);
    visdc->RegisterField("solution", &(solver_ref->GetSolution()));
    visdc->RegisterField("strain-energy-density", &tGF_ref);
    visdc->RegisterField("strain-energy-density-grad", &tGF_grad_ref);
    visdc->SetCycle(0);
    visdc->SetTime(0.0);
    visdc->Save();
    // MFEM_ABORT(" ");
  }

  x.SetTrueVector();
  double init_strain_energy = 0.0;

  if (physics == 1)
  {
    tStraincoeff.SetU(&(solver->GetSolution()));
    tGF.ProjectCoefficient(tStraincoeff	);
    GetScalarDerivative(&tGF, &tGF_grad, dim);
    init_strain_energy = tGF.ComputeIntegral();
  }
  VisItDataCollection *visdc = new VisItDataCollection("tmop-pde"+std::to_string(jobid), PMesh);
  visdc->RegisterField("solution", &(solver->GetSolution()));
  if (physics == 1)
  {
    visdc->RegisterField("strain-energy-density", &tGF);
    visdc->RegisterField("strain-energy-density-grad", &tGF_grad);
  }
  visdc->SetCycle(0);
  visdc->SetTime(0.0);
  visdc->Save();
  int save_freq = 10;


  if (method == 0)
  {
    auto init_l2_error = discreteSol.ComputeL2Error(*trueSolution);
    auto init_grad_error = discreteSol.ComputeGradError(trueSolutionGrad);
    auto init_h1_error = discreteSol.ComputeH1Error(trueSolution, trueSolutionGrad);
    {
      solver_strong->SetDesignVarFromUpdatedLocations(x);
      solver_strong->FSolve();
      ParGridFunction &discreteSol_strong = solver_strong->GetSolution();
      init_l2_error = discreteSol_strong.ComputeL2Error(*trueSolution);
      init_grad_error = discreteSol_strong.ComputeGradError(trueSolutionGrad);
      init_h1_error = discreteSol_strong.ComputeH1Error(trueSolution, trueSolutionGrad);
    }

    ParNonlinearForm a(pfespace);
    a.AddDomainIntegrator(tmop_integ);
    {
      Array<int> ess_bdr(PMesh->bdr_attributes.Max());
      ess_bdr = 1;
      //a.SetEssentialBC(ess_bdr);
    }
    IterativeSolver::PrintLevel newton_print;
    newton_print.Errors().Warnings().Iterations();
    // set the TMOP Integrator
    tmma->SetOperator(a);
    // Set change limits on dx
    tmma->SetUpperBound(max_ch);
    tmma->SetLowerBound(max_ch);
    // Set true vector so that it can be zeroed out
    {
      Vector & trueBounds = gridfuncLSBoundIndicator.GetTrueVector();
      tmma->SetTrueDofs(trueBounds);
    }
    // Set QoI and Solver and weight
    if (weight_1 != 0.0)
    {
      tmma->SetQuantityOfInterest(&QoIEvaluator);
      tmma->SetDiffusionSolver(reinterpret_cast<Diffusion_Solver*>(solver));       // TODO change to base class
      tmma->SetQoIWeight(weight_1);
      tmma->SetVectorHelmholtzFilter(filterSolver);
    }

    // Set min jac
    tmma->SetMinimumDeterminantThreshold(1e-7);

    // Set line search factors
    tmma->SetLineSearchNormFactor(ls_norm_fac);
    tmma->SetLineSearchEnergyFactor(ls_energy_fac);

    tmma->SetPrintLevel(newton_print);

    const real_t init_energy = tmma->GetEnergy(x.GetTrueVector(), true);
    const real_t init_metric_energy = tmma->GetEnergy(x.GetTrueVector(), false);
    const real_t init_qoi_energy = init_energy - init_metric_energy;

    // Set max # iterations
    bool save_after_every_iteration = true;
    if (save_after_every_iteration)
    {
      tmma->SetDataCollectionObjectandMesh(visdc, PMesh, save_freq);
    }
    tmma->SetMaxIter(max_it);
    if (filter)
    {
      tmma->MultFilter(x.GetTrueVector());
    }
    else
    {
      tmma->Mult(x.GetTrueVector());
    }
    x.SetFromTrueVector();
    if (!save_after_every_iteration)
    {
      if (physics == 1)
      {
        tStraincoeff.SetU(&(solver->GetSolution()));
        tGF.ProjectCoefficient(tStraincoeff	);
        GetScalarDerivative(&tGF, &tGF_grad, dim);
      }
      visdc->SetCycle(1);
      visdc->SetTime(1.0);
      visdc->Save();
    }


    // Visualize the mesh displacement.
    if (visualization)
    {
      x0 -= x;
      socketstream vis;
      common::VisualizeField(vis, "localhost", 19916, x0,
                              "Displacements", 800, 000, 400, 400, "jRmclAppppppppppppp]]]]]]]]]]]]]]]");

      ParaViewDataCollection paraview_dc("NativeMeshOptimizer", PMesh);
      paraview_dc.SetLevelsOfDetail(1);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetCycle(0);
      paraview_dc.SetTime(1.0);
      //paraview_dc.RegisterField("Solution",&x_gf);
      paraview_dc.Save();
    }

    {
      ostringstream mesh_name;
      mesh_name << "optimized.mesh";
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      PMesh->PrintAsSerial(mesh_ofs);
    }


    solver->SetDesignVarFromUpdatedLocations(x);
    solver->FSolve();
    discreteSol = solver->GetSolution();
    if (visualization)
    {
        socketstream vis;
        common::VisualizeField(vis, "localhost", 19916, discreteSol,
                              "Final Solver Solution", 400, 480, 400, 400, "jRmclAppppppppppppp]]]]]]]]]]]]]]]");
    }

    auto final_l2_error = discreteSol.ComputeL2Error(*trueSolution);
    auto final_grad_error = discreteSol.ComputeGradError(trueSolutionGrad);
    auto final_h1_error = discreteSol.ComputeH1Error(trueSolution, trueSolutionGrad);

    const real_t final_energy = tmma->GetEnergy(x.GetTrueVector(), true);
    const real_t final_metric_energy = tmma->GetEnergy(x.GetTrueVector(), false);
    const real_t final_qoi_energy = final_energy - final_metric_energy;

    discreteSol.ProjectCoefficient(*trueSolution);
    if (visualization)
    {
        socketstream vis;
        common::VisualizeField(vis, "localhost", 19916, discreteSol,
                              "Final Projected Solution", 400, 000, 400, 400, "jRmclAppppppppppppp]]]]]]]]]]]]]]]");
    }
    if (myid == 0)
    {
      std::cout << "Initial L2 error: " << " " << init_l2_error << " " << std::endl;
      std::cout << "Final   L2 error: " << " " << final_l2_error << " " << std::endl;

      std::cout << "Initial Grad error: " << " " << init_grad_error << " " << std::endl;
      std::cout << "Final   Grad error: " << " " << final_grad_error << " " << std::endl;

      std::cout << "Initial H1 error: " << " " << init_h1_error << " " << std::endl;
      std::cout << "Final   H1 error: " << " " << final_h1_error << " " << std::endl;

      std::cout << "Initial Total/Metric/QOI Energy: " << init_energy << " " << init_metric_energy << " " << init_qoi_energy << std::endl;
      std::cout << "Final   Total/Metric/QOI Energy: " << final_energy << " " << final_metric_energy << " " << final_qoi_energy << std::endl;
      std::cout << "Initial metric/qoi energy (unscaled): " << init_metric_energy/weight_tmop << " " <<
      init_qoi_energy/weight_1 << " " << std::endl;
      std::cout << "Final metric/qoi energy (unscaled): " << final_metric_energy/weight_tmop << " " <<
      final_qoi_energy/weight_1 << " " << std::endl;
    }

    int ne_glob = PMesh->GetGlobalNE();
    real_t min_l2_solver, min_grad_solver;
    int min_err_iter;
    tmma->GetMinErrInfo(min_l2_solver, min_grad_solver, min_err_iter);
    if (myid == 0)
    {
      std::cout << "k10info: " << qoitype << " "  << ftype << " " <<
                              mesh_poly_deg << " " <<
                              init_mesh_size << " " << ne_glob << " " <<
                              init_l2_error << " " << init_grad_error << " " << final_l2_error << " " << final_grad_error << " " << min_l2_solver << " " << min_grad_solver <<
                              std::endl;
    }

    if (visualization && adapt_coeff)
    {

         VisVectorField(adapt_coeff, PMesh, &orifield);
        socketstream vis;
        common::VisualizeField(vis, "localhost", 19916, orifield,
                              "Orientation", 800, 480, 400, 400, "jRmclAevvppp]]]]]]]]]]]]]]]");
          visdcori->SetCycle(1);
          visdcori->SetTime(1.0);
          visdcori->Save();
    }
  }
  else
  {
    int cycle_count = 1;
    double final_strain_energy = 0.0;
    for(int i=1;i<max_it;i++)
    {
      filterSolver->setLoadGridFunction(gridfuncOptVar);
      filterSolver->FSolve();
      ParGridFunction & filteredDesign = filterSolver->GetSolution();

      solver->SetDesign( filteredDesign );
      solver->FSolve();

      ParGridFunction & discreteSol = solver->GetSolution();

      QoIEvaluator.SetDesign( filteredDesign );
      MeshQualityEvaluator.SetDesign( filteredDesign );

      QoIEvaluator.SetDiscreteSol( discreteSol );
      QoIEvaluator.SetIntegrationRules(&IntRulesLo, quad_order);

      double ObjVal = QoIEvaluator.EvalQoI();
      double meshQualityVal = MeshQualityEvaluator.EvalQoI();

      double val = weight_1 * ObjVal+ weight_tmop * meshQualityVal;

      QoIEvaluator.EvalQoIGrad();
      MeshQualityEvaluator.EvalQoIGrad();

      ParLinearForm * dQdu = QoIEvaluator.GetDQDu();
      ParLinearForm * dQdxExpl = QoIEvaluator.GetDQDx();
      ParLinearForm * dMeshQdxExpl = MeshQualityEvaluator.GetDQDx();

      solver->ASolve( *dQdu );

      ParLinearForm * dQdxImpl = solver->GetImplicitDqDx();

      ParLinearForm dQdx(pfespace); dQdx = 0.0;
      ParLinearForm dQdx_physics(pfespace); dQdx_physics = 0.0;
      ParLinearForm dQdx_filtered(pfespace); dQdx_filtered = 0.0;
      dQdx_physics.Add(1.0, *dQdxExpl);
      dQdx_physics.Add(1.0, *dQdxImpl);

      dQdx_filtered.Add(weight_1, *dQdxExpl);
      dQdx_filtered.Add(weight_1, *dQdxImpl);
      dQdx_filtered.Add(weight_tmop, *dMeshQdxExpl);

      HypreParVector *truedQdx_physics = dQdx_physics.ParallelAssemble();
      ParGridFunction dQdx_physicsGF(pfespace, truedQdx_physics);

      //std::cout << dQdx_filtered.Norml2() << " k101-filt1\n";
      filterSolver->ASolve(dQdx_filtered);
      ParLinearForm * dQdxImplfilter = filterSolver->GetImplicitDqDx();

      dQdx.Add(1.0, *dQdxImplfilter);
      //std::cout << dQdxImplfilter->Norml2() << " k101-filt2\n";

      HypreParVector *truedQdx = dQdx.ParallelAssemble();

      HypreParVector *truedQdx_Expl = dQdxExpl->ParallelAssemble();
      HypreParVector *truedQdx_Impl = dQdxImpl->ParallelAssemble();

      // Construct grid function from hypre vector
      ParGridFunction dQdx_ExplGF(pfespace, truedQdx_Expl);
      ParGridFunction dQdx_ImplGF(pfespace, truedQdx_Impl);


      objgrad = *truedQdx;

      //----------------------------------------------------------------------------------------------------------

      if(dQduFD)
      {
        double epsilon = 1e-8;
        ParGridFunction tFD_sens(fespace_scalar); tFD_sens = 0.0;
        for( int Ia = 0; Ia<discreteSol.Size(); Ia++)
        {
          if (myid == 0)
          {
            std::cout<<"iter: "<< Ia<< " out of: "<<discreteSol.Size() <<std::endl;
          }
          discreteSol[Ia] +=epsilon;

          // QoIEvaluator_FD1(PMesh, qoiType, 1);
          QoIEvaluator_FD1.setTrueSolCoeff(  trueSolution );
          if(qoiType == QoIType::ENERGY){QoIEvaluator_FD1.setTrueSolCoeff( QCoef );}
          QoIEvaluator_FD1.setTrueSolGradCoeff(trueSolutionGrad);
          QoIEvaluator_FD1.SetDesign( gridfuncOptVar );
          QoIEvaluator_FD1.SetDiscreteSol( discreteSol );
          QoIEvaluator_FD1.SetNodes(x0);
          QoIEvaluator_FD1.SetIntegrationRules(&IntRulesLo, quad_order);

          double ObjVal_FD1 = QoIEvaluator_FD1.EvalQoI();

          discreteSol[Ia] -=2.0*epsilon;

          QoIEvaluator_FD2.setTrueSolCoeff(  trueSolution );
          if(qoiType == QoIType::ENERGY){QoIEvaluator_FD2.setTrueSolCoeff( QCoef );}
          QoIEvaluator_FD2.setTrueSolGradCoeff(trueSolutionGrad);
          QoIEvaluator_FD2.SetDesign( gridfuncOptVar );
          QoIEvaluator_FD2.SetDiscreteSol( discreteSol );
          QoIEvaluator_FD2.SetNodes(x0);
          QoIEvaluator_FD2.SetIntegrationRules(&IntRulesLo, quad_order);

          double ObjVal_FD2 = QoIEvaluator_FD2.EvalQoI();

          discreteSol[Ia] +=epsilon;

          tFD_sens[Ia] = (ObjVal_FD1-ObjVal_FD2)/(2.0*epsilon);
        }
        dQdu->Print();
        std::cout<<"  ----------  FD Diff ------------"<<std::endl;
        tFD_sens.Print();

        std::cout<<"  ---------- dQdu Analytic - FD Diff ------------"<<std::endl;
        ParGridFunction tFD_diff(fespace_scalar); tFD_diff = 0.0;
        tFD_diff = *dQdu;
        tFD_diff -=tFD_sens;
        //tFD_diff.Print();
        std::cout<<"norm: "<<tFD_diff.Norml2()<<std::endl;
      }

      if(dQdxFD)
      {
        // nodes are p
        // det(J) is order d*p-1
        double epsilon = 1e-8;
        ParGridFunction tFD_sens(pfespace); tFD_sens = 0.0;
        Array<double> GLLVec;
        int nqpts;
        {
          const IntegrationRule *ir = &IntRulesLo.Get(Geometry::SQUARE, 8);
          nqpts = ir->GetNPoints();
          // std::cout << nqpts << " k10c\n";
          for (int e = 0; e < PMesh->GetNE(); e++)
          {
            ElementTransformation *T = PMesh->GetElementTransformation(e);
            for (int q = 0; q < ir->GetNPoints(); q++)
            {
              const IntegrationPoint &ip = ir->IntPoint(q);
              T->SetIntPoint(&ip);
              double disc_val = discreteSol.GetValue(e, ip);
              double exact_val = trueSolution->Eval( *T, ip );
              GLLVec.Append(disc_val-exact_val);
            }
          }
        }
        std::cout << nqpts << " " << GLLVec.Size() << " k10c\n";
        // MFEM_ABORT(" ");

        for( int Ia = 0; Ia<gridfuncOptVar.Size(); Ia++)
        {
          if(gridfuncLSBoundIndicator[Ia] == 1.0)
          {
            (*dQdxExpl)[Ia] = 0.0;

            continue;
          }

          std::cout<<"iter: "<< Ia<< " out of: "<<gridfuncOptVar.Size() <<std::endl;
          double fac = 1.0-gridfuncLSBoundIndicator[Ia];
          gridfuncOptVar[Ia] +=(fac)*epsilon;

          QuantityOfInterest QoIEvaluator_FD1(PMesh, qoiType, 1,physics_deg);
          QoIEvaluator_FD1.setTrueSolCoeff(  trueSolution );
          if(qoiType == QoIType::ENERGY){QoIEvaluator_FD1.setTrueSolCoeff( QCoef );}
          QoIEvaluator_FD1.setTrueSolGradCoeff(trueSolutionGrad);
          QoIEvaluator_FD1.SetDesign( gridfuncOptVar );
          QoIEvaluator_FD1.SetDiscreteSol( discreteSol );
          QoIEvaluator_FD1.SetNodes(x0);
          QoIEvaluator_FD1.SetGLLVec(GLLVec);
          QoIEvaluator_FD1.SetNqptsPerEl(nqpts);
          QoIEvaluator_FD1.SetIntegrationRules(&IntRulesLo, quad_order);

          double ObjVal_FD1 = QoIEvaluator_FD1.EvalQoI();

          gridfuncOptVar[Ia] -=(fac)*2.0*epsilon;

          QuantityOfInterest QoIEvaluator_FD2(PMesh, qoiType, 1,physics_deg);
          QoIEvaluator_FD2.setTrueSolCoeff(  trueSolution );
          // QoIEvaluator_FD2.setTrueSolCoeff(  &zerocoeff );
          if(qoiType == QoIType::ENERGY){QoIEvaluator_FD2.setTrueSolCoeff( QCoef );}
          QoIEvaluator_FD2.setTrueSolGradCoeff(trueSolutionGrad);
          QoIEvaluator_FD2.SetDesign( gridfuncOptVar );
          QoIEvaluator_FD2.SetDiscreteSol( discreteSol );
          QoIEvaluator_FD2.SetNodes(x0);
          QoIEvaluator_FD2.SetGLLVec(GLLVec);
          QoIEvaluator_FD2.SetNqptsPerEl(nqpts);
          QoIEvaluator_FD2.SetIntegrationRules(&IntRulesLo, quad_order);

          double ObjVal_FD2 = QoIEvaluator_FD2.EvalQoI();

          gridfuncOptVar[Ia] +=(fac)*epsilon;

          tFD_sens[Ia] = (ObjVal_FD1-ObjVal_FD2)/(2.0*epsilon);
        }

        dQdxExpl->Print();
        std::cout<<"  ----------  FD Diff ------------"<<std::endl;
        tFD_sens.Print();

        std::cout<<"  ---------- dQdx Analytic - FD Diff ------------"<<std::endl;
        ParGridFunction tFD_diff(pfespace); tFD_diff = 0.0;
        tFD_diff = *dQdxExpl;
        tFD_diff -=tFD_sens;
        tFD_diff.Print();
        std::cout<<"norm: "<<tFD_diff.Norml2()<<std::endl;
        for( int Ia = 0; Ia<gridfuncOptVar.Size(); Ia++)
        {
          tFD_diff[Ia] *= (1.0-gridfuncLSBoundIndicator[Ia]);
        }
        // tFD_diff.Print();
        std::cout<<"norm: "<<tFD_diff.Norml2()<<std::endl;
      }

      if(dQdxFD_global)
      {
        double epsilon = 1e-8;
        ParGridFunction tFD_sens(pfespace); tFD_sens = 0.0;
        for( int Ia = 0; Ia<gridfuncOptVar.Size(); Ia++)
        {
          if(gridfuncLSBoundIndicator[Ia] == 1.0)
          {
            dQdx_physics[Ia] = 0.0;
            dQdx[Ia] = 0.0;

            continue;
          }
          std::cout<<"iter: "<< Ia<< " out of: "<<gridfuncOptVar.Size() <<std::endl;
          double fac = 1.0-gridfuncLSBoundIndicator[Ia];
          gridfuncOptVar[Ia] +=fac*epsilon;

          solver_FD1.SetDesign( gridfuncOptVar );
          solver_FD1.FSolve();
          ParGridFunction & discreteSol_1 = solver_FD1.GetSolution();

          QoIEvaluator_FD1.setTrueSolCoeff(  trueSolution );
          if(qoiType == QoIType::ENERGY){QoIEvaluator_FD1.setTrueSolCoeff( QCoef );}
          QoIEvaluator_FD1.setTrueSolGradCoeff(trueSolutionGrad);
          QoIEvaluator_FD1.SetDesign( gridfuncOptVar );
          QoIEvaluator_FD1.SetDiscreteSol( discreteSol_1 );
          QoIEvaluator_FD1.SetNodes(x0);
          QoIEvaluator_FD1.SetIntegrationRules(&IntRulesLo, quad_order);

          double ObjVal_FD1 = QoIEvaluator_FD1.EvalQoI();

          gridfuncOptVar[Ia] -=fac*2.0*epsilon;

          solver_FD2.SetDesign( gridfuncOptVar );
          solver_FD2.FSolve();
          ParGridFunction & discreteSol_2 = solver_FD2.GetSolution();

          QoIEvaluator_FD2.setTrueSolCoeff(  trueSolution );
          if(qoiType == QoIType::ENERGY){QoIEvaluator_FD2.setTrueSolCoeff( QCoef );}
          QoIEvaluator_FD2.setTrueSolGradCoeff(trueSolutionGrad);
          QoIEvaluator_FD2.SetDesign( gridfuncOptVar );
          QoIEvaluator_FD2.SetDiscreteSol( discreteSol_2 );
          QoIEvaluator_FD2.SetNodes(x0);
          QoIEvaluator_FD2.SetIntegrationRules(&IntRulesLo, quad_order);

          double ObjVal_FD2 = QoIEvaluator_FD2.EvalQoI();

          gridfuncOptVar[Ia] +=fac*epsilon;

          tFD_sens[Ia] = (ObjVal_FD1-ObjVal_FD2)/(2.0*epsilon);
        }

        dQdx.Print();
        std::cout<<"  ----------  FD Diff - Global ------------"<<std::endl;
        tFD_sens.Print();

        std::cout<<"  ---------- dQdx Analytic - FD Diff ------------"<<std::endl;
        ParGridFunction tFD_diff(pfespace); tFD_diff = 0.0;
        tFD_diff = dQdx;
        tFD_diff -=tFD_sens;
        tFD_diff.Print();
        std::cout<<"norm: "<<tFD_diff.Norml2()<<std::endl;
        for( int Ia = 0; Ia<gridfuncOptVar.Size(); Ia++)
        {
          tFD_diff[Ia] *= (1.0-gridfuncLSBoundIndicator[Ia]);
        }
        // tFD_diff.Print();
        std::cout<<"norm: "<<tFD_diff.Norml2()<<std::endl;

        paraview_dc.SetCycle(i);
        paraview_dc.SetTime(i*1.0);
        //paraview_dc.RegisterField("ObjGrad",&objGradGF);
        paraview_dc.RegisterField("Solution",&x_gf);
        paraview_dc.RegisterField("SolutionD",&discreteSol   );
        paraview_dc.RegisterField("Sensitivity",&dQdx_physicsGF);
        paraview_dc.RegisterField("SensitivityFD",&tFD_sens);
        paraview_dc.RegisterField("SensitivityDiff",&tFD_diff);
        paraview_dc.RegisterField("SensitivityExpl",&dQdx_ExplGF);
        paraview_dc.RegisterField("SensitivityImpl",&dQdx_ImplGF);
        paraview_dc.Save();

        std::cout<<"expl: "<<dQdxExpl->Norml2()<<std::endl;
        std::cout<<"impl: "<<dQdxImpl->Norml2()<<std::endl;
      }

      if( BreakAfterFirstIt )
      {
        mfem_error("break before update");
      }

      //----------------------------------------------------------------------------------------------------------
      gridfuncOptVar.SetTrueVector();
      Vector & trueBounds = gridfuncLSBoundIndicator.GetTrueVector();

      // impose desing variable bounds - set xxmin and xxmax
      xxmin=trueOptvar; xxmin-=max_ch;
      xxmax=trueOptvar; xxmax+=max_ch;
      for(int li=0;li<xxmin.Size();li++){
        if( trueBounds[li] ==1.0)
        {
          xxmin[li] = -1e-8;
          xxmax[li] =  1e-8;
        }
      }

      Vector Xi = x0;
      Xi += filteredDesign;
      PMesh->SetNodes(Xi);
      PMesh->DeleteGeometricFactors();

      if (i % save_freq == 0)
      {
        if (physics == 1)
        {
           tStraincoeff.SetU(&(solver->GetSolution()));
           tGF.ProjectCoefficient(tStraincoeff	);
           GetScalarDerivative(&tGF, &tGF_grad, dim);
        }
         visdc->SetCycle(cycle_count++);
         visdc->SetTime(cycle_count*1.0);
         visdc->Save();
      }

      // StrainEnergyDensityCoefficient tStraincoeff(&firstLameCoef, &secondLameCoef, &discreteSol);
      if (physics == 1)
      {
        tStraincoeff.SetU(&discreteSol);
        tGF.ProjectCoefficient(tStraincoeff	);
      }

      x_gf.ProjectCoefficient(*trueSolution);
      //ParGridFunction objGradGF(pfespace); objGradGF = objgrad;
      paraview_dc.SetCycle(i);
      paraview_dc.SetTime(i*1.0);
      //paraview_dc.RegisterField("ObjGrad",&objGradGF);
      paraview_dc.RegisterField("SolutionD",&discreteSol   );
      if (physics == 1)
      {
        paraview_dc.RegisterField("StrainEnergyDensity",&tGF);
      }
      //paraview_dc.RegisterField("Solution",&x_gf);
      //paraview_dc.RegisterField("Sensitivity",&dQdx_physicsGF);
      paraview_dc.Save();

      double localGradNormSquared = std::pow(objgrad.Norml2(), 2);
      double globGradNorm;
#ifdef MFEM_USE_MPI
      MPI_Allreduce(&localGradNormSquared, &globGradNorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
      globGradNorm = std::sqrt(globGradNorm);

      if (myid == 0)
      {
        std:cout<<"Iter: "<<i<<" obj: "<<val<<" with: "<<ObjVal<<" | "<<meshQualityVal<<" objGrad_Norm: "<<globGradNorm<<std::endl;
      }

#ifdef MFEM_USE_PETSC
      double  conDummy = -0.1;
      mmaPetsc->Update(trueOptvar,objgrad,&conDummy,&volgrad,xxmin,xxmax);
#else
      mfem:Vector conDummy(1);  conDummy= -0.1;
      //std::cout << trueOptvar.Norml2() << " k10-dxpre\n";
      mma->Update(i, objgrad, conDummy, volgrad, xxmin,xxmax, trueOptvar);
      //std::cout << trueOptvar.Norml2() << " k10-dxpost\n";
#endif
      gridfuncOptVar.SetFromTrueVector();

      // std::string tDesingName = "DesingVarVec";
      // desingVarVec.Save( tDesingName.c_str() );

      // std::string tFieldName = "FieldVec";
      // tPreassureGF.Save( tFieldName.c_str() );
    }

    if (visualization)
    {
        x0 -= x;
        socketstream vis;
        common::VisualizeField(vis, "localhost", 19916, x0,
                              "Displacements", 400, 400, 300, 300, "jRmclA");
    }

    {
      ostringstream mesh_name;
      mesh_name << "optimized.mesh";
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      PMesh->PrintAsSerial(mesh_ofs);
    }

    {
      if (physics == 1)
      {
        tStraincoeff.SetU(&(solver->GetSolution()));
        tGF.ProjectCoefficient(tStraincoeff	);
        GetScalarDerivative(&tGF, &tGF_grad, dim);
        final_strain_energy = tGF.ComputeIntegral();
        if (myid == 0)
        {
          std::cout << init_strain_energy << " " << final_strain_energy
          << " " << strain_energy_ref << " strain-energy\n";
        }
      }
      VisItDataCollection *visdc = new VisItDataCollection("tmop-pde-final"+std::to_string(jobid), PMesh);
      visdc->RegisterField("solution", &(solver->GetSolution()));
      if (physics == 1)
      {
        visdc->RegisterField("strain-energy-density", &tGF);
        visdc->RegisterField("strain-energy-density-grad", &tGF_grad);
      }
      visdc->SetCycle(0);
      visdc->SetTime(0.0);
      visdc->Save();
      if (physics == 1)
      {
        GetMaxDisplacement(&solver->GetSolution(), maxdisp_opt);
        if (myid == 0)
        {
          std::cout << "Reference maximum displacement: " << maxdisp_ref(0) << " " << maxdisp_ref(1) << " " << maxdisp_ref(2) << std::endl;
          std::cout << "Initial maximum displacement: " << maxdisp_init(0) << " " << maxdisp_init(1) << " " << maxdisp_init(2) << std::endl;
          std::cout << "Final maximum displacement: " << maxdisp_opt(0) << " " << maxdisp_opt(1) << " " << maxdisp_opt(2) << std::endl;
        }
      }
    }
  }

  delete solver;
  delete PMesh;

  return 0;
}

// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_SPACING
#define MFEM_SPACING

#include "../linalg/vector.hpp"

namespace mfem
{

typedef enum {UNIFORM, LINEAR, GEOMETRIC, BELL, GAUSSIAN, LOGARITHMIC}
SPACING_TYPE;

class SpacingFunction
{
public:
   SpacingFunction(int n_,
                   bool r=false)
   {
      n = n_;
      reverse = r;
   }

   virtual double Eval(int p) = 0;

   virtual void SetSize(int size) = 0;

   void SetReverse(bool r) { reverse = r; }

   void EvalAll(Vector & s)
   {
      s.SetSize(n);
      for (int i=0; i<n; ++i)
      {
         s[i] = Eval(i);
      }
   }

   // The format is
   // SPACING_TYPE numIntParam numDoubleParam {int params} {double params}
   virtual void Print(std::ostream &os) = 0;

protected:
   int n;
   bool reverse;
};

class UniformSpacingFunction : public SpacingFunction
{
public:
   UniformSpacingFunction(int n_)
      : SpacingFunction(n_)
   {
      CalculateSpacing();
   }

   virtual void SetSize(int size) override
   {
      n = size;
      CalculateSpacing();
   }

   virtual double Eval(int p) override
   {
      return s;
   }

   virtual void Print(std::ostream &os) override
   {
      // SPACING_TYPE numIntParam numDoubleParam {int params} {double params}
      os << UNIFORM << " 2 0 " << n << " " << (int) reverse << "\n";
   }

private:
   double s;

   void CalculateSpacing()
   {
      // Spacing is 1 / n
      s = 1.0 / ((double) n);
   }
};

class LinearSpacingFunction : public SpacingFunction
{
public:
   LinearSpacingFunction(int n_, bool r_, double s_)
      : SpacingFunction(n_, r_), s(s_)
   {
      MFEM_ASSERT(0.0 < s && s < 1.0, "Initial spacing must be in (0,1)");
      CalculateDifference();
   }

   virtual void SetSize(int size) override
   {
      n = size;
      CalculateDifference();
   }

   virtual double Eval(int p) override
   {
      MFEM_ASSERT(p>=0 && p<n, "Access element " << p
                  << " of spacing function, size = " << n);
      const int i = reverse ? n - 1 - p : p;
      return s + (i * d);
   }

   virtual void Print(std::ostream &os) override
   {
      // SPACING_TYPE numIntParam numDoubleParam {int params} {double params}
      os << LINEAR << " 2 1 " << n << " " << (int) reverse << " " << s << "\n";
   }

private:
   double s, d;

   void CalculateDifference()
   {
      // Spacings are s, s + d, ..., s + (n-1)d, which must sum to 1:
      // 1 = ns + dn(n-1)/2
      d = 2.0 * (1.0 - (n * s)) / ((double) (n*(n-1)));

      if (s + ((n-1) * d) <= 0.0)
      {
         MFEM_ABORT("Invalid linear spacing parameters");
      }
   }
};

// Spacing of interval i is s*r^i for 0 <= i < n, with
//     s + s*r + s*r^2 + ... + s*r^(n-1) = 1
//     s * (r^n - 1) / (r - 1) = 1
// The initial spacing s and number of intervals n are inputs, and r is solved
// for by Newton's method.
class GeometricSpacingFunction : public SpacingFunction
{
public:
   GeometricSpacingFunction(int n_, bool r_, double s_)
      : SpacingFunction(n_, r_), s(s_)
   {
      CalculateSpacing();
   }

   virtual void SetSize(int size) override
   {
      n = size;
      CalculateSpacing();
   }

   virtual double Eval(int p) override
   {
      const int i = reverse ? n - 1 - p : p;
      return s * std::pow(r, i);
   }

   virtual void Print(std::ostream &os) override
   {
      // SPACING_TYPE numIntParam numDoubleParam {int params} {double params}
      os << GEOMETRIC << " 2 1 " << n << " " << (int) reverse
         << " " << s << "\n";
   }

private:
   double s;  // Initial spacing
   double r;  // Ratio

   void CalculateSpacing();
};

class BellSpacingFunction : public SpacingFunction
{
public:
   BellSpacingFunction(int n_, bool r_, double s0_, double s1_)
      : SpacingFunction(n_, r_), s0(s0_), s1(s1_)
   {
      CalculateSpacing();
   }

   virtual void SetSize(int size) override
   {
      n = size;
      CalculateSpacing();
   }

   virtual double Eval(int p) override
   {
      const int i = reverse ? n - 1 - p : p;
      return s[i];
   }

   virtual void Print(std::ostream &os) override
   {
      // SPACING_TYPE numIntParam numDoubleParam {int params} {double params}
      os << BELL << " 2 2 " << n << " " << (int) reverse << " "
         << s0 << " " << s1 << "\n";
   }

private:
   double s0, s1;
   Vector s;

   void CalculateSpacing();
};

// GaussianSpacingFunction fits a Gaussian function of the general form
// g(x) = a exp(-(x-m)^2 / c^2) for some scalar parameters a, m, c.
class GaussianSpacingFunction : public SpacingFunction
{
public:
   GaussianSpacingFunction(int n_, bool r_, double s0_, double s1_)
      : SpacingFunction(n_, r_), s0(s0_), s1(s1_)
   {
      CalculateSpacing();
   }

   virtual void SetSize(int size) override
   {
      n = size;
      CalculateSpacing();
   }

   virtual double Eval(int p) override
   {
      const int i = reverse ? n - 1 - p : p;
      return s[i];
   }

   virtual void Print(std::ostream &os) override
   {
      // SPACING_TYPE numIntParam numDoubleParam {int params} {double params}
      os << GAUSSIAN << " 2 2 " << n << " " << (int) reverse << " "
         << s0 << " " << s1 << "\n";
   }

private:
   double s0, s1;
   Vector s;

   void CalculateSpacing();
};

class LogarithmicSpacingFunction : public SpacingFunction
{
public:
   LogarithmicSpacingFunction(int n_, bool r_, bool sym_=false, double b_=10.0)
      : SpacingFunction(n_, r_), sym(sym_), logBase(b_)
   {
      CalculateSpacing();
   }

   virtual void SetSize(int size) override
   {
      n = size;
      CalculateSpacing();
   }

   virtual double Eval(int p) override
   {
      const int i = reverse ? n - 1 - p : p;
      return s[i];
   }

   virtual void Print(std::ostream &os) override
   {
      // SPACING_TYPE numIntParam numDoubleParam {int params} {double params}
      os << LOGARITHMIC << " 3 1 " << n << " " << (int) reverse << " "
         << (int) sym << " " << logBase << "\n";
   }

private:
   bool sym;
   double logBase;
   Vector s;

   void CalculateSpacing();
   void CalculateSymmetric();
   void CalculateNonsymmetric();
};

SpacingFunction* GetSpacingFunction(const SPACING_TYPE type,
                                    Array<int> const& ipar,
                                    Array<double> const& dpar);
}
#endif

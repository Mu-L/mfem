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

#ifndef MFEM_FE
#define MFEM_FE

#include "../config/config.hpp"
#include "../general/array.hpp"
#include "../linalg/linalg.hpp"
#include "intrules.hpp"
#include "geom.hpp"

#include <map>

namespace mfem
{

/// Possible basis types. Note that not all elements can use all BasisType(s).
class BasisType
{
public:
   enum
   {
      Invalid         = -1,
      GaussLegendre   = 0,  ///< Open type
      GaussLobatto    = 1,  ///< Closed type
      Positive        = 2,  ///< Bernstein polynomials
      OpenUniform     = 3,  ///< Nodes: x_i = (i+1)/(n+1), i=0,...,n-1
      ClosedUniform   = 4,  ///< Nodes: x_i = i/(n-1),     i=0,...,n-1
      OpenHalfUniform = 5,  ///< Nodes: x_i = (i+1/2)/n,   i=0,...,n-1
      NumBasisTypes   = 6   /**< Keep track of maximum types to prevent
                                 hard-coding */
   };
   /** @brief If the input does not represents a valid BasisType, abort with an
       error; otherwise return the input. */
   static int Check(int b_type)
   {
      MFEM_VERIFY(0 <= b_type && b_type < NumBasisTypes,
                  "unknown BasisType: " << b_type);
      return b_type;
   }
   /** @brief If the input does not represents a valid nodal BasisType, abort
       with an error; otherwise return the input. */
   static int CheckNodal(int b_type)
   {
      MFEM_VERIFY(Check(b_type) != Positive,
                  "invalid nodal BasisType: " << Name(b_type));
      return b_type;
   }
   /** @brief Get the corresponding Quadrature1D constant, when that makes
       sense; otherwise return Quadrature1D::Invalid. */
   static int GetQuadrature1D(int b_type)
   {
      switch (b_type)
      {
         case GaussLegendre:   return Quadrature1D::GaussLegendre;
         case GaussLobatto:    return Quadrature1D::GaussLobatto;
         case Positive:        return Quadrature1D::ClosedUniform; // <-----
         case OpenUniform:     return Quadrature1D::OpenUniform;
         case ClosedUniform:   return Quadrature1D::ClosedUniform;
         case OpenHalfUniform: return Quadrature1D::OpenHalfUniform;
      }
      return Quadrature1D::Invalid;
   }
   /// Return the nodal BasisType corresponding to the Quadrature1D type.
   static int GetNodalBasis(int qpt_type)
   {
      switch (qpt_type)
      {
         case Quadrature1D::GaussLegendre:   return GaussLegendre;
         case Quadrature1D::GaussLobatto:    return GaussLobatto;
         case Quadrature1D::OpenUniform:     return OpenUniform;
         case Quadrature1D::ClosedUniform:   return ClosedUniform;
         case Quadrature1D::OpenHalfUniform: return OpenHalfUniform;
      }
      return Invalid;
   }
   /// Check and convert a BasisType constant to a string identifier.
   static const char *Name(int b_type)
   {
      static const char *name[] =
      {
         "Gauss-Legendre", "Gauss-Lobatto", "Positive (Bernstein)",
         "Open uniform", "Closed uniform", "Open half uniform"
      };
      return name[Check(b_type)];
   }
   /// Check and convert a BasisType constant to a char basis identifier.
   static char GetChar(int b_type)
   {
      static const char ident[] = { 'g', 'G', 'P', 'u', 'U', 'o' };
      return ident[Check(b_type)];
   }
   /// Convert char basis identifier to a BasisType constant.
   static int GetType(char b_ident)
   {
      switch (b_ident)
      {
         case 'g': return GaussLegendre;
         case 'G': return GaussLobatto;
         case 'P': return Positive;
         case 'u': return OpenUniform;
         case 'U': return ClosedUniform;
         case 'o': return OpenHalfUniform;
      }
      MFEM_ABORT("unknown BasisType identifier");
      return -1;
   }
};

// Base and derived classes for finite elements

/// Describes the function space on each element
class FunctionSpace
{
public:
   enum
   {
      Pk, ///< Polynomials of order k
      Qk, ///< Tensor products of polynomials of order k
      rQk ///< Refined tensor products of polynomials of order k
   };
};

class ElementTransformation;
class Coefficient;
class VectorCoefficient;
class MatrixCoefficient;
class KnotVector;

/// Abstract class for Finite Elements
class FiniteElement
{
protected:
   int Dim;      ///< Dimension of reference space
   Geometry::Type GeomType; ///< Geometry::Type of the reference element
   int FuncSpace, RangeType, MapType,
       DerivType, DerivRangeType, DerivMapType;
   mutable
   int Dof,      ///< Number of degrees of freedom
       Order;    ///< Order/degree of the shape functions
   mutable int Orders[Geometry::MaxDim]; ///< Anisotropic orders
   IntegrationRule Nodes;
#ifndef MFEM_THREAD_SAFE
   mutable DenseMatrix vshape; // Dof x Dim
#endif

public:
   /// Enumeration for RangeType and DerivRangeType
   enum RangeT { SCALAR, VECTOR };

   /** @brief Enumeration for MapType: defines how reference functions are
       mapped to physical space.

       A reference function \f$ \hat u(\hat x) \f$ can be mapped to a function 
      \f$ u(x) \f$ on a general physical element in following where:
       - \f$ x = T(\hat x) \f$ is the image of the reference point \f$ \hat x \f$
       - \f$ J = J(\hat x) \f$ is the Jacobian matrix of the transformation T
       - \f$ w = w(\hat x) = det(J) \f$ is the transformation weight factor for square J
       - \f$ w = w(\hat x) = det(J^t J)^{1/2} \f$ is the transformation weight factor in general
   */
   enum MapT { 
          VALUE,     /**< For scalar fields; preserves point values
                          \f$ u(x) = \hat u(\hat x) \f$ */
          INTEGRAL,  /**< For scalar fields; preserves volume integrals
                          \f$ u(x) = (1/w) \hat u(\hat x) \f$ */
          H_DIV,     /**< For vector fields; preserves surface integrals of the
                          normal component \f$ u(x) = (J/w) \hat u(\hat x) \f$ */
          H_CURL     /**< For vector fields; preserves line integrals of the
                          tangential component 
                          \f$ u(x) = J^{-t} \hat u(\hat x) \f$ (square J), 
                          \f$ u(x) = J(J^t J)^{-1} \hat u(\hat x) \f$ (general J) */
        };

   /** @brief Enumeration for DerivType: defines which derivative method
       is implemented.

       Each FiniteElement class implements up to one type of derivative.  The
       value returned by GetDerivType() indicates which derivative method is
       implemented.
   */
   enum DerivT { 
          NONE, ///< No derivatives implemented
          GRAD, ///< Implements CalcDShape methods
          DIV,  ///< Implements CalcDivShape methods
          CURL  ///< Implements CalcCurlShape methods
        };

   /** @brief Construct FiniteElement with given
       @param D    Reference space dimension
       @param G    Geometry type (of type Geometry::Type)
       @param Do   Number of degrees of freedom in the FiniteElement
       @param O    Order/degree of the FiniteElement
       @param F    FunctionSpace type of the FiniteElement
    */
   FiniteElement(int D, Geometry::Type G, int Do, int O,
                 int F = FunctionSpace::Pk);

   /// Returns the reference space dimension for the finite element
   int GetDim() const { return Dim; }

   /// Returns the Geometry::Type of the reference element
   Geometry::Type GetGeomType() const { return GeomType; }

   /// Returns the number of degrees of freedom in the finite element
   int GetDof() const { return Dof; }

   /** @brief Returns the order of the finite element. In the case of
       anisotropic orders, returns the maximum order. */
   int GetOrder() const { return Order; }

   /** @brief Returns true if the FiniteElement basis *may be using* different
       orders/degrees in different spatial directions. */
   bool HasAnisotropicOrders() const { return Orders[0] != -1; }

   /// Returns an array containing the anisotropic orders/degrees.
   const int *GetAnisotropicOrders() const { return Orders; }

   /// Returns the type of FunctionSpace on the element.
   int Space() const { return FuncSpace; }

   /// Returns the FiniteElement::RangeT of the element, one of {SCALAR, VECTOR}.
   int GetRangeType() const { return RangeType; }

   /** @brief Returns the FiniteElement::RangeT of the element derivative, either 
       SCALAR or VECTOR. */
   int GetDerivRangeType() const { return DerivRangeType; }

   /** @brief Returns the FiniteElement::MapT of the element describing how reference 
       functions are mapped to physical space, one of {VALUE, INTEGRAL  
       H_DIV, H_CURL}. */
   int GetMapType() const { return MapType; }


   /** @brief Returns the FiniteElement::MapT of the element describing how reference 
       functions are mapped to physical space, one of {VALUE, INTEGRAL  
       H_DIV, H_CURL}. */
   int GetDerivType() const { return DerivType; }

   /** @brief Returns the FiniteElement::MapT of the element describing how reference 
       function derivatives are mapped to physical space, one of {VALUE,   
       INTEGRAL, H_DIV, H_CURL}. */
   int GetDerivMapType() const { return DerivMapType; }

   /** @brief Evaluate the values of all shape functions of a scalar finite
       element in reference space at the given point @a ip. */
   /** The size (#Dof) of the result Vector @a shape must be set in advance. */
   virtual void CalcShape(const IntegrationPoint &ip,
                          Vector &shape) const = 0;

   /** @brief Evaluate the values of all shape functions of a scalar finite
       element in physical space at the point described by @a Trans. */
   /** The size (#Dof) of the result Vector @a shape must be set in advance. */
   void CalcPhysShape(ElementTransformation &Trans, Vector &shape) const;

   /** @brief Evaluate the gradients of all shape functions of a scalar finite
       element in reference space at the given point @a ip. */
   /** Each row of the result DenseMatrix @a dshape contains the derivatives of
       one shape function. The size (#Dof x #Dim) of @a dshape must be set in
       advance.  */
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const = 0;

   /** @brief Evaluate the gradients of all shape functions of a scalar finite
       element in physical space at the point described by @a Trans. */
   /** Each row of the result DenseMatrix @a dshape contains the derivatives of
       one shape function. The size (#Dof x SDim) of @a dshape must be set in
       advance, where SDim >= #Dim is the physical space dimension as described
       by @a Trans. */
   void CalcPhysDShape(ElementTransformation &Trans, DenseMatrix &dshape) const;

   /// Get a const reference to the nodes of the element
   const IntegrationRule & GetNodes() const { return Nodes; }

   // virtual functions for finite elements on vector spaces

   /** @brief Evaluate the values of all shape functions of a *vector* finite
       element in reference space at the given point @a ip. */
   /** Each row of the result DenseMatrix @a shape contains the components of
       one vector shape function. The size (#Dof x #Dim) of @a shape must be set
       in advance. */
   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;

   /** @brief Evaluate the values of all shape functions of a *vector* finite
       element in physical space at the point described by @a Trans. */
   /** Each row of the result DenseMatrix @a shape contains the components of
       one vector shape function. The size (#Dof x SDim) of @a shape must be set
       in advance, where SDim >= #Dim is the physical space dimension as
       described by @a Trans. */
   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const;

   /// Equivalent to the CalcVShape() method with the same arguments.
   void CalcPhysVShape(ElementTransformation &Trans, DenseMatrix &shape) const
   { CalcVShape(Trans, shape); }

   /** @brief Evaluate the divergence of all shape functions of a *vector*
       finite element in reference space at the given point @a ip. */
   /** The size (#Dof) of the result Vector @a divshape must be set in advance.
    */
   virtual void CalcDivShape(const IntegrationPoint &ip,
                             Vector &divshape) const;

   /** @brief Evaluate the divergence of all shape functions of a *vector*
       finite element in physical space at the point described by @a Trans. */
   /** The size (#Dof) of the result Vector @a divshape must be set in advance.
    */
   void CalcPhysDivShape(ElementTransformation &Trans, Vector &divshape) const;

   /** @brief Evaluate the curl of all shape functions of a *vector* finite
       element in reference space at the given point @a ip. */
   /** Each row of the result DenseMatrix @a curl_shape contains the components
       of the curl of one vector shape function. The size (#Dof x CDim) of
       @a curl_shape must be set in advance, where CDim = 3 for #Dim = 3 and
       CDim = 1 for #Dim = 2. */
   virtual void CalcCurlShape(const IntegrationPoint &ip,
                              DenseMatrix &curl_shape) const;

   /** @brief Evaluate the curl of all shape functions of a *vector* finite
       element in physical space at the point described by @a Trans. */
   /** Each row of the result DenseMatrix @a curl_shape contains the components
       of the curl of one vector shape function. The size (#Dof x CDim) of
       @a curl_shape must be set in advance, where CDim = 3 for #Dim = 3 and
       CDim = 1 for #Dim = 2. */
   void CalcPhysCurlShape(ElementTransformation &Trans,
                          DenseMatrix &curl_shape) const;

   /** @brief Get the dofs associated with the given @a face.  
       @a *dofs is set to an internal array of the local dofc on the 
       face, while *ndofs is set to the number of dofs on that face.
   */
   virtual void GetFaceDofs(int face, int **dofs, int *ndofs) const;

   /** each row of h contains the upper triangular part of the hessian
       of one shape function; the order in 2D is {u_xx, u_xy, u_yy} */
   virtual void CalcHessian (const IntegrationPoint &ip,
                             DenseMatrix &h) const;

   /** @brief Return the local interpolation matrix @a I (Dof x Dof) where the
       fine element is the image of the base geometry under the given
       transformation. */
   virtual void GetLocalInterpolation(ElementTransformation &Trans,
                                      DenseMatrix &I) const;

   /** @brief Return a local restriction matrix @a R (Dof x Dof) mapping fine
       dofs to coarse dofs.

       The fine element is the image of the base geometry under the given
       transformation, @a Trans.

       The assumption in this method is that a subset of the coarse dofs can be
       expressed only in terms of the dofs of the given fine element.

       Rows in @a R corresponding to coarse dofs that cannot be expressed in
       terms of the fine dofs will be marked as invalid by setting the first
       entry (column 0) in the row to infinity().

       This method assumes that the dimensions of @a R are set before it is
       called. */
   virtual void GetLocalRestriction(ElementTransformation &Trans,
                                    DenseMatrix &R) const;

   /** @brief Return interpolation matrix, @a I, which maps dofs from a coarse
       element, @a fe, to the fine dofs on @a this finite element. */
   /** @a Trans represents the mapping from the reference element of @a this
       element into a subset of the reference space of the element @a fe, thus
       allowing the "coarse" FiniteElement to be different from the "fine"
       FiniteElement as when h-refinement is combined with p-refinement or
       p-derefinement. It is assumed that both finite elements use the same
       FiniteElement::MapT. */
   virtual void GetTransferMatrix(const FiniteElement &fe,
                                  ElementTransformation &Trans,
                                  DenseMatrix &I) const;

   /** @brief Given a coefficient and a transformation, compute its projection
       (approximation) in the local finite dimensional space in terms
       of the degrees of freedom. */
   virtual void Project (Coefficient &coeff,
                         ElementTransformation &Trans, Vector &dofs) const;

   /** @brief Given a vector coefficient and a transformation, compute its
       projection (approximation) in the local finite dimensional space
       in terms of the degrees of freedom. (VectorFiniteElements) */
   virtual void Project (VectorCoefficient &vc,
                         ElementTransformation &Trans, Vector &dofs) const;

   /** @brief Given a matrix coefficient and a transformation, compute an approximation
       ("projection") in the local finite dimensional space in terms of the
       degrees of freedom. For VectorFiniteElements, the rows of the coefficient
       are projected in the vector space. */
   virtual void ProjectMatrixCoefficient(
      MatrixCoefficient &mc, ElementTransformation &T, Vector &dofs) const;

   /** @brief Project a delta function centered on the given @a vertex in
       the local finite dimensional space represented by the @a dofs. */
   virtual void ProjectDelta(int vertex, Vector &dofs) const;

   /** @brief Compute the embedding/projection matrix from the given FiniteElement
       onto 'this' FiniteElement. The ElementTransformation is included to
       support cases when the projection depends on it. */
   virtual void Project(const FiniteElement &fe, ElementTransformation &Trans,
                        DenseMatrix &I) const;

   /** @brief Compute the discrete gradient matrix from the given FiniteElement onto
       'this' FiniteElement. The ElementTransformation is included to support
       cases when the matrix depends on it. */
   virtual void ProjectGrad(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &grad) const;

   /** @brief Compute the discrete curl matrix from the given FiniteElement onto
       'this' FiniteElement. The ElementTransformation is included to support
       cases when the matrix depends on it. */
   virtual void ProjectCurl(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &curl) const;

   /** @brief Compute the discrete divergence matrix from the given FiniteElement onto
       'this' FiniteElement. The ElementTransformation is included to support
       cases when the matrix depends on it. */
   virtual void ProjectDiv(const FiniteElement &fe,
                           ElementTransformation &Trans,
                           DenseMatrix &div) const;

   /// Deconstruct the FiniteElement
   virtual ~FiniteElement () { }

   /** @brief Return true if the BasisType of @a b_type is closed 
       (has Quadrature1D points on the boundary). */
   static bool IsClosedType(int b_type)
   {
      const int q_type = BasisType::GetQuadrature1D(b_type);
      return ((q_type != Quadrature1D::Invalid) &&
              (Quadrature1D::CheckClosed(q_type) != Quadrature1D::Invalid));
   }

   /** @brief Return true if the BasisType of @a b_type is open 
       (doesn't have Quadrature1D points on the boundary). */
   static bool IsOpenType(int b_type)
   {
      const int q_type = BasisType::GetQuadrature1D(b_type);
      return ((q_type != Quadrature1D::Invalid) &&
              (Quadrature1D::CheckOpen(q_type) != Quadrature1D::Invalid));
   }

   /** @brief Ensure that the BasisType of @a b_type is closed 
       (has Quadrature1D points on the boundary). */
   static int VerifyClosed(int b_type)
   {
      MFEM_VERIFY(IsClosedType(b_type),
                  "invalid closed basis type: " << b_type);
      return b_type;
   }

   /** @brief Ensure that the BasisType of @a b_type is open 
       (doesn't have Quadrature1D points on the boundary). */   
   static int VerifyOpen(int b_type)
   {
      MFEM_VERIFY(IsOpenType(b_type), "invalid open basis type: " << b_type);
      return b_type;
   }

   /** @brief Ensure that the BasisType of @a b_type nodal 
       (satisfies the interpolation property). */ 
   static int VerifyNodal(int b_type)
   {
      return BasisType::CheckNodal(b_type);
   }
};

class ScalarFiniteElement : public FiniteElement
{
protected:
#ifndef MFEM_THREAD_SAFE
   mutable Vector c_shape;
#endif

   static const ScalarFiniteElement &CheckScalarFE(const FiniteElement &fe)
   {
      if (fe.GetRangeType() != SCALAR)
      { mfem_error("'fe' must be a ScalarFiniteElement"); }
      return static_cast<const ScalarFiniteElement &>(fe);
   }

public:
   /** @brief Construct FiniteElement with given
       @param D    Reference space dimension
       @param G    Geometry type (of type Geometry::Type)
       @param Do   Number of degrees of freedom in the FiniteElement
       @param O    Order/degree of the FiniteElement
       @param F    FunctionSpace type of the FiniteElement
    */  
   ScalarFiniteElement(int D, Geometry::Type G, int Do, int O,
                       int F = FunctionSpace::Pk)
#ifdef MFEM_THREAD_SAFE
      : FiniteElement(D, G, Do, O, F)
   { DerivType = GRAD; DerivRangeType = VECTOR; DerivMapType = H_CURL; }
#else
      : FiniteElement(D, G, Do, O, F), c_shape(Dof)
   { DerivType = GRAD; DerivRangeType = VECTOR; DerivMapType = H_CURL; }
#endif

   /** @brief Set the FiniteElement::MapT of the element to either VALUE or INTEGRAL.
       Also sets the FiniteElement::DerivT to GRAD if the FiniteElement::MapT is VALUE. */
   void SetMapType(int M)
   {
      MFEM_VERIFY(M == VALUE || M == INTEGRAL, "unknown MapType");
      MapType = M;
      DerivType = (M == VALUE) ? GRAD : NONE;
   }


   /** @brief Get the matrix @a I that defines nodal interpolation  
       @a between this element and the refined element @a fine_fe. */
   void NodalLocalInterpolation(ElementTransformation &Trans,
                                DenseMatrix &I,
                                const ScalarFiniteElement &fine_fe) const;

   /** @brief Get matrix @a I "Interpolation" defined through local 
       L2-projection in the space defined by the @a fine_fe.  */
   /** If the "fine" elements cannot represent all basis functions of the
       "coarse" element, then boundary values from different sub-elements are
       generally different. */
   void ScalarLocalInterpolation(ElementTransformation &Trans,
                                 DenseMatrix &I,
                                 const ScalarFiniteElement &fine_fe) const;
};

class NodalFiniteElement : public ScalarFiniteElement
{
protected:
   void ProjectCurl_2D(const FiniteElement &fe,
                       ElementTransformation &Trans,
                       DenseMatrix &curl) const;

public:
   /** @brief Construct FiniteElement with given
       @param D    Reference space dimension
       @param G    Geometry type (of type Geometry::Type)
       @param Do   Number of degrees of freedom in the FiniteElement
       @param O    Order/degree of the FiniteElement
       @param F    FunctionSpace type of the FiniteElement
   */
   NodalFiniteElement(int D, Geometry::Type G, int Do, int O,
                      int F = FunctionSpace::Pk)
      : ScalarFiniteElement(D, G, Do, O, F) { }

   virtual void GetLocalInterpolation(ElementTransformation &Trans,
                                      DenseMatrix &I) const
   { NodalLocalInterpolation(Trans, I, *this); }

   virtual void GetLocalRestriction(ElementTransformation &Trans,
                                    DenseMatrix &R) const;

   virtual void GetTransferMatrix(const FiniteElement &fe,
                                  ElementTransformation &Trans,
                                  DenseMatrix &I) const
   { CheckScalarFE(fe).NodalLocalInterpolation(Trans, I, *this); }

   virtual void Project (Coefficient &coeff,
                         ElementTransformation &Trans, Vector &dofs) const;

   virtual void Project (VectorCoefficient &vc,
                         ElementTransformation &Trans, Vector &dofs) const;

   // (mc.height x mc.width) @ DOFs -> (Dof x mc.width x mc.height) in dofs
   virtual void ProjectMatrixCoefficient(
      MatrixCoefficient &mc, ElementTransformation &T, Vector &dofs) const;

   virtual void Project(const FiniteElement &fe, ElementTransformation &Trans,
                        DenseMatrix &I) const;

   virtual void ProjectGrad(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &grad) const;

   virtual void ProjectDiv(const FiniteElement &fe,
                           ElementTransformation &Trans,
                           DenseMatrix &div) const;
};


class PositiveFiniteElement : public ScalarFiniteElement
{
public:
   /** @brief Construct FiniteElement with given
       @param D    Reference space dimension
       @param G    Geometry type (of type Geometry::Type)
       @param Do   Number of degrees of freedom in the FiniteElement
       @param O    Order/degree of the FiniteElement
       @param F    FunctionSpace type of the FiniteElement
   */  
   PositiveFiniteElement(int D, Geometry::Type G, int Do, int O,
                         int F = FunctionSpace::Pk) :
      ScalarFiniteElement(D, G, Do, O, F)
   { }

   virtual void GetLocalInterpolation(ElementTransformation &Trans,
                                      DenseMatrix &I) const
   { ScalarLocalInterpolation(Trans, I, *this); }

   virtual void GetTransferMatrix(const FiniteElement &fe,
                                  ElementTransformation &Trans,
                                  DenseMatrix &I) const
   { CheckScalarFE(fe).ScalarLocalInterpolation(Trans, I, *this); }

   using FiniteElement::Project;

   // Low-order monotone "projection" (actually it is not a projection): the
   // dofs are set to be the Coefficient values at the nodes.
   virtual void Project(Coefficient &coeff,
                        ElementTransformation &Trans, Vector &dofs) const;

   virtual void Project (VectorCoefficient &vc,
                         ElementTransformation &Trans, Vector &dofs) const;

   virtual void Project(const FiniteElement &fe, ElementTransformation &Trans,
                        DenseMatrix &I) const;
};

class VectorFiniteElement : public FiniteElement
{
   // Hide the scalar functions CalcShape and CalcDShape.
private:
   /// Overrides the scalar CalcShape function to print an error.
   virtual void CalcShape(const IntegrationPoint &ip,
                          Vector &shape) const;

   /// Overrides the scalar CalcDShape function to print an error.
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;

protected:
#ifndef MFEM_THREAD_SAFE
   mutable DenseMatrix J, Jinv;
   mutable DenseMatrix curlshape, curlshape_J;
#endif
   void SetDerivMembers();

   void CalcVShape_RT(ElementTransformation &Trans,
                      DenseMatrix &shape) const;

   void CalcVShape_ND(ElementTransformation &Trans,
                      DenseMatrix &shape) const;

   void Project_RT(const double *nk, const Array<int> &d2n,
                   VectorCoefficient &vc, ElementTransformation &Trans,
                   Vector &dofs) const;

   // project the rows of the matrix coefficient in an RT space
   void ProjectMatrixCoefficient_RT(
      const double *nk, const Array<int> &d2n,
      MatrixCoefficient &mc, ElementTransformation &T, Vector &dofs) const;

   void Project_RT(const double *nk, const Array<int> &d2n,
                   const FiniteElement &fe, ElementTransformation &Trans,
                   DenseMatrix &I) const;

   // rotated gradient in 2D
   void ProjectGrad_RT(const double *nk, const Array<int> &d2n,
                       const FiniteElement &fe, ElementTransformation &Trans,
                       DenseMatrix &grad) const;

   // Compute the curl as a discrete operator from ND FE (fe) to ND FE (this).
   // The natural FE for the range is RT, so this is an approximation.
   void ProjectCurl_ND(const double *tk, const Array<int> &d2t,
                       const FiniteElement &fe, ElementTransformation &Trans,
                       DenseMatrix &curl) const;

   void ProjectCurl_RT(const double *nk, const Array<int> &d2n,
                       const FiniteElement &fe, ElementTransformation &Trans,
                       DenseMatrix &curl) const;

   void Project_ND(const double *tk, const Array<int> &d2t,
                   VectorCoefficient &vc, ElementTransformation &Trans,
                   Vector &dofs) const;

   /// project the rows of the matrix coefficient in an ND space
   void ProjectMatrixCoefficient_ND(
      const double *tk, const Array<int> &d2t,
      MatrixCoefficient &mc, ElementTransformation &T, Vector &dofs) const;

   void Project_ND(const double *tk, const Array<int> &d2t,
                   const FiniteElement &fe, ElementTransformation &Trans,
                   DenseMatrix &I) const;

   void ProjectGrad_ND(const double *tk, const Array<int> &d2t,
                       const FiniteElement &fe, ElementTransformation &Trans,
                       DenseMatrix &grad) const;

   void LocalInterpolation_RT(const VectorFiniteElement &cfe,
                              const double *nk, const Array<int> &d2n,
                              ElementTransformation &Trans,
                              DenseMatrix &I) const;

   void LocalInterpolation_ND(const VectorFiniteElement &cfe,
                              const double *tk, const Array<int> &d2t,
                              ElementTransformation &Trans,
                              DenseMatrix &I) const;

   void LocalRestriction_RT(const double *nk, const Array<int> &d2n,
                            ElementTransformation &Trans,
                            DenseMatrix &R) const;

   void LocalRestriction_ND(const double *tk, const Array<int> &d2t,
                            ElementTransformation &Trans,
                            DenseMatrix &R) const;

   static const VectorFiniteElement &CheckVectorFE(const FiniteElement &fe)
   {
      if (fe.GetRangeType() != VECTOR)
      { mfem_error("'fe' must be a VectorFiniteElement"); }
      return static_cast<const VectorFiniteElement &>(fe);
   }

public:
   VectorFiniteElement (int D, Geometry::Type G, int Do, int O, int M,
                        int F = FunctionSpace::Pk) :
#ifdef MFEM_THREAD_SAFE
      FiniteElement(D, G, Do, O, F)
   { RangeType = VECTOR; MapType = M; SetDerivMembers(); }
#else
      FiniteElement(D, G, Do, O, F), Jinv(D)
   { RangeType = VECTOR; MapType = M; SetDerivMembers(); }
#endif
};

/// A 0D point finite element
class PointFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the FiniteElement
   PointFiniteElement();

   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;

   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};

/// A 1D linear element with nodes on the endpoints
class Linear1DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the FiniteElement
   Linear1DFiniteElement();

   /** virtual function which evaluates the values of all
       shape functions at a given point ip and stores
       them in the vector shape of dimension Dof (2) */
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;

   /** virtual function which evaluates the derivatives of all
       shape functions at a given point ip and stores them in
       the matrix dshape (Dof x Dim) (2 x 1) so that each row
       contains the derivative of one shape function */
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};

/// A 2D linear element on triangle with nodes at the vertices of the triangle
class Linear2DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the FiniteElement
   Linear2DFiniteElement();

   /** virtual function which evaluates the values of all
       shape functions at a given point ip and stores
       them in the vector shape of dimension Dof (3) */
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;

   /** virtual function which evaluates the values of all
       partial derivatives of all shape functions at a given
       point ip and stores them in the matrix dshape (Dof x Dim) (3 x 2)
       so that each row contains the derivatives of one shape function */
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const
   { dofs = 0.0; dofs(vertex) = 1.0; }
};

/// A 2D bi-linear element on a square with nodes at the vertices of the square
class BiLinear2DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the FiniteElement
   BiLinear2DFiniteElement();

   /** virtual function which evaluates the values of all
       shape functions at a given point ip and stores
       them in the vector shape of dimension Dof (4) */
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;

   /** virtual function which evaluates the values of all
       partial derivatives of all shape functions at a given
       point ip and stores them in the matrix dshape (Dof x Dim) (4 x 2)
       so that each row contains the derivatives of one shape function */
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void CalcHessian (const IntegrationPoint &ip,
                             DenseMatrix &h) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const
   { dofs = 0.0; dofs(vertex) = 1.0; } // { dofs = 1.0; }
};

/// A linear element on a triangle with nodes at the 3 "Gaussian" points
class GaussLinear2DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the FiniteElement
   GaussLinear2DFiniteElement();
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const;
};

/// A 2D bi-linear element on a square with nodes at the "Gaussian" points
class GaussBiLinear2DFiniteElement : public NodalFiniteElement
{
private:
   static const double p[2];

public:
   /// Construct the FiniteElement
   GaussBiLinear2DFiniteElement();
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const;
};

/** @brief A  2D linear element on a square with 3 nodes at the 
    vertices of the lower left triangle */
class P1OnQuadFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the FiniteElement
   P1OnQuadFiniteElement();
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const
   { dofs = 1.0; }
};

/// A 1D quadractic finite element with uniformly spaced nodes
class Quad1DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the FiniteElement
   Quad1DFiniteElement();

   /** virtual function which evaluates the values of all
       shape functions at a given point ip and stores
       them in the vector shape of dimension Dof (3) */
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;

   /** virtual function which evaluates the derivatives of all
       shape functions at a given point ip and stores them in
       the matrix dshape (Dof x Dim) (3 x 1) so that each row
       contains the derivative of one shape function */
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};

/// A 1D quadratic positive element utilizing the 2nd order Bernstein basis
class QuadPos1DFiniteElement : public PositiveFiniteElement
{
public:
  /// Construct the FiniteElement
   QuadPos1DFiniteElement();
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};

/** @brief A 2D quadratic element on triangle with nodes at the
    vertices and midpoints of the triangle. */
class Quad2DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the FiniteElement
   Quad2DFiniteElement();

   /** virtual function which evaluates the values of all
       shape functions at a given point ip and stores
       them in the vector shape of dimension Dof (6) */
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;

   /** virtual function which evaluates the values of all
       partial derivatives of all shape functions at a given
       point ip and stores them in the matrix dshape (Dof x Dim) (6 x 2)
       so that each row contains the derivatives of one shape function */
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;

   virtual void CalcHessian (const IntegrationPoint &ip,
                             DenseMatrix &h) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const;
};

/// A quadratic element on triangle with nodes at the "Gaussian" points
class GaussQuad2DFiniteElement : public NodalFiniteElement
{
private:
   static const double p[2];
   DenseMatrix A;
   mutable DenseMatrix D;
   mutable Vector pol;
public:
   /// Construct the FiniteElement
   GaussQuad2DFiniteElement();
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   // virtual void ProjectDelta(int vertex, Vector &dofs) const;
};

/// A 2D bi-quadratic element on a square with uniformly spaced nodes
class BiQuad2DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the FiniteElement
   BiQuad2DFiniteElement();

   /** virtual function which evaluates the values of all
       shape functions at a given point ip and stores
       them in the vector shape of dimension Dof (9) */
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;

   /** virtual function which evaluates the values of all
       partial derivatives of all shape functions at a given
       point ip and stores them in the matrix dshape (Dof x Dim) (9 x 2)
       so that each row contains the derivatives of one shape function */
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const;
};


/// A 2D positive bi-quadratic element on a square utilizing the 2nd order Bernstein basis
class BiQuadPos2DFiniteElement : public PositiveFiniteElement
{
public:
   /// Construct the FiniteElement
   BiQuadPos2DFiniteElement();
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void GetLocalInterpolation(ElementTransformation &Trans,
                                      DenseMatrix &I) const;
   using FiniteElement::Project;
   virtual void Project(Coefficient &coeff, ElementTransformation &Trans,
                        Vector &dofs) const;
   virtual void Project(VectorCoefficient &vc, ElementTransformation &Trans,
                        Vector &dofs) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const
   { dofs = 0.; dofs(vertex) = 1.; }
};

/// A 2D bi-quadratic element on a square with nodes at the 9 "Gaussian" points
class GaussBiQuad2DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the FiniteElement
   GaussBiQuad2DFiniteElement();
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   // virtual void ProjectDelta(int vertex, Vector &dofs) const { dofs = 1.; }
};


/// A 2D bi-cubic element on a square with uniformly spaces nodes
class BiCubic2DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the FiniteElement
   BiCubic2DFiniteElement();
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;

   /// Compute the Hessian of second order partial derivatives at @a ip.
   virtual void CalcHessian (const IntegrationPoint &ip,
                             DenseMatrix &h) const;
};

/// A 1D cubic element with uniformly spaced nodes
class Cubic1DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the FiniteElement
   Cubic1DFiniteElement();

   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;

   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};

/// A 2D cubic element on a triangle with uniformly spaced nodes
class Cubic2DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the FiniteElement
   Cubic2DFiniteElement();

   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;

   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;

   virtual void CalcHessian (const IntegrationPoint &ip,
                             DenseMatrix &h) const;
};

/// A 3D cubic element on a tetrahedron with 20 nodes at the thirds of the tetrahedron
class Cubic3DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the FiniteElement
   Cubic3DFiniteElement();

   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;

   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};

/// A 2D constant element on a triangle
class P0TriangleFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the FiniteElement
   P0TriangleFiniteElement();

   /// evaluate shape function - constant 1
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;

   /// evaluate derivatives of shape function - constant 0
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const
   { dofs(0) = 1.0; }
};


/// A 2D constant element on a square
class P0QuadFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the FiniteElement
   P0QuadFiniteElement();
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const
   { dofs(0) = 1.0; }
};


/** @brief A 3D linear element on a tetrahedron with nodes at the 
    vertices of the tetrahedron */
class Linear3DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the FiniteElement
   Linear3DFiniteElement();

   /** @brief virtual function which evaluates the values of all
       shape functions at a given point ip and stores
       them in the vector shape of dimension Dof (4) */
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;

   /** @brief virtual function which evaluates the values of all
       partial derivatives of all shape functions at a given
       point ip and stores them in the matrix dshape (Dof x Dim) (4 x 3)
       so that each row contains the derivatives of one shape function */
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;

   virtual void ProjectDelta(int vertex, Vector &dofs) const
   { dofs = 0.0; dofs(vertex) = 1.0; }

   /** @brief Get the dofs associated with the given @a face.  
       @a *dofs is set to an internal array of the local dofc on the 
       face, while *ndofs is set to the number of dofs on that face.
   */
   virtual void GetFaceDofs(int face, int **dofs, int *ndofs) const;
};

/// A 3D quadratic element on a tetrahedron with uniformly spaced nodes
class Quadratic3DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the FiniteElement
   Quadratic3DFiniteElement();

   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;

   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};

/// A 3D tri-linear element on a cube with nodes at the vertices of the cube
class TriLinear3DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the FiniteElement
   TriLinear3DFiniteElement();

   /** virtual function which evaluates the values of all
       shape functions at a given point ip and stores
       them in the vector shape of dimension Dof (8) */
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;

   /** virtual function which evaluates the values of all
       partial derivatives of all shape functions at a given
       point ip and stores them in the matrix dshape (Dof x Dim) (8 x 3)
       so that each row contains the derivatives of one shape function */
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;

   virtual void ProjectDelta(int vertex, Vector &dofs) const
   { dofs = 0.0; dofs(vertex) = 1.0; }
};


/// A 2D Crouzeix-Raviart element on triangle
class CrouzeixRaviartFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the FiniteElement
   CrouzeixRaviartFiniteElement();
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const
   { dofs = 1.0; }
};

/// A 2D Crouzeix-Raviart finite element on square
class CrouzeixRaviartQuadFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the FiniteElement
   CrouzeixRaviartQuadFiniteElement();
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};

class P0SegmentFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the FiniteElement with dummy order @a Ord
   P0SegmentFiniteElement(int Ord = 0);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};

/** @brief A 2D 1st Raviart-Thomas vector element on a triangle */
class RT0TriangleFiniteElement : public VectorFiniteElement
{
private:
   static const double nk[3][2];

public:
   /// Construct the FiniteElement
   RT0TriangleFiniteElement();

   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;

   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const
   { CalcVShape_RT(Trans, shape); }

   virtual void CalcDivShape(const IntegrationPoint &ip,
                             Vector &divshape) const;

   virtual void GetLocalInterpolation (ElementTransformation &Trans,
                                       DenseMatrix &I) const;

   using FiniteElement::Project;

   virtual void Project (VectorCoefficient &vc,
                         ElementTransformation &Trans, Vector &dofs) const;
};

/** @brief A 2D 1st Raviart-Thomas vector element on a square*/
class RT0QuadFiniteElement : public VectorFiniteElement
{
private:
   static const double nk[4][2];

public:
   /// Construct the FiniteElement
   RT0QuadFiniteElement();

   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;

   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const
   { CalcVShape_RT(Trans, shape); }

   virtual void CalcDivShape(const IntegrationPoint &ip,
                             Vector &divshape) const;

   virtual void GetLocalInterpolation (ElementTransformation &Trans,
                                       DenseMatrix &I) const;

   using FiniteElement::Project;

   virtual void Project (VectorCoefficient &vc,
                         ElementTransformation &Trans, Vector &dofs) const;
};

/** @brief A 2D 2nd Raviart-Thomas vector element on a triangle */
class RT1TriangleFiniteElement : public VectorFiniteElement
{
private:
   static const double nk[8][2];

public:
   /// Construct the FiniteElement
   RT1TriangleFiniteElement();

   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;

   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const
   { CalcVShape_RT(Trans, shape); }

   virtual void CalcDivShape(const IntegrationPoint &ip,
                             Vector &divshape) const;

   virtual void GetLocalInterpolation (ElementTransformation &Trans,
                                       DenseMatrix &I) const;

   using FiniteElement::Project;

   virtual void Project (VectorCoefficient &vc,
                         ElementTransformation &Trans, Vector &dofs) const;
};

/** @brief A 2D 2nd Raviart-Thomas vector element on a square */
class RT1QuadFiniteElement : public VectorFiniteElement
{
private:
   static const double nk[12][2];

public:
   /// Construct the FiniteElement
   RT1QuadFiniteElement();

   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;

   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const
   { CalcVShape_RT(Trans, shape); }

   virtual void CalcDivShape(const IntegrationPoint &ip,
                             Vector &divshape) const;

   virtual void GetLocalInterpolation (ElementTransformation &Trans,
                                       DenseMatrix &I) const;

   using FiniteElement::Project;

   virtual void Project (VectorCoefficient &vc,
                         ElementTransformation &Trans, Vector &dofs) const;
};

/** @brief A 2D 3rd Raviart-Thomas vector element on a triangle */
class RT2TriangleFiniteElement : public VectorFiniteElement
{
private:
   static const double M[15][15];
public:
   /// Construct the FiniteElement
   RT2TriangleFiniteElement();

   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;

   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const
   { CalcVShape_RT(Trans, shape); }

   virtual void CalcDivShape(const IntegrationPoint &ip,
                             Vector &divshape) const;
};

/** @brief A 2D 3rd Raviart-Thomas vector element on a square */
class RT2QuadFiniteElement : public VectorFiniteElement
{
private:
   static const double nk[24][2];
   static const double pt[4];
   static const double dpt[3];

public:
   /// Construct the FiniteElement
   RT2QuadFiniteElement();

   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;

   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const
   { CalcVShape_RT(Trans, shape); }

   virtual void CalcDivShape(const IntegrationPoint &ip,
                             Vector &divshape) const;

   virtual void GetLocalInterpolation (ElementTransformation &Trans,
                                       DenseMatrix &I) const;

   using FiniteElement::Project;

   virtual void Project (VectorCoefficient &vc,
                         ElementTransformation &Trans, Vector &dofs) const;
};

/// A 1D linear element with nodes at 1/3 and 2/3 (trace of RT1)
class P1SegmentFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the FiniteElement
   P1SegmentFiniteElement();
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};

/// A 1D quadratic element with nodes at the Gaussian points (trace of RT2)
class P2SegmentFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the FiniteElement
   P2SegmentFiniteElement();
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};

/// A 1D element with uniform nodes
class Lagrange1DFiniteElement : public NodalFiniteElement
{
private:
   Vector rwk;
#ifndef MFEM_THREAD_SAFE
   mutable Vector rxxk;
#endif
public:
   /// Construct the FiniteElement with the provided @a degree
   Lagrange1DFiniteElement (int degree);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};


class P1TetNonConfFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the FiniteElement
   P1TetNonConfFiniteElement();
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};

/// A 3D constant element on a tetrahedron
class P0TetFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the FiniteElement
   P0TetFiniteElement ();
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const
   { dofs(0) = 1.0; }
};

/// A 3D constant element on a cube
class P0HexFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the FiniteElement
   P0HexFiniteElement ();
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const
   { dofs(0) = 1.0; }
};

/** @brief Tensor products of 1D Lagrange1DFiniteElement 
    (only degree 2 is functional) */
class LagrangeHexFiniteElement : public NodalFiniteElement
{
private:
   Lagrange1DFiniteElement * fe1d;
   int dof1d;
   int *I, *J, *K;
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape1dx, shape1dy, shape1dz;
   mutable DenseMatrix dshape1dx, dshape1dy, dshape1dz;
#endif

public:
   /// Construct the FiniteElement with the provided @a degree
   LagrangeHexFiniteElement (int degree);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   ~LagrangeHexFiniteElement ();
};


/// A 1D refined linear element
class RefinedLinear1DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the FiniteElement
   RefinedLinear1DFiniteElement();

   /** virtual function which evaluates the values of all
       shape functions at a given point ip and stores
       them in the vector shape of dimension Dof (3) */
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;

   /** virtual function which evaluates the derivatives of all
       shape functions at a given point ip and stores them in
       the matrix dshape (Dof x Dim) (3 x 1) so that each row
       contains the derivative of one shape function */
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};

/// A 2D refined linear element on a triangle
class RefinedLinear2DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the FiniteElement
   RefinedLinear2DFiniteElement();

   /** virtual function which evaluates the values of all
       shape functions at a given point ip and stores
       them in the vector shape of dimension Dof (6) */
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;

   /** virtual function which evaluates the values of all
       partial derivatives of all shape functions at a given
       point ip and stores them in the matrix dshape (Dof x Dim) (6 x 2)
       so that each row contains the derivatives of one shape function */
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};

/// A 2D refined linear element on a tetrahedron
class RefinedLinear3DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the FiniteElement
   RefinedLinear3DFiniteElement();

   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;

   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};

/// A 2D refined bi-linear FE on a square
class RefinedBiLinear2DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the FiniteElement
   RefinedBiLinear2DFiniteElement();

   /** virtual function which evaluates the values of all
       shape functions at a given point ip and stores
       them in the vector shape of dimension Dof (9) */
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;

   /** virtual function which evaluates the values of all
       partial derivatives of all shape functions at a given
       point ip and stores them in the matrix dshape (Dof x Dim) (9 x 2)
       so that each row contains the derivatives of one shape function */
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};

/// A 3D refined tri-linear element on a cube
class RefinedTriLinear3DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the FiniteElement
   RefinedTriLinear3DFiniteElement();

   /** virtual function which evaluates the values of all
       shape functions at a given point ip and stores
       them in the vector shape of dimension Dof (9) */
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;

   /** virtual function which evaluates the values of all
       partial derivatives of all shape functions at a given
       point ip and stores them in the matrix dshape (Dof x Dim) (9 x 2)
       so that each row contains the derivatives of one shape function */
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};


/// A 3D 1st order Nedelec element on a cube
class Nedelec1HexFiniteElement : public VectorFiniteElement
{
private:
   static const double tk[12][3];

public:
   /// Construct the FiniteElement
   Nedelec1HexFiniteElement();
   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;
   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const
   { CalcVShape_ND(Trans, shape); }
   virtual void CalcCurlShape(const IntegrationPoint &ip,
                              DenseMatrix &curl_shape) const;
   virtual void GetLocalInterpolation (ElementTransformation &Trans,
                                       DenseMatrix &I) const;
   using FiniteElement::Project;
   virtual void Project (VectorCoefficient &vc,
                         ElementTransformation &Trans, Vector &dofs) const;
};


/// A 3D 1st order Nedelec element on a tetrahedron
class Nedelec1TetFiniteElement : public VectorFiniteElement
{
private:
   static const double tk[6][3];

public:
   /// Construct the FiniteElement
   Nedelec1TetFiniteElement();
   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;
   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const
   { CalcVShape_ND(Trans, shape); }
   virtual void CalcCurlShape(const IntegrationPoint &ip,
                              DenseMatrix &curl_shape) const;
   virtual void GetLocalInterpolation (ElementTransformation &Trans,
                                       DenseMatrix &I) const;
   using FiniteElement::Project;
   virtual void Project (VectorCoefficient &vc,
                         ElementTransformation &Trans, Vector &dofs) const;
};


/// A 3D 0th order Raviert-Thomas element on a cube
class RT0HexFiniteElement : public VectorFiniteElement
{
private:
   static const double nk[6][3];

public:
   /// Construct the FiniteElement
   RT0HexFiniteElement();

   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;

   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const
   { CalcVShape_RT(Trans, shape); }

   virtual void CalcDivShape(const IntegrationPoint &ip,
                             Vector &divshape) const;

   virtual void GetLocalInterpolation (ElementTransformation &Trans,
                                       DenseMatrix &I) const;

   using FiniteElement::Project;

   virtual void Project (VectorCoefficient &vc,
                         ElementTransformation &Trans, Vector &dofs) const;
};


/// A 3D 1st order Raviert-Thomas element on a cube
class RT1HexFiniteElement : public VectorFiniteElement
{
private:
   static const double nk[36][3];

public:
   /// Construct the FiniteElement   
   RT1HexFiniteElement();

   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;

   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const
   { CalcVShape_RT(Trans, shape); }

   virtual void CalcDivShape(const IntegrationPoint &ip,
                             Vector &divshape) const;

   virtual void GetLocalInterpolation (ElementTransformation &Trans,
                                       DenseMatrix &I) const;

   using FiniteElement::Project;

   virtual void Project (VectorCoefficient &vc,
                         ElementTransformation &Trans, Vector &dofs) const;
};


/// A 3D 0th order Raviert-Thomas element on a tetrahedron
class RT0TetFiniteElement : public VectorFiniteElement
{
private:
   static const double nk[4][3];

public:
   /// Construct the FiniteElement
   RT0TetFiniteElement();

   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;

   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const
   { CalcVShape_RT(Trans, shape); }

   virtual void CalcDivShape(const IntegrationPoint &ip,
                             Vector &divshape) const;

   virtual void GetLocalInterpolation (ElementTransformation &Trans,
                                       DenseMatrix &I) const;

   using FiniteElement::Project;

   virtual void Project (VectorCoefficient &vc,
                         ElementTransformation &Trans, Vector &dofs) const;
};


class RotTriLinearHexFiniteElement : public NodalFiniteElement
{
public:
   RotTriLinearHexFiniteElement();
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};


/// Class for computing 1D special polynomials and their associated basis functions
class Poly_1D
{
public:
   enum EvalType
   {
      ChangeOfBasis = 0, // Use change of basis, O(p^2) Evals
      Barycentric   = 1, // Use barycentric Lagrangian interpolation, O(p) Evals
      Positive      = 2, // Fast evaluation of Bernstein polynomials
      NumEvalTypes  = 3  // Keep count of the number of eval types
   };

   class Basis
   {
   private:
      int etype;
      DenseMatrixInverse Ai;
      mutable Vector x, w;

   public:
      /// Create a nodal or positive (Bernstein) basis
      Basis(const int p, const double *nodes, EvalType etype = Barycentric);
      void Eval(const double x, Vector &u) const;
      void Eval(const double x, Vector &u, Vector &d) const;
   };

private:
   typedef std::map< int, Array<double*>* > PointsMap;
   typedef std::map< int, Array<Basis*>* > BasisMap;

   PointsMap points_container;
   BasisMap  bases_container;

   static Array2D<int> binom;

   static void CalcMono(const int p, const double x, double *u);
   static void CalcMono(const int p, const double x, double *u, double *d);

   static void CalcLegendre(const int p, const double x, double *u);
   static void CalcLegendre(const int p, const double x, double *u, double *d);

   static void CalcChebyshev(const int p, const double x, double *u);
   static void CalcChebyshev(const int p, const double x, double *u, double *d);
   static void CalcChebyshev(const int p, const double x, double *u, double *d,
                             double *dd);

   QuadratureFunctions1D quad_func;

public:
   Poly_1D() { }

   /** @brief Get a pointer to an array containing the binomial coefficients "p
       choose k" for k=0,...,p for the given p. */
   static const int *Binom(const int p);

   /** @brief Get the coordinates of the points of the given BasisType,
       @a btype.

       @param[in] p      The polynomial degree; the number of points is `p+1`.
       @param[in] btype  The BasisType.

       @return A pointer to an array containing the `p+1` coordinates of the
               points. Returns NULL if the BasisType has no associated set of
               points. */
   const double *GetPoints(const int p, const int btype);

   /// Get coordinates of an open (GaussLegendre) set of points if degree @a p
   const double *OpenPoints(const int p,
                            const int btype = BasisType::GaussLegendre)
   { return GetPoints(p, btype); }

   /// Get coordinates of a closed (GaussLegendre) set of points if degree @a p
   const double *ClosedPoints(const int p,
                              const int btype = BasisType::GaussLobatto)
   { return GetPoints(p, btype); }

   /** @brief Get a Poly_1D::Basis object of the given degree and BasisType,
       @a btype.

       @param[in] p      The polynomial degree of the basis.
       @param[in] btype  The BasisType.

       @return A reference to an object of type Poly_1D::Basis that represents
               the requested basis type. */
   Basis &GetBasis(const int p, const int btype);

   /** @brief Evaluate the values of a hierarchical 1D basis at point x
       hierarchical = k-th basis function is degree k polynomial */
   static void CalcBasis(const int p, const double x, double *u)
   // { CalcMono(p, x, u); }
   // Bernstein basis is not hierarchical --> does not work for triangles
   //  and tetrahedra
   // { CalcBernstein(p, x, u); }
   // { CalcLegendre(p, x, u); }
   { CalcChebyshev(p, x, u); }

   /// Evaluate the values and derivatives of a hierarchical 1D basis at point @a x
   static void CalcBasis(const int p, const double x, double *u, double *d)
   // { CalcMono(p, x, u, d); }
   // { CalcBernstein(p, x, u, d); }
   // { CalcLegendre(p, x, u, d); }
   { CalcChebyshev(p, x, u, d); }

   /// Evaluate the values, derivatives and second derivatives of a hierarchical 1D basis at point x
   static void CalcBasis(const int p, const double x, double *u, double *d,
                         double *dd)
   // { CalcMono(p, x, u, d); }
   // { CalcBernstein(p, x, u, d); }
   // { CalcLegendre(p, x, u, d); }
   { CalcChebyshev(p, x, u, d, dd); }

   /// Evaluate a representation of a Delta function at point x
   static double CalcDelta(const int p, const double x)
   { return pow(x, (double) p); }

   /** @brief Compute the points for the Chebyshev polynomials of order @a p
       and place them in the already allocated @a x array. */
   static void ChebyshevPoints(const int p, double *x);

   /** @brief Compute the @a p terms in the expansion of the binomial (x + y)^p
       and store them in the allready allocated @a u array. */
   static void CalcBinomTerms(const int p, const double x, const double y,
                              double *u);
   /** @brief Compute the terms in the expansion of the binomial (x + y)^p and 
       their derivatives with respect to x assuming that dy/dx = -1.  Store the
       results in the already allocated @a u and @a d arrays.*/
   static void CalcBinomTerms(const int p, const double x, const double y,
                              double *u, double *d);
   /** @brief Compute the derivatives (w.r.t. x) of the terms in the expansion 
       of the binomial (x + y)^p assuming that dy/dx = -1.  Store the results
       in the already allocated @a d array.*/
   static void CalcDBinomTerms(const int p, const double x, const double y,
                               double *d);

   /** @brief Compute the values of the Bernstein basis functions of order
       @a p at coordinate @a x and store the results in the already allocated
       @a u array. */
   static void CalcBernstein(const int p, const double x, double *u)
   { CalcBinomTerms(p, x, 1. - x, u); }

   /** @brief Compute the values and derivatives of the Bernstein basis functions 
       of order @a p at coordinate @a x and store the results in the already allocated
       @a u and @a d arrays. */   
   static void CalcBernstein(const int p, const double x, double *u, double *d)
   { CalcBinomTerms(p, x, 1. - x, u, d); }

   ~Poly_1D();
};

extern Poly_1D poly1d;


/// An element defined as an ND tensor product of 1D elements on a segement, square, or cube
class TensorBasisElement
{
protected:
   int b_type;
   Array<int> dof_map;
   Poly_1D::Basis &basis1d;

public:
   enum DofMapType
   {
      L2_DOF_MAP = 0,
      H1_DOF_MAP = 1
   };

   TensorBasisElement(const int dims, const int p, const int btype,
                      const DofMapType dmtype);

   int GetBasisType() const { return b_type; }

   const Poly_1D::Basis& GetBasis1D() const { return basis1d; }

   /** @brief Get an Array<int> that maps lexicographically ordered indices to
       the indices of the respective nodes/dofs/basis functions. If the dofs are
       ordered lexicographically, i.e. the mapping is identity, the returned
       Array will be empty. */
   const Array<int> &GetDofMap() const { return dof_map; }

   static Geometry::Type GetTensorProductGeometry(int dim)
   {
      switch (dim)
      {
         case 1: return Geometry::SEGMENT;
         case 2: return Geometry::SQUARE;
         case 3: return Geometry::CUBE;
         default:
            MFEM_ABORT("invalid dimension: " << dim);
            return Geometry::INVALID;
      }
   }

   /// Return @a base raised to the power @a dim.
   static int Pow(int base, int dim)
   {
      switch (dim)
      {
         case 1: return base;
         case 2: return base*base;
         case 3: return base*base*base;
         default: MFEM_ABORT("invalid dimension: " << dim); return -1;
      }
   }
};

class NodalTensorFiniteElement : public NodalFiniteElement,
   public TensorBasisElement
{
public:
   NodalTensorFiniteElement(const int dims, const int p, const int btype,
                            const DofMapType dmtype);
};

class PositiveTensorFiniteElement : public PositiveFiniteElement,
   public TensorBasisElement
{
public:
   PositiveTensorFiniteElement(const int dims, const int p,
                               const DofMapType dmtype);
};

/// Arbitrary H1 elements in 1D
class H1_SegmentElement : public NodalTensorFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, dshape_x;
#endif

public:
   /// Construct the FiniteElement of order @a p and BasisType @a btype
   H1_SegmentElement(const int p, const int btype = BasisType::GaussLobatto);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const;
};


/// Arbitrary H1 elements in 2D on a square
class H1_QuadrilateralElement : public NodalTensorFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, dshape_x, dshape_y;
#endif

public:
   /// Construct the FiniteElement of order @a p and BasisType @a btype
   H1_QuadrilateralElement(const int p,
                           const int btype = BasisType::GaussLobatto);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const;
};


/// Arbitrary H1 elements in 3D on a cube
class H1_HexahedronElement : public NodalTensorFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, shape_z, dshape_x, dshape_y, dshape_z;
#endif

public:
   /// Construct the FiniteElement of order @a p and BasisType @a btype
   H1_HexahedronElement(const int p, const int btype = BasisType::GaussLobatto);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const;
};

/// Arbitrary order H1 elements in 1D utilizing the Bernstein basis
class H1Pos_SegmentElement : public PositiveTensorFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   // This is to share scratch space between invocations, which helps
   // speed things up, but with OpenMP, we need one copy per thread.
   // Right now, we solve this by allocating this space within each function
   // call every time we call it.  Alternatively, we should do some sort
   // thread private thing.  Brunner, Jan 2014
   mutable Vector shape_x, dshape_x;
#endif

public:
   /// Construct the FiniteElement of order @a p
   H1Pos_SegmentElement(const int p);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const;
};


/// Arbitrary order H1 elements in 2D utilizing the Bernstein basis on a square
class H1Pos_QuadrilateralElement : public PositiveTensorFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   // See comment in H1Pos_SegmentElement
   mutable Vector shape_x, shape_y, dshape_x, dshape_y;
#endif

public:
   /// Construct the FiniteElement of order @a p
   H1Pos_QuadrilateralElement(const int p);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const;
};


/// Arbitrary order H1 elements in 3D utilizing the Bernstein basis on a cube
class H1Pos_HexahedronElement : public PositiveTensorFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   // See comment in H1Pos_SegementElement.
   mutable Vector shape_x, shape_y, shape_z, dshape_x, dshape_y, dshape_z;
#endif

public:
   /// Construct the FiniteElement of order @a p
   H1Pos_HexahedronElement(const int p);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const;
};


/// Arbitrary order H1 elements in 2D on a tiangle
class H1_TriangleElement : public NodalFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, shape_l, dshape_x, dshape_y, dshape_l, u;
   mutable Vector ddshape_x, ddshape_y, ddshape_l;
   mutable DenseMatrix du, ddu;
#endif
   DenseMatrixInverse Ti;

public:
   /// Construct the FiniteElement of order @a p and BasisType @a btype
   H1_TriangleElement(const int p, const int btype = BasisType::GaussLobatto);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void CalcHessian(const IntegrationPoint &ip,
                            DenseMatrix &ddshape) const;
};


/// Arbitrary order H1 elements in 3D  on a tetrahedron
class H1_TetrahedronElement : public NodalFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, shape_z, shape_l;
   mutable Vector dshape_x, dshape_y, dshape_z, dshape_l, u;
   mutable Vector ddshape_x, ddshape_y, ddshape_z, ddshape_l;
   mutable DenseMatrix du, ddu;
#endif
   DenseMatrixInverse Ti;

public:
   /// Construct the FiniteElement of order @a p and BasisType @a btype
   H1_TetrahedronElement(const int p,
                         const int btype = BasisType::GaussLobatto);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void CalcHessian(const IntegrationPoint &ip,
                            DenseMatrix &ddshape) const;
};


/// Arbitrary order H1 elements in 2D utilizing the Bernstein basis on a triangle
class H1Pos_TriangleElement : public PositiveFiniteElement
{
protected:
#ifndef MFEM_THREAD_SAFE
   mutable Vector m_shape, dshape_1d;
   mutable DenseMatrix m_dshape;
#endif
   Array<int> dof_map;

public:
   /// Construct the FiniteElement of order @a p
   H1Pos_TriangleElement(const int p);

   // The size of shape is (p+1)(p+2)/2 (dof).
   static void CalcShape(const int p, const double x, const double y,
                         double *shape);

   // The size of dshape_1d is p+1; the size of dshape is (dof x dim).
   static void CalcDShape(const int p, const double x, const double y,
                          double *dshape_1d, double *dshape);

   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};


/// Arbitrary order H1 elements in 3D utilizing the Bernstein basis on a tetrahedron
class H1Pos_TetrahedronElement : public PositiveFiniteElement
{
protected:
#ifndef MFEM_THREAD_SAFE
   mutable Vector m_shape, dshape_1d;
   mutable DenseMatrix m_dshape;
#endif
   Array<int> dof_map;

public:
   /// Construct the FiniteElement of order @a p
   H1Pos_TetrahedronElement(const int p);

   // The size of shape is (p+1)(p+2)(p+3)/6 (dof).
   static void CalcShape(const int p, const double x, const double y,
                         const double z, double *shape);

   // The size of dshape_1d is p+1; the size of dshape is (dof x dim).
   static void CalcDShape(const int p, const double x, const double y,
                          const double z, double *dshape_1d, double *dshape);

   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};


/// Arbitrary order H1 elements in 3D on a wedge
class H1_WedgeElement : public NodalFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector t_shape, s_shape;
   mutable DenseMatrix t_dshape, s_dshape;
#endif
   Array<int> t_dof, s_dof;

   H1_TriangleElement TriangleFE;
   H1_SegmentElement  SegmentFE;

public:
   /// Construct the FiniteElement of order @a p and BasisType @a btype
   H1_WedgeElement(const int p,
                   const int btype = BasisType::GaussLobatto);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};

/// Class for linear FE on wedge
class BiLinear3DFiniteElement : public H1_WedgeElement
{
public:
   /// Construct a linear FE on wedge
   BiLinear3DFiniteElement() : H1_WedgeElement(1) {}
};

/// Class for quadratic FE on wedge
class BiQuadratic3DFiniteElement : public H1_WedgeElement
{
public:
   /// Construct a quadratic FE on wedge
   BiQuadratic3DFiniteElement() : H1_WedgeElement(2) {}
};

/// Class for cubic FE on wedge
class BiCubic3DFiniteElement : public H1_WedgeElement
{
public:
   /// Construct a cubic FE on wedge
   BiCubic3DFiniteElement() : H1_WedgeElement(3) {}
};

/// Arbitrary order H1 elements in 3D utilizing the Bernstein basis on a wedge
class H1Pos_WedgeElement : public PositiveFiniteElement
{
protected:
#ifndef MFEM_THREAD_SAFE
   mutable Vector t_shape, s_shape;
   mutable DenseMatrix t_dshape, s_dshape;
#endif
   Array<int> t_dof, s_dof;

   H1Pos_TriangleElement TriangleFE;
   H1Pos_SegmentElement  SegmentFE;

public:
   /// Construct the FiniteElement of order @a p
   H1Pos_WedgeElement(const int p);

   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};


/// Arbitrary L2 elements in 1D on a segment
class L2_SegmentElement : public NodalTensorFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, dshape_x;
#endif

public:
   /// Construct the FiniteElement of order @a p and BasisType @a btype
   L2_SegmentElement(const int p, const int btype = BasisType::GaussLegendre);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const;
};

/// Arbitrary order L2 elements in 1D utilizing the Bernstein basis on a segment
class L2Pos_SegmentElement : public PositiveTensorFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, dshape_x;
#endif

public:
   /// Construct the FiniteElement of order @a p
   L2Pos_SegmentElement(const int p);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const;
};


/// Arbitrary order L2 elements in 2D on a square
class L2_QuadrilateralElement : public NodalTensorFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, dshape_x, dshape_y;
#endif

public:
   /// Construct the FiniteElement of order @a p and BasisType @a btype
   L2_QuadrilateralElement(const int p,
                           const int btype = BasisType::GaussLegendre);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const;
   virtual void ProjectCurl(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &curl) const
   { ProjectCurl_2D(fe, Trans, curl); }
};

/// Arbitrary order L2 elements in 2D utilizing the Bernstein basis on a square
class L2Pos_QuadrilateralElement : public PositiveTensorFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, dshape_x, dshape_y;
#endif

public:
   /// Construct the FiniteElement of order @a p
   L2Pos_QuadrilateralElement(const int p);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const;
};

/// Arbitrary order L2 elements in 3D on a cube
class L2_HexahedronElement : public NodalTensorFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, shape_z, dshape_x, dshape_y, dshape_z;
#endif

public:
   /// Construct the FiniteElement of order @a p and BasisType @a btype
   L2_HexahedronElement(const int p,
                        const int btype = BasisType::GaussLegendre);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const;
};


/// Arbitrary order L2 elements in 3D utilizing the Bernstein basis on a cube
class L2Pos_HexahedronElement : public PositiveTensorFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, shape_z, dshape_x, dshape_y, dshape_z;
#endif

public:
   /// Construct the FiniteElement of order @a p
   L2Pos_HexahedronElement(const int p);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const;
};


/// Arbitrary order L2 elements in 2D on a triangle
class L2_TriangleElement : public NodalFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, shape_l, dshape_x, dshape_y, dshape_l, u;
   mutable DenseMatrix du;
#endif
   DenseMatrixInverse Ti;

public:
   /// Construct the FiniteElement of order @a p and BasisType @a btype
   L2_TriangleElement(const int p,
                      const int btype = BasisType::GaussLegendre);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const;
   virtual void ProjectCurl(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &curl) const
   { ProjectCurl_2D(fe, Trans, curl); }
};

/// Arbitrary order L2 elements in 2D utilizing the Bernstein basis on a triangle
class L2Pos_TriangleElement : public PositiveFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector dshape_1d;
#endif

public:
   /// Construct the FiniteElement of order @a p
   L2Pos_TriangleElement(const int p);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const;
};


/// Arbitrary order L2 elements in 3D on a tetrahedron
class L2_TetrahedronElement : public NodalFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, shape_z, shape_l;
   mutable Vector dshape_x, dshape_y, dshape_z, dshape_l, u;
   mutable DenseMatrix du;
#endif
   DenseMatrixInverse Ti;

public:
   /// Construct the FiniteElement of order @a p and BasisType @a btype
   L2_TetrahedronElement(const int p,
                         const int btype = BasisType::GaussLegendre);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const;
};


/// Arbitrary order L2 elements in 3D utilizing the Bernstein basis on a tetrahedron
class L2Pos_TetrahedronElement : public PositiveFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector dshape_1d;
#endif

public:
   /// Construct the FiniteElement of order @a p
   L2Pos_TetrahedronElement(const int p);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const;
};


/// Arbitrary order L2 elements in 3D on a wedge
class L2_WedgeElement : public NodalFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector t_shape, s_shape;
   mutable DenseMatrix t_dshape, s_dshape;
#endif
   Array<int> t_dof, s_dof;

   L2_TriangleElement TriangleFE;
   L2_SegmentElement  SegmentFE;

public:
   /// Construct the FiniteElement of order @a p and BasisType @a btype
   L2_WedgeElement(const int p,
                   const int btype = BasisType::GaussLegendre);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};

/// A 0th order L2 element on a Wedge
class P0WedgeFiniteElement : public L2_WedgeElement
{
public:
   P0WedgeFiniteElement () : L2_WedgeElement(0) {}
};

/// Arbitrary order L2 elements in 3D utilizing the Bernstein basis on a wedge
class L2Pos_WedgeElement : public PositiveFiniteElement
{
protected:
#ifndef MFEM_THREAD_SAFE
   mutable Vector t_shape, s_shape;
   mutable DenseMatrix t_dshape, s_dshape;
#endif
   Array<int> t_dof, s_dof;

   L2Pos_TriangleElement TriangleFE;
   L2Pos_SegmentElement  SegmentFE;

public:
   /// Construct the FiniteElement of order @a p
   L2Pos_WedgeElement(const int p);

   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};

/// Arbitrary order Raviart-Thomas elements in 2D on a square
class RT_QuadrilateralElement : public VectorFiniteElement
{
private:
   static const double nk[8];

   Poly_1D::Basis &cbasis1d, &obasis1d;
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_cx, shape_ox, shape_cy, shape_oy;
   mutable Vector dshape_cx, dshape_cy;
#endif
   Array<int> dof_map, dof2nk;

public:
   /** @brief Construct the FiniteElement of order @a p and closed and open
       BasisType @a cb_type and @a ob_type */
   RT_QuadrilateralElement(const int p,
                           const int cb_type = BasisType::GaussLobatto,
                           const int ob_type = BasisType::GaussLegendre);
   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;
   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const
   { CalcVShape_RT(Trans, shape); }
   virtual void CalcDivShape(const IntegrationPoint &ip,
                             Vector &divshape) const;
   virtual void GetLocalInterpolation(ElementTransformation &Trans,
                                      DenseMatrix &I) const
   { LocalInterpolation_RT(*this, nk, dof2nk, Trans, I); }
   virtual void GetLocalRestriction(ElementTransformation &Trans,
                                    DenseMatrix &R) const
   { LocalRestriction_RT(nk, dof2nk, Trans, R); }
   virtual void GetTransferMatrix(const FiniteElement &fe,
                                  ElementTransformation &Trans,
                                  DenseMatrix &I) const
   { LocalInterpolation_RT(CheckVectorFE(fe), nk, dof2nk, Trans, I); }
   using FiniteElement::Project;
   virtual void Project(VectorCoefficient &vc,
                        ElementTransformation &Trans, Vector &dofs) const
   { Project_RT(nk, dof2nk, vc, Trans, dofs); }
   virtual void ProjectMatrixCoefficient(
      MatrixCoefficient &mc, ElementTransformation &T, Vector &dofs) const
   { ProjectMatrixCoefficient_RT(nk, dof2nk, mc, T, dofs); }
   virtual void Project(const FiniteElement &fe, ElementTransformation &Trans,
                        DenseMatrix &I) const
   { Project_RT(nk, dof2nk, fe, Trans, I); }
   // Gradient + rotation = Curl: H1 -> H(div)
   virtual void ProjectGrad(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &grad) const
   { ProjectGrad_RT(nk, dof2nk, fe, Trans, grad); }
   // Curl = Gradient + rotation: H1 -> H(div)
   virtual void ProjectCurl(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &curl) const
   { ProjectGrad_RT(nk, dof2nk, fe, Trans, curl); }
};

/// Arbitrary order Raviart-Thomas elements in 3D on a cube
class RT_HexahedronElement : public VectorFiniteElement
{
   static const double nk[18];

   Poly_1D::Basis &cbasis1d, &obasis1d;
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_cx, shape_ox, shape_cy, shape_oy, shape_cz, shape_oz;
   mutable Vector dshape_cx, dshape_cy, dshape_cz;
#endif
   Array<int> dof_map, dof2nk;

public:
   /** @brief Construct the FiniteElement of order @a p and closed and open
       BasisType @a cb_type and @a ob_type */   
   RT_HexahedronElement(const int p,
                        const int cb_type = BasisType::GaussLobatto,
                        const int ob_type = BasisType::GaussLegendre);

   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;
   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const
   { CalcVShape_RT(Trans, shape); }
   virtual void CalcDivShape(const IntegrationPoint &ip,
                             Vector &divshape) const;
   virtual void GetLocalInterpolation(ElementTransformation &Trans,
                                      DenseMatrix &I) const
   { LocalInterpolation_RT(*this, nk, dof2nk, Trans, I); }
   virtual void GetLocalRestriction(ElementTransformation &Trans,
                                    DenseMatrix &R) const
   { LocalRestriction_RT(nk, dof2nk, Trans, R); }
   virtual void GetTransferMatrix(const FiniteElement &fe,
                                  ElementTransformation &Trans,
                                  DenseMatrix &I) const
   { LocalInterpolation_RT(CheckVectorFE(fe), nk, dof2nk, Trans, I); }
   using FiniteElement::Project;
   virtual void Project(VectorCoefficient &vc,
                        ElementTransformation &Trans, Vector &dofs) const
   { Project_RT(nk, dof2nk, vc, Trans, dofs); }
   virtual void ProjectMatrixCoefficient(
      MatrixCoefficient &mc, ElementTransformation &T, Vector &dofs) const
   { ProjectMatrixCoefficient_RT(nk, dof2nk, mc, T, dofs); }
   virtual void Project(const FiniteElement &fe, ElementTransformation &Trans,
                        DenseMatrix &I) const
   { Project_RT(nk, dof2nk, fe, Trans, I); }
   virtual void ProjectCurl(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &curl) const
   { ProjectCurl_RT(nk, dof2nk, fe, Trans, curl); }
};


/// Arbitrary order Raviart-Thomas elements in 2D on a triangle
class RT_TriangleElement : public VectorFiniteElement
{
   static const double nk[6], c;

#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, shape_l;
   mutable Vector dshape_x, dshape_y, dshape_l;
   mutable DenseMatrix u;
   mutable Vector divu;
#endif
   Array<int> dof2nk;
   DenseMatrixInverse Ti;

public:
   /// Construct the FiniteElement of order @a p
   RT_TriangleElement(const int p);
   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;
   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const
   { CalcVShape_RT(Trans, shape); }
   virtual void CalcDivShape(const IntegrationPoint &ip,
                             Vector &divshape) const;
   virtual void GetLocalInterpolation(ElementTransformation &Trans,
                                      DenseMatrix &I) const
   { LocalInterpolation_RT(*this, nk, dof2nk, Trans, I); }
   virtual void GetLocalRestriction(ElementTransformation &Trans,
                                    DenseMatrix &R) const
   { LocalRestriction_RT(nk, dof2nk, Trans, R); }
   virtual void GetTransferMatrix(const FiniteElement &fe,
                                  ElementTransformation &Trans,
                                  DenseMatrix &I) const
   { LocalInterpolation_RT(CheckVectorFE(fe), nk, dof2nk, Trans, I); }
   using FiniteElement::Project;
   virtual void Project(VectorCoefficient &vc,
                        ElementTransformation &Trans, Vector &dofs) const
   { Project_RT(nk, dof2nk, vc, Trans, dofs); }
   virtual void ProjectMatrixCoefficient(
      MatrixCoefficient &mc, ElementTransformation &T, Vector &dofs) const
   { ProjectMatrixCoefficient_RT(nk, dof2nk, mc, T, dofs); }
   virtual void Project(const FiniteElement &fe, ElementTransformation &Trans,
                        DenseMatrix &I) const
   { Project_RT(nk, dof2nk, fe, Trans, I); }
   // Gradient + rotation = Curl: H1 -> H(div)
   virtual void ProjectGrad(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &grad) const
   { ProjectGrad_RT(nk, dof2nk, fe, Trans, grad); }
   // Curl = Gradient + rotation: H1 -> H(div)
   virtual void ProjectCurl(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &curl) const
   { ProjectGrad_RT(nk, dof2nk, fe, Trans, curl); }
};


/// Arbitrary order Raviart-Thomas elements in 3D on a tetrahedron
class RT_TetrahedronElement : public VectorFiniteElement
{
   static const double nk[12], c;

#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, shape_z, shape_l;
   mutable Vector dshape_x, dshape_y, dshape_z, dshape_l;
   mutable DenseMatrix u;
   mutable Vector divu;
#endif
   Array<int> dof2nk;
   DenseMatrixInverse Ti;

public:
   /// Construct the FiniteElement of order @a p
   RT_TetrahedronElement(const int p);
   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;
   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const
   { CalcVShape_RT(Trans, shape); }
   virtual void CalcDivShape(const IntegrationPoint &ip,
                             Vector &divshape) const;
   virtual void GetLocalInterpolation(ElementTransformation &Trans,
                                      DenseMatrix &I) const
   { LocalInterpolation_RT(*this, nk, dof2nk, Trans, I); }
   virtual void GetLocalRestriction(ElementTransformation &Trans,
                                    DenseMatrix &R) const
   { LocalRestriction_RT(nk, dof2nk, Trans, R); }
   virtual void GetTransferMatrix(const FiniteElement &fe,
                                  ElementTransformation &Trans,
                                  DenseMatrix &I) const
   { LocalInterpolation_RT(CheckVectorFE(fe), nk, dof2nk, Trans, I); }
   using FiniteElement::Project;
   virtual void Project(VectorCoefficient &vc,
                        ElementTransformation &Trans, Vector &dofs) const
   { Project_RT(nk, dof2nk, vc, Trans, dofs); }
   virtual void ProjectMatrixCoefficient(
      MatrixCoefficient &mc, ElementTransformation &T, Vector &dofs) const
   { ProjectMatrixCoefficient_RT(nk, dof2nk, mc, T, dofs); }
   virtual void Project(const FiniteElement &fe, ElementTransformation &Trans,
                        DenseMatrix &I) const
   { Project_RT(nk, dof2nk, fe, Trans, I); }
   virtual void ProjectCurl(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &curl) const
   { ProjectCurl_RT(nk, dof2nk, fe, Trans, curl); }
};


/// Arbitrary order Nedelec elements in 3D on a cube
class ND_HexahedronElement : public VectorFiniteElement
{
   static const double tk[18];

   Poly_1D::Basis &cbasis1d, &obasis1d;
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_cx, shape_ox, shape_cy, shape_oy, shape_cz, shape_oz;
   mutable Vector dshape_cx, dshape_cy, dshape_cz;
#endif
   Array<int> dof_map, dof2tk;

public:
   /** @brief Construct the FiniteElement of order @a p and closed and open
    BasisType @a cb_type and @a ob_type */
   ND_HexahedronElement(const int p,
                        const int cb_type = BasisType::GaussLobatto,
                        const int ob_type = BasisType::GaussLegendre);

   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;

   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const
   { CalcVShape_ND(Trans, shape); }

   virtual void CalcCurlShape(const IntegrationPoint &ip,
                              DenseMatrix &curl_shape) const;

   virtual void GetLocalInterpolation(ElementTransformation &Trans,
                                      DenseMatrix &I) const
   { LocalInterpolation_ND(*this, tk, dof2tk, Trans, I); }

   virtual void GetLocalRestriction(ElementTransformation &Trans,
                                    DenseMatrix &R) const
   { LocalRestriction_ND(tk, dof2tk, Trans, R); }

   virtual void GetTransferMatrix(const FiniteElement &fe,
                                  ElementTransformation &Trans,
                                  DenseMatrix &I) const
   { LocalInterpolation_ND(CheckVectorFE(fe), tk, dof2tk, Trans, I); }

   using FiniteElement::Project;

   virtual void Project(VectorCoefficient &vc,
                        ElementTransformation &Trans, Vector &dofs) const
   { Project_ND(tk, dof2tk, vc, Trans, dofs); }

   virtual void ProjectMatrixCoefficient(
      MatrixCoefficient &mc, ElementTransformation &T, Vector &dofs) const
   { ProjectMatrixCoefficient_ND(tk, dof2tk, mc, T, dofs); }

   virtual void Project(const FiniteElement &fe,
                        ElementTransformation &Trans,
                        DenseMatrix &I) const
   { Project_ND(tk, dof2tk, fe, Trans, I); }

   virtual void ProjectGrad(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &grad) const
   { ProjectGrad_ND(tk, dof2tk, fe, Trans, grad); }

   virtual void ProjectCurl(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &curl) const
   { ProjectCurl_ND(tk, dof2tk, fe, Trans, curl); }
};


/// Arbitrary order Nedelec elements in 2D on a square
class ND_QuadrilateralElement : public VectorFiniteElement
{
   static const double tk[8];

   Poly_1D::Basis &cbasis1d, &obasis1d;
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_cx, shape_ox, shape_cy, shape_oy;
   mutable Vector dshape_cx, dshape_cy;
#endif
   Array<int> dof_map, dof2tk;

public:
   /** @brief Construct the FiniteElement of order @a p and closed and open
       BasisType @a cb_type and @a ob_type */   
   ND_QuadrilateralElement(const int p,
                           const int cb_type = BasisType::GaussLobatto,
                           const int ob_type = BasisType::GaussLegendre);
   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;
   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const
   { CalcVShape_ND(Trans, shape); }
   virtual void CalcCurlShape(const IntegrationPoint &ip,
                              DenseMatrix &curl_shape) const;
   virtual void GetLocalInterpolation(ElementTransformation &Trans,
                                      DenseMatrix &I) const
   { LocalInterpolation_ND(*this, tk, dof2tk, Trans, I); }
   virtual void GetLocalRestriction(ElementTransformation &Trans,
                                    DenseMatrix &R) const
   { LocalRestriction_ND(tk, dof2tk, Trans, R); }
   virtual void GetTransferMatrix(const FiniteElement &fe,
                                  ElementTransformation &Trans,
                                  DenseMatrix &I) const
   { LocalInterpolation_ND(CheckVectorFE(fe), tk, dof2tk, Trans, I); }
   using FiniteElement::Project;
   virtual void Project(VectorCoefficient &vc,
                        ElementTransformation &Trans, Vector &dofs) const
   { Project_ND(tk, dof2tk, vc, Trans, dofs); }
   virtual void ProjectMatrixCoefficient(
      MatrixCoefficient &mc, ElementTransformation &T, Vector &dofs) const
   { ProjectMatrixCoefficient_ND(tk, dof2tk, mc, T, dofs); }
   virtual void Project(const FiniteElement &fe,
                        ElementTransformation &Trans,
                        DenseMatrix &I) const
   { Project_ND(tk, dof2tk, fe, Trans, I); }
   virtual void ProjectGrad(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &grad) const
   { ProjectGrad_ND(tk, dof2tk, fe, Trans, grad); }
};


/// Arbitrary order Nedelec elements in 3D on a tetrahedron
class ND_TetrahedronElement : public VectorFiniteElement
{
   static const double tk[18], c;

#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, shape_z, shape_l;
   mutable Vector dshape_x, dshape_y, dshape_z, dshape_l;
   mutable DenseMatrix u;
#endif
   Array<int> dof2tk;
   DenseMatrixInverse Ti;

public:
   /// Construct the FiniteElement of order @a p
   ND_TetrahedronElement(const int p);
   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;
   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const
   { CalcVShape_ND(Trans, shape); }
   virtual void CalcCurlShape(const IntegrationPoint &ip,
                              DenseMatrix &curl_shape) const;
   virtual void GetLocalInterpolation(ElementTransformation &Trans,
                                      DenseMatrix &I) const
   { LocalInterpolation_ND(*this, tk, dof2tk, Trans, I); }
   virtual void GetLocalRestriction(ElementTransformation &Trans,
                                    DenseMatrix &R) const
   { LocalRestriction_ND(tk, dof2tk, Trans, R); }
   virtual void GetTransferMatrix(const FiniteElement &fe,
                                  ElementTransformation &Trans,
                                  DenseMatrix &I) const
   { LocalInterpolation_ND(CheckVectorFE(fe), tk, dof2tk, Trans, I); }
   using FiniteElement::Project;
   virtual void Project(VectorCoefficient &vc,
                        ElementTransformation &Trans, Vector &dofs) const
   { Project_ND(tk, dof2tk, vc, Trans, dofs); }
   virtual void ProjectMatrixCoefficient(
      MatrixCoefficient &mc, ElementTransformation &T, Vector &dofs) const
   { ProjectMatrixCoefficient_ND(tk, dof2tk, mc, T, dofs); }
   virtual void Project(const FiniteElement &fe,
                        ElementTransformation &Trans,
                        DenseMatrix &I) const
   { Project_ND(tk, dof2tk, fe, Trans, I); }
   virtual void ProjectGrad(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &grad) const
   { ProjectGrad_ND(tk, dof2tk, fe, Trans, grad); }

   virtual void ProjectCurl(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &curl) const
   { ProjectCurl_ND(tk, dof2tk, fe, Trans, curl); }
};

/// Arbitrary order Nedelec elements in 2D on a triangle
class ND_TriangleElement : public VectorFiniteElement
{
   static const double tk[8], c;

#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, shape_l;
   mutable Vector dshape_x, dshape_y, dshape_l;
   mutable DenseMatrix u;
   mutable Vector curlu;
#endif
   Array<int> dof2tk;
   DenseMatrixInverse Ti;

public:
   /// Construct the FiniteElement of order @a p
   ND_TriangleElement(const int p);
   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;
   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const
   { CalcVShape_ND(Trans, shape); }
   virtual void CalcCurlShape(const IntegrationPoint &ip,
                              DenseMatrix &curl_shape) const;
   virtual void GetLocalInterpolation(ElementTransformation &Trans,
                                      DenseMatrix &I) const
   { LocalInterpolation_ND(*this, tk, dof2tk, Trans, I); }
   virtual void GetLocalRestriction(ElementTransformation &Trans,
                                    DenseMatrix &R) const
   { LocalRestriction_ND(tk, dof2tk, Trans, R); }
   virtual void GetTransferMatrix(const FiniteElement &fe,
                                  ElementTransformation &Trans,
                                  DenseMatrix &I) const
   { LocalInterpolation_ND(CheckVectorFE(fe), tk, dof2tk, Trans, I); }
   using FiniteElement::Project;
   virtual void Project(VectorCoefficient &vc,
                        ElementTransformation &Trans, Vector &dofs) const
   { Project_ND(tk, dof2tk, vc, Trans, dofs); }
   virtual void ProjectMatrixCoefficient(
      MatrixCoefficient &mc, ElementTransformation &T, Vector &dofs) const
   { ProjectMatrixCoefficient_ND(tk, dof2tk, mc, T, dofs); }
   virtual void Project(const FiniteElement &fe,
                        ElementTransformation &Trans,
                        DenseMatrix &I) const
   { Project_ND(tk, dof2tk, fe, Trans, I); }
   virtual void ProjectGrad(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &grad) const
   { ProjectGrad_ND(tk, dof2tk, fe, Trans, grad); }
};


/// Arbitrary order Nedelec elements in 1D on a segment
class ND_SegmentElement : public VectorFiniteElement
{
   static const double tk[1];

   Poly_1D::Basis &obasis1d;
   Array<int> dof2tk;

public:
   /** @brief Construct the FiniteElement of order @a p and open
       BasisType @a ob_type */   
   ND_SegmentElement(const int p, const int ob_type = BasisType::GaussLegendre);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const
   { obasis1d.Eval(ip.x, shape); }
   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;
   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const
   { CalcVShape_ND(Trans, shape); }
   // virtual void CalcCurlShape(const IntegrationPoint &ip,
   //                            DenseMatrix &curl_shape) const;
   virtual void GetLocalInterpolation(ElementTransformation &Trans,
                                      DenseMatrix &I) const
   { LocalInterpolation_ND(*this, tk, dof2tk, Trans, I); }
   virtual void GetLocalRestriction(ElementTransformation &Trans,
                                    DenseMatrix &R) const
   { LocalRestriction_ND(tk, dof2tk, Trans, R); }
   virtual void GetTransferMatrix(const FiniteElement &fe,
                                  ElementTransformation &Trans,
                                  DenseMatrix &I) const
   { LocalInterpolation_ND(CheckVectorFE(fe), tk, dof2tk, Trans, I); }
   using FiniteElement::Project;
   virtual void Project(VectorCoefficient &vc,
                        ElementTransformation &Trans, Vector &dofs) const
   { Project_ND(tk, dof2tk, vc, Trans, dofs); }
   virtual void ProjectMatrixCoefficient(
      MatrixCoefficient &mc, ElementTransformation &T, Vector &dofs) const
   { ProjectMatrixCoefficient_ND(tk, dof2tk, mc, T, dofs); }
   virtual void Project(const FiniteElement &fe,
                        ElementTransformation &Trans,
                        DenseMatrix &I) const
   { Project_ND(tk, dof2tk, fe, Trans, I); }
   virtual void ProjectGrad(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &grad) const
   { ProjectGrad_ND(tk, dof2tk, fe, Trans, grad); }
};


/// An arbitrary order and dimension NURBS element
class NURBSFiniteElement : public ScalarFiniteElement
{
protected:
   mutable Array <const KnotVector*> kv;
   mutable const int *ijk;
   mutable int patch, elem;
   mutable Vector weights;

public:
   /** @brief Construct FiniteElement with given
       @param D    Reference space dimension
       @param G    Geometry type (of type Geometry::Type)
       @param Do   Number of degrees of freedom in the FiniteElement
       @param O    Order/degree of the FiniteElement
       @param F    FunctionSpace type of the FiniteElement
    */   
   NURBSFiniteElement(int D, Geometry::Type G, int Do, int O, int F)
      : ScalarFiniteElement(D, G, Do, O, F)
   {
      ijk = NULL;
      patch = elem = -1;
      kv.SetSize(Dim);
      weights.SetSize(Dof);
      weights = 1.0;
   }

   void                 Reset      ()         const { patch = elem = -1; }
   void                 SetIJK     (const int *IJK) const { ijk = IJK; }
   int                  GetPatch   ()         const { return patch; }
   void                 SetPatch   (int p)    const { patch = p; }
   int                  GetElement ()         const { return elem; }
   void                 SetElement (int e)    const { elem = e; }
   Array <const KnotVector*> &KnotVectors()   const { return kv; }
   Vector              &Weights    ()         const { return weights; }
   /// Update the NURBSFiniteElement according to the currently set knot vectors
   virtual void         SetOrder   ()         const { }
};


/// An arbitrary order 1D NURBS element on a segment
class NURBS1DFiniteElement : public NURBSFiniteElement
{
protected:
   mutable Vector shape_x;

public:
   /// Construct the FiniteElement of order @a p
   NURBS1DFiniteElement(int p)
      : NURBSFiniteElement(1, Geometry::SEGMENT, p + 1, p, FunctionSpace::Qk),
        shape_x(p + 1) { }

   virtual void SetOrder() const;
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};

/// An arbitrary order 2D NURBS element on a square
class NURBS2DFiniteElement : public NURBSFiniteElement
{
protected:
   mutable Vector u, shape_x, shape_y, dshape_x, dshape_y;

public:
   /// Construct the FiniteElement of order @a p
   NURBS2DFiniteElement(int p)
      : NURBSFiniteElement(2, Geometry::SQUARE, (p + 1)*(p + 1), p,
                           FunctionSpace::Qk),
        u(Dof), shape_x(p + 1), shape_y(p + 1), dshape_x(p + 1), dshape_y(p + 1)
   { Orders[0] = Orders[1] = p; }

   /// Construct the FiniteElement with x-order @a px and y-order @a py
   NURBS2DFiniteElement(int px, int py)
      : NURBSFiniteElement(2, Geometry::SQUARE, (px + 1)*(py + 1),
                           std::max(px, py), FunctionSpace::Qk),
        u(Dof), shape_x(px + 1), shape_y(py + 1), dshape_x(px + 1),
        dshape_y(py + 1)
   { Orders[0] = px; Orders[1] = py; }

   virtual void SetOrder() const;
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};

/// An arbitrary order 3D NURBS element on a cube
class NURBS3DFiniteElement : public NURBSFiniteElement
{
protected:
   mutable Vector u, shape_x, shape_y, shape_z, dshape_x, dshape_y, dshape_z;

public:
   /// Construct the FiniteElement of order @a p
   NURBS3DFiniteElement(int p)
      : NURBSFiniteElement(3, Geometry::CUBE, (p + 1)*(p + 1)*(p + 1), p,
                           FunctionSpace::Qk),
        u(Dof), shape_x(p + 1), shape_y(p + 1), shape_z(p + 1),
        dshape_x(p + 1), dshape_y(p + 1), dshape_z(p + 1)
   { Orders[0] = Orders[1] = Orders[2] = p; }

   /// Construct the FiniteElement with x-order @a px and y-order @a py and z-order @a pz
   NURBS3DFiniteElement(int px, int py, int pz)
      : NURBSFiniteElement(3, Geometry::CUBE, (px + 1)*(py + 1)*(pz + 1),
                           std::max(std::max(px,py),pz), FunctionSpace::Qk),
        u(Dof), shape_x(px + 1), shape_y(py + 1), shape_z(pz + 1),
        dshape_x(px + 1), dshape_y(py + 1), dshape_z(pz + 1)
   { Orders[0] = px; Orders[1] = py; Orders[2] = pz; }

   virtual void SetOrder() const;
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};

} // namespace mfem

#endif

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

#include "mfem.hpp"
#include "catch.hpp"

using namespace mfem;

TEST_CASE("H1 Segment Finite Element",
          "[H1_SegmentElement]"
          "[NodalFiniteElement]"
          "[ScalarFiniteElement]"
          "[FiniteElement]")
{
   int p = 1;

   H1_SegmentElement fe(p);

   SECTION("Attributes")
   {
      REQUIRE( fe.GetDim()            == 1                     );
      REQUIRE( fe.GetGeomType()       == Geometry::SEGMENT     );
      REQUIRE( fe.GetDof()            == p+1                   );
      REQUIRE( fe.GetOrder()          == p                     );
      REQUIRE( fe.Space()             == (int) FunctionSpace::Pk     );
      REQUIRE( fe.GetRangeType()      == (int) FiniteElement::SCALAR );
      REQUIRE( fe.GetMapType()        == (int) FiniteElement::VALUE  );
      REQUIRE( fe.GetDerivType()      == (int) FiniteElement::GRAD   );
      REQUIRE( fe.GetDerivRangeType() == (int) FiniteElement::VECTOR );
      REQUIRE( fe.GetDerivMapType()   == (int) FiniteElement::H_CURL );
   }
}

TEST_CASE("H1 Triangle Finite Element",
          "[H1_TriangleElement]"
          "[NodalFiniteElement]"
          "[ScalarFiniteElement]"
          "[FiniteElement]")
{
   for (int p = 1; p<=2; p++)
   {
      H1_TriangleElement fe(p);

      if (p == 1)
      {
         SECTION("Attributes")
         {
            REQUIRE( fe.GetDim()            == 2                           );
            REQUIRE( fe.GetGeomType()       == Geometry::TRIANGLE          );
            REQUIRE( fe.Space()             == (int) FunctionSpace::Pk     );
            REQUIRE( fe.GetRangeType()      == (int) FiniteElement::SCALAR );
            REQUIRE( fe.GetMapType()        == (int) FiniteElement::VALUE  );
            REQUIRE( fe.GetDerivType()      == (int) FiniteElement::GRAD   );
            REQUIRE( fe.GetDerivRangeType() == (int) FiniteElement::VECTOR );
            REQUIRE( fe.GetDerivMapType()   == (int) FiniteElement::H_CURL );
         }
      }
      SECTION("Sizes for p = " + std::to_string(p))
      {
         REQUIRE( fe.GetDof()   == (p+1)*(p+2)/2 );
         REQUIRE( fe.GetOrder() == p             );
      }
   }
}

TEST_CASE("H1 Quadrilateral Finite Element",
          "[H1_QuadrilateralElement]"
          "[NodalFiniteElement]"
          "[ScalarFiniteElement]"
          "[FiniteElement]")
{
   for (int p = 1; p<=2; p++)
   {
      H1_QuadrilateralElement fe(p);

      if (p == 1)
      {
         SECTION("Attributes")
         {
            REQUIRE( fe.GetDim()            == 2                           );
            REQUIRE( fe.GetGeomType()       == Geometry::SQUARE            );
            REQUIRE( fe.Space()             == (int) FunctionSpace::Qk     );
            REQUIRE( fe.GetRangeType()      == (int) FiniteElement::SCALAR );
            REQUIRE( fe.GetMapType()        == (int) FiniteElement::VALUE  );
            REQUIRE( fe.GetDerivType()      == (int) FiniteElement::GRAD   );
            REQUIRE( fe.GetDerivRangeType() == (int) FiniteElement::VECTOR );
            REQUIRE( fe.GetDerivMapType()   == (int) FiniteElement::H_CURL );
         }
      }
      SECTION("Sizes for p = " + std::to_string(p))
      {
         REQUIRE( fe.GetDof()   == (int)pow(p+1,2) );
         REQUIRE( fe.GetOrder() == p               );
      }
   }
}

TEST_CASE("H1 Tetrahedron Finite Element",
          "[H1_TetrahedronElement]"
          "[NodalFiniteElement]"
          "[ScalarFiniteElement]"
          "[FiniteElement]")
{
   for (int p = 1; p<=3; p++)
   {
      H1_TetrahedronElement fe(p);

      if (p == 1)
      {
         SECTION("Attributes")
         {
            REQUIRE( fe.GetDim()            == 3                           );
            REQUIRE( fe.GetGeomType()       == Geometry::TETRAHEDRON       );
            REQUIRE( fe.Space()             == (int) FunctionSpace::Pk     );
            REQUIRE( fe.GetRangeType()      == (int) FiniteElement::SCALAR );
            REQUIRE( fe.GetMapType()        == (int) FiniteElement::VALUE  );
            REQUIRE( fe.GetDerivType()      == (int) FiniteElement::GRAD   );
            REQUIRE( fe.GetDerivRangeType() == (int) FiniteElement::VECTOR );
            REQUIRE( fe.GetDerivMapType()   == (int) FiniteElement::H_CURL );
         }
      }
      SECTION("Sizes for p = " + std::to_string(p))
      {
         REQUIRE( fe.GetDof()   == (p+1)*(p+2)*(p+3)/6 );
         REQUIRE( fe.GetOrder() == p                   );
      }
   }
}

TEST_CASE("H1 Hexahedron Finite Element",
          "[H1_HexahedronElement]"
          "[NodalFiniteElement]"
          "[ScalarFiniteElement]"
          "[FiniteElement]")
{
   for (int p = 1; p<=3; p++)
   {
      H1_HexahedronElement fe(p);

      if (p == 1)
      {
         SECTION("Attributes")
         {
            REQUIRE( fe.GetDim()            == 3                           );
            REQUIRE( fe.GetGeomType()       == Geometry::CUBE              );
            REQUIRE( fe.Space()             == (int) FunctionSpace::Qk     );
            REQUIRE( fe.GetRangeType()      == (int) FiniteElement::SCALAR );
            REQUIRE( fe.GetMapType()        == (int) FiniteElement::VALUE  );
            REQUIRE( fe.GetDerivType()      == (int) FiniteElement::GRAD   );
            REQUIRE( fe.GetDerivRangeType() == (int) FiniteElement::VECTOR );
            REQUIRE( fe.GetDerivMapType()   == (int) FiniteElement::H_CURL );
         }
      }
      SECTION("Sizes for p = " + std::to_string(p))
      {
         REQUIRE( fe.GetDof()   == (int)pow(p+1,3) );
         REQUIRE( fe.GetOrder() == p               );
      }
   }
}

TEST_CASE("H1 Wedge Finite Element",
          "[H1_WedgeElement]"
          "[NodalFiniteElement]"
          "[ScalarFiniteElement]"
          "[FiniteElement]")
{
   for (int p = 1; p<=3; p++)
   {
      H1_WedgeElement fe(p);

      if (p == 1)
      {
         SECTION("Attributes")
         {
            REQUIRE( fe.GetDim()            == 3                           );
            REQUIRE( fe.GetGeomType()       == Geometry::PRISM             );
            REQUIRE( fe.Space()             == (int) FunctionSpace::Qk     );
            REQUIRE( fe.GetRangeType()      == (int) FiniteElement::SCALAR );
            REQUIRE( fe.GetMapType()        == (int) FiniteElement::VALUE  );
            REQUIRE( fe.GetDerivType()      == (int) FiniteElement::GRAD   );
            REQUIRE( fe.GetDerivRangeType() == (int) FiniteElement::VECTOR );
            REQUIRE( fe.GetDerivMapType()   == (int) FiniteElement::H_CURL );
         }
      }
      SECTION("Sizes for p = " + std::to_string(p))
      {
         REQUIRE( fe.GetDof()   == (p+1)*(p+1)*(p+2)/2 );
         REQUIRE( fe.GetOrder() == p                   );
      }
   }
}

TEST_CASE("Nedelec Segment Finite Element",
          "[ND_SegmentElement]"
          "[VectorFiniteElement]"
          "[FiniteElement]")
{
   int p = 1;

   ND_SegmentElement fe(p);

   SECTION("Attributes")
   {
      REQUIRE( fe.GetDim()            == 1                     );
      REQUIRE( fe.GetGeomType()       == Geometry::SEGMENT     );
      REQUIRE( fe.GetDof()            == p                     );
      REQUIRE( fe.GetOrder()          == p-1                   );
      REQUIRE( fe.Space()             == (int) FunctionSpace::Pk      );
      REQUIRE( fe.GetRangeType()      == (int) FiniteElement::VECTOR  );
      REQUIRE( fe.GetMapType()        == (int) FiniteElement::H_CURL  );
      REQUIRE( fe.GetDerivType()      == (int) FiniteElement::NONE    );
      REQUIRE( fe.GetDerivRangeType() == (int) FiniteElement::SCALAR  );
      REQUIRE( fe.GetDerivMapType()   == (int) FiniteElement::INTEGRAL);
   }
}

TEST_CASE("Nedelec Triangular Finite Element",
          "[ND_TriangleElement]"
          "[VectorFiniteElement]"
          "[FiniteElement]")
{
   for (int p = 1; p<=2; p++)
   {
      ND_TriangleElement fe(p);

      if (p == 1)
      {
         SECTION("Attributes")
         {
            REQUIRE( fe.GetDim()            == 2                            );
            REQUIRE( fe.GetGeomType()       == Geometry::TRIANGLE           );
            REQUIRE( fe.Space()             == (int) FunctionSpace::Pk      );
            REQUIRE( fe.GetRangeType()      == (int) FiniteElement::VECTOR  );
            REQUIRE( fe.GetMapType()        == (int) FiniteElement::H_CURL  );
            REQUIRE( fe.GetDerivType()      == (int) FiniteElement::CURL    );
            REQUIRE( fe.GetDerivRangeType() == (int) FiniteElement::SCALAR  );
            REQUIRE( fe.GetDerivMapType()   == (int) FiniteElement::INTEGRAL);
         }
      }
      SECTION("Sizes for p = " + std::to_string(p))
      {
         REQUIRE( fe.GetDof()   == p*(p+2) );
         REQUIRE( fe.GetOrder() == p       );
      }
   }
}

TEST_CASE("Nedelec Quadrilateral Finite Element",
          "[ND_QuadrilateralElement]"
          "[VectorFiniteElement]"
          "[FiniteElement]")
{
   for (int p = 1; p<=2; p++)
   {
      ND_QuadrilateralElement fe(p);

      if (p == 1)
      {
         SECTION("Attributes")
         {
            REQUIRE( fe.GetDim()            == 2                            );
            REQUIRE( fe.GetGeomType()       == Geometry::SQUARE             );
            REQUIRE( fe.Space()             == (int) FunctionSpace::Qk      );
            REQUIRE( fe.GetRangeType()      == (int) FiniteElement::VECTOR  );
            REQUIRE( fe.GetMapType()        == (int) FiniteElement::H_CURL  );
            REQUIRE( fe.GetDerivType()      == (int) FiniteElement::CURL    );
            REQUIRE( fe.GetDerivRangeType() == (int) FiniteElement::SCALAR  );
            REQUIRE( fe.GetDerivMapType()   == (int) FiniteElement::INTEGRAL);
         }
      }
      SECTION("Sizes for p = " + std::to_string(p))
      {
         REQUIRE( fe.GetDof()   == 2*p*(p+1) );
         REQUIRE( fe.GetOrder() == p         );
      }
   }
}

TEST_CASE("Nedelec Tetrahedron Finite Element",
          "[ND_TetrahedronElement]"
          "[VectorFiniteElement]"
          "[FiniteElement]")
{
   for (int p = 1; p<=3; p++)
   {
      ND_TetrahedronElement fe(p);

      if (p == 1)
      {
         SECTION("Attributes")
         {
            REQUIRE( fe.GetDim()            == 3                           );
            REQUIRE( fe.GetGeomType()       == Geometry::TETRAHEDRON       );
            REQUIRE( fe.Space()             == (int) FunctionSpace::Pk     );
            REQUIRE( fe.GetRangeType()      == (int) FiniteElement::VECTOR );
            REQUIRE( fe.GetMapType()        == (int) FiniteElement::H_CURL );
            REQUIRE( fe.GetDerivType()      == (int) FiniteElement::CURL   );
            REQUIRE( fe.GetDerivRangeType() == (int) FiniteElement::VECTOR );
            REQUIRE( fe.GetDerivMapType()   == (int) FiniteElement::H_DIV  );
         }
      }
      SECTION("Sizes for p = " + std::to_string(p))
      {
         REQUIRE( fe.GetDof()   == p*(p+2)*(p+3)/2 );
         REQUIRE( fe.GetOrder() == p               );
      }
   }
}

TEST_CASE("Nedelec Hexahedron Finite Element",
          "[ND_HexahedronElement]"
          "[VectorFiniteElement]"
          "[FiniteElement]")
{
   for (int p = 1; p<=3; p++)
   {
      ND_HexahedronElement fe(p);

      if (p == 1)
      {
         SECTION("Attributes")
         {
            REQUIRE( fe.GetDim()            == 3                           );
            REQUIRE( fe.GetGeomType()       == Geometry::CUBE              );
            REQUIRE( fe.Space()             == (int) FunctionSpace::Qk     );
            REQUIRE( fe.GetRangeType()      == (int) FiniteElement::VECTOR );
            REQUIRE( fe.GetMapType()        == (int) FiniteElement::H_CURL );
            REQUIRE( fe.GetDerivType()      == (int) FiniteElement::CURL   );
            REQUIRE( fe.GetDerivRangeType() == (int) FiniteElement::VECTOR );
            REQUIRE( fe.GetDerivMapType()   == (int) FiniteElement::H_DIV  );
         }
      }
      SECTION("Sizes for p = " + std::to_string(p))
      {
         REQUIRE( fe.GetDof()   == 3*p*(int)pow(p+1,2) );
         REQUIRE( fe.GetOrder() == p                   );
      }
   }
}

TEST_CASE("Raviart-Thomas Triangular Finite Element",
          "[RT_TRiangleElement]"
          "[VectorFiniteElement]"
          "[FiniteElement]")
{
   for (int p = 1; p<=2; p++)
   {
      RT_TriangleElement fe(p-1);

      if (p == 1)
      {
         SECTION("Attributes")
         {
            REQUIRE( fe.GetDim()            == 2                             );
            REQUIRE( fe.GetGeomType()       == Geometry::TRIANGLE            );
            REQUIRE( fe.Space()             == (int) FunctionSpace::Pk       );
            REQUIRE( fe.GetRangeType()      == (int) FiniteElement::VECTOR   );
            REQUIRE( fe.GetMapType()        == (int) FiniteElement::H_DIV    );
            REQUIRE( fe.GetDerivType()      == (int) FiniteElement::DIV      );
            REQUIRE( fe.GetDerivRangeType() == (int) FiniteElement::SCALAR   );
            REQUIRE( fe.GetDerivMapType()   == (int) FiniteElement::INTEGRAL );
         }
      }
      SECTION("Sizes for p = " + std::to_string(p))
      {
         REQUIRE( fe.GetDof()   == p*(p+2) );
         REQUIRE( fe.GetOrder() == p       );
      }
   }
}

TEST_CASE("Raviart-Thomas Quadrilateral Finite Element",
          "[RT_QuadrilateralElement]"
          "[VectorFiniteElement]"
          "[FiniteElement]")
{
   for (int p = 1; p<=2; p++)
   {
      RT_QuadrilateralElement fe(p-1);

      if (p == 1)
      {
         SECTION("Attributes")
         {
            REQUIRE( fe.GetDim()            == 2                             );
            REQUIRE( fe.GetGeomType()       == Geometry::SQUARE              );
            REQUIRE( fe.Space()             == (int) FunctionSpace::Qk       );
            REQUIRE( fe.GetRangeType()      == (int) FiniteElement::VECTOR   );
            REQUIRE( fe.GetMapType()        == (int) FiniteElement::H_DIV    );
            REQUIRE( fe.GetDerivType()      == (int) FiniteElement::DIV      );
            REQUIRE( fe.GetDerivRangeType() == (int) FiniteElement::SCALAR   );
            REQUIRE( fe.GetDerivMapType()   == (int) FiniteElement::INTEGRAL );
         }
      }
      SECTION("Sizes for p = " + std::to_string(p))
      {
         REQUIRE( fe.GetDof()   == 2*p*(p+1) );
         REQUIRE( fe.GetOrder() == p         );
      }
   }
}

TEST_CASE("Raviart-Thomas Tetrahedron Finite Element",
          "[RT_TetrahedronElement]"
          "[VectorFiniteElement]"
          "[FiniteElement]")
{
   for (int p = 1; p<=3; p++)
   {
      RT_TetrahedronElement fe(p-1);

      if (p == 1)
      {
         SECTION("Attributes")
         {
            REQUIRE( fe.GetDim()            == 3                             );
            REQUIRE( fe.GetGeomType()       == Geometry::TETRAHEDRON         );
            REQUIRE( fe.Space()             == (int) FunctionSpace::Pk       );
            REQUIRE( fe.GetRangeType()      == (int) FiniteElement::VECTOR   );
            REQUIRE( fe.GetMapType()        == (int) FiniteElement::H_DIV    );
            REQUIRE( fe.GetDerivType()      == (int) FiniteElement::DIV      );
            REQUIRE( fe.GetDerivRangeType() == (int) FiniteElement::SCALAR   );
            REQUIRE( fe.GetDerivMapType()   == (int) FiniteElement::INTEGRAL );
         }
      }
      SECTION("Sizes for p = " + std::to_string(p))
      {
         REQUIRE( fe.GetDof()   == p*(p+1)*(p+3)/2 );
         REQUIRE( fe.GetOrder() == p               );
      }
   }
}

TEST_CASE("Raviart-Thomas Hexahedron Finite Element",
          "[RT_HexahedronElement]"
          "[VectorFiniteElement]"
          "[FiniteElement]")
{
   for (int p = 1; p<=3; p++)
   {
      RT_HexahedronElement fe(p-1);

      if (p == 1)
      {
         SECTION("Attributes")
         {
            REQUIRE( fe.GetDim()            == 3                             );
            REQUIRE( fe.GetGeomType()       == Geometry::CUBE                );
            REQUIRE( fe.Space()             == (int) FunctionSpace::Qk       );
            REQUIRE( fe.GetRangeType()      == (int) FiniteElement::VECTOR   );
            REQUIRE( fe.GetMapType()        == (int) FiniteElement::H_DIV    );
            REQUIRE( fe.GetDerivType()      == (int) FiniteElement::DIV      );
            REQUIRE( fe.GetDerivRangeType() == (int) FiniteElement::SCALAR   );
            REQUIRE( fe.GetDerivMapType()   == (int) FiniteElement::INTEGRAL );
         }
      }
      SECTION("Sizes for p = " + std::to_string(p))
      {
         REQUIRE( fe.GetDof()   == 3*(p+1)*(int)pow(p,2) );
         REQUIRE( fe.GetOrder() == p                     );
      }
   }
}

TEST_CASE("L2 Segment Finite Element",
          "[L2_SegmentElement]"
          "[NodalFiniteElement]"
          "[ScalarFiniteElement]"
          "[FiniteElement]")
{
   int p = 1;

   L2_SegmentElement fe(p);

   SECTION("Attributes")
   {
      REQUIRE( fe.GetDim()            == 1                     );
      REQUIRE( fe.GetGeomType()       == Geometry::SEGMENT     );
      REQUIRE( fe.GetDof()            == (p+1)                 );
      REQUIRE( fe.GetOrder()          == p                     );
      REQUIRE( fe.Space()             == (int) FunctionSpace::Pk     );
      REQUIRE( fe.GetRangeType()      == (int) FiniteElement::SCALAR );
      REQUIRE( fe.GetMapType()        == (int) FiniteElement::VALUE  );
      REQUIRE( fe.GetDerivType()      == (int) FiniteElement::GRAD   );
      REQUIRE( fe.GetDerivRangeType() == (int) FiniteElement::VECTOR );
      REQUIRE( fe.GetDerivMapType()   == (int) FiniteElement::H_CURL );
   }
}

TEST_CASE("L2 Triangle Finite Element",
          "[L2_TriangleElement]"
          "[NodalFiniteElement]"
          "[ScalarFiniteElement]"
          "[FiniteElement]")
{
   for (int p = 1; p<=2; p++)
   {
      L2_TriangleElement fe(p);

      if (p == 1)
      {
         SECTION("Attributes")
         {
            REQUIRE( fe.GetDim()            == 2                           );
            REQUIRE( fe.GetGeomType()       == Geometry::TRIANGLE          );
            REQUIRE( fe.Space()             == (int) FunctionSpace::Pk     );
            REQUIRE( fe.GetRangeType()      == (int) FiniteElement::SCALAR );
            REQUIRE( fe.GetMapType()        == (int) FiniteElement::VALUE  );
            REQUIRE( fe.GetDerivType()      == (int) FiniteElement::GRAD   );
            REQUIRE( fe.GetDerivRangeType() == (int) FiniteElement::VECTOR );
            REQUIRE( fe.GetDerivMapType()   == (int) FiniteElement::H_CURL );
         }
      }
      SECTION("Sizes for p = " + std::to_string(p))
      {
         REQUIRE( fe.GetDof()   == (p+1)*(p+2)/2 );
         REQUIRE( fe.GetOrder() == p             );
      }
   }
}

TEST_CASE("L2 Quadrilateral Finite Element",
          "[L2_QuadrilateralElement]"
          "[NodalFiniteElement]"
          "[ScalarFiniteElement]"
          "[FiniteElement]")
{
   for (int p = 1; p<=2; p++)
   {
      L2_QuadrilateralElement fe(p);

      if (p == 1)
      {
         SECTION("Attributes")
         {
            REQUIRE( fe.GetDim()            == 2                           );
            REQUIRE( fe.GetGeomType()       == Geometry::SQUARE            );
            REQUIRE( fe.Space()             == (int) FunctionSpace::Qk     );
            REQUIRE( fe.GetRangeType()      == (int) FiniteElement::SCALAR );
            REQUIRE( fe.GetMapType()        == (int) FiniteElement::VALUE  );
            REQUIRE( fe.GetDerivType()      == (int) FiniteElement::GRAD   );
            REQUIRE( fe.GetDerivRangeType() == (int) FiniteElement::VECTOR );
            REQUIRE( fe.GetDerivMapType()   == (int) FiniteElement::H_CURL );
         }
      }
      SECTION("Sizes for p = " + std::to_string(p))
      {
         REQUIRE( fe.GetDof()   == (int)pow(p+1,2) );
         REQUIRE( fe.GetOrder() == p               );
      }
   }
}

TEST_CASE("L2 Tetrahedron Finite Element",
          "[L2_TetrahedronElement]"
          "[NodalFiniteElement]"
          "[ScalarFiniteElement]"
          "[FiniteElement]")
{
   for (int p = 1; p<=3; p++)
   {
      L2_TetrahedronElement fe(p);

      if (p == 1)
      {
         SECTION("Attributes")
         {
            REQUIRE( fe.GetDim()            == 3                           );
            REQUIRE( fe.GetGeomType()       == Geometry::TETRAHEDRON       );
            REQUIRE( fe.Space()             == (int) FunctionSpace::Pk     );
            REQUIRE( fe.GetRangeType()      == (int) FiniteElement::SCALAR );
            REQUIRE( fe.GetMapType()        == (int) FiniteElement::VALUE  );
            REQUIRE( fe.GetDerivType()      == (int) FiniteElement::GRAD   );
            REQUIRE( fe.GetDerivRangeType() == (int) FiniteElement::VECTOR );
            REQUIRE( fe.GetDerivMapType()   == (int) FiniteElement::H_CURL );
         }
      }
      SECTION("Sizes for p = " + std::to_string(p))
      {
         REQUIRE( fe.GetDof()   == (p+1)*(p+2)*(p+3)/6 );
         REQUIRE( fe.GetOrder() == p                   );
      }
   }
}

TEST_CASE("L2 Hexahedron Finite Element",
          "[L2_HexahedronElement]"
          "[NodalFiniteElement]"
          "[ScalarFiniteElement]"
          "[FiniteElement]")
{
   for (int p = 1; p<=3; p++)
   {
      L2_HexahedronElement fe(p);

      if (p == 1)
      {
         SECTION("Attributes")
         {
            REQUIRE( fe.GetDim()            == 3                           );
            REQUIRE( fe.GetGeomType()       == Geometry::CUBE              );
            REQUIRE( fe.Space()             == (int) FunctionSpace::Qk     );
            REQUIRE( fe.GetRangeType()      == (int) FiniteElement::SCALAR );
            REQUIRE( fe.GetMapType()        == (int) FiniteElement::VALUE  );
            REQUIRE( fe.GetDerivType()      == (int) FiniteElement::GRAD   );
            REQUIRE( fe.GetDerivRangeType() == (int) FiniteElement::VECTOR );
            REQUIRE( fe.GetDerivMapType()   == (int) FiniteElement::H_CURL );
         }
      }
      SECTION("Sizes for p = " + std::to_string(p))
      {
         REQUIRE( fe.GetDof()   == (int)pow(p+1,3) );
         REQUIRE( fe.GetOrder() == p               );
      }
   }
}

TEST_CASE("L2 Wedge Finite Element",
          "[L2_WedgeElement]"
          "[NodalFiniteElement]"
          "[ScalarFiniteElement]"
          "[FiniteElement]")
{
   for (int p = 1; p<=3; p++)
   {
      L2_WedgeElement fe(p);

      if (p == 1)
      {
         SECTION("Attributes")
         {
            REQUIRE( fe.GetDim()            == 3                           );
            REQUIRE( fe.GetGeomType()       == Geometry::PRISM             );
            REQUIRE( fe.Space()             == (int) FunctionSpace::Qk     );
            REQUIRE( fe.GetRangeType()      == (int) FiniteElement::SCALAR );
            REQUIRE( fe.GetMapType()        == (int) FiniteElement::VALUE  );
            REQUIRE( fe.GetDerivType()      == (int) FiniteElement::GRAD   );
            REQUIRE( fe.GetDerivRangeType() == (int) FiniteElement::VECTOR );
            REQUIRE( fe.GetDerivMapType()   == (int) FiniteElement::H_CURL );
         }
      }
      SECTION("Sizes for p = " + std::to_string(p))
      {
         REQUIRE( fe.GetDof()   == (p+1)*(p+1)*(p+2)/2 );
         REQUIRE( fe.GetOrder() == p                   );
      }
   }
}

#include "mfem.hpp"

using namespace mfem;
using namespace std;

class AzimuthalECoefficient : public Coefficient
{
private:
   const GridFunction * vgf;
public:
   AzimuthalECoefficient(const GridFunction * vgf_)
      : Coefficient(), vgf(vgf_) {}
   virtual real_t Eval(ElementTransformation &T,
                       const IntegrationPoint &ip);
};

class ParallelECoefficient : public Coefficient
{
private:
   const GridFunction * vgf;
public:
   ParallelECoefficient(const GridFunction * vgf_)
      : Coefficient(), vgf(vgf_) {}
   virtual real_t Eval(ElementTransformation &T,
                       const IntegrationPoint &ip);
};

class EpsilonMatrixCoefficient : public MatrixArrayCoefficient
{
private:
   Mesh * mesh = nullptr;
   ParMesh * pmesh = nullptr;
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Array<ParGridFunction * > pgfs;
   Array<GridFunctionCoefficient * > gf_cfs;
   GridFunction * vgf = nullptr;
   int dim;
   int sdim;
public:
   EpsilonMatrixCoefficient(const char * filename, Mesh * mesh_, ParMesh * pmesh_,
                            real_t scale = 1.0);

   // Visualize the components of the matrix coefficient
   // in separate GLVis windows for each component
   void VisualizeMatrixCoefficient();
   // Update the Gridfunctions after mesh refinement
   void Update();

   ~EpsilonMatrixCoefficient();

};

class DielectricTensorComponentCoefficient : public Coefficient
{
private:
   real_t delta, a0, a1;
   bool use_imag;
   int row, col;

public:
   /// Constructor
   /// @param delta_    Imaginary scaling factor
   /// @param a0_       Constant in P(r)
   /// @param a1_       Linear term in P(r)
   /// @param row_      row index of the dielectric tensor
   /// @param col_      column index of the dielectric tensor
   /// @param use_imag_ If true, returns imaginary part; otherwise real part
   DielectricTensorComponentCoefficient(real_t delta_, real_t a0_, real_t a1_, int row_, int col_, bool use_imag_ = false);
   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip);

private:
   real_t ComputeRealPart(const Vector &x);
   real_t ComputeImagPart(const Vector &x);
};

void VisualizeMatrixArrayCoefficient(MatrixArrayCoefficient &mc, ParMesh *pmesh, int order, bool paraview = false, const char *name = nullptr);

void ComputeB(const Vector &x, Vector &b);
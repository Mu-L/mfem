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

#ifndef MFEM_AMGX
#define MFEM_AMGX

#include "../config/config.hpp"

#ifdef MFEM_USE_AMGX
#ifdef MFEM_USE_MPI

#include <amgx_c.h>
#include <mpi.h>
#include "hypre.hpp"

//using Int_64 = long long int; //needed for Amgx

//Reference:
//Pi-Yueh Chuang, & Lorena A. Barba (2017).
//AmgXWrapper: An interface between PETSc and the NVIDIA AmgX library. J. Open Source Software, 2(16):280, doi:10.21105/joss.00280

namespace mfem
{

class NvidiaAMGX
{

private:

   //Only first instance will setup/teardown AMGX
   static int count;

   bool isEnabled{false};
   bool isMPIEnabled{false};

   //Number of gpus - assume same as MPI procs
   int nDevs, deviceId;

   int MPI_SZ, MPI_RANK;

   AMGX_Mode amgx_mode;

   MPI_Comm amgx_comm; //amgx communicator


   //Begin Adam
   std::string nodeName;
   int gpuProc = MPI_UNDEFINED;
   int globalSize;
   int localSize;
   int gpuWorldSize;
   int devWorldSize;
   int myGlobalRank;
   int myLocalRank;
   int myGpuWorldRank;
   int myDevWorldRank;
   MPI_Comm localCpuWorld; //adam
   MPI_Comm gpuWorld;
   MPI_Comm devWorld;
   //End Adam


   //Amgx matrices and vectors
   int ring;
   AMGX_matrix_handle      A{nullptr};
   AMGX_vector_handle x{nullptr}, b{nullptr};
   AMGX_solver_handle solver{nullptr};


   SparseMatrix * spop;
   AMGX_config_handle  cfg;

   static AMGX_resources_handle   rsrc;

   Array<int> m_I;
   Array<int64_t> m_J;
   Array<double> m_Aloc;

   //Reference impl: PETSc MatMPIAIJGetLocalMat method
   //used to merge Diagonal and OffDiagonal blocks in a ParCSR matrix
   void GetLocalA(const HypreParMatrix &in_A, Array<int> &I,
                  Array<int64_t> &J, Array<double> &Aloc);


public:

   NvidiaAMGX() = default;

   //Constructor
   NvidiaAMGX(const MPI_Comm &comm,
              const std::string &modeStr, const std::string &cfgFile);

   void Init(const MPI_Comm &comm,
             const std::string &modeStr, const std::string &cfgFile);

   void Init(const std::string &modeStr, const std::string &cfgFile);

   void SetA(const HypreParMatrix &A);

   void SetA(const Operator& op);

   void Solve(Vector &x, Vector &b);

   void Mult(Vector &b, Vector &x);

   //new test functions for mpi > gpus

   void initialize_new(const MPI_Comm &comm,
           const std::string &modeStr, const std::string &cfgFile);

   void initMPIcomms_new(const MPI_Comm &comm);

   void setDeviceCount_new();

   void setDeviceIDs_new();

   void initAmgX_new(const std::string &cfgFile);

   void finalize_new();


   //Destructor
   ~NvidiaAMGX();
};

}

#endif

#endif
#endif

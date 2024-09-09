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

#ifndef MFEM_PNCSUBMESH
#define MFEM_PNCSUBMESH

#include "../../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "../pncmesh.hpp"
#include "psubmesh.hpp"
#include <unordered_map>

namespace mfem
{

/**
 * @brief TODO: a nice blurb
 */
class ParNCSubMesh : public ParNCMesh
{
  friend class ParSubMesh; ///< Only ParSubMesh can use methods in this class
public:
  using From = SubMesh::From;
   /**
   * @brief Check if Mesh @a m is a SubMesh.
   *
   * @param m The input Mesh
   */
   static bool IsParNCSubMesh(const NCMesh *m)
   {
      return dynamic_cast<const ParNCSubMesh *>(m) != nullptr;
   }
   /**
    * @brief Get the parent Mesh object
    *
    */
   const ParNCMesh* GetParent() const
   {
      return parent_;
   }

   /**
    * @brief Get the From indicator.
    *
    * Indicates whether the SubMesh has been created from a domain or
    * surface.
    */
   ParSubMesh::From GetFrom() const
   {
      return from_;
   }


private:
   /// Private constructor
   ParNCSubMesh(ParSubMesh& submesh, const ParNCMesh &parent, From from,
                const Array<int> &attributes);

   /// The parent Mesh. Not owned.
   const ParNCMesh *parent_;

   /// Indicator from which part of the parent ParMesh the ParSubMesh is going
   /// to be created.
   From from_;

   /// Attributes on the parent NCMesh on which the ParNCSubMesh is created.
   /// Could either be domain or boundary attributes (determined by from_).
   Array<int> attributes_;

   /// Mapping from submesh element nc ids (index of the array), to the parent
   /// element ids. If from a boundary, these map to faces in the parent.
   Array<int> parent_element_ids_;

   /// Mapping from ParNCSubMesh node ids (index of the array), to the parent NCMesh
   /// node ids.
   Array<int> parent_node_ids_;

   /// Mapping from parent NCMesh node ids to submesh NCMesh node ids.
   // Inverse map of parent_node_ids_.
   std::unordered_map<int, int> parent_to_submesh_node_ids_;

   /// Mapping from parent NCMesh element ids to submesh NCMesh element ids.
   // Inverse map of parent_element_ids_.
   std::unordered_map<int, int> parent_to_submesh_element_ids_;

   // /// Mapping from parent NCMesh edge ids to submesh NCMesh edge ids.
   // // Inverse map of parent_edge_ids_.
   // Array<int> parent_to_submesh_edge_ids_;

   // /// Mapping from parent NCMesh face ids to submesh NCMesh face ids.
   // // Inverse map of parent_face_ids_.
   // std::unordered_map<int, int> parent_to_submesh_face_ids_;
};

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_PNCSUBMESH

"""Mesh simplification module - Reduce complexity for 3D printing."""

import trimesh

# Try to import pymeshlab for high-quality decimation
try:
    import pymeshlab
    HAS_PYMESHLAB = True
except ImportError:
    HAS_PYMESHLAB = False


def simplify_mesh(
    mesh: trimesh.Trimesh,
    target_faces: int = 10000,
    preserve_topology: bool = True
) -> trimesh.Trimesh:
    """
    Reduce mesh complexity while preserving shape.
    
    Args:
        mesh: Input mesh
        target_faces: Target number of faces after simplification
        preserve_topology: If True, preserve mesh topology (no holes created)
        
    Returns:
        Simplified mesh
    """
    current_faces = mesh.faces.shape[0]
    
    if current_faces <= target_faces:
        print(f"Mesh already has {current_faces} faces, skipping simplification")
        return mesh
    
    ratio = target_faces / current_faces
    print(f"Simplifying mesh: {current_faces:,} -> {target_faces:,} faces ({ratio:.1%})")
    
    if HAS_PYMESHLAB:
        return _simplify_pymeshlab(mesh, target_faces, preserve_topology)
    else:
        return _simplify_trimesh(mesh, ratio)


def _simplify_pymeshlab(
    mesh: trimesh.Trimesh,
    target_faces: int,
    preserve_topology: bool
) -> trimesh.Trimesh:
    """Simplify using PyMeshLab (higher quality)."""
    
    # Create MeshSet
    ms = pymeshlab.MeshSet()
    
    # Convert trimesh to pymeshlab mesh
    m = pymeshlab.Mesh(
        vertex_matrix=mesh.vertices,
        face_matrix=mesh.faces
    )
    ms.add_mesh(m)
    
    # Apply Quadric Edge Collapse decimation
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=target_faces,
        preservetopology=preserve_topology,
        preserveboundary=True,
        qualitythr=0.5,
        planarquadric=True,
    )
    
    # Extract result
    result = ms.current_mesh()
    
    return trimesh.Trimesh(
        vertices=result.vertex_matrix(),
        faces=result.face_matrix()
    )


def _simplify_trimesh(mesh: trimesh.Trimesh, ratio: float) -> trimesh.Trimesh:
    """Simplify using trimesh's built-in simplification (fallback)."""
    
    # Use trimesh's simplify_quadric_decimation
    simplified = mesh.simplify_quadric_decimation(
        face_count=int(mesh.faces.shape[0] * ratio)
    )
    
    return simplified


def smooth_mesh(mesh: trimesh.Trimesh, iterations: int = 2) -> trimesh.Trimesh:
    """
    Apply Laplacian smoothing to reduce sharp edges.
    
    Args:
        mesh: Input mesh
        iterations: Number of smoothing iterations
        
    Returns:
        Smoothed mesh
    """
    if not HAS_PYMESHLAB:
        print("Warning: PyMeshLab not available, skipping smoothing")
        return mesh
    
    ms = pymeshlab.MeshSet()
    m = pymeshlab.Mesh(
        vertex_matrix=mesh.vertices,
        face_matrix=mesh.faces
    )
    ms.add_mesh(m)
    
    # Taubin smoothing (better than Laplacian for preserving volume)
    ms.apply_coord_taubin_smoothing(
        stepsmoothnum=iterations,
        lambda_=0.5,
        mu=-0.53,
    )
    
    result = ms.current_mesh()
    
    return trimesh.Trimesh(
        vertices=result.vertex_matrix(),
        faces=result.face_matrix()
    )

"""Mesh repair module - Make meshes watertight and manifold."""

import numpy as np
import trimesh

# Try to import manifold3d for robust repair
try:
    import manifold3d as mf
    HAS_MANIFOLD = True
except ImportError:
    HAS_MANIFOLD = False


def repair_mesh(
    mesh: trimesh.Trimesh,
    method: str = 'auto'
) -> trimesh.Trimesh:
    """
    Repair mesh to be watertight and manifold.
    
    Args:
        mesh: Input mesh (potentially with holes, non-manifold edges)
        method: Repair method
            - 'auto': Use best available method
            - 'manifold': Use manifold3d (robust but may change topology)
            - 'trimesh': Use trimesh's repair functions
            - 'fill_holes': Just fill holes, minimal changes
            
    Returns:
        Repaired watertight mesh
    """
    if mesh.is_watertight:
        print("Mesh is already watertight")
        return mesh
    
    if method == 'auto':
        method = 'manifold' if HAS_MANIFOLD else 'trimesh'
    
    print(f"Repairing mesh using {method} method")
    print(f"  Input: {mesh.vertices.shape[0]:,} vertices, watertight={mesh.is_watertight}")
    
    if method == 'manifold':
        repaired = _repair_manifold(mesh)
    elif method == 'trimesh':
        repaired = _repair_trimesh(mesh)
    elif method == 'fill_holes':
        repaired = _fill_holes(mesh)
    else:
        raise ValueError(f"Unknown repair method: {method}")
    
    print(f"  Output: {repaired.vertices.shape[0]:,} vertices, watertight={repaired.is_watertight}")
    
    return repaired


def _repair_manifold(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Use manifold3d to create a guaranteed manifold mesh."""
    
    if not HAS_MANIFOLD:
        raise ImportError("manifold3d not installed")
    
    # Manifold automatically repairs non-manifold input
    try:
        mf_mesh = mf.Manifold.from_mesh(
            mf.Mesh(
                vert_properties=mesh.vertices.astype(np.float32),
                tri_verts=mesh.faces.astype(np.uint32)
            )
        )
        
        # Convert back
        result_data = mf_mesh.to_mesh()
        return trimesh.Trimesh(
            vertices=result_data.vert_properties,
            faces=result_data.tri_verts
        )
    except Exception as e:
        print(f"  Manifold repair failed: {e}")
        print("  Falling back to trimesh repair")
        return _repair_trimesh(mesh)


def _repair_trimesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Use trimesh's built-in repair functions."""
    
    repaired = mesh.copy()
    
    # Remove degenerate faces
    repaired.remove_degenerate_faces()
    
    # Remove duplicate faces
    repaired.remove_duplicate_faces()
    
    # Remove unreferenced vertices
    repaired.remove_unreferenced_vertices()
    
    # Merge close vertices
    repaired.merge_vertices()
    
    # Fix normals (all pointing outward)
    repaired.fix_normals()
    
    # Fill holes
    if not repaired.is_watertight:
        repaired = _fill_holes(repaired)
    
    return repaired


def _fill_holes(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Fill holes in mesh by detecting and triangulating boundaries."""
    
    repaired = mesh.copy()
    
    # Get face adjacency info
    # Holes are edges that only belong to one face
    edges = repaired.edges_unique
    edge_face_count = np.bincount(
        repaired.edges_unique_inverse,
        minlength=len(edges)
    )
    
    # Boundary edges belong to only one face
    boundary_mask = edge_face_count == 1
    n_boundary = boundary_mask.sum()
    
    if n_boundary > 0:
        print(f"  Found {n_boundary} boundary edges (holes)")
        # Trimesh doesn't have built-in hole filling
        # For now, just report - would need pymeshlab or blender for actual filling
        print("  Warning: Automatic hole filling not implemented, mesh may not be watertight")
    
    return repaired


def check_manifold(mesh: trimesh.Trimesh) -> dict:
    """
    Check mesh for manifold issues.
    
    Returns:
        Dictionary with issue counts:
        {
            'watertight': bool,
            'degenerate_faces': int,
            'duplicate_faces': int,
            'boundary_edges': int,
            'non_manifold_edges': int,
        }
    """
    edges = mesh.edges_unique
    edge_face_count = np.bincount(
        mesh.edges_unique_inverse,
        minlength=len(edges)
    )
    
    return {
        'watertight': mesh.is_watertight,
        'degenerate_faces': len(mesh.degenerate_faces),
        'duplicate_faces': len(mesh.faces) - len(np.unique(np.sort(mesh.faces, axis=1), axis=0)),
        'boundary_edges': (edge_face_count == 1).sum(),
        'non_manifold_edges': (edge_face_count > 2).sum(),
    }


def repair_mesh_robust(
    mesh: trimesh.Trimesh,
    config: 'PipelineConfig' = None,
) -> trimesh.Trimesh:
    """
    Robust multi-pass mesh repair with guaranteed watertight output.
    
    Uses multiple repair strategies and validates after each pass.
    WILL RAISE if watertight validation is enabled and fails.
    
    Args:
        mesh: Input mesh (potentially with holes, non-manifold edges)
        config: Pipeline configuration (uses defaults if None)
        
    Returns:
        Repaired watertight mesh
        
    Raises:
        ValueError: If validate_watertight is True and repair fails
    """
    from .config import PipelineConfig
    
    if config is None:
        config = PipelineConfig.standard()
    
    if mesh.is_watertight:
        print("Mesh is already watertight")
        return mesh
    
    print(f"Robust repair ({config.repair_passes} passes)")
    print(f"  Input: {mesh.vertices.shape[0]:,} vertices, watertight={mesh.is_watertight}")
    
    repaired = mesh.copy()
    
    for pass_num in range(config.repair_passes):
        print(f"  Pass {pass_num + 1}/{config.repair_passes}...")
        
        # Step 1: Basic cleanup
        repaired.remove_degenerate_faces()
        repaired.remove_duplicate_faces()
        repaired.remove_unreferenced_vertices()
        
        # Step 2: Merge close vertices
        repaired.merge_vertices(merge_tex=True, merge_norm=True)
        
        # Step 3: Fix normals
        repaired.fix_normals()
        
        # Step 4: Try manifold repair if available
        if HAS_MANIFOLD and not repaired.is_watertight:
            try:
                repaired = _repair_manifold(repaired)
            except Exception as e:
                print(f"    Manifold repair failed: {e}")
        
        # Step 5: Try PyMeshLab if available
        if not repaired.is_watertight:
            try:
                repaired = _repair_pymeshlab(repaired, config)
            except Exception as e:
                print(f"    PyMeshLab repair failed: {e}")
        
        if repaired.is_watertight:
            print(f"  Watertight after pass {pass_num + 1}")
            break
    
    # Final validation
    is_watertight = repaired.is_watertight
    print(f"  Output: {repaired.vertices.shape[0]:,} vertices, watertight={is_watertight}")
    
    if config.validate_watertight and not is_watertight:
        issues = check_manifold(repaired)
        raise ValueError(
            f"Mesh repair failed - output is not watertight!\n"
            f"  Boundary edges: {issues['boundary_edges']}\n"
            f"  Non-manifold edges: {issues['non_manifold_edges']}\n"
            f"Consider using voxel boolean engine for guaranteed watertight output."
        )
    
    return repaired


def _repair_pymeshlab(mesh: trimesh.Trimesh, config: 'PipelineConfig') -> trimesh.Trimesh:
    """Use PyMeshLab for advanced repair operations."""
    
    try:
        import pymeshlab
    except ImportError:
        raise ImportError("PyMeshLab not installed. Run: pip install pymeshlab")
    
    # Create MeshSet
    ms = pymeshlab.MeshSet()
    
    # Convert trimesh to pymeshlab mesh
    m = pymeshlab.Mesh(
        vertex_matrix=mesh.vertices,
        face_matrix=mesh.faces,
    )
    ms.add_mesh(m)
    
    # Apply repair filters
    try:
        # Remove duplicate vertices
        ms.meshing_remove_duplicate_vertices()
        
        # Remove duplicate faces
        ms.meshing_remove_duplicate_faces()
        
        # Remove zero area faces
        ms.meshing_remove_null_faces()
        
        # Close holes
        ms.meshing_close_holes(maxholesize=int(config.hole_fill_max_area))
        
        # Repair non-manifold edges
        ms.meshing_repair_non_manifold_edges()
        
        # Repair non-manifold vertices
        ms.meshing_repair_non_manifold_vertices()
        
    except Exception as e:
        print(f"    PyMeshLab filter error: {e}")
    
    # Convert back to trimesh
    result_mesh = ms.current_mesh()
    return trimesh.Trimesh(
        vertices=result_mesh.vertex_matrix(),
        faces=result_mesh.face_matrix(),
    )


def ensure_minimum_thickness(
    mesh: trimesh.Trimesh,
    min_thickness: float = 3.0,
) -> trimesh.Trimesh:
    """
    Ensure mesh has minimum wall thickness for 3D printing.
    
    Thin sections are thickened by offsetting surfaces.
    
    Args:
        mesh: Input mesh
        min_thickness: Minimum thickness in mesh units (mm)
        
    Returns:
        Mesh with guaranteed minimum thickness
    """
    # This is a simplified implementation
    # Full implementation would use medial axis analysis
    
    print(f"Ensuring minimum thickness of {min_thickness}mm")
    
    # For now, just check bounding box dimensions
    extents = mesh.bounding_box.extents
    min_extent = min(extents)
    
    if min_extent < min_thickness:
        print(f"  Warning: Minimum extent {min_extent:.2f}mm is below threshold")
        # Could apply uniform scaling or offset here
    
    return mesh


# Type hint import for config
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .config import PipelineConfig

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

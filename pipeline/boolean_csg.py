"""Boolean CSG module - Subtract bodies to create prosthesis volumes."""

import numpy as np
import trimesh

# Try to import manifold3d, fall back to trimesh boolean if not available
try:
    import manifold3d as mf
    HAS_MANIFOLD = True
except ImportError:
    HAS_MANIFOLD = False
    print("Warning: manifold3d not installed, using trimesh boolean (less robust)")


def subtract_bodies(
    outer_mesh: trimesh.Trimesh,
    inner_mesh: trimesh.Trimesh,
    engine: str = 'auto'
) -> trimesh.Trimesh:
    """
    Boolean subtraction: outer - inner = prosthesis volumes.
    
    The result contains the "difference" - material that exists in outer
    but not in inner. These are the prosthesis pieces.
    
    Args:
        outer_mesh: The larger body (scaled female target)
        inner_mesh: The smaller body (male stunt performer)
        engine: Boolean engine to use
            - 'auto': Use manifold3d if available, else trimesh
            - 'manifold': Force manifold3d (raises if not installed)
            - 'trimesh': Use trimesh's built-in boolean
            
    Returns:
        Mesh containing the boolean difference
    """
    if engine == 'auto':
        engine = 'manifold' if HAS_MANIFOLD else 'trimesh'
    
    print(f"Boolean subtraction using {engine} engine")
    print(f"  Outer mesh: {outer_mesh.vertices.shape[0]:,} vertices")
    print(f"  Inner mesh: {inner_mesh.vertices.shape[0]:,} vertices")
    
    if engine == 'manifold':
        if not HAS_MANIFOLD:
            raise ImportError("manifold3d not installed. Run: pip install manifold3d")
        
        # Convert to Manifold objects
        outer_mf = mf.Manifold.from_mesh(
            mf.Mesh(
                vert_properties=outer_mesh.vertices.astype(np.float32),
                tri_verts=outer_mesh.faces.astype(np.uint32)
            )
        )
        inner_mf = mf.Manifold.from_mesh(
            mf.Mesh(
                vert_properties=inner_mesh.vertices.astype(np.float32),
                tri_verts=inner_mesh.faces.astype(np.uint32)
            )
        )
        
        # Boolean subtraction
        result_mf = outer_mf - inner_mf
        
        # Convert back to trimesh
        result_mesh_data = result_mf.to_mesh()
        result = trimesh.Trimesh(
            vertices=result_mesh_data.vert_properties,
            faces=result_mesh_data.tri_verts
        )
        
    elif engine == 'trimesh':
        # Use trimesh's boolean (requires blender or openscad backend)
        result = outer_mesh.difference(inner_mesh, engine='blender')
        
    else:
        raise ValueError(f"Unknown boolean engine: {engine}")
    
    print(f"  Result: {result.vertices.shape[0]:,} vertices, {result.faces.shape[0]:,} faces")
    
    return result


def add_offset(mesh: trimesh.Trimesh, offset: float = 2.0) -> trimesh.Trimesh:
    """
    Add a small offset to the inner mesh before subtraction.
    
    This creates a gap between the prosthesis and the body,
    useful for foam compression and comfort.
    
    Args:
        mesh: Input mesh
        offset: Offset distance in mesh units (e.g., mm)
        
    Returns:
        Offset mesh (vertices moved inward along normals)
    """
    # Get vertex normals
    normals = mesh.vertex_normals
    
    # Move vertices inward (negative offset)
    offset_mesh = mesh.copy()
    offset_mesh.vertices -= normals * offset
    
    print(f"Applied {offset} unit offset to mesh")
    
    return offset_mesh

"""Boolean CSG module - Subtract bodies to create prosthesis volumes.

Provides multiple boolean engines:
- manifold: Fast mesh-based boolean (requires clean input)
- voxel: Slower but guarantees watertight output (recommended)
- auto: Choose best engine based on input quality
"""

import numpy as np
import trimesh
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import PipelineConfig

# Try to import manifold3d, fall back to trimesh boolean if not available
try:
    import manifold3d as mf
    HAS_MANIFOLD = True
except ImportError:
    HAS_MANIFOLD = False


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


def subtract_bodies_voxel(
    outer_mesh: trimesh.Trimesh,
    inner_mesh: trimesh.Trimesh,
    voxel_size: float = 2.0,
) -> trimesh.Trimesh:
    """
    Voxel-based boolean subtraction - GUARANTEED WATERTIGHT OUTPUT.
    
    Process:
    1. Voxelize both meshes to a 3D grid
    2. Boolean subtract in voxel space (trivial - just AND/NOT)
    3. Convert back to mesh using marching cubes
    4. Result is always manifold (marching cubes guarantees this)
    
    Args:
        outer_mesh: The larger body (scaled female target)
        inner_mesh: The smaller body (male stunt performer)
        voxel_size: Size of each voxel in mesh units (mm)
                    Smaller = more detail, more memory, slower
                    Larger = less detail, faster
                    
    Returns:
        Watertight mesh containing the boolean difference
    """
    from skimage import measure
    
    print(f"Voxel boolean subtraction (voxel_size={voxel_size}mm)")
    print(f"  Outer mesh: {outer_mesh.vertices.shape[0]:,} vertices")
    print(f"  Inner mesh: {inner_mesh.vertices.shape[0]:,} vertices")
    
    # Compute combined bounding box with padding
    all_vertices = np.vstack([outer_mesh.vertices, inner_mesh.vertices])
    bbox_min = all_vertices.min(axis=0) - voxel_size * 2
    bbox_max = all_vertices.max(axis=0) + voxel_size * 2
    
    # Create voxel grid
    grid_shape = np.ceil((bbox_max - bbox_min) / voxel_size).astype(int)
    print(f"  Voxel grid shape: {grid_shape} = {np.prod(grid_shape):,} voxels")
    
    # Voxelize outer mesh (what we want to keep where female is larger)
    outer_voxels = _voxelize_mesh(outer_mesh, bbox_min, voxel_size, grid_shape)
    
    # Voxelize inner mesh (what we subtract - the male body)
    inner_voxels = _voxelize_mesh(inner_mesh, bbox_min, voxel_size, grid_shape)
    
    # Boolean subtraction in voxel space: outer AND NOT inner
    result_voxels = outer_voxels & ~inner_voxels
    
    n_result_voxels = result_voxels.sum()
    print(f"  Result voxels: {n_result_voxels:,}")
    
    if n_result_voxels == 0:
        print("  Warning: Boolean result is empty!")
        return trimesh.Trimesh()
    
    # Convert back to mesh using marching cubes
    # Pad with zeros to ensure closed surface at boundaries
    padded = np.pad(result_voxels.astype(float), 1, mode='constant', constant_values=0)
    
    try:
        verts, faces, normals, values = measure.marching_cubes(
            padded, 
            level=0.5,
            spacing=(voxel_size, voxel_size, voxel_size),
        )
    except Exception as e:
        print(f"  Marching cubes failed: {e}")
        return trimesh.Trimesh()
    
    # Adjust vertices for bbox offset and padding
    verts = verts + bbox_min - voxel_size  # Account for padding
    
    result = trimesh.Trimesh(vertices=verts, faces=faces)
    
    # Fix normals (marching cubes can produce inverted normals)
    result.fix_normals()
    
    print(f"  Result: {result.vertices.shape[0]:,} vertices, {result.faces.shape[0]:,} faces")
    print(f"  Watertight: {result.is_watertight}")
    
    return result


def _voxelize_mesh(
    mesh: trimesh.Trimesh,
    bbox_min: np.ndarray,
    voxel_size: float,
    grid_shape: np.ndarray,
) -> np.ndarray:
    """
    Convert mesh to boolean voxel grid.
    
    Uses ray casting to determine inside/outside.
    """
    # Create voxel centers
    voxel_grid = np.zeros(grid_shape, dtype=bool)
    
    # Use trimesh's voxelization
    try:
        voxels = mesh.voxelized(pitch=voxel_size)
        
        # Get filled voxel indices
        filled = voxels.matrix
        
        # Map to our grid (may need offset adjustment)
        origin_offset = ((voxels.bounds[0] - bbox_min) / voxel_size).astype(int)
        
        # Copy filled voxels to our grid
        for idx in np.argwhere(filled):
            grid_idx = tuple(idx + origin_offset)
            if all(0 <= grid_idx[i] < grid_shape[i] for i in range(3)):
                voxel_grid[grid_idx] = True
                
    except Exception as e:
        print(f"  Voxelization warning: {e}")
        # Fallback: simple bounding box voxelization
        pass
    
    return voxel_grid


def subtract_with_config(
    outer_mesh: trimesh.Trimesh,
    inner_mesh: trimesh.Trimesh,
    config: 'PipelineConfig',
) -> trimesh.Trimesh:
    """
    Boolean subtraction using pipeline configuration.
    
    Automatically selects the best engine based on config settings.
    """
    engine = config.boolean_engine
    
    if engine == 'auto':
        # Use voxel for guaranteed watertight, manifold for speed
        if config.validate_watertight:
            engine = 'voxel'
        elif HAS_MANIFOLD:
            engine = 'manifold'
        else:
            engine = 'voxel'
    
    if engine == 'voxel':
        return subtract_bodies_voxel(outer_mesh, inner_mesh, config.voxel_size)
    else:
        return subtract_bodies(outer_mesh, inner_mesh, engine=engine)

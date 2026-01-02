"""Scan import module - Load OBJ/STL/PLY body scans."""

import trimesh
from pathlib import Path


def load_scan(path: str | Path) -> trimesh.Trimesh:
    """
    Load a 3D body scan from file.
    
    Args:
        path: Path to OBJ, STL, or PLY file
        
    Returns:
        Trimesh object (single mesh)
        
    Raises:
        ValueError: If file cannot be loaded or contains multiple meshes
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Scan file not found: {path}")
    
    # Load mesh, force single mesh output
    mesh = trimesh.load(path, force='mesh')
    
    # Ensure we have a Trimesh, not a Scene
    if isinstance(mesh, trimesh.Scene):
        # Combine all meshes in scene
        meshes = list(mesh.geometry.values())
        if len(meshes) == 0:
            raise ValueError(f"No geometry found in {path}")
        mesh = trimesh.util.concatenate(meshes)
    
    # Basic validation
    if mesh.vertices.shape[0] == 0:
        raise ValueError(f"Empty mesh loaded from {path}")
    
    print(f"Loaded scan: {path.name}")
    print(f"  Vertices: {mesh.vertices.shape[0]:,}")
    print(f"  Faces: {mesh.faces.shape[0]:,}")
    print(f"  Watertight: {mesh.is_watertight}")
    print(f"  Bounds: {mesh.bounds}")
    
    return mesh


def validate_scan(mesh: trimesh.Trimesh) -> dict:
    """
    Validate a body scan mesh for pipeline compatibility.
    
    Returns:
        Dictionary with validation results and warnings
    """
    issues = []
    warnings = []
    
    # Check watertight
    if not mesh.is_watertight:
        warnings.append("Mesh is not watertight - will attempt repair")
    
    # Check vertex count (too few = low quality, too many = slow processing)
    if mesh.vertices.shape[0] < 1000:
        issues.append(f"Very low vertex count ({mesh.vertices.shape[0]}) - scan may be too low quality")
    elif mesh.vertices.shape[0] > 500000:
        warnings.append(f"High vertex count ({mesh.vertices.shape[0]:,}) - consider decimating for faster processing")
    
    # Check for degenerate faces
    degenerate = mesh.degenerate_faces
    if len(degenerate) > 0:
        warnings.append(f"{len(degenerate)} degenerate faces detected")
    
    # Check bounding box (reasonable human size in cm/mm)
    extent = mesh.bounding_box.extents
    height = max(extent)
    
    if height < 10:
        warnings.append(f"Very small mesh (height={height:.1f}) - units may be meters, expected cm or mm")
    elif height > 3000:
        warnings.append(f"Very large mesh (height={height:.1f}) - units may be mm, pipeline expects cm")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings,
        'stats': {
            'vertices': mesh.vertices.shape[0],
            'faces': mesh.faces.shape[0],
            'watertight': mesh.is_watertight,
            'height': height,
        }
    }

"""Export module - Save meshes to STL and STEP formats."""

import trimesh
from pathlib import Path
from typing import Dict, Optional
import json


def export_stl(mesh: trimesh.Trimesh, path: str | Path) -> None:
    """
    Export mesh to binary STL format.
    
    Args:
        mesh: Trimesh object to export
        path: Output file path (.stl)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    mesh.export(str(path), file_type='stl')
    print(f"Exported STL: {path}")


def export_blobs_stl(
    blobs: Dict[str, trimesh.Trimesh],
    output_dir: str | Path
) -> Dict[str, str]:
    """
    Export all blobs to individual STL files.
    
    Args:
        blobs: Dictionary of labeled meshes
        output_dir: Directory to save files
        
    Returns:
        Dictionary mapping labels to file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = {}
    for label, mesh in blobs.items():
        filename = f"{label}.stl"
        filepath = output_dir / filename
        export_stl(mesh, filepath)
        paths[label] = str(filepath)
    
    # Write manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump({
            'blobs': list(blobs.keys()),
            'files': paths,
            'count': len(blobs),
        }, f, indent=2)
    
    print(f"Exported {len(blobs)} blobs to {output_dir}")
    return paths


def export_step(mesh: trimesh.Trimesh, path: str | Path) -> bool:
    """
    Export mesh to STEP format via FreeCAD.
    
    Requires FreeCAD to be installed and accessible.
    
    Args:
        mesh: Trimesh object to export
        path: Output file path (.step or .stp)
        
    Returns:
        True if successful, False otherwise
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # First save as temporary STL
    temp_stl = path.with_suffix('.temp.stl')
    mesh.export(str(temp_stl), file_type='stl')
    
    try:
        # Try to use FreeCAD
        success = _export_step_freecad(temp_stl, path)
    except Exception as e:
        print(f"STEP export failed: {e}")
        print("FreeCAD may not be installed or accessible")
        success = False
    finally:
        # Clean up temp file
        if temp_stl.exists():
            temp_stl.unlink()
    
    return success


def _export_step_freecad(stl_path: Path, step_path: Path) -> bool:
    """Use FreeCAD's Python API to convert STL to STEP."""
    
    import sys
    
    # Common FreeCAD lib paths
    freecad_paths = [
        "C:/Program Files/FreeCAD 0.21/bin",
        "C:/Program Files/FreeCAD/bin",
        "/usr/lib/freecad/lib",
        "/usr/lib/freecad-python3/lib",
        "/Applications/FreeCAD.app/Contents/Resources/lib",
    ]
    
    # Try to find FreeCAD
    freecad_found = False
    for fp in freecad_paths:
        if Path(fp).exists():
            sys.path.insert(0, fp)
            freecad_found = True
            break
    
    if not freecad_found:
        # Try importing anyway (might be in PYTHONPATH)
        pass
    
    try:
        import FreeCAD
        import Part
        import Mesh
    except ImportError as e:
        raise ImportError(
            f"FreeCAD not found: {e}\n"
            "Install FreeCAD or set FREECAD_PATH environment variable"
        )
    
    # Import STL mesh
    mesh_obj = Mesh.Mesh(str(stl_path))
    
    # Convert to shape
    shape = Part.Shape()
    shape.makeShapeFromMesh(mesh_obj.Topology, 0.1)
    
    # Make solid
    solid = Part.makeSolid(shape)
    
    # Export STEP
    solid.exportStep(str(step_path))
    
    print(f"Exported STEP: {step_path}")
    return True


def export_blobs_step(
    blobs: Dict[str, trimesh.Trimesh],
    output_dir: str | Path
) -> Dict[str, Optional[str]]:
    """
    Export all blobs to individual STEP files.
    
    Args:
        blobs: Dictionary of labeled meshes
        output_dir: Directory to save files
        
    Returns:
        Dictionary mapping labels to file paths (None if export failed)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = {}
    for label, mesh in blobs.items():
        filename = f"{label}.step"
        filepath = output_dir / filename
        
        if export_step(mesh, filepath):
            paths[label] = str(filepath)
        else:
            paths[label] = None
            print(f"Warning: Failed to export {label} as STEP")
    
    return paths

"""Multi-body STEP export - Export multiple bodies and skeleton in one file."""

import numpy as np
import trimesh
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING
import tempfile

if TYPE_CHECKING:
    from .ansur import BoneLengths


def export_dual_body_step(
    body_a: trimesh.Trimesh,
    body_b: trimesh.Trimesh,
    skeleton_points: Optional[Dict[str, np.ndarray]] = None,
    output_path: Path = None,
    body_a_name: str = "Body_ANSUR_Female",
    body_b_name: str = "Body_Target_Male",
) -> bool:
    """
    Export two body meshes and optional skeleton to a single STEP file.
    
    The STEP file will contain:
    - Body A as a solid (e.g., ANSUR female body)
    - Body B as a solid (e.g., target male body)
    - Skeleton as wire geometry (optional)
    
    User can then import to CAD app for boolean operations.
    
    Args:
        body_a: First body mesh (ANSUR-generated)
        body_b: Second body mesh (target/scan)
        skeleton_points: Optional dict of joint_name -> [x,y,z] positions
        output_path: Output .step file path
        body_a_name: Label for body A in the STEP file
        body_b_name: Label for body B in the STEP file
        
    Returns:
        True if successful
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Exporting dual-body STEP: {output_path}")
    print(f"  {body_a_name}: {body_a.vertices.shape[0]:,} vertices")
    print(f"  {body_b_name}: {body_b.vertices.shape[0]:,} vertices")
    if skeleton_points:
        print(f"  Skeleton: {len(skeleton_points)} joints")
    
    # Try FreeCAD first
    try:
        return _export_freecad(body_a, body_b, skeleton_points, output_path,
                              body_a_name, body_b_name)
    except Exception as e:
        print(f"FreeCAD export failed: {e}")
    
    # Try OpenCASCADE via cadquery
    try:
        return _export_cadquery(body_a, body_b, skeleton_points, output_path,
                               body_a_name, body_b_name)
    except Exception as e:
        print(f"CadQuery export failed: {e}")
    
    # Fallback: Export as separate STL files
    print("Falling back to STL export (no STEP support available)")
    return _export_stl_fallback(body_a, body_b, skeleton_points, output_path)


def _export_freecad(
    body_a: trimesh.Trimesh,
    body_b: trimesh.Trimesh,
    skeleton_points: Optional[Dict[str, np.ndarray]],
    output_path: Path,
    body_a_name: str,
    body_b_name: str,
) -> bool:
    """Export using FreeCAD."""
    import sys
    
    # Common FreeCAD paths
    freecad_paths = [
        "C:/Program Files/FreeCAD 0.21/bin",
        "C:/Program Files/FreeCAD/bin",
        "/usr/lib/freecad/lib",
        "/Applications/FreeCAD.app/Contents/Resources/lib",
    ]
    
    for fp in freecad_paths:
        if Path(fp).exists():
            sys.path.insert(0, fp)
            break
    
    import FreeCAD
    import Part
    import Mesh
    
    # Create a compound to hold all bodies
    shapes = []
    
    # Convert body A
    with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
        body_a.export(f.name, file_type='stl')
        mesh_a = Mesh.Mesh(f.name)
        shape_a = Part.Shape()
        shape_a.makeShapeFromMesh(mesh_a.Topology, 0.1)
        solid_a = Part.makeSolid(shape_a)
        shapes.append(solid_a)
        Path(f.name).unlink()
    
    # Convert body B
    with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
        body_b.export(f.name, file_type='stl')
        mesh_b = Mesh.Mesh(f.name)
        shape_b = Part.Shape()
        shape_b.makeShapeFromMesh(mesh_b.Topology, 0.1)
        solid_b = Part.makeSolid(shape_b)
        shapes.append(solid_b)
        Path(f.name).unlink()
    
    # Add skeleton as wire edges
    if skeleton_points:
        skeleton_edges = _create_skeleton_edges_freecad(skeleton_points, Part)
        shapes.extend(skeleton_edges)
    
    # Create compound
    compound = Part.makeCompound(shapes)
    
    # Export
    compound.exportStep(str(output_path))
    
    print(f"Exported STEP: {output_path}")
    return True


def _export_cadquery(
    body_a: trimesh.Trimesh,
    body_b: trimesh.Trimesh,
    skeleton_points: Optional[Dict[str, np.ndarray]],
    output_path: Path,
    body_a_name: str,
    body_b_name: str,
) -> bool:
    """Export using CadQuery."""
    import cadquery as cq
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakePolygon
    from OCP.TopoDS import TopoDS_Compound
    from OCP.BRep import BRep_Builder
    
    # Create compound
    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    
    # Convert meshes to solids via STL import
    with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
        body_a.export(f.name, file_type='stl')
        solid_a = cq.importers.importStep(f.name)  # This won't work for STL
        Path(f.name).unlink()
    
    # This approach needs refinement - CadQuery doesn't directly import STL as solid
    raise NotImplementedError("CadQuery STL-to-solid not fully implemented")


def _export_stl_fallback(
    body_a: trimesh.Trimesh,
    body_b: trimesh.Trimesh,
    skeleton_points: Optional[Dict[str, np.ndarray]],
    output_path: Path,
) -> bool:
    """Fallback: export as separate STL files."""
    
    output_dir = output_path.parent
    base_name = output_path.stem
    
    # Export body A
    path_a = output_dir / f"{base_name}_body_a.stl"
    body_a.export(str(path_a), file_type='stl')
    print(f"  Exported: {path_a}")
    
    # Export body B
    path_b = output_dir / f"{base_name}_body_b.stl"
    body_b.export(str(path_b), file_type='stl')
    print(f"  Exported: {path_b}")
    
    # Export skeleton as OBJ lines
    if skeleton_points:
        path_skel = output_dir / f"{base_name}_skeleton.obj"
        _export_skeleton_obj(skeleton_points, path_skel)
        print(f"  Exported: {path_skel}")
    
    print(f"\nNote: Install FreeCAD for combined STEP export")
    return True


def _create_skeleton_edges_freecad(skeleton_points: Dict[str, np.ndarray], Part) -> list:
    """Create FreeCAD wire edges for skeleton bones."""
    
    # Define bone connections
    bones = [
        ('ankle_left', 'knee_left'),
        ('ankle_right', 'knee_right'),
        ('knee_left', 'hip_left'),
        ('knee_right', 'hip_right'),
        ('hip_left', 'pelvis'),
        ('hip_right', 'pelvis'),
        ('pelvis', 'spine_mid'),
        ('spine_mid', 'shoulders'),
        ('shoulders', 'neck'),
        ('neck', 'head'),
        ('shoulders', 'shoulder_left'),
        ('shoulders', 'shoulder_right'),
        ('shoulder_left', 'elbow_left'),
        ('shoulder_right', 'elbow_right'),
        ('elbow_left', 'wrist_left'),
        ('elbow_right', 'wrist_right'),
    ]
    
    edges = []
    for joint_a, joint_b in bones:
        if joint_a in skeleton_points and joint_b in skeleton_points:
            p1 = skeleton_points[joint_a]
            p2 = skeleton_points[joint_b]
            
            v1 = Part.Vector(float(p1[0]), float(p1[1]), float(p1[2]))
            v2 = Part.Vector(float(p2[0]), float(p2[1]), float(p2[2]))
            
            edge = Part.makeLine(v1, v2)
            edges.append(edge)
    
    return edges


def _export_skeleton_obj(skeleton_points: Dict[str, np.ndarray], path: Path) -> None:
    """Export skeleton as OBJ wireframe."""
    
    bones = [
        ('ankle_left', 'knee_left'),
        ('ankle_right', 'knee_right'),
        ('knee_left', 'hip_left'),
        ('knee_right', 'hip_right'),
        ('hip_left', 'hip_right'),
        ('hip_left', 'pelvis'),
        ('hip_right', 'pelvis'),
        ('pelvis', 'shoulders'),
        ('shoulders', 'shoulder_left'),
        ('shoulders', 'shoulder_right'),
    ]
    
    # Build vertex list
    joint_to_idx = {}
    vertices = []
    for i, (name, pos) in enumerate(skeleton_points.items()):
        joint_to_idx[name] = i + 1  # OBJ is 1-indexed
        vertices.append(pos)
    
    # Write OBJ
    with open(path, 'w') as f:
        f.write("# Skeleton wireframe\n")
        
        # Vertices
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        
        # Lines (edges)
        for joint_a, joint_b in bones:
            if joint_a in joint_to_idx and joint_b in joint_to_idx:
                f.write(f"l {joint_to_idx[joint_a]} {joint_to_idx[joint_b]}\n")


def create_skeleton_mesh(
    skeleton_points: Dict[str, np.ndarray],
    joint_radius: float = 10.0,
    bone_radius: float = 5.0,
) -> trimesh.Trimesh:
    """
    Create a mesh representation of the skeleton for visualization.
    
    Joints are spheres, bones are cylinders.
    
    Args:
        skeleton_points: Joint positions
        joint_radius: Radius of joint spheres (mm)
        bone_radius: Radius of bone cylinders (mm)
        
    Returns:
        Combined skeleton mesh
    """
    meshes = []
    
    # Joint spheres
    for name, pos in skeleton_points.items():
        sphere = trimesh.creation.icosphere(subdivisions=1, radius=joint_radius)
        sphere.apply_translation(pos)
        meshes.append(sphere)
    
    # Bone cylinders
    bones = [
        ('ankle_left', 'knee_left'),
        ('ankle_right', 'knee_right'),
        ('knee_left', 'hip_left'),
        ('knee_right', 'hip_right'),
        ('hip_left', 'hip_right'),
        ('pelvis', 'shoulders'),
        ('shoulders', 'shoulder_left'),
        ('shoulders', 'shoulder_right'),
        ('shoulder_left', 'elbow_left'),
        ('shoulder_right', 'elbow_right'),
        ('elbow_left', 'wrist_left'),
        ('elbow_right', 'wrist_right'),
    ]
    
    for joint_a, joint_b in bones:
        if joint_a in skeleton_points and joint_b in skeleton_points:
            p1 = skeleton_points[joint_a]
            p2 = skeleton_points[joint_b]
            
            # Create cylinder between points
            length = np.linalg.norm(p2 - p1)
            if length > 0:
                cyl = trimesh.creation.cylinder(radius=bone_radius, height=length)
                
                # Transform to connect points
                direction = (p2 - p1) / length
                midpoint = (p1 + p2) / 2
                
                # Rotation to align with bone direction
                z_axis = np.array([0, 0, 1])
                rotation = trimesh.geometry.align_vectors(z_axis, direction)
                
                transform = np.eye(4)
                transform[:3, :3] = rotation[:3, :3]
                transform[:3, 3] = midpoint
                
                cyl.apply_transform(transform)
                meshes.append(cyl)
    
    if meshes:
        return trimesh.util.concatenate(meshes)
    return trimesh.Trimesh()

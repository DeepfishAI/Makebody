"""MakeHuman integration - Generate body meshes from anthropometric measurements.

MakeHuman is an open-source tool for making 3D human characters.
This module provides integration to generate body meshes from ANSUR-style measurements.

Two approaches:
1. MakeHuman standalone app with scripting (requires MakeHuman installed)
2. makehuman-pyside6 Python package (pip installable)

Usage:
    from pipeline.makehuman_gen import generate_body_from_measurements
    
    mesh = generate_body_from_measurements({
        'stature': 1650,  # mm
        'waistcircumference': 750,
        'chestcircumference': 900,
        ...
    })
"""

import numpy as np
import trimesh
from pathlib import Path
from typing import Dict, Optional, TYPE_CHECKING
import subprocess
import tempfile
import json

if TYPE_CHECKING:
    from .ansur import ANSURSubject


# Try to import makehuman packages
try:
    # Option 1: MakeHuman Python package (pip install makehuman-pyside6)
    import makehuman
    HAS_MAKEHUMAN_PKG = True
except ImportError:
    HAS_MAKEHUMAN_PKG = False

# Common MakeHuman installation paths
MAKEHUMAN_PATHS = [
    "C:/Program Files/MakeHuman",
    "C:/Program Files (x86)/MakeHuman",
    "/Applications/MakeHuman.app/Contents/MacOS",
    "/usr/share/makehuman",
    "~/makehuman",
]


def find_makehuman_install() -> Optional[Path]:
    """Find MakeHuman installation directory."""
    for p in MAKEHUMAN_PATHS:
        path = Path(p).expanduser()
        if path.exists():
            return path
    return None


def ansur_to_makehuman_params(subject: 'ANSURSubject') -> Dict[str, float]:
    """
    Convert ANSUR measurements to MakeHuman modifier parameters.
    
    MakeHuman uses normalized 0-1 parameters for body morphing.
    This maps ANSUR absolute measurements (mm) to MakeHuman relative scales.
    
    Args:
        subject: ANSUR subject with measurements
        
    Returns:
        Dictionary of MakeHuman modifier names to values (0-1)
    """
    # Reference values for normalization (approximate average adult)
    REF_HEIGHT = 1700  # mm
    REF_WAIST = 800    # mm
    REF_CHEST = 950    # mm
    REF_HIP = 950      # mm
    
    # Compute normalized values
    height_factor = subject.stature / REF_HEIGHT
    waist_factor = subject.waistcircumference / REF_WAIST if subject.waistcircumference > 0 else 1.0
    chest_factor = subject.chestcircumference / REF_CHEST if subject.chestcircumference > 0 else 1.0
    hip_factor = subject.hipbreadth * 2.5 / REF_HIP if subject.hipbreadth > 0 else 1.0
    
    # Map to MakeHuman modifier ranges (typically 0-1 or -1 to 1)
    params = {
        # Height/Scale
        'macrodetails/Height': np.clip((height_factor - 0.8) / 0.4, 0, 1),
        
        # Weight/mass distribution
        'macrodetails-universal/Weight': np.clip((waist_factor - 0.7) / 0.6, 0, 1),
        'macrodetails-universal/Muscle': 0.3,  # Neutral
        
        # Proportions
        'macrodetails-proportions/BodyProportions': 0.5,  # Neutral
        
        # Torso
        'torso/torso-scale-vert': np.clip((height_factor - 0.8) / 0.4, 0, 1),
        'torso/torso-scale-horiz': np.clip((chest_factor - 0.8) / 0.4, 0, 1),
        
        # Waist
        'measure/measure-waist-circ-decr|incr': np.clip((waist_factor - 0.8) / 0.4 - 0.5, -1, 1),
        
        # Hips
        'hip/hip-scale-horiz-decr|incr': np.clip((hip_factor - 0.8) / 0.4 - 0.5, -1, 1),
        
        # Gender (0 = female, 1 = male)
        'macrodetails/Gender': 0.0 if subject.gender.lower() == 'female' else 1.0,
        
        # Age (0 = young, 1 = old)
        'macrodetails/Age': 0.5,  # Middle age default
    }
    
    return params


def generate_body_from_measurements(
    measurements: Dict[str, float],
    gender: str = 'female',
    output_path: Optional[Path] = None,
) -> trimesh.Trimesh:
    """
    Generate a body mesh from anthropometric measurements.
    
    Args:
        measurements: Dictionary of measurement name -> value (in mm)
            Required: 'stature'
            Optional: 'waistcircumference', 'chestcircumference', 'hipbreadth', etc.
        gender: 'male' or 'female'
        output_path: Optional path to save the mesh
        
    Returns:
        Generated body mesh as trimesh.Trimesh
    """
    # Create pseudo ANSURSubject for conversion
    from .ansur import ANSURSubject
    
    subject = ANSURSubject(
        subject_id=0,
        gender=gender,
        stature=measurements.get('stature', 1650),
        acromialheight=measurements.get('acromialheight', 1400),
        acromionradialelength=measurements.get('acromionradialelength', 320),
        radialestylionlength=measurements.get('radialestylionlength', 250),
        wristheight=measurements.get('wristheight', 830),
        trochanterionheight=measurements.get('trochanterionheight', 880),
        iliocristaleheight=measurements.get('iliocristaleheight', 1000),
        kneeheightmidpatella=measurements.get('kneeheightmidpatella', 480),
        lateralmalleolusheight=measurements.get('lateralmalleolusheight', 70),
        tibialheight=measurements.get('tibialheight', 430),
        cervicaleheight=measurements.get('cervicaleheight', 1450),
        suprasternaleheight=measurements.get('suprasternaleheight', 1380),
        waistcircumference=measurements.get('waistcircumference', 750),
        hipbreadth=measurements.get('hipbreadth', 350),
        chestcircumference=measurements.get('chestcircumference', 900),
    )
    
    return generate_body_from_ansur(subject, output_path)


def generate_body_from_ansur(
    subject: 'ANSURSubject',
    output_path: Optional[Path] = None,
) -> trimesh.Trimesh:
    """
    Generate a body mesh from an ANSUR subject.
    
    Tries multiple backends:
    1. MakeHuman Python package (if installed)
    2. MakeHuman CLI (if installed)
    3. Fallback: Generate simple parametric body
    
    Args:
        subject: ANSUR subject with measurements
        output_path: Optional path to save the mesh
        
    Returns:
        Generated body mesh
    """
    # Get MakeHuman parameters
    params = ansur_to_makehuman_params(subject)
    
    print(f"Generating body for {subject.gender}, stature={subject.stature}mm")
    
    # Try MakeHuman package first
    if HAS_MAKEHUMAN_PKG:
        try:
            mesh = _generate_with_mh_package(params, subject.gender)
            if output_path:
                mesh.export(str(output_path))
            return mesh
        except Exception as e:
            print(f"MakeHuman package failed: {e}")
    
    # Try MakeHuman CLI
    mh_path = find_makehuman_install()
    if mh_path:
        try:
            mesh = _generate_with_mh_cli(params, subject.gender, mh_path)
            if output_path:
                mesh.export(str(output_path))
            return mesh
        except Exception as e:
            print(f"MakeHuman CLI failed: {e}")
    
    # Fallback: Generate simple parametric body
    print("Using fallback parametric body generator")
    mesh = _generate_parametric_body(subject)
    if output_path:
        mesh.export(str(output_path))
    return mesh


def _generate_with_mh_package(params: Dict[str, float], gender: str) -> trimesh.Trimesh:
    """Generate using makehuman Python package."""
    raise NotImplementedError(
        "MakeHuman package integration not yet implemented.\n"
        "Install makehuman-pyside6 and run MakeHuman GUI for now."
    )


def _generate_with_mh_cli(
    params: Dict[str, float],
    gender: str,
    mh_path: Path,
) -> trimesh.Trimesh:
    """Generate using MakeHuman command line."""
    
    # Write params to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(params, f)
        params_file = f.name
    
    # Output to temp file
    with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f:
        output_file = f.name
    
    try:
        # Call MakeHuman CLI (varies by version)
        cmd = [
            str(mh_path / "makehuman"),  # or makehuman.exe on Windows
            "--nogui",
            "--script", "generate_from_params.py",  # Would need custom script
            params_file,
            output_file,
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            raise RuntimeError(f"MakeHuman CLI failed: {result.stderr}")
        
        # Load generated mesh
        mesh = trimesh.load(output_file)
        return mesh
        
    finally:
        # Cleanup
        Path(params_file).unlink(missing_ok=True)
        Path(output_file).unlink(missing_ok=True)


def _generate_parametric_body(subject: 'ANSURSubject') -> trimesh.Trimesh:
    """
    Generate a simple parametric body as fallback.
    
    This is a very simplified body - just stacked ellipsoids.
    Good enough for testing the pipeline, but not production quality.
    """
    from .ansur import BoneLengths
    
    bones = subject.bone_lengths()
    
    # Build body from simple shapes
    meshes = []
    
    # Torso (ellipsoid)
    torso_height = bones.spine
    torso_width = subject.chestcircumference / (2 * np.pi) if subject.chestcircumference > 0 else 150
    torso_depth = torso_width * 0.7
    
    torso = trimesh.creation.capsule(
        height=torso_height,
        radius=torso_width,
    )
    torso.apply_translation([0, 0, subject.iliocristaleheight + torso_height / 2])
    meshes.append(torso)
    
    # Head (sphere)
    head_radius = 100  # mm
    head = trimesh.creation.icosphere(subdivisions=2, radius=head_radius)
    head.apply_translation([0, 0, bones.total_height - head_radius * 0.8])
    meshes.append(head)
    
    # Legs (cylinders)
    for side in [-1, 1]:
        hip_x = side * 80
        
        # Upper leg
        thigh = trimesh.creation.cylinder(radius=70, height=bones.femur)
        thigh.apply_translation([hip_x, 0, subject.kneeheightmidpatella + bones.femur / 2])
        meshes.append(thigh)
        
        # Lower leg  
        shin = trimesh.creation.cylinder(radius=50, height=bones.tibia)
        shin.apply_translation([hip_x, 0, subject.lateralmalleolusheight + bones.tibia / 2])
        meshes.append(shin)
    
    # Arms (cylinders)
    for side in [-1, 1]:
        shoulder_x = side * (torso_width + 30)
        shoulder_z = subject.acromialheight
        
        # Upper arm
        upper_arm = trimesh.creation.cylinder(radius=40, height=bones.humerus)
        upper_arm.apply_translation([shoulder_x + side * bones.humerus * 0.3, 0, shoulder_z - bones.humerus * 0.5])
        upper_arm.apply_transform(trimesh.transformations.rotation_matrix(
            np.radians(30 * side), [0, 1, 0]
        ))
        meshes.append(upper_arm)
    
    # Combine all parts
    combined = trimesh.util.concatenate(meshes)
    
    # Make watertight via convex hull (simplified but guaranteed solid)
    # Note: This loses detail but ensures printability
    try:
        combined = combined.convex_hull
    except:
        pass
    
    print(f"Generated parametric body: {combined.vertices.shape[0]:,} vertices")
    
    return combined


# Cloud API integration (for services requiring API keys)
class BodyGeneratorAPI:
    """
    Interface for cloud-based body generation APIs.
    
    This is a placeholder for services like:
    - Custom cloud body generator
    - Other commercial body generation APIs
    """
    
    def __init__(self, api_key: str, endpoint: str = None):
        self.api_key = api_key
        self.endpoint = endpoint or "https://api.example.com/body/generate"
    
    def generate(self, measurements: Dict[str, float]) -> trimesh.Trimesh:
        """Generate body via API call."""
        import requests
        
        response = requests.post(
            self.endpoint,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"measurements": measurements},
        )
        response.raise_for_status()
        
        # Assuming API returns mesh data or URL
        data = response.json()
        
        if 'mesh_url' in data:
            return trimesh.load(data['mesh_url'])
        elif 'mesh_data' in data:
            # Would need to parse mesh format
            raise NotImplementedError("Direct mesh data parsing not implemented")
        else:
            raise ValueError("API response missing mesh data")

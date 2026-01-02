"""Height-band mesh scaler - Scale mesh by height region with smooth blending."""

import numpy as np
import trimesh
from typing import Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .ansur import BoneLengths


def scale_by_height_bands(
    mesh: trimesh.Trimesh,
    source_bones: 'BoneLengths',
    target_bones: 'BoneLengths',
    blend_distance: float = 50.0,
) -> trimesh.Trimesh:
    """
    Scale mesh using height-based bone region assignment.
    
    Automatically assigns vertices to bone regions based on Z-coordinate,
    then applies per-region scale factors with smooth blending at boundaries.
    
    This is the AUTOMATIC rough sizing step - no skeleton rig required.
    
    Args:
        mesh: Input mesh to scale
        source_bones: Bone lengths of source body (e.g., ANSUR female)
        target_bones: Bone lengths of target body (e.g., scanned male)
        blend_distance: Distance over which to blend between regions (mm)
        
    Returns:
        Scaled copy of mesh
    """
    # Compute scale factors
    scale_factors = source_bones.scale_factors_to(target_bones)
    
    # Get mesh bounds
    bounds = mesh.bounds
    z_min = bounds[0, 2]  # Feet
    z_max = bounds[1, 2]  # Head top
    total_height = z_max - z_min
    
    # Compute height band boundaries (as fractions of total height)
    # These are approximate - based on typical human proportions
    bands = compute_height_bands(source_bones, z_min)
    
    print(f"Height-band scaling:")
    print(f"  Total height: {total_height:.1f}mm")
    print(f"  Bands: {bands}")
    print(f"  Scale factors: {scale_factors}")
    
    # Create scaled copy
    scaled = mesh.copy()
    vertices = scaled.vertices.copy()
    
    # For each vertex, compute blended scale factor based on Z height
    for i, vertex in enumerate(vertices):
        z = vertex[2]
        
        # Determine scale factor with blending
        scale = compute_blended_scale(z, bands, scale_factors, blend_distance)
        
        # Scale vertex relative to its height band's anchor point
        anchor_z = get_band_anchor(z, bands)
        
        # Scale XY (width) and Z (height) independently
        # Z uses the scale factor, XY uses sqrt for proportional width
        xy_scale = np.sqrt(scale)  # Width scales as sqrt of height
        
        vertices[i, 0] *= xy_scale  # X
        vertices[i, 1] *= xy_scale  # Y
        
        # Z scaling relative to anchor
        vertices[i, 2] = anchor_z + (z - anchor_z) * scale
    
    scaled.vertices = vertices
    
    # Fix any resulting issues
    scaled.fix_normals()
    
    print(f"  Output: {scaled.vertices.shape[0]:,} vertices")
    
    return scaled


def compute_height_bands(
    bones: 'BoneLengths',
    z_offset: float = 0.0,
) -> Dict[str, Tuple[float, float, str]]:
    """
    Compute height band boundaries from bone lengths.
    
    Returns:
        Dictionary mapping band name to (z_min, z_max, scale_key)
    """
    # Work from ground up
    ankle = z_offset
    knee = ankle + bones.tibia
    hip = knee + bones.femur
    shoulder = hip + bones.spine
    head = shoulder + (bones.total_height - bones.spine - bones.femur - bones.tibia)
    
    return {
        'lower_leg': (ankle, knee, 'tibia'),
        'upper_leg': (knee, hip, 'femur'),
        'torso': (hip, shoulder, 'spine'),
        'head_neck': (shoulder, head, 'spine'),  # Use spine scale for head too
    }


def compute_blended_scale(
    z: float,
    bands: Dict[str, Tuple[float, float, str]],
    scale_factors: Dict[str, float],
    blend_distance: float,
) -> float:
    """
    Compute scale factor for a given Z height with smooth blending.
    
    At band boundaries, blends between adjacent scale factors.
    """
    # Find which band(s) this Z belongs to
    for band_name, (z_min, z_max, scale_key) in bands.items():
        if z_min <= z <= z_max:
            scale = scale_factors.get(scale_key, 1.0)
            
            # Check if near boundary - blend with adjacent band
            dist_from_min = z - z_min
            dist_from_max = z_max - z
            
            if dist_from_min < blend_distance:
                # Near lower boundary - blend with band below
                prev_scale = get_adjacent_scale(band_name, bands, scale_factors, 'below')
                if prev_scale is not None:
                    t = dist_from_min / blend_distance
                    t = smooth_step(t)  # Smooth interpolation
                    scale = prev_scale * (1 - t) + scale * t
                    
            elif dist_from_max < blend_distance:
                # Near upper boundary - blend with band above
                next_scale = get_adjacent_scale(band_name, bands, scale_factors, 'above')
                if next_scale is not None:
                    t = dist_from_max / blend_distance
                    t = smooth_step(t)
                    scale = next_scale * (1 - t) + scale * t
            
            return scale
    
    # Fallback - use total height scale
    return scale_factors.get('total_height', 1.0)


def get_adjacent_scale(
    band_name: str,
    bands: Dict[str, Tuple[float, float, str]],
    scale_factors: Dict[str, float],
    direction: str,
) -> float:
    """Get scale factor from adjacent band."""
    band_order = ['lower_leg', 'upper_leg', 'torso', 'head_neck']
    
    try:
        idx = band_order.index(band_name)
        if direction == 'below' and idx > 0:
            adj_band = band_order[idx - 1]
        elif direction == 'above' and idx < len(band_order) - 1:
            adj_band = band_order[idx + 1]
        else:
            return None
        
        _, _, scale_key = bands[adj_band]
        return scale_factors.get(scale_key, 1.0)
    except (ValueError, KeyError):
        return None


def get_band_anchor(z: float, bands: Dict[str, Tuple[float, float, str]]) -> float:
    """Get the anchor point (lower boundary) for scaling at given Z."""
    for band_name, (z_min, z_max, _) in bands.items():
        if z_min <= z <= z_max:
            return z_min
    return 0.0


def smooth_step(t: float) -> float:
    """Smooth interpolation (ease in-out)."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def scale_arms_separately(
    mesh: trimesh.Trimesh,
    arm_scale: float,
    shoulder_width_threshold: float = 150.0,
) -> trimesh.Trimesh:
    """
    Scale arm vertices separately based on X distance from center.
    
    Vertices far from body center (|X| > threshold) are considered arms.
    
    Args:
        mesh: Input mesh
        arm_scale: Scale factor for arms
        shoulder_width_threshold: X distance beyond which is "arm" (mm)
        
    Returns:
        Mesh with scaled arms
    """
    scaled = mesh.copy()
    vertices = scaled.vertices.copy()
    
    # Find arm vertices (far from center in X)
    x_center = (mesh.bounds[0, 0] + mesh.bounds[1, 0]) / 2
    
    for i, vertex in enumerate(vertices):
        x_dist = abs(vertex[0] - x_center)
        if x_dist > shoulder_width_threshold:
            # This is an arm vertex - scale relative to shoulder
            # (simplified - assumes shoulder is at threshold distance)
            arm_offset = x_dist - shoulder_width_threshold
            sign = 1 if vertex[0] > x_center else -1
            
            new_x_dist = shoulder_width_threshold + arm_offset * arm_scale
            vertices[i, 0] = x_center + sign * new_x_dist
    
    scaled.vertices = vertices
    return scaled

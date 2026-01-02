"""Scaling module - Scale source mesh to match target frame size."""

import numpy as np
import trimesh
from .landmarks import LandmarkSet, compute_heights


def scale_to_frame(
    source_mesh: trimesh.Trimesh,
    source_landmarks: LandmarkSet,
    target_landmarks: LandmarkSet,
    mode: str = 'uniform'
) -> trimesh.Trimesh:
    """
    Scale source mesh to match target skeleton frame.
    
    Args:
        source_mesh: Mesh to scale (e.g., female body)
        source_landmarks: Landmarks on source mesh
        target_landmarks: Landmarks on target mesh (e.g., male body)
        mode: Scaling mode
            - 'uniform': Single scale factor based on total height
            - 'segmented': Different scale for torso vs legs (future)
            
    Returns:
        Scaled copy of source mesh
    """
    source_heights = compute_heights(source_landmarks)
    target_heights = compute_heights(target_landmarks)
    
    if mode == 'uniform':
        # Simple uniform scaling based on total height
        scale_factor = target_heights['total_height'] / source_heights['total_height']
        
        print(f"Scaling source to target frame")
        print(f"  Source height: {source_heights['total_height']:.1f}")
        print(f"  Target height: {target_heights['total_height']:.1f}")
        print(f"  Scale factor: {scale_factor:.3f}")
        
        # Create scaled copy
        scaled = source_mesh.copy()
        
        # Scale around ankle midpoint (feet stay grounded)
        ankle_l = np.array(source_landmarks['ankle_left'])
        ankle_r = np.array(source_landmarks['ankle_right'])
        ankle_mid = (ankle_l + ankle_r) / 2
        
        # Translate to origin, scale, translate back
        scaled.vertices -= ankle_mid
        scaled.vertices *= scale_factor
        
        # Translate to target ankle position
        target_ankle_l = np.array(target_landmarks['ankle_left'])
        target_ankle_r = np.array(target_landmarks['ankle_right'])
        target_ankle_mid = (target_ankle_l + target_ankle_r) / 2
        scaled.vertices += target_ankle_mid
        
        return scaled
    
    elif mode == 'segmented':
        # TODO: Implement segment-aware scaling
        # Scale torso and legs independently for better fit
        raise NotImplementedError("Segmented scaling not yet implemented")
    
    else:
        raise ValueError(f"Unknown scaling mode: {mode}")


def compute_volume_ratio(mesh1: trimesh.Trimesh, mesh2: trimesh.Trimesh) -> float:
    """
    Compute the volume ratio between two meshes.
    
    Useful for estimating total prosthesis volume.
    
    Returns:
        mesh1.volume / mesh2.volume
    """
    if mesh1.is_watertight and mesh2.is_watertight:
        return abs(mesh1.volume) / abs(mesh2.volume)
    else:
        # Fall back to bounding box volume
        bb1 = mesh1.bounding_box.volume
        bb2 = mesh2.bounding_box.volume
        return bb1 / bb2


def scale_to_frame_segmented(
    source_mesh: trimesh.Trimesh,
    source_bones: 'BoneLengths',
    target_bones: 'BoneLengths',
    landmarks: dict = None,
) -> trimesh.Trimesh:
    """
    Scale mesh segment-by-segment to match target skeleton.
    
    Uses bone length ratios to scale different body segments independently.
    This preserves proportions better than uniform scaling when body
    proportions differ significantly.
    
    Args:
        source_mesh: Mesh to scale (e.g., ANSUR-generated female body)
        source_bones: Bone lengths of source body (from ANSUR)
        target_bones: Bone lengths to match (from 3D scan landmarks)
        landmarks: Optional landmarks for segment boundary detection
        
    Returns:
        Scaled copy of source mesh
    """
    from .ansur import BoneLengths
    
    # Compute per-segment scale factors
    scale_factors = source_bones.scale_factors_to(target_bones)
    
    print("Segmented scaling:")
    for bone, factor in scale_factors.items():
        print(f"  {bone}: x{factor:.3f}")
    
    # For now, use weighted average scale factor
    # Full implementation would deform mesh per-segment
    avg_scale = np.mean([
        scale_factors['femur'],
        scale_factors['tibia'],
        scale_factors['humerus'],
        scale_factors['forearm'],
        scale_factors['spine'],
    ])
    
    print(f"  Average scale factor: {avg_scale:.3f}")
    
    # Create scaled copy
    scaled = source_mesh.copy()
    
    # Scale around center of mass
    centroid = scaled.centroid
    scaled.vertices -= centroid
    scaled.vertices *= avg_scale
    scaled.vertices += centroid
    
    # TODO: Implement true segmented scaling with mesh deformation
    # This would require:
    # 1. Segment the mesh by body part (legs, torso, arms)
    # 2. Scale each segment independently
    # 3. Blend at segment boundaries
    
    return scaled


def scale_by_height_ratio(
    source_mesh: trimesh.Trimesh,
    source_height: float,
    target_height: float,
) -> trimesh.Trimesh:
    """
    Simple uniform scaling based on height ratio.
    
    Args:
        source_mesh: Mesh to scale
        source_height: Height of source in mesh units
        target_height: Height to match
        
    Returns:
        Scaled mesh
    """
    scale_factor = target_height / source_height
    
    print(f"Height-based scaling: {source_height:.1f} -> {target_height:.1f} (x{scale_factor:.3f})")
    
    scaled = source_mesh.copy()
    
    # Scale around bottom center (feet stay grounded)
    bounds = scaled.bounds
    bottom_center = np.array([
        (bounds[0, 0] + bounds[1, 0]) / 2,
        (bounds[0, 1] + bounds[1, 1]) / 2,
        bounds[0, 2],  # Bottom Z
    ])
    
    scaled.vertices -= bottom_center
    scaled.vertices *= scale_factor
    scaled.vertices += bottom_center
    
    return scaled


# Type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .ansur import BoneLengths

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

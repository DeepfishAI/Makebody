"""Alignment module - Register source mesh to target using landmarks."""

import numpy as np
import trimesh
from .landmarks import LandmarkSet, landmarks_to_matrix


def compute_rigid_transform(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    """
    Compute optimal rigid transformation (rotation + translation) from source to target.
    
    Uses Kabsch algorithm (SVD-based) for least-squares alignment.
    
    Args:
        source_points: Nx3 array of source landmark positions
        target_points: Nx3 array of target landmark positions
        
    Returns:
        4x4 homogeneous transformation matrix
    """
    assert source_points.shape == target_points.shape
    
    # Center both point sets
    source_centroid = source_points.mean(axis=0)
    target_centroid = target_points.mean(axis=0)
    
    source_centered = source_points - source_centroid
    target_centered = target_points - target_centroid
    
    # Compute covariance matrix
    H = source_centered.T @ target_centered
    
    # SVD
    U, S, Vt = np.linalg.svd(H)
    
    # Compute rotation
    R = Vt.T @ U.T
    
    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute translation
    t = target_centroid - R @ source_centroid
    
    # Build 4x4 homogeneous matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    
    return T


def align_meshes(
    source_mesh: trimesh.Trimesh,
    source_landmarks: LandmarkSet,
    target_landmarks: LandmarkSet,
) -> trimesh.Trimesh:
    """
    Align source mesh to target using landmark-based rigid registration.
    
    Args:
        source_mesh: The mesh to transform (e.g., female body)
        source_landmarks: Landmarks on the source mesh
        target_landmarks: Landmarks on the target mesh (e.g., male body)
        
    Returns:
        Transformed copy of source mesh aligned to target
    """
    # Convert landmarks to point matrices
    source_pts = landmarks_to_matrix(source_landmarks)
    target_pts = landmarks_to_matrix(target_landmarks)
    
    # Compute rigid transform
    T = compute_rigid_transform(source_pts, target_pts)
    
    # Apply to mesh (creates a copy)
    aligned = source_mesh.copy()
    aligned.apply_transform(T)
    
    # Compute alignment error
    aligned_pts = (T[:3, :3] @ source_pts.T).T + T[:3, 3]
    error = np.linalg.norm(aligned_pts - target_pts, axis=1).mean()
    
    print(f"Alignment complete")
    print(f"  Mean landmark error: {error:.2f} units")
    
    return aligned

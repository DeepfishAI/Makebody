"""Landmark handling - Load and process body landmarks."""

import json
import numpy as np
from pathlib import Path
from typing import TypedDict


class LandmarkSet(TypedDict):
    """Type definition for landmark coordinates."""
    head_top: list[float]
    shoulder_left: list[float]
    shoulder_right: list[float]
    hip_left: list[float]
    hip_right: list[float]
    ankle_left: list[float]
    ankle_right: list[float]
    # Optional additional landmarks
    knee_left: list[float] | None
    knee_right: list[float] | None
    chest_center: list[float] | None


def load_landmarks(path: str | Path) -> dict[str, LandmarkSet]:
    """
    Load landmark definitions from JSON file.
    
    Expected format:
    {
        "male": {
            "head_top": [x, y, z],
            "shoulder_left": [x, y, z],
            ...
        },
        "female": {
            "head_top": [x, y, z],
            ...
        }
    }
    
    Args:
        path: Path to landmarks JSON file
        
    Returns:
        Dictionary with 'male' and 'female' landmark sets
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Landmarks file not found: {path}")
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    # Validate required keys
    if 'male' not in data or 'female' not in data:
        raise ValueError("Landmarks file must contain 'male' and 'female' keys")
    
    required = ['head_top', 'shoulder_left', 'shoulder_right', 'hip_left', 'hip_right', 'ankle_left', 'ankle_right']
    
    for key in ['male', 'female']:
        for landmark in required:
            if landmark not in data[key]:
                raise ValueError(f"Missing required landmark '{landmark}' in {key} data")
            # Convert to numpy arrays for easier math
            data[key][landmark] = np.array(data[key][landmark], dtype=float)
    
    return data


def compute_heights(landmarks: LandmarkSet) -> dict[str, float]:
    """
    Compute key height measurements from landmarks.
    
    Returns:
        Dictionary with:
        - total_height: head_top to ankle midpoint
        - torso_height: shoulder midpoint to hip midpoint
        - leg_height: hip midpoint to ankle midpoint
        - shoulder_width: shoulder left to right
        - hip_width: hip left to right
    """
    head = np.array(landmarks['head_top'])
    shoulder_l = np.array(landmarks['shoulder_left'])
    shoulder_r = np.array(landmarks['shoulder_right'])
    hip_l = np.array(landmarks['hip_left'])
    hip_r = np.array(landmarks['hip_right'])
    ankle_l = np.array(landmarks['ankle_left'])
    ankle_r = np.array(landmarks['ankle_right'])
    
    # Compute midpoints
    shoulder_mid = (shoulder_l + shoulder_r) / 2
    hip_mid = (hip_l + hip_r) / 2
    ankle_mid = (ankle_l + ankle_r) / 2
    
    return {
        'total_height': float(np.linalg.norm(head - ankle_mid)),
        'torso_height': float(np.linalg.norm(shoulder_mid - hip_mid)),
        'leg_height': float(np.linalg.norm(hip_mid - ankle_mid)),
        'shoulder_width': float(np.linalg.norm(shoulder_l - shoulder_r)),
        'hip_width': float(np.linalg.norm(hip_l - hip_r)),
    }


def landmarks_to_matrix(landmarks: LandmarkSet) -> np.ndarray:
    """
    Convert landmarks dict to Nx3 matrix for alignment algorithms.
    
    Returns:
        Numpy array of shape (N, 3) with landmark coordinates
    """
    keys = ['head_top', 'shoulder_left', 'shoulder_right', 'hip_left', 'hip_right', 'ankle_left', 'ankle_right']
    points = [np.array(landmarks[k]) for k in keys]
    return np.vstack(points)

"""Blob separation module - Isolate and label disconnected prosthesis pieces."""

import numpy as np
import trimesh
from typing import Dict, List, Tuple


def separate_and_label(
    mesh: trimesh.Trimesh,
    min_volume: float = 10.0
) -> Dict[str, trimesh.Trimesh]:
    """
    Separate mesh into disconnected components and label them by body region.
    
    Args:
        mesh: Input mesh (result of boolean subtraction)
        min_volume: Minimum volume threshold to keep a component (filters noise)
        
    Returns:
        Dictionary mapping region names to meshes:
        {
            'chest_left': Trimesh,
            'chest_right': Trimesh,
            'hip_left': Trimesh,
            ...
        }
    """
    # Split into disconnected components
    components = mesh.split(only_watertight=False)
    
    print(f"Found {len(components)} disconnected components")
    
    # Filter small components (noise)
    filtered = []
    for comp in components:
        if comp.is_watertight:
            vol = abs(comp.volume)
        else:
            vol = comp.bounding_box.volume
        
        if vol >= min_volume:
            filtered.append(comp)
        else:
            print(f"  Filtered small component (volume={vol:.1f})")
    
    print(f"Kept {len(filtered)} components after filtering")
    
    # Label components by centroid position
    labeled = {}
    for comp in filtered:
        label = classify_blob(comp)
        
        # Handle duplicates by adding suffix
        if label in labeled:
            i = 2
            while f"{label}_{i}" in labeled:
                i += 1
            label = f"{label}_{i}"
        
        labeled[label] = comp
        print(f"  {label}: {comp.vertices.shape[0]:,} vertices, centroid={comp.centroid}")
    
    return labeled


def classify_blob(mesh: trimesh.Trimesh) -> str:
    """
    Classify a blob by its centroid position.
    
    Assumes standard anatomical orientation:
    - Y is up (height)
    - X is left-right
    - Z is front-back
    
    Args:
        mesh: Single connected component
        
    Returns:
        Label string: 'chest_left', 'hip_right', 'buttocks', etc.
    """
    cx, cy, cz = mesh.centroid
    
    # Get bounding box for context
    bounds = mesh.bounds
    height = bounds[1, 1] - bounds[0, 1]  # Y extent
    
    # Determine vertical region based on centroid Y
    # Normalize assuming total height ~170 units (cm)
    y_normalized = cy / 170.0  # Rough normalization
    
    # Determine left/right
    if cx < -5:
        side = 'left'
    elif cx > 5:
        side = 'right'
    else:
        side = 'center'
    
    # Determine front/back
    if cz > 5:
        depth = 'front'
    elif cz < -5:
        depth = 'back'
    else:
        depth = ''
    
    # Classify by height region
    if y_normalized > 0.75:  # Upper torso
        if side == 'center':
            return 'upper_torso'
        return f'chest_{side}'
    
    elif y_normalized > 0.55:  # Mid torso
        if depth == 'back':
            return 'back_padding'
        return f'waist_{side}' if side != 'center' else 'waist'
    
    elif y_normalized > 0.45:  # Hip region
        if depth == 'back':
            return 'buttocks' if side == 'center' else f'buttock_{side}'
        return f'hip_{side}' if side != 'center' else 'hip'
    
    elif y_normalized > 0.25:  # Upper leg
        return f'thigh_{side}' if side != 'center' else 'thigh'
    
    else:  # Lower leg
        return f'calf_{side}' if side != 'center' else 'calf'


def get_blob_stats(blobs: Dict[str, trimesh.Trimesh]) -> Dict[str, dict]:
    """
    Get statistics for each blob (useful for manufacturing).
    
    Returns:
        Dictionary with stats per blob:
        {
            'chest_left': {
                'volume': float,
                'surface_area': float,
                'bounding_box': [width, height, depth],
                'centroid': [x, y, z]
            },
            ...
        }
    """
    stats = {}
    for label, mesh in blobs.items():
        bbox_extents = mesh.bounding_box.extents
        
        if mesh.is_watertight:
            volume = abs(mesh.volume)
        else:
            volume = None
        
        stats[label] = {
            'volume': volume,
            'surface_area': mesh.area,
            'bounding_box': bbox_extents.tolist(),
            'centroid': mesh.centroid.tolist(),
            'vertices': mesh.vertices.shape[0],
            'faces': mesh.faces.shape[0],
            'watertight': mesh.is_watertight,
        }
    
    return stats

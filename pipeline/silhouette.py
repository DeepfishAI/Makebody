"""Silhouette generator - Create 2D body profile from ANSUR measurements."""

import numpy as np
from typing import List, Tuple, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from .ansur import ANSURSubject


def circumference_to_width(circumference: float, shape_ratio: float = 1.3) -> float:
    """
    Convert circumference to front-view width.
    
    Assumes elliptical cross-section with given front:side ratio.
    
    Args:
        circumference: Measured circumference in mm
        shape_ratio: Ratio of front width to side depth (typically 1.2-1.5)
        
    Returns:
        Front-view width in mm
    """
    if circumference <= 0:
        return 0
    # For ellipse: C ≈ π × √(2(a² + b²)) where a/b = shape_ratio
    # Simplified: width ≈ C / (π × factor)
    return circumference / (np.pi * shape_ratio)


def generate_silhouette_points(subject: 'ANSURSubject') -> List[Tuple[float, float]]:
    """
    Generate 2D body silhouette points from ANSUR measurements.
    
    Returns points for the RIGHT side of the body (positive X).
    Mirror for full silhouette.
    
    Args:
        subject: ANSUR subject with measurements
        
    Returns:
        List of (height_z, half_width_x) tuples from feet to head
    """
    # Extract measurements with safe defaults
    def safe_get(val, default=0):
        return val if val and val > 0 else default
    
    stature = safe_get(subject.stature, 1700)
    
    # Key measurement points
    # Format: (height, half_width)
    points = []
    
    # Feet (ground level)
    points.append((0, 40))  # Foot width estimate
    
    # Ankles
    ankle_height = safe_get(subject.lateralmalleolusheight, 70)
    # Ankle circumference not always in ANSUR, estimate from stature
    ankle_width = stature * 0.04  # ~4% of height
    points.append((ankle_height, ankle_width / 2))
    
    # Calves (widest part of lower leg, ~35% up shin)
    calf_height = ankle_height + (subject.kneeheightmidpatella - ankle_height) * 0.35
    calf_width = stature * 0.07  # ~7% of height
    points.append((calf_height, calf_width / 2))
    
    # Knees
    knee_height = safe_get(subject.kneeheightmidpatella, 480)
    knee_width = stature * 0.06  # Narrower at knee
    points.append((knee_height, knee_width / 2))
    
    # Mid-thigh
    thigh_height = knee_height + (subject.trochanterionheight - knee_height) * 0.5
    thigh_width = stature * 0.10  # ~10% of height
    points.append((thigh_height, thigh_width / 2))
    
    # Crotch / Hip joint
    hip_height = safe_get(subject.trochanterionheight, 880)
    hip_width = safe_get(subject.hipbreadth, stature * 0.20)
    points.append((hip_height, hip_width / 2))
    
    # Hips / Buttocks (widest below waist)
    buttock_height = safe_get(subject.iliocristaleheight, 1000) - 50
    buttock_width = hip_width * 1.1  # Slightly wider than hip joints
    points.append((buttock_height, buttock_width / 2))
    
    # Waist (narrowest point)
    # Estimate waist height as midpoint between iliocristale and bust
    waist_height = safe_get(subject.iliocristaleheight, 1000) + 50
    waist_circ = safe_get(subject.waistcircumference, stature * 0.45)
    waist_width = circumference_to_width(waist_circ)
    points.append((waist_height, waist_width / 2))
    
    # Lower chest / ribcage
    lower_chest_height = waist_height + 100
    chest_circ = safe_get(subject.chestcircumference, stature * 0.55)
    lower_chest_width = circumference_to_width(chest_circ) * 0.95
    points.append((lower_chest_height, lower_chest_width / 2))
    
    # Bust / Chest (widest torso point)
    bust_height = safe_get(subject.suprasternaleheight, 1380) - 100
    chest_width = circumference_to_width(chest_circ)
    points.append((bust_height, chest_width / 2))
    
    # Shoulders
    shoulder_height = safe_get(subject.acromialheight, 1400)
    # Use biacromialbreadth if available, else estimate
    shoulder_width = stature * 0.25  # ~25% of height for shoulders
    points.append((shoulder_height, shoulder_width / 2))
    
    # Neck base
    neck_height = safe_get(subject.cervicaleheight, 1450)
    neck_width = stature * 0.07  # Neck width
    points.append((neck_height, neck_width / 2))
    
    # Chin level
    chin_height = stature - 120  # ~12cm below top
    chin_width = stature * 0.08
    points.append((chin_height, chin_width / 2))
    
    # Head (widest)
    head_center_height = stature - 80
    head_width = stature * 0.09  # Face width
    points.append((head_center_height, head_width / 2))
    
    # Top of head
    points.append((stature, 20))  # Narrow at crown
    
    return points


def silhouette_to_svg_path(points: List[Tuple[float, float]], scale: float = 0.25) -> str:
    """
    Convert silhouette points to SVG path string.
    
    Creates a closed path by mirroring the right side.
    
    Args:
        points: List of (height, half_width) points
        scale: Scale factor for display
        
    Returns:
        SVG path 'd' attribute string
    """
    if not points:
        return ""
    
    # Sort by height (bottom to top)
    sorted_points = sorted(points, key=lambda p: p[0])
    
    # Create smooth curve using quadratic bezier
    # Right side (bottom to top)
    path_parts = []
    
    # Start at bottom left
    first = sorted_points[0]
    path_parts.append(f"M {-first[1] * scale} {-first[0] * scale}")
    
    # Draw left side (mirrored, bottom to top)
    for i, (z, x) in enumerate(sorted_points[1:], 1):
        path_parts.append(f"L {-x * scale} {-z * scale}")
    
    # Draw right side (top to bottom)
    for z, x in reversed(sorted_points):
        path_parts.append(f"L {x * scale} {-z * scale}")
    
    # Close path
    path_parts.append("Z")
    
    return " ".join(path_parts)


def silhouette_to_canvas_points(
    points: List[Tuple[float, float]],
    canvas_width: float = 400,
    canvas_height: float = 600,
    padding: float = 20,
) -> Dict:
    """
    Convert silhouette points to canvas coordinates.
    
    Args:
        points: List of (height, half_width) points
        canvas_width: Canvas width in pixels
        canvas_height: Canvas height in pixels
        padding: Padding around the figure
        
    Returns:
        Dict with 'left' and 'right' point arrays for drawing
    """
    if not points:
        return {'left': [], 'right': []}
    
    sorted_points = sorted(points, key=lambda p: p[0])
    
    # Find bounds
    max_height = max(p[0] for p in sorted_points)
    max_width = max(p[1] for p in sorted_points) * 2
    
    # Calculate scale
    available_width = canvas_width - 2 * padding
    available_height = canvas_height - 2 * padding
    
    scale = min(available_width / max_width, available_height / max_height)
    
    center_x = canvas_width / 2
    bottom_y = canvas_height - padding
    
    # Generate points
    left_points = []
    right_points = []
    
    for z, half_x in sorted_points:
        y = bottom_y - z * scale
        x_offset = half_x * scale
        
        left_points.append([center_x - x_offset, y])
        right_points.append([center_x + x_offset, y])
    
    return {
        'left': left_points,
        'right': right_points,
        'scale': scale,
        'center_x': center_x,
    }


def generate_silhouette_json(subject: 'ANSURSubject') -> Dict:
    """
    Generate silhouette data as JSON-serializable dict.
    
    Returns data ready for frontend canvas rendering.
    """
    points = generate_silhouette_points(subject)
    canvas_data = silhouette_to_canvas_points(points)
    
    return {
        'raw_points': [[p[0], p[1]] for p in points],  # (height, half_width)
        'canvas': canvas_data,
        'stature': subject.stature,
        'gender': subject.gender,
    }

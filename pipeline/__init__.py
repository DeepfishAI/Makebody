"""Prosthesis Pipeline - Generate 3D-printable prosthesis from body scans."""

from .config import PipelineConfig, DEFAULT_CONFIG
from .scan_import import load_scan
from .landmarks import load_landmarks, compute_heights
from .alignment import align_meshes
from .scaling import scale_to_frame, scale_to_frame_segmented, scale_by_height_ratio
from .boolean_csg import subtract_bodies, subtract_bodies_voxel, subtract_with_config
from .blob_separation import separate_and_label
from .simplification import simplify_mesh
from .repair import repair_mesh, repair_mesh_robust, check_manifold
from .step_export import export_stl, export_step, export_blobs_stl

# ANSUR data handling
from .ansur import (
    ANSURSubject,
    BoneLengths,
    load_ansur_dataset,
    find_closest_subject,
    compute_scale_factors,
)

__all__ = [
    # Configuration
    'PipelineConfig',
    'DEFAULT_CONFIG',
    
    # Core pipeline
    'load_scan',
    'load_landmarks',
    'compute_heights',
    'align_meshes',
    'scale_to_frame',
    'scale_to_frame_segmented',
    'scale_by_height_ratio',
    'subtract_bodies',
    'subtract_bodies_voxel',
    'subtract_with_config',
    'separate_and_label',
    'simplify_mesh',
    'repair_mesh',
    'repair_mesh_robust',
    'check_manifold',
    'export_stl',
    'export_step',
    'export_blobs_stl',
    
    # ANSUR
    'ANSURSubject',
    'BoneLengths',
    'load_ansur_dataset',
    'find_closest_subject',
    'compute_scale_factors',
]


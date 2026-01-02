"""Prosthesis Pipeline - Generate 3D-printable prosthesis from body scans."""

from .scan_import import load_scan
from .landmarks import load_landmarks, compute_heights
from .alignment import align_meshes
from .scaling import scale_to_frame
from .boolean_csg import subtract_bodies
from .blob_separation import separate_and_label
from .simplification import simplify_mesh
from .repair import repair_mesh
from .step_export import export_stl, export_step

__all__ = [
    'load_scan',
    'load_landmarks',
    'compute_heights',
    'align_meshes',
    'scale_to_frame',
    'subtract_bodies',
    'separate_and_label',
    'simplify_mesh',
    'repair_mesh',
    'export_stl',
    'export_step',
]

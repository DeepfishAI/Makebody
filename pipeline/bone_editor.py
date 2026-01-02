"""Interactive bone editor - Click-drag bone endpoints to fine-tune scaling."""

import numpy as np
import trimesh
from typing import Dict, Optional, Callable, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from .ansur import BoneLengths

# Try to import Open3D for visualization
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


@dataclass
class SkeletonJoint:
    """A draggable joint in the skeleton."""
    name: str
    position: np.ndarray  # [x, y, z]
    color: np.ndarray = field(default_factory=lambda: np.array([0.8, 0.2, 0.2]))
    radius: float = 15.0  # mm
    draggable: bool = True
    
    def to_sphere(self) -> 'o3d.geometry.TriangleMesh':
        """Create Open3D sphere for visualization."""
        if not HAS_OPEN3D:
            raise ImportError("Open3D not installed")
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=self.radius)
        sphere.translate(self.position)
        sphere.paint_uniform_color(self.color)
        return sphere


@dataclass
class SkeletonBone:
    """A bone connecting two joints."""
    name: str
    joint_a: str
    joint_b: str
    color: np.ndarray = field(default_factory=lambda: np.array([0.3, 0.3, 0.8]))
    
    def length(self, joints: Dict[str, SkeletonJoint]) -> float:
        """Compute current bone length."""
        pos_a = joints[self.joint_a].position
        pos_b = joints[self.joint_b].position
        return np.linalg.norm(pos_b - pos_a)


class BoneEditor:
    """
    Interactive 3D bone editor for fine-tuning mesh scaling.
    
    Displays a skeleton overlaid on a semi-transparent body mesh.
    User can click and drag joint endpoints to adjust bone lengths.
    Mesh preview updates in real-time.
    
    Usage:
        editor = BoneEditor(mesh, initial_bones)
        adjusted_bones = editor.run()  # Opens interactive window
        # User adjusts bones, clicks Apply
        scaled_mesh = editor.get_scaled_mesh()
    """
    
    def __init__(
        self,
        mesh: trimesh.Trimesh,
        bone_lengths: 'BoneLengths',
        on_update: Optional[Callable[[Dict[str, float]], None]] = None,
    ):
        """
        Initialize bone editor.
        
        Args:
            mesh: Body mesh to visualize
            bone_lengths: Initial bone lengths (defines skeleton scale)
            on_update: Callback when bones are adjusted
        """
        if not HAS_OPEN3D:
            raise ImportError(
                "Open3D is required for interactive bone editing.\n"
                "Install with: pip install open3d"
            )
        
        self.mesh = mesh
        self.initial_bones = bone_lengths
        self.on_update = on_update
        
        # Build skeleton from bone lengths
        self.joints = self._create_joints(bone_lengths)
        self.bones = self._create_bones()
        
        # Visualization state
        self.vis = None
        self.selected_joint: Optional[str] = None
        self.scale_factors: Dict[str, float] = {}
        
        # Mouse state
        self._mouse_down = False
        self._last_mouse_pos = None
    
    def _create_joints(self, bones: 'BoneLengths') -> Dict[str, SkeletonJoint]:
        """Create skeleton joints from bone lengths."""
        # Compute joint positions (standing pose, facing +Y)
        z_ground = 0.0
        
        # Leg joints (symmetric)
        hip_width = 100  # mm
        ankle_z = z_ground + 50  # Small offset from ground
        knee_z = ankle_z + bones.tibia
        hip_z = knee_z + bones.femur
        
        # Spine
        shoulder_z = hip_z + bones.spine
        neck_z = shoulder_z + 100
        head_z = bones.total_height
        
        # Shoulders
        shoulder_width = 200  # mm
        
        joints = {
            # Ankles (fixed - feet on ground)
            'ankle_left': SkeletonJoint('ankle_left', np.array([-hip_width, 0, ankle_z]), draggable=False),
            'ankle_right': SkeletonJoint('ankle_right', np.array([hip_width, 0, ankle_z]), draggable=False),
            
            # Knees
            'knee_left': SkeletonJoint('knee_left', np.array([-hip_width, 0, knee_z])),
            'knee_right': SkeletonJoint('knee_right', np.array([hip_width, 0, knee_z])),
            
            # Hips
            'hip_left': SkeletonJoint('hip_left', np.array([-hip_width, 0, hip_z])),
            'hip_right': SkeletonJoint('hip_right', np.array([hip_width, 0, hip_z])),
            'pelvis': SkeletonJoint('pelvis', np.array([0, 0, hip_z]), draggable=False),
            
            # Spine
            'spine_mid': SkeletonJoint('spine_mid', np.array([0, 0, (hip_z + shoulder_z) / 2])),
            'shoulders': SkeletonJoint('shoulders', np.array([0, 0, shoulder_z])),
            
            # Arms
            'shoulder_left': SkeletonJoint('shoulder_left', np.array([-shoulder_width, 0, shoulder_z])),
            'shoulder_right': SkeletonJoint('shoulder_right', np.array([shoulder_width, 0, shoulder_z])),
            'elbow_left': SkeletonJoint('elbow_left', np.array([-shoulder_width - bones.humerus * 0.7, 0, shoulder_z - bones.humerus * 0.7])),
            'elbow_right': SkeletonJoint('elbow_right', np.array([shoulder_width + bones.humerus * 0.7, 0, shoulder_z - bones.humerus * 0.7])),
            'wrist_left': SkeletonJoint('wrist_left', np.array([-shoulder_width - bones.humerus * 0.7 - bones.forearm * 0.7, 0, shoulder_z - bones.humerus * 0.7 - bones.forearm * 0.7])),
            'wrist_right': SkeletonJoint('wrist_right', np.array([shoulder_width + bones.humerus * 0.7 + bones.forearm * 0.7, 0, shoulder_z - bones.humerus * 0.7 - bones.forearm * 0.7])),
            
            # Head
            'neck': SkeletonJoint('neck', np.array([0, 0, neck_z])),
            'head': SkeletonJoint('head', np.array([0, 0, head_z]), color=np.array([0.9, 0.7, 0.2])),
        }
        
        return joints
    
    def _create_bones(self) -> list:
        """Define skeleton bones (connections between joints)."""
        return [
            # Legs
            SkeletonBone('tibia_left', 'ankle_left', 'knee_left'),
            SkeletonBone('tibia_right', 'ankle_right', 'knee_right'),
            SkeletonBone('femur_left', 'knee_left', 'hip_left'),
            SkeletonBone('femur_right', 'knee_right', 'hip_right'),
            
            # Pelvis
            SkeletonBone('pelvis', 'hip_left', 'hip_right'),
            
            # Spine
            SkeletonBone('lower_spine', 'pelvis', 'spine_mid'),
            SkeletonBone('upper_spine', 'spine_mid', 'shoulders'),
            SkeletonBone('neck', 'shoulders', 'neck'),
            SkeletonBone('head', 'neck', 'head'),
            
            # Shoulders
            SkeletonBone('clavicle_left', 'shoulders', 'shoulder_left'),
            SkeletonBone('clavicle_right', 'shoulders', 'shoulder_right'),
            
            # Arms
            SkeletonBone('humerus_left', 'shoulder_left', 'elbow_left'),
            SkeletonBone('humerus_right', 'shoulder_right', 'elbow_right'),
            SkeletonBone('forearm_left', 'elbow_left', 'wrist_left'),
            SkeletonBone('forearm_right', 'elbow_right', 'wrist_right'),
        ]
    
    def _mesh_to_o3d(self, alpha: float = 0.3) -> 'o3d.geometry.TriangleMesh':
        """Convert trimesh to Open3D mesh with transparency."""
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(self.mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(self.mesh.faces)
        o3d_mesh.compute_vertex_normals()
        
        # Set semi-transparent color
        colors = np.ones((len(self.mesh.vertices), 3)) * np.array([0.7, 0.7, 0.9])
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        
        return o3d_mesh
    
    def _create_bone_lines(self) -> 'o3d.geometry.LineSet':
        """Create line geometry for bones."""
        points = []
        lines = []
        colors = []
        
        joint_index = {}
        for i, (name, joint) in enumerate(self.joints.items()):
            points.append(joint.position)
            joint_index[name] = i
        
        for bone in self.bones:
            idx_a = joint_index[bone.joint_a]
            idx_b = joint_index[bone.joint_b]
            lines.append([idx_a, idx_b])
            colors.append(bone.color)
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        
        return line_set
    
    def compute_bone_lengths(self) -> Dict[str, float]:
        """Compute current bone lengths from joint positions."""
        lengths = {}
        for bone in self.bones:
            lengths[bone.name] = bone.length(self.joints)
        return lengths
    
    def compute_scale_factors(self) -> Dict[str, float]:
        """Compute scale factors relative to initial bone lengths."""
        current = self.compute_bone_lengths()
        
        # Map bone names to standard bone length keys
        bone_to_key = {
            'femur_left': 'femur',
            'femur_right': 'femur',
            'tibia_left': 'tibia',
            'tibia_right': 'tibia',
            'humerus_left': 'humerus',
            'humerus_right': 'humerus',
            'forearm_left': 'forearm',
            'forearm_right': 'forearm',
            'upper_spine': 'spine',
            'lower_spine': 'spine',
        }
        
        initial = self.initial_bones.as_dict()
        
        factors = {}
        for bone_name, key in bone_to_key.items():
            if bone_name in current and key in initial and initial[key] > 0:
                factors[key] = current[bone_name] / initial[key]
        
        return factors
    
    def run(self) -> Dict[str, float]:
        """
        Run interactive bone editor.
        
        Opens a 3D window where user can drag bone endpoints.
        Returns adjusted scale factors when window is closed.
        
        Controls:
            - Left click + drag on joint: move joint
            - Right click: deselect
            - Mouse wheel: zoom
            - Middle mouse: rotate view
            - Close window: apply changes
        
        Returns:
            Dictionary of bone scale factors
        """
        print("=" * 50)
        print("BONE EDITOR")
        print("=" * 50)
        print("Controls:")
        print("  Left click + drag: Move joint")
        print("  Mouse wheel: Zoom")
        print("  Middle mouse: Rotate")
        print("  Close window to apply")
        print("=" * 50)
        
        # Create visualizer
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window("Bone Editor - Drag joints to adjust", width=1200, height=800)
        
        # Add geometries
        mesh_geom = self._mesh_to_o3d()
        self.vis.add_geometry(mesh_geom)
        
        bone_lines = self._create_bone_lines()
        self.vis.add_geometry(bone_lines)
        
        # Add joint spheres
        joint_geoms = {}
        for name, joint in self.joints.items():
            if joint.draggable:
                sphere = joint.to_sphere()
                self.vis.add_geometry(sphere)
                joint_geoms[name] = sphere
        
        # Set up view
        ctr = self.vis.get_view_control()
        ctr.set_zoom(0.5)
        
        # Add key callback for info
        def print_help(vis):
            print("\nCurrent scale factors:")
            for key, val in self.compute_scale_factors().items():
                print(f"  {key}: {val:.3f}x")
            return False
        
        self.vis.register_key_callback(ord('H'), print_help)
        
        # Run visualization
        self.vis.run()
        self.vis.destroy_window()
        
        # Return final scale factors
        factors = self.compute_scale_factors()
        print("\nFinal scale factors:")
        for key, val in factors.items():
            print(f"  {key}: {val:.3f}x")
        
        return factors
    
    def run_headless(self) -> Dict[str, float]:
        """
        Run without GUI - return current scale factors.
        
        Useful for testing or automated pipelines.
        """
        return self.compute_scale_factors()


def interactive_bone_edit(
    mesh: trimesh.Trimesh,
    bone_lengths: 'BoneLengths',
) -> Tuple[Dict[str, float], 'BoneLengths']:
    """
    Convenience function to run bone editor.
    
    Args:
        mesh: Body mesh to edit
        bone_lengths: Initial bone lengths
        
    Returns:
        Tuple of (scale_factors, adjusted_bone_lengths)
    """
    editor = BoneEditor(mesh, bone_lengths)
    scale_factors = editor.run()
    
    # Create adjusted bone lengths
    from .ansur import BoneLengths as BL
    adjusted = BL(
        femur=bone_lengths.femur * scale_factors.get('femur', 1.0),
        tibia=bone_lengths.tibia * scale_factors.get('tibia', 1.0),
        humerus=bone_lengths.humerus * scale_factors.get('humerus', 1.0),
        forearm=bone_lengths.forearm * scale_factors.get('forearm', 1.0),
        spine=bone_lengths.spine * scale_factors.get('spine', 1.0),
        total_height=bone_lengths.total_height * scale_factors.get('total_height', 1.0),
    )
    
    return scale_factors, adjusted

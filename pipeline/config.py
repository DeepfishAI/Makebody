"""Pipeline configuration module - Centralized resolution and quality controls."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class PipelineConfig:
    """
    Configuration for the prosthesis generation pipeline.
    
    All resolution/granularity parameters are adjustable to balance
    between processing speed, detail preservation, and output quality.
    
    Attributes:
        voxel_size: Size of voxels for voxel-based boolean operations (mm).
                    Smaller = more detail, slower. Larger = faster, less detail.
        target_faces: Target face count after mesh simplification.
                      Higher = more detail, larger file size.
        min_thickness: Minimum wall thickness for printability (mm).
                       Thin sections below this are thickened.
        point_density: Points per mm² for point cloud operations.
        
        repair_passes: Number of repair iterations to attempt.
        hole_fill_max_area: Maximum hole area to auto-fill (mm²).
        
        boolean_engine: Which boolean engine to use.
            - 'manifold': Fast, good for clean inputs
            - 'voxel': Slower but guaranteed watertight output
            - 'auto': Choose based on input quality
    """
    
    # Resolution controls (THE KEY ADJUSTABLE PARAMETERS)
    voxel_size: float = 2.0           # mm - voxel grid resolution
    target_faces: int = 50000         # target triangle count
    min_thickness: float = 3.0        # mm - minimum wall thickness
    point_density: float = 1.0        # points per mm²
    
    # Mesh repair settings
    repair_passes: int = 3            # iterations of repair
    hole_fill_max_area: float = 100.0 # mm² - auto-fill holes smaller than this
    merge_tolerance: float = 0.01     # mm - merge vertices closer than this
    
    # Boolean engine selection
    boolean_engine: Literal['manifold', 'voxel', 'auto'] = 'auto'
    
    # Output settings
    output_format: Literal['stl', 'step', 'both'] = 'stl'
    validate_watertight: bool = True  # Fail if output not watertight
    
    # Blob filtering
    min_blob_volume: float = 10.0     # mm³ - discard blobs smaller than this
    
    # Comfort offset
    body_offset: float = 0.0          # mm - gap between prosthesis and body
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.voxel_size <= 0:
            raise ValueError(f"voxel_size must be positive, got {self.voxel_size}")
        if self.target_faces < 100:
            raise ValueError(f"target_faces too low: {self.target_faces}")
        if self.min_thickness < 0:
            raise ValueError(f"min_thickness cannot be negative: {self.min_thickness}")
    
    @classmethod
    def preview(cls) -> 'PipelineConfig':
        """
        Fast preview quality - for quick iteration.
        
        Low resolution, fast processing. Good for checking alignment
        and rough prosthesis shapes before final export.
        """
        return cls(
            voxel_size=5.0,           # Coarse voxels
            target_faces=10000,       # Low poly
            min_thickness=5.0,        # Thick walls (less detail)
            repair_passes=1,          # Minimal repair
            boolean_engine='voxel',   # Guaranteed to work
            validate_watertight=True,
        )
    
    @classmethod
    def standard(cls) -> 'PipelineConfig':
        """
        Standard quality - balanced speed and detail.
        
        Good for most use cases. Reasonable processing time
        with acceptable detail for 3D printing.
        """
        return cls(
            voxel_size=2.0,           # Medium voxels
            target_faces=50000,       # Medium poly
            min_thickness=3.0,        # Standard thickness
            repair_passes=3,          # Thorough repair
            boolean_engine='auto',    # Choose best engine
            validate_watertight=True,
        )
    
    @classmethod
    def production(cls) -> 'PipelineConfig':
        """
        Production quality - maximum detail.
        
        High resolution for final production pieces.
        Slower processing but best visual quality.
        """
        return cls(
            voxel_size=1.0,           # Fine voxels
            target_faces=200000,      # High poly
            min_thickness=2.0,        # Thin walls allowed
            repair_passes=5,          # Very thorough repair
            boolean_engine='auto',
            validate_watertight=True,
            output_format='both',     # STL and STEP
        )
    
    @classmethod
    def from_preset(cls, preset: str) -> 'PipelineConfig':
        """Create config from named preset."""
        presets = {
            'preview': cls.preview,
            'standard': cls.standard,
            'production': cls.production,
        }
        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}. Choose from: {list(presets.keys())}")
        return presets[preset]()
    
    def with_overrides(self, **kwargs) -> 'PipelineConfig':
        """
        Create a new config with specific values overridden.
        
        Example:
            config = PipelineConfig.standard().with_overrides(voxel_size=1.5)
        """
        from dataclasses import asdict
        current = asdict(self)
        current.update(kwargs)
        return PipelineConfig(**current)
    
    def describe(self) -> str:
        """Human-readable description of current settings."""
        lines = [
            "Pipeline Configuration:",
            f"  Resolution:",
            f"    Voxel size:    {self.voxel_size} mm",
            f"    Target faces:  {self.target_faces:,}",
            f"    Min thickness: {self.min_thickness} mm",
            f"  Boolean engine:  {self.boolean_engine}",
            f"  Repair passes:   {self.repair_passes}",
            f"  Output format:   {self.output_format}",
            f"  Watertight check: {self.validate_watertight}",
        ]
        return "\n".join(lines)


# Default configuration instance
DEFAULT_CONFIG = PipelineConfig.standard()

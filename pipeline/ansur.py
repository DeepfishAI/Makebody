"""ANSUR data module - Parse anthropometric data and extract bone segment lengths."""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, List


@dataclass
class BoneLengths:
    """Bone segment lengths extracted from anthropometric measurements."""
    
    femur: float        # Thigh bone (hip to knee)
    tibia: float        # Shin bone (knee to ankle)
    humerus: float      # Upper arm (shoulder to elbow)
    forearm: float      # Lower arm (elbow to wrist)
    spine: float        # Torso (neck to hip)
    total_height: float # Overall stature
    
    def as_dict(self) -> Dict[str, float]:
        return {
            'femur': self.femur,
            'tibia': self.tibia,
            'humerus': self.humerus,
            'forearm': self.forearm,
            'spine': self.spine,
            'total_height': self.total_height,
        }
    
    def scale_factors_to(self, target: 'BoneLengths') -> Dict[str, float]:
        """
        Compute per-segment scale factors to match target bone lengths.
        
        Returns:
            Dictionary of scale factors per bone segment.
            Multiply source segment by factor to match target.
        """
        return {
            'femur': target.femur / self.femur if self.femur > 0 else 1.0,
            'tibia': target.tibia / self.tibia if self.tibia > 0 else 1.0,
            'humerus': target.humerus / self.humerus if self.humerus > 0 else 1.0,
            'forearm': target.forearm / self.forearm if self.forearm > 0 else 1.0,
            'spine': target.spine / self.spine if self.spine > 0 else 1.0,
            'total_height': target.total_height / self.total_height if self.total_height > 0 else 1.0,
        }


@dataclass 
class ANSURSubject:
    """
    Single subject from ANSUR II dataset.
    
    Contains key measurements needed for skeleton extraction.
    All measurements are in millimeters.
    """
    
    subject_id: int
    gender: str
    stature: float                    # Total height
    
    # Shoulder/arm landmarks
    acromialheight: float             # Top of shoulder
    acromionradialelength: float      # Shoulder to elbow (along arm)
    radialestylionlength: float       # Forearm length
    wristheight: float                # Wrist height from ground
    
    # Hip/leg landmarks
    trochanterionheight: float        # Hip joint (top of femur)
    iliocristaleheight: float         # Top of hip bone
    kneeheightmidpatella: float       # Knee height
    lateralmalleolusheight: float     # Ankle height
    tibialheight: float               # Shin bone reference
    
    # Spine landmarks
    cervicaleheight: float            # Base of neck (C7)
    suprasternaleheight: float        # Top of sternum
    
    # Additional useful measurements  
    waistcircumference: float = 0.0
    hipbreadth: float = 0.0
    chestcircumference: float = 0.0
    
    def bone_lengths(self) -> BoneLengths:
        """
        Extract bone segment lengths from anthropometric measurements.
        
        These are approximate bone lengths derived from external landmarks.
        """
        return BoneLengths(
            femur=self.trochanterionheight - self.kneeheightmidpatella,
            tibia=self.kneeheightmidpatella - self.lateralmalleolusheight,
            humerus=self.acromionradialelength,  # Direct measurement
            forearm=self.radialestylionlength,   # Direct measurement
            spine=self.cervicaleheight - self.iliocristaleheight,
            total_height=self.stature,
        )
    
    def skeleton_points(self) -> Dict[str, np.ndarray]:
        """
        Generate approximate skeleton joint positions.
        
        Assumes subject is standing upright facing +Y direction,
        centered at X=0, feet at Z=0.
        
        Returns:
            Dictionary of joint name -> [x, y, z] position
        """
        # Approximate body width for lateral positioning
        shoulder_half_width = 200  # mm, approximate
        hip_half_width = 100       # mm, approximate
        
        return {
            # Spine
            'head_top': np.array([0, 0, self.stature]),
            'c7_vertebra': np.array([0, 0, self.cervicaleheight]),
            'sternum_top': np.array([0, 0, self.suprasternaleheight]),
            'pelvis': np.array([0, 0, self.iliocristaleheight]),
            
            # Left arm
            'shoulder_left': np.array([-shoulder_half_width, 0, self.acromialheight]),
            'elbow_left': np.array([-shoulder_half_width - 50, 0, self.acromialheight - self.acromionradialelength]),
            'wrist_left': np.array([-shoulder_half_width - 80, 0, self.wristheight]),
            
            # Right arm
            'shoulder_right': np.array([shoulder_half_width, 0, self.acromialheight]),
            'elbow_right': np.array([shoulder_half_width + 50, 0, self.acromialheight - self.acromionradialelength]),
            'wrist_right': np.array([shoulder_half_width + 80, 0, self.wristheight]),
            
            # Left leg
            'hip_left': np.array([-hip_half_width, 0, self.trochanterionheight]),
            'knee_left': np.array([-hip_half_width, 0, self.kneeheightmidpatella]),
            'ankle_left': np.array([-hip_half_width, 0, self.lateralmalleolusheight]),
            
            # Right leg
            'hip_right': np.array([hip_half_width, 0, self.trochanterionheight]),
            'knee_right': np.array([hip_half_width, 0, self.kneeheightmidpatella]),
            'ankle_right': np.array([hip_half_width, 0, self.lateralmalleolusheight]),
        }


def load_ansur_dataset(csv_path: Path) -> pd.DataFrame:
    """
    Load ANSUR II dataset from CSV file.
    
    Args:
        csv_path: Path to ANSUR_II_FEMALE_Public.csv or ANSUR_II_MALE_Public.csv
        
    Returns:
        DataFrame with all measurements
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"ANSUR data file not found: {csv_path}")
    
    # Try different encodings
    for encoding in ['utf-8', 'latin-1', 'cp1252']:
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Could not read {csv_path} with any known encoding")
    
    # Normalize column names to lowercase
    df.columns = df.columns.str.lower().str.strip()
    
    print(f"Loaded {len(df)} subjects from {csv_path.name}")
    return df


def row_to_subject(row: pd.Series) -> ANSURSubject:
    """Convert a DataFrame row to ANSURSubject object."""
    
    def safe_get(name: str, default: float = 0.0) -> float:
        """Safely get value, handling missing columns."""
        if name in row.index:
            val = row[name]
            return float(val) if pd.notna(val) else default
        return default
    
    return ANSURSubject(
        subject_id=int(row.get('subjectid', 0)),
        gender=str(row.get('gender', 'Unknown')),
        stature=safe_get('stature'),
        acromialheight=safe_get('acromialheight'),
        acromionradialelength=safe_get('acromionradialelength'),
        radialestylionlength=safe_get('radialestylionlength'),
        wristheight=safe_get('wristheight'),
        trochanterionheight=safe_get('trochanterionheight'),
        iliocristaleheight=safe_get('iliocristaleheight'),
        kneeheightmidpatella=safe_get('kneeheightmidpatella'),
        lateralmalleolusheight=safe_get('lateralmalleolusheight'),
        tibialheight=safe_get('tibialheight'),
        cervicaleheight=safe_get('cervicaleheight'),
        suprasternaleheight=safe_get('suprasternaleheight'),
        waistcircumference=safe_get('waistcircumference'),
        hipbreadth=safe_get('hipbreadth'),
        chestcircumference=safe_get('chestcircumference'),
    )


def find_closest_subject(
    target_measurements: Dict[str, float],
    dataset: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
) -> ANSURSubject:
    """
    Find the ANSUR subject that best matches target measurements.
    
    Args:
        target_measurements: Dictionary of measurement_name -> value
            e.g., {'stature': 1650, 'waistcircumference': 750}
        dataset: ANSUR DataFrame
        weights: Optional weights for each measurement (higher = more important)
        
    Returns:
        ANSURSubject closest to the target measurements
    """
    if weights is None:
        weights = {k: 1.0 for k in target_measurements}
    
    # Compute weighted distance for each row
    distances = np.zeros(len(dataset))
    
    for measure, target_val in target_measurements.items():
        col_name = measure.lower()
        if col_name in dataset.columns:
            col_vals = dataset[col_name].fillna(target_val)
            weight = weights.get(measure, 1.0)
            # Normalized squared difference
            distances += weight * ((col_vals - target_val) / target_val) ** 2
    
    # Find minimum distance
    best_idx = distances.argmin()
    best_row = dataset.iloc[best_idx]
    
    print(f"Found closest match: Subject {best_row.get('subjectid', best_idx)}")
    
    return row_to_subject(best_row)


def get_statistics(dataset: pd.DataFrame, column: str) -> Dict[str, float]:
    """Get mean, std, min, max for a measurement column."""
    col = dataset[column.lower()]
    return {
        'mean': col.mean(),
        'std': col.std(),
        'min': col.min(),
        'max': col.max(),
        'median': col.median(),
    }


def compute_scale_factors(
    source_bones: BoneLengths,
    target_bones: BoneLengths,
) -> Dict[str, float]:
    """
    Compute scaling factors to transform source skeleton to match target.
    
    Args:
        source_bones: Bone lengths of the body to be scaled (e.g., female)
        target_bones: Bone lengths to match (e.g., male stunt performer)
        
    Returns:
        Dictionary of per-segment scale factors
    """
    factors = source_bones.scale_factors_to(target_bones)
    
    print("Scale factors (source -> target):")
    for bone, factor in factors.items():
        direction = "↑" if factor > 1.0 else "↓" if factor < 1.0 else "="
        print(f"  {bone}: {factor:.3f} {direction}")
    
    return factors

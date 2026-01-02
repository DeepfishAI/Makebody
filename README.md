# Prosthesis Pipeline

Python pipeline for generating 3D-printable prosthesis pieces from body scan boolean subtraction.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python main.py --male scans/male.obj --female scans/female.obj --landmarks landmarks.json --output output/
```

## Pipeline Stages

1. **Import** - Load OBJ/STL/PLY body scans
2. **Align** - Register female to male skeletal landmarks
3. **Scale** - Stretch female to male frame height
4. **Subtract** - Boolean CSG: Female − Male = prosthesis volumes
5. **Separate** - Isolate disconnected blobs
6. **Simplify** - Reduce mesh complexity
7. **Repair** - Make watertight
8. **Export** - Output STL/STEP files

## Landmark JSON Format

```json
{
  "male": {
    "head_top": [0, 180, 0],
    "shoulder_left": [-20, 150, 0],
    "shoulder_right": [20, 150, 0],
    "hip_left": [-15, 100, 0],
    "hip_right": [15, 100, 0],
    "ankle_left": [-10, 0, 0],
    "ankle_right": [10, 0, 0]
  },
  "female": {
    "head_top": [0, 165, 0],
    "shoulder_left": [-18, 140, 0],
    "shoulder_right": [18, 140, 0],
    "hip_left": [-16, 92, 0],
    "hip_right": [16, 92, 0],
    "ankle_left": [-8, 0, 0],
    "ankle_right": [8, 0, 0]
  }
}
```

## Output

Individual prosthesis pieces in `output/`:
- `chest_left.stl`
- `chest_right.stl`
- `hip_left.stl`
- `hip_right.stl`
- `buttocks.stl`
- etc.

## Requirements

- Python 3.10+
- FreeCAD (optional, for STEP export)

#!/usr/bin/env python3
"""
Prosthesis Pipeline CLI - Generate 3D-printable prosthesis from body scans.

Usage:
    python main.py --male scans/male.obj --female scans/female.obj --landmarks landmarks.json --output output/
    
With resolution presets:
    python main.py --male scans/male.obj --female scans/female.obj --landmarks landmarks.json --resolution preview
    python main.py --male scans/male.obj --female scans/female.obj --landmarks landmarks.json --resolution production

Override specific settings:
    python main.py ... --voxel-size 1.5 --target-faces 100000 --boolean-engine voxel
"""

import click
import json
from pathlib import Path

from pipeline import (
    PipelineConfig,
    load_scan,
    load_landmarks,
    align_meshes,
    scale_to_frame,
    subtract_bodies,
    subtract_with_config,
    separate_and_label,
    simplify_mesh,
    repair_mesh,
    repair_mesh_robust,
    export_blobs_stl,
)
from pipeline.blob_separation import get_blob_stats


@click.command()
@click.option('--male', '-m', required=True, type=click.Path(exists=True),
              help='Path to male stunt performer body scan (OBJ/STL/PLY)')
@click.option('--female', '-f', required=True, type=click.Path(exists=True),
              help='Path to female target body (OBJ/STL/PLY)')
@click.option('--landmarks', '-l', required=True, type=click.Path(exists=True),
              help='Path to landmarks JSON file')
@click.option('--output', '-o', default='output/', type=click.Path(),
              help='Output directory for prosthesis STL files')
# Resolution controls
@click.option('--resolution', '-r', 
              type=click.Choice(['preview', 'standard', 'production']),
              default='standard',
              help='Quality preset (affects voxel size, face count, etc.)')
@click.option('--voxel-size', type=float, default=None,
              help='Override voxel size in mm (smaller=more detail, slower)')
@click.option('--target-faces', type=int, default=None,
              help='Override target face count for simplification')
@click.option('--boolean-engine', 
              type=click.Choice(['manifold', 'voxel', 'auto']),
              default=None,
              help='Boolean engine (voxel guarantees watertight output)')
# Additional options
@click.option('--simplify/--no-simplify', default=True,
              help='Simplify output meshes')
@click.option('--min-volume', default=None, type=float,
              help='Minimum volume for a blob to be kept (mm³)')
@click.option('--offset', default=0.0, type=float,
              help='Offset from body surface for comfort gap (mm)')
@click.option('--strict/--no-strict', default=True,
              help='Fail if output is not watertight')
def main(male, female, landmarks, output, resolution, voxel_size, target_faces,
         boolean_engine, simplify, min_volume, offset, strict):
    """
    Generate prosthesis pieces from body scans.
    
    Takes a male stunt performer scan and a female target body,
    scales the female to match the male's frame, then subtracts
    the male from the female to produce prosthesis "blobs".
    
    OUTPUT IS GUARANTEED WATERTIGHT when using --boolean-engine voxel
    (which is the default when --strict is enabled).
    """
    # Build configuration from preset + overrides
    config = PipelineConfig.from_preset(resolution)
    
    # Apply CLI overrides
    overrides = {}
    if voxel_size is not None:
        overrides['voxel_size'] = voxel_size
    if target_faces is not None:
        overrides['target_faces'] = target_faces
    if boolean_engine is not None:
        overrides['boolean_engine'] = boolean_engine
    if min_volume is not None:
        overrides['min_blob_volume'] = min_volume
    if offset > 0:
        overrides['body_offset'] = offset
    overrides['validate_watertight'] = strict
    
    if overrides:
        config = config.with_overrides(**overrides)
    
    print("=" * 60)
    print("PROSTHESIS PIPELINE")
    print("=" * 60)
    print(config.describe())
    print("=" * 60)
    
    # Step 1: Load scans
    print("\n[1/7] Loading body scans...")
    male_mesh = load_scan(male)
    female_mesh = load_scan(female)
    
    # Step 2: Load landmarks
    print("\n[2/7] Loading landmarks...")
    lm = load_landmarks(landmarks)
    
    # Step 3: Align female to male
    print("\n[3/7] Aligning bodies...")
    female_aligned = align_meshes(female_mesh, lm['female'], lm['male'])
    
    # Step 4: Scale female to male frame
    print("\n[4/7] Scaling to match frame...")
    female_scaled = scale_to_frame(female_aligned, lm['female'], lm['male'])
    
    # Optional: Add offset for comfort
    if config.body_offset > 0:
        from pipeline.boolean_csg import add_offset
        male_mesh = add_offset(male_mesh, config.body_offset)
    
    # Step 5: Boolean subtraction (using config for engine selection)
    print("\n[5/7] Boolean subtraction (Female - Male)...")
    prosthesis = subtract_with_config(female_scaled, male_mesh, config)
    
    # Step 6: Separate and label blobs
    print("\n[6/7] Separating prosthesis blobs...")
    blobs = separate_and_label(prosthesis, min_volume=config.min_blob_volume)
    
    # Simplify and repair each blob
    if simplify:
        print("\n[7a/7] Simplifying meshes...")
        blobs = {
            label: simplify_mesh(mesh, target_faces=config.target_faces)
            for label, mesh in blobs.items()
        }
    
    print("\n[7b/7] Repairing meshes (ensuring watertight)...")
    repaired_blobs = {}
    for label, mesh in blobs.items():
        try:
            repaired_blobs[label] = repair_mesh_robust(mesh, config)
        except ValueError as e:
            print(f"  ERROR repairing {label}: {e}")
            if strict:
                raise
            repaired_blobs[label] = mesh  # Keep unrepaired if not strict
    blobs = repaired_blobs
    
    # Step 7: Export
    print("\n[7c/7] Exporting STL files...")
    output_dir = Path(output)
    export_blobs_stl(blobs, output_dir)
    
    # Write stats
    stats = get_blob_stats(blobs)
    stats_path = output_dir / "stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote stats to {stats_path}")
    
    # Write config used
    config_path = output_dir / "config.json"
    from dataclasses import asdict
    with open(config_path, 'w') as f:
        json.dump(asdict(config), f, indent=2)
    print(f"Wrote config to {config_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"\nGenerated {len(blobs)} prosthesis pieces:")
    for label, s in stats.items():
        vol_str = f"{s['volume']:.1f}" if s['volume'] else "N/A"
        wt = "✓" if s.get('watertight', False) else "✗"
        print(f"  - {label}: {s['vertices']:,} vertices, volume={vol_str}mm³, watertight={wt}")
    print(f"\nOutput directory: {output_dir.absolute()}")


if __name__ == '__main__':
    main()

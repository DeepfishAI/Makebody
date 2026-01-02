#!/usr/bin/env python3
"""
Prosthesis Pipeline CLI - Generate 3D-printable prosthesis from body scans.

Usage:
    python main.py --male scans/male.obj --female scans/female.obj --landmarks landmarks.json --output output/
"""

import click
import json
from pathlib import Path

from pipeline import (
    load_scan,
    load_landmarks,
    align_meshes,
    scale_to_frame,
    subtract_bodies,
    separate_and_label,
    simplify_mesh,
    repair_mesh,
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
@click.option('--simplify/--no-simplify', default=True,
              help='Simplify output meshes')
@click.option('--target-faces', default=10000, type=int,
              help='Target face count for simplification')
@click.option('--min-volume', default=10.0, type=float,
              help='Minimum volume for a blob to be kept')
@click.option('--offset', default=0.0, type=float,
              help='Offset from body surface (for comfort gap)')
def main(male, female, landmarks, output, simplify, target_faces, min_volume, offset):
    """
    Generate prosthesis pieces from body scans.
    
    Takes a male stunt performer scan and a female target body,
    scales the female to match the male's frame, then subtracts
    the male from the female to produce prosthesis "blobs".
    """
    print("=" * 60)
    print("PROSTHESIS PIPELINE")
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
    if offset > 0:
        from pipeline.boolean_csg import add_offset
        male_mesh = add_offset(male_mesh, offset)
    
    # Step 5: Boolean subtraction
    print("\n[5/7] Boolean subtraction (Female - Male)...")
    prosthesis = subtract_bodies(female_scaled, male_mesh)
    
    # Step 6: Separate and label blobs
    print("\n[6/7] Separating prosthesis blobs...")
    blobs = separate_and_label(prosthesis, min_volume=min_volume)
    
    # Simplify and repair each blob
    if simplify:
        print("\n[7a/7] Simplifying meshes...")
        blobs = {
            label: simplify_mesh(mesh, target_faces=target_faces)
            for label, mesh in blobs.items()
        }
    
    print("\n[7b/7] Repairing meshes...")
    blobs = {
        label: repair_mesh(mesh)
        for label, mesh in blobs.items()
    }
    
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
    
    # Summary
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"\nGenerated {len(blobs)} prosthesis pieces:")
    for label, s in stats.items():
        vol_str = f"{s['volume']:.1f}" if s['volume'] else "N/A"
        print(f"  - {label}: {s['vertices']:,} vertices, volume={vol_str}")
    print(f"\nOutput directory: {output_dir.absolute()}")


if __name__ == '__main__':
    main()

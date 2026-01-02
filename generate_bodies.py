#!/usr/bin/env python3
"""
Prosthesis Body Generator - Generate two body solids for CAD boolean operations.

Simplified workflow:
1. Generate/load ANSUR body (female target shape)
2. Load target body scan (male stunt performer)
3. Scale ANSUR body to match target skeleton
4. Export both bodies + skeleton as single file for CAD import

Usage:
    python generate_bodies.py --ansur-csv DATA/ANSUR_II_FEMALE.csv --subject-id 10001 \
                              --target scans/male.obj --output prosthesis_set.step

    python generate_bodies.py --measurements '{"stature": 1650, "waistcircumference": 750}' \
                              --target scans/male.obj --output prosthesis_set.step --interactive
"""

import click
import json
from pathlib import Path

from pipeline import (
    PipelineConfig,
    load_scan,
    load_ansur_dataset,
    find_closest_subject,
    generate_body_from_ansur,
    scale_by_height_bands,
    BoneEditor,
)
from pipeline.dual_export import export_dual_body_step, create_skeleton_mesh


@click.command()
@click.option('--ansur-csv', type=click.Path(exists=True),
              help='Path to ANSUR CSV file')
@click.option('--subject-id', type=int, default=None,
              help='ANSUR subject ID to use')
@click.option('--measurements', type=str, default=None,
              help='JSON string of measurements (alternative to ANSUR)')
@click.option('--target', '-t', required=True, type=click.Path(exists=True),
              help='Path to target body scan (OBJ/STL/PLY)')
@click.option('--output', '-o', default='prosthesis_set.step',
              help='Output file path (.step or .stl)')
@click.option('--interactive/--no-interactive', default=False,
              help='Open interactive bone editor for fine-tuning')
@click.option('--resolution', type=click.Choice(['preview', 'standard', 'production']),
              default='standard', help='Quality preset')
def main(ansur_csv, subject_id, measurements, target, output, interactive, resolution):
    """
    Generate two body solids for prosthesis creation.
    
    Outputs a single file containing:
    - Body A: ANSUR-based target shape (e.g., female actress)
    - Body B: Scanned body to match (e.g., male stunt performer)
    - Skeleton overlay for reference
    
    Import the output into your CAD app to perform boolean subtraction.
    """
    print("=" * 60)
    print("PROSTHESIS BODY GENERATOR")
    print("=" * 60)
    
    config = PipelineConfig.from_preset(resolution)
    print(config.describe())
    print("=" * 60)
    
    # Step 1: Get ANSUR subject
    print("\n[1/5] Loading ANSUR subject...")
    
    if ansur_csv:
        # Load from ANSUR dataset
        dataset = load_ansur_dataset(ansur_csv)
        
        if subject_id:
            # Find specific subject
            row = dataset[dataset['subjectid'] == subject_id].iloc[0]
            from pipeline.ansur import row_to_subject
            ansur_subject = row_to_subject(row)
        else:
            # Use first subject or average
            from pipeline.ansur import row_to_subject
            ansur_subject = row_to_subject(dataset.iloc[0])
            
    elif measurements:
        # Create from provided measurements
        meas = json.loads(measurements)
        from pipeline.ansur import ANSURSubject
        ansur_subject = ANSURSubject(
            subject_id=0,
            gender=meas.get('gender', 'female'),
            stature=meas.get('stature', 1650),
            acromialheight=meas.get('acromialheight', 1400),
            acromionradialelength=meas.get('acromionradialelength', 320),
            radialestylionlength=meas.get('radialestylionlength', 250),
            wristheight=meas.get('wristheight', 830),
            trochanterionheight=meas.get('trochanterionheight', 880),
            iliocristaleheight=meas.get('iliocristaleheight', 1000),
            kneeheightmidpatella=meas.get('kneeheightmidpatella', 480),
            lateralmalleolusheight=meas.get('lateralmalleolusheight', 70),
            tibialheight=meas.get('tibialheight', 430),
            cervicaleheight=meas.get('cervicaleheight', 1450),
            suprasternaleheight=meas.get('suprasternaleheight', 1380),
            waistcircumference=meas.get('waistcircumference', 750),
            hipbreadth=meas.get('hipbreadth', 350),
            chestcircumference=meas.get('chestcircumference', 900),
        )
    else:
        raise click.UsageError("Must provide --ansur-csv or --measurements")
    
    print(f"  Subject: {ansur_subject.subject_id}")
    print(f"  Gender: {ansur_subject.gender}")
    print(f"  Stature: {ansur_subject.stature}mm")
    
    # Step 2: Generate ANSUR body mesh
    print("\n[2/5] Generating ANSUR body mesh...")
    ansur_body = generate_body_from_ansur(ansur_subject)
    ansur_bones = ansur_subject.bone_lengths()
    skeleton_points = ansur_subject.skeleton_points()
    
    print(f"  Generated: {ansur_body.vertices.shape[0]:,} vertices")
    
    # Step 3: Load target body scan
    print("\n[3/5] Loading target body scan...")
    target_body = load_scan(target)
    print(f"  Loaded: {target_body.vertices.shape[0]:,} vertices")
    
    # Step 4: Scale ANSUR body to match target
    print("\n[4/5] Scaling ANSUR body to target skeleton...")
    
    # Compute target bone lengths from scan bounds (simplified)
    target_height = target_body.bounds[1, 2] - target_body.bounds[0, 2]
    height_ratio = target_height / ansur_subject.stature
    
    from pipeline.ansur import BoneLengths
    target_bones = BoneLengths(
        femur=ansur_bones.femur * height_ratio,
        tibia=ansur_bones.tibia * height_ratio,
        humerus=ansur_bones.humerus * height_ratio,
        forearm=ansur_bones.forearm * height_ratio,
        spine=ansur_bones.spine * height_ratio,
        total_height=target_height,
    )
    
    # Apply height-band scaling
    ansur_body_scaled = scale_by_height_bands(
        ansur_body, ansur_bones, target_bones
    )
    
    # Update skeleton points to match scaled body
    for joint_name in skeleton_points:
        skeleton_points[joint_name] = skeleton_points[joint_name] * height_ratio
    
    # Interactive adjustment
    if interactive:
        print("\n[4b/5] Interactive bone adjustment...")
        try:
            editor = BoneEditor(ansur_body_scaled, target_bones)
            scale_factors = editor.run()
            
            # Re-apply scaling with adjusted bones
            adjusted_bones = BoneLengths(
                femur=target_bones.femur * scale_factors.get('femur', 1.0),
                tibia=target_bones.tibia * scale_factors.get('tibia', 1.0),
                humerus=target_bones.humerus * scale_factors.get('humerus', 1.0),
                forearm=target_bones.forearm * scale_factors.get('forearm', 1.0),
                spine=target_bones.spine * scale_factors.get('spine', 1.0),
                total_height=target_bones.total_height,
            )
            ansur_body_scaled = scale_by_height_bands(
                ansur_body, ansur_bones, adjusted_bones
            )
        except ImportError:
            print("  Skipping interactive mode (Open3D not installed)")
    
    # Step 5: Export
    print("\n[5/5] Exporting bodies...")
    output_path = Path(output)
    
    success = export_dual_body_step(
        body_a=ansur_body_scaled,
        body_b=target_body,
        skeleton_points=skeleton_points,
        output_path=output_path,
        body_a_name="ANSUR_Target_Body",
        body_b_name="Scan_Reference_Body",
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"\nOutput: {output_path.absolute()}")
    print("\nNext steps:")
    print("  1. Import into CAD app (Fusion 360, SolidWorks, etc.)")
    print("  2. Boolean subtract: ANSUR body - Scan body = Prosthesis")
    print("  3. Cleanup and export for 3D printing")


if __name__ == '__main__':
    main()

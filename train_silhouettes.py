#!/usr/bin/env python3
"""
Training/Batch Generator - Pre-generate silhouettes for all ANSUR subjects.

This script processes the entire ANSUR dataset and generates:
1. Silhouette data (JSON) for each subject
2. PNG preview images for quick browsing
3. Summary statistics

Usage:
    python train_silhouettes.py --gender female --output silhouettes/
    python train_silhouettes.py --gender both --output silhouettes/ --images
"""

import json
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from pipeline import load_ansur_dataset
from pipeline.ansur import row_to_subject
from pipeline.silhouette import generate_silhouette_points, silhouette_to_canvas_points

# Optional: PIL for image generation
try:
    from PIL import Image, ImageDraw
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def generate_silhouette_png(
    subject_data: dict,
    output_path: Path,
    width: int = 200,
    height: int = 400,
) -> None:
    """Generate a PNG preview of the silhouette."""
    if not HAS_PIL:
        return
    
    points = subject_data['raw_points']
    if not points:
        return
    
    # Create image
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Find bounds
    max_height = max(p[0] for p in points)
    max_width = max(p[1] for p in points) * 2
    
    # Scale
    padding = 10
    scale = min((width - 2 * padding) / max_width, (height - 2 * padding) / max_height)
    center_x = width / 2
    bottom_y = height - padding
    
    # Sort by height
    sorted_pts = sorted(points, key=lambda p: p[0])
    
    # Build polygon
    polygon = []
    
    # Left side
    for h, w in sorted_pts:
        x = center_x - w * scale
        y = bottom_y - h * scale
        polygon.append((x, y))
    
    # Right side (reversed)
    for h, w in reversed(sorted_pts):
        x = center_x + w * scale
        y = bottom_y - h * scale
        polygon.append((x, y))
    
    # Draw filled polygon
    draw.polygon(polygon, fill=(88, 166, 255, 128), outline=(88, 166, 255, 255))
    
    img.save(output_path)


def process_subject(row, gender: str, output_dir: Path, generate_images: bool):
    """Process a single subject."""
    try:
        subject = row_to_subject(row)
        subject_id = int(row.get('subjectid', 0))
        
        # Generate silhouette points
        points = generate_silhouette_points(subject)
        canvas_data = silhouette_to_canvas_points(points)
        
        # Build data
        data = {
            'subject_id': subject_id,
            'gender': gender,
            'stature': subject.stature,
            'raw_points': [[p[0], p[1]] for p in points],
            'canvas': canvas_data,
        }
        
        # Save JSON
        json_path = output_dir / 'json' / f"{subject_id}.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(data, f)
        
        # Generate image if requested
        if generate_images:
            img_path = output_dir / 'images' / f"{subject_id}.png"
            img_path.parent.mkdir(parents=True, exist_ok=True)
            generate_silhouette_png(data, img_path)
        
        return subject_id, True, None
        
    except Exception as e:
        return row.get('subjectid', 'unknown'), False, str(e)


def run_training(
    gender: str = 'both',
    output_dir: Path = Path('silhouettes'),
    generate_images: bool = False,
    max_workers: int = 4,
    limit: int = None,
):
    """
    Process all ANSUR subjects and generate silhouettes.
    
    Args:
        gender: 'female', 'male', or 'both'
        output_dir: Directory to save output
        generate_images: Whether to generate PNG previews
        max_workers: Number of parallel workers
        limit: Optional limit on number of subjects to process
    """
    print("=" * 60)
    print("SILHOUETTE TRAINING")
    print("=" * 60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    genders_to_process = ['female', 'male'] if gender == 'both' else [gender]
    
    all_results = []
    
    for g in genders_to_process:
        print(f"\nProcessing {g} subjects...")
        
        # Load dataset
        data_dir = Path(__file__).parent / "DATA"
        csv_path = data_dir / f"ANSUR_II_{g.upper()}_Public (1).csv"
        
        if not csv_path.exists():
            print(f"  Dataset not found: {csv_path}")
            continue
        
        df = load_ansur_dataset(csv_path)
        
        if limit:
            df = df.head(limit)
        
        total = len(df)
        print(f"  Found {total} subjects")
        
        # Create output subdirectory for gender
        gender_dir = output_dir / g
        gender_dir.mkdir(parents=True, exist_ok=True)
        
        # Process subjects
        start_time = time.time()
        success = 0
        failed = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_subject, row, g, gender_dir, generate_images): i
                for i, (_, row) in enumerate(df.iterrows())
            }
            
            for i, future in enumerate(as_completed(futures)):
                subject_id, ok, error = future.result()
                
                if ok:
                    success += 1
                else:
                    failed += 1
                    print(f"    ERROR: Subject {subject_id}: {error}")
                
                # Progress
                if (i + 1) % 100 == 0 or (i + 1) == total:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    print(f"    Progress: {i + 1}/{total} ({rate:.1f}/sec)")
        
        all_results.append({
            'gender': g,
            'total': total,
            'success': success,
            'failed': failed,
        })
        
        print(f"  Complete: {success} success, {failed} failed")
    
    # Write summary
    summary = {
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'results': all_results,
        'images_generated': generate_images,
    }
    
    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary written to: {summary_path}")
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Generate silhouettes for ANSUR dataset')
    parser.add_argument('--gender', choices=['female', 'male', 'both'], default='both',
                        help='Which gender(s) to process')
    parser.add_argument('--output', '-o', default='silhouettes',
                        help='Output directory')
    parser.add_argument('--images', action='store_true',
                        help='Generate PNG preview images')
    parser.add_argument('--workers', '-w', type=int, default=4,
                        help='Number of parallel workers')
    parser.add_argument('--limit', '-n', type=int, default=None,
                        help='Limit number of subjects (for testing)')
    
    args = parser.parse_args()
    
    run_training(
        gender=args.gender,
        output_dir=Path(args.output),
        generate_images=args.images,
        max_workers=args.workers,
        limit=args.limit,
    )


if __name__ == '__main__':
    main()

"""
Makebody Browser UI - ANSUR subject browser and body generation interface.

Run with: python app.py
Open: http://localhost:5000
"""

from flask import Flask, render_template, jsonify, request, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
from pathlib import Path
import json
import tempfile

from pipeline import (
    load_ansur_dataset,
    load_scan,
    generate_body_from_ansur,
    scale_by_height_bands,
    BoneLengths,
)
from pipeline.ansur import row_to_subject, ANSURSubject
from pipeline.dual_export import export_dual_body_step

app = Flask(__name__)
CORS(app)

# Data paths
DATA_DIR = Path(__file__).parent / "DATA"
ANSUR_FEMALE = DATA_DIR / "ANSUR_II_FEMALE_Public (1).csv"
ANSUR_MALE = DATA_DIR / "ANSUR_II_MALE_Public (1).csv"

# Cache loaded datasets
_datasets = {}


def get_dataset(gender: str) -> pd.DataFrame:
    """Load and cache ANSUR dataset."""
    if gender not in _datasets:
        path = ANSUR_FEMALE if gender == 'female' else ANSUR_MALE
        if path.exists():
            _datasets[gender] = load_ansur_dataset(path)
        else:
            _datasets[gender] = pd.DataFrame()
    return _datasets[gender]


@app.route('/')
def index():
    """Main page with ANSUR browser."""
    return render_template('index.html')


@app.route('/api/subjects')
def list_subjects():
    """List available ANSUR subjects."""
    gender = request.args.get('gender', 'female')
    limit = int(request.args.get('limit', 100))
    
    df = get_dataset(gender)
    
    if df.empty:
        return jsonify({'error': 'Dataset not found', 'subjects': []})
    
    # Return key fields for browsing
    subjects = []
    for _, row in df.head(limit).iterrows():
        subjects.append({
            'id': int(row.get('subjectid', 0)),
            'stature': float(row.get('stature', 0)),
            'weight_lbs': float(row.get('weightlbs', 0)),
            'age': int(row.get('age', 0)),
            'waist': float(row.get('waistcircumference', 0)),
            'chest': float(row.get('chestcircumference', 0)),
            'hip': float(row.get('hipbreadth', 0)),
        })
    
    return jsonify({
        'gender': gender,
        'count': len(subjects),
        'subjects': subjects,
    })


@app.route('/api/subject/<int:subject_id>')
def get_subject(subject_id: int):
    """Get full details for a specific subject."""
    gender = request.args.get('gender', 'female')
    df = get_dataset(gender)
    
    if df.empty:
        return jsonify({'error': 'Dataset not found'}), 404
    
    row = df[df['subjectid'] == subject_id]
    if row.empty:
        return jsonify({'error': 'Subject not found'}), 404
    
    row = row.iloc[0]
    subject = row_to_subject(row)
    bones = subject.bone_lengths()
    
    return jsonify({
        'id': subject_id,
        'gender': subject.gender,
        'stature': subject.stature,
        'measurements': {
            'acromialheight': subject.acromialheight,
            'cervicaleheight': subject.cervicaleheight,
            'waistcircumference': subject.waistcircumference,
            'chestcircumference': subject.chestcircumference,
            'hipbreadth': subject.hipbreadth,
            'kneeheightmidpatella': subject.kneeheightmidpatella,
            'trochanterionheight': subject.trochanterionheight,
        },
        'bones': bones.as_dict(),
    })


@app.route('/api/skeleton/<int:subject_id>')
def get_skeleton(subject_id: int):
    """Get skeleton joint positions for visualization."""
    gender = request.args.get('gender', 'female')
    df = get_dataset(gender)
    
    if df.empty:
        return jsonify({'error': 'Dataset not found'}), 404
    
    row = df[df['subjectid'] == subject_id]
    if row.empty:
        return jsonify({'error': 'Subject not found'}), 404
    
    subject = row_to_subject(row.iloc[0])
    skeleton = subject.skeleton_points()
    
    # Convert numpy arrays to lists for JSON
    skeleton_json = {k: v.tolist() for k, v in skeleton.items()}
    
    return jsonify({
        'subject_id': subject_id,
        'joints': skeleton_json,
        'bones': [
            ['ankle_left', 'knee_left'],
            ['ankle_right', 'knee_right'],
            ['knee_left', 'hip_left'],
            ['knee_right', 'hip_right'],
            ['hip_left', 'hip_right'],
            ['hip_left', 'pelvis'],
            ['hip_right', 'pelvis'],
            ['pelvis', 'shoulders'],
            ['shoulders', 'shoulder_left'],
            ['shoulders', 'shoulder_right'],
            ['shoulder_left', 'elbow_left'],
            ['shoulder_right', 'elbow_right'],
            ['elbow_left', 'wrist_left'],
            ['elbow_right', 'wrist_right'],
            ['shoulders', 'neck'],
            ['neck', 'head'],
        ],
    })


@app.route('/api/generate', methods=['POST'])
def generate_bodies():
    """Generate bodies and export."""
    data = request.json
    
    subject_id = data.get('subject_id')
    gender = data.get('gender', 'female')
    target_height = data.get('target_height')  # mm
    bone_adjustments = data.get('bone_adjustments', {})
    
    # Get ANSUR subject
    df = get_dataset(gender)
    row = df[df['subjectid'] == subject_id].iloc[0]
    ansur_subject = row_to_subject(row)
    
    # Generate ANSUR body
    ansur_body = generate_body_from_ansur(ansur_subject)
    ansur_bones = ansur_subject.bone_lengths()
    skeleton_points = ansur_subject.skeleton_points()
    
    # Compute target bones
    if target_height:
        height_ratio = target_height / ansur_subject.stature
    else:
        height_ratio = 1.0
    
    target_bones = BoneLengths(
        femur=ansur_bones.femur * height_ratio * bone_adjustments.get('femur', 1.0),
        tibia=ansur_bones.tibia * height_ratio * bone_adjustments.get('tibia', 1.0),
        humerus=ansur_bones.humerus * height_ratio * bone_adjustments.get('humerus', 1.0),
        forearm=ansur_bones.forearm * height_ratio * bone_adjustments.get('forearm', 1.0),
        spine=ansur_bones.spine * height_ratio * bone_adjustments.get('spine', 1.0),
        total_height=target_height or ansur_subject.stature,
    )
    
    # Scale body
    scaled_body = scale_by_height_bands(ansur_body, ansur_bones, target_bones)
    
    # Update skeleton for scaled body
    for joint_name in skeleton_points:
        skeleton_points[joint_name] = skeleton_points[joint_name] * height_ratio
    
    # Export to temp file
    with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
        output_path = Path(f.name)
    
    scaled_body.export(str(output_path), file_type='stl')
    
    return jsonify({
        'success': True,
        'download_url': f'/api/download/{output_path.name}',
        'vertices': scaled_body.vertices.shape[0],
        'faces': scaled_body.faces.shape[0],
    })


@app.route('/api/download/<filename>')
def download_file(filename):
    """Download generated file."""
    path = Path(tempfile.gettempdir()) / filename
    if path.exists():
        return send_file(path, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404


if __name__ == '__main__':
    print("=" * 50)
    print("MAKEBODY - Browser UI")
    print("=" * 50)
    print("Open http://localhost:5000 in your browser")
    print("=" * 50)
    app.run(debug=True, port=5000)

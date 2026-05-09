# RapidAid — Accident Detection System

**Vision-Based Automated Accident Detection and Victim Condition Assessment**

## Overview

RapidAid analyzes CCTV footage to detect:
1. **Vehicles involved** in an accident (not parked/background vehicles)
2. **Victims** lying on the ground or trapped (not bystanders/helpers)
3. **Accident zone** — tight bounding area around the crash scene

All detections are highlighted on the frame and returned as structured JSON data.

## Architecture

```
Frame/Video → Vehicle Detection → Person Detection → Accident Classification
                                                         ↓
            Annotated Frame + JSON ← Report Generation ← Victim Classification
                                                         ↓
                                                    Zone Computation
```

### Modules
| Module | Purpose | Model |
|--------|---------|-------|
| Vehicle Detector | Detect all vehicles with segmentation masks | YOLOv8s-seg |
| Person Detector | Detect all persons with skeleton keypoints | YOLOv8s-pose |
| Accident Classifier | Score vehicle pairs for crash involvement | Multi-signal scoring |
| Victim Classifier | Classify persons as victim/bystander | Keypoint + spatial analysis |
| Zone Calculator | Compute consistent accident zone | Geometric algorithm |

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# YOLO weights are auto-downloaded on first run
```

## Usage

### Process a single image
```bash
python main.py --image data/test_frames/accident.jpg
```

### Process a video
```bash
python main.py --video data/test_videos/crash.mp4
```

### Options
```bash
--no-stop      # Don't stop at first accident (analyze full video)
--no-display   # Don't show results (save only)
```

## Output

### Annotated Frame
- **Red** — Vehicles involved in the accident
- **Orange** — Victims (trapped, fallen, or nearby)
- **Blue** — Accident zone boundary

### JSON Report
```json
{
  "accident_detected": true,
  "involved_vehicles": [
    {"type": "Car", "confidence": 87.8, "crash_score": 0.72, "bbox": [...]}
  ],
  "accident_zone": [x1, y1, x2, y2],
  "victims_in_zone": 2,
  "victims": [
    {"id": 1, "status": "fallen", "confidence": 82.3, "bbox": [...]}
  ]
}
```

## Project Structure
```
AccidentDetection/
├── config/           # Settings, thresholds, vehicle class mappings
├── models/           # Detection engines (vehicle, person, accident, victim, zone)
├── pipeline/         # Orchestration (frame processor, video processor, reports)
├── utils/            # Geometry, visualization, helpers
├── weights/          # Model weights (auto-downloaded)
├── outputs/          # Annotated frames and JSON reports
├── data/             # Test images and videos
├── main.py           # CLI entry point
└── requirements.txt
```

## GPU Training (Colab)

To improve accuracy with custom-trained models:
1. Use the training notebook in `notebooks/training.ipynb`
2. Train on accident datasets (CADP, DoTA, CCD)
3. Download trained weights to `weights/` directory

# RapidAid-Accident-Detection-System
Vision-Based Automated Accident Detection and Victim Condition Assessment

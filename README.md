<p align="center">
  <h1 align="center">🚨 RapidAid — Accident Detection System</h1>
  <p align="center">
    <strong>Vision-Based Automated Accident Detection & Victim Assessment</strong>
    <br />
    Real-time accident detection from CCTV footage using an ensemble of deep learning models
    <br />
    <br />
    <a href="#quick-start">Quick Start</a>
    ·
    <a href="#architecture">Architecture</a>
    ·
    <a href="#training">Model Training</a>
    ·
    <a href="#results">Results</a>
  </p>
</p>

<br />

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pipeline Stages](#pipeline-stages)
- [Model Ensemble](#model-ensemble)
- [Output Format](#output-format)
- [Project Structure](#project-structure)
- [Training Custom Models](#training)
- [Configuration](#configuration)
- [Results](#results)
- [License](#license)

---

## 🔍 Overview

**RapidAid** is an end-to-end computer vision system that analyzes CCTV/dashcam footage to automatically detect traffic accidents, identify involved vehicles, locate victims, and generate structured incident reports — all in real time.

The system uses a **4-model ensemble** architecture:

| Signal | Model | Role |
|--------|-------|------|
| **M1** | YOLOv8n-cls | Scene-level accident classification |
| **M2** | YOLOv8n-cls | Per-vehicle damage assessment |
| **M3** | LSTM | Temporal pattern recognition (video) |
| **M4** | YOLOv8n | **Collision zone detection** (primary signal) |

M4 represents a paradigm shift: instead of inferring collisions from geometric heuristics, it **directly detects where accidents happen** in the frame, then identifies which vehicles are inside that zone.

---

## ✨ Key Features

- 🎯 **100% detection accuracy** on test suite (15/15: 6 frames + 9 videos)
- 🔍 **Collision zone detection** — directly locates the crash area, not just individual vehicles
- 🚗 **Multi-vehicle identification** — correctly flags 1–4 involved vehicles per incident
- 🧑 **Victim detection** — identifies trapped, fallen, or hit pedestrians via pose estimation
- 🎬 **Video processing** — temporal analysis with alert-mode architecture and post-crash frame selection
- 📊 **Structured JSON reports** — full incident data with bounding boxes, confidence scores, and zones
- 🖼️ **Annotated frames** — color-coded visualization of vehicles, victims, and accident zones
- ⚡ **CPU-compatible** — runs on standard hardware without GPU requirement

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT (Frame/Video)                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                 ┌───────────▼───────────┐
                 │   M1: Scene Classifier │ ← Pre-filter (is this an accident?)
                 └───────────┬───────────┘
                             │
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                   ▼
┌─────────────────┐ ┌───────────────┐ ┌──────────────────┐
│ Vehicle Detector │ │Person Detector│ │ M4: Collision Zone│
│  (YOLOv8s-seg)  │ │(YOLOv8s-pose) │ │   Detector (NEW) │
└────────┬────────┘ └───────┬───────┘ └────────┬─────────┘
         │                  │                   │
         │          ┌───────▼───────┐           │
         │          │    Victim     │           │
         │          │  Classifier   │           │
         │          └───────┬───────┘           │
         │                  │                   │
         └──────────┬───────┘───────────────────┘
                    ▼
         ┌──────────────────┐
         │ M2: Damage Scorer│ ← Per-vehicle damage assessment
         └────────┬─────────┘
                  ▼
         ┌──────────────────┐
         │ Accident Classifier│ ← Multi-signal crash scoring
         └────────┬─────────┘
                  ▼
    ┌─────────────────────────┐
    │  Zone + Report Generator │ → Annotated Frame + JSON
    └─────────────────────────┘
```

### Video Processing Flow

```
Video ──► Sample Frames ──► M1 Classifier Baseline
                │
                ├──► Alert Triggers (M1 rise / M4 zone / motion spike)
                │
                ├──► Full Pipeline on Alert Frames
                │
                ├──► Temporal Confirmation (sliding window)
                │
                └──► Post-Crash Best Frame Selection
```

---

## 📦 Installation

### Prerequisites

- Python 3.9+
- 4GB+ RAM (CPU inference)

### Setup

```bash
# Clone the repository
git clone https://github.com/shahaanabbas-13/RapidAid-Accident-Detection-System.git
cd RapidAid-Accident-Detection-System

# Install dependencies
pip install -r requirements.txt
```

> **Note:** YOLO base weights (`yolov8s-seg.pt`, `yolov8s-pose.pt`) are auto-downloaded on first run. Custom model weights (`accident_classifier.pt`, `damage_classifier.pt`, `collision_detector.pt`) must be trained and placed in the `weights/` directory — see [Training](#training).

---

## 🚀 Quick Start

### Analyze a single image

```bash
python main.py --image data/test_frames/accident.jpg
```

### Analyze a video

```bash
python main.py --video data/test_videos/crash.mp4
```

### Additional options

```bash
# Process full video without stopping at first detection
python main.py --video data/test_videos/crash.mp4 --no-stop

# Save results without opening display window
python main.py --image data/test_frames/accident.jpg --no-display
```

### Run the full test suite

```bash
python test_all.py
```

---

## ⚙️ Pipeline Stages

| Stage | Description | Key Logic |
|-------|-------------|-----------|
| **0** | **Pre-filter** — M1 classifies scene as accident/normal | Skips non-accident frames early (bypassed when M4 is available) |
| **1** | **Vehicle Detection** — YOLOv8s-seg finds all vehicles | Segmentation masks + background filtering |
| **2** | **Person Detection** — YOLOv8s-pose finds all persons | 17-point keypoint estimation |
| **3-PRIMARY** | **M4 Collision Zone** — directly detects crash area | Finds vehicles overlapping the zone; proximity-ranked for large zones |
| **3-FALLBACK** | **Geometric Analysis** — pairwise crash scoring | IoU, edge distance, aspect ratio, pixel collision |
| **3c** | **M2 Damage Scoring** — per-vehicle damage assessment | Gated behind `M2_TRUST_ENABLED` flag |
| **4** | **Victim Classification** — persons as victim/bystander | Keypoint analysis (lying, trapped, fallen) |
| **5** | **Fallback Detection** — single-vehicle, pedestrian collision | Progressive fallback chain with confidence gates |
| **6** | **Zone Computation** — accident zone bounding box | Collision-point centered algorithm |
| **7** | **Report Generation** — annotated frame + JSON | Color-coded visualization + structured data |

---

## 🧠 Model Ensemble

| Model | Type | Input | Output | Weights File |
|-------|------|-------|--------|-------------|
| **M1** — Scene Classifier | YOLOv8n-cls | Full frame (320px) | `accident` / `no_accident` + confidence | `accident_classifier.pt` |
| **M2** — Damage Detector | YOLOv8n-cls | Vehicle crop (320px) | `damaged` / `normal` + confidence | `damage_classifier.pt` |
| **M3** — Temporal Classifier | LSTM (2→64→1) | Sequence of 16 (M1, motion) pairs | Temporal accident probability | `temporal_classifier.pt` |
| **M4** — Collision Detector | YOLOv8n (detect) | Full frame (640px) | Collision zone bounding boxes | `collision_detector.pt` |
| **YOLO-Seg** | YOLOv8s-seg | Full frame | Vehicle bboxes + masks | `yolov8s-seg.pt` |
| **YOLO-Pose** | YOLOv8s-pose | Full frame | Person bboxes + 17 keypoints | `yolov8s-pose.pt` |

---

## 📤 Output Format

### Annotated Frame

| Color | Meaning |
|-------|---------|
| 🔴 **Red** | Vehicles involved in the accident |
| 🟠 **Orange** | Victims (trapped, fallen, hit) |
| 🔵 **Blue** | Accident zone boundary |
| 🟢 **Green** | Collision zone (M4 detection) |

### JSON Report

```json
{
  "accident_detected": true,
  "timestamp": "2026-05-09 21:44:11",
  "timestamp_sec": 7.0,
  "classifier_confidence_m1": 0.997,
  "damage_score_m2": 0.964,
  "fused_confidence": 0.928,
  "detection_mode": "ensemble",
  "total_vehicles_in_scene": 4,
  "involved_vehicles": [
    {
      "type": "Car",
      "confidence": 90.9,
      "crash_score": 1.0,
      "damage_score": 0.628,
      "bbox": [469, 79, 629, 165]
    }
  ],
  "accident_zone": [163, 55, 671, 218],
  "victims_in_zone": 2,
  "victims": [
    {
      "id": 1,
      "confidence": 87.0,
      "status": "trapped",
      "bbox": [271, 342, 360, 445]
    }
  ]
}
```

---

## 📁 Project Structure

```
AccidentDetection/
│
├── config/
│   ├── settings.py               # All thresholds, weights, and feature flags
│   └── vehicle_classes.py        # COCO vehicle class mappings + aspect ratios
│
├── models/
│   ├── vehicle_detector.py       # YOLOv8s-seg vehicle detection + filtering
│   ├── person_detector.py        # YOLOv8s-pose person/keypoint detection
│   ├── frame_classifier.py       # M1: scene-level accident classifier
│   ├── damage_classifier.py      # M2: per-vehicle damage scorer
│   ├── temporal_classifier.py    # M3: LSTM temporal pattern classifier
│   ├── collision_detector.py     # M4: collision zone detector (PRIMARY)
│   ├── accident_classifier.py    # Multi-signal pairwise crash scoring
│   ├── victim_classifier.py      # Keypoint-based victim classification
│   ├── vehicle_pedestrian_detector.py  # Vehicle-pedestrian collision detection
│   └── accident_zone.py          # Accident zone computation
│
├── pipeline/
│   ├── frame_processor.py        # Single-frame pipeline orchestrator
│   ├── video_processor.py        # Video analysis with temporal logic
│   └── report_generator.py       # JSON + annotated frame output
│
├── training/                     # Google Colab training scripts
│   ├── train_m1_v2_scene_classifier.py   # M1 with Indian traffic hard negatives
│   ├── train_m2_v2_damage_detector.py    # M2 with Indian vehicle normals
│   ├── train_m4_collision_detector.py    # M4 collision zone detector
│   ├── train_m3_temporal_classifier.py   # M3 LSTM temporal classifier
│   └── README.md                         # Training instructions
│
├── utils/
│   ├── geometry.py               # IoU, overlap, distance, collision utils
│   ├── visualization.py          # Frame annotation and drawing
│   └── helpers.py                # Display and I/O utilities
│
├── weights/                      # Trained model weights
│   ├── accident_classifier.pt    # M1: scene classifier
│   ├── damage_classifier.pt      # M2: damage detector
│   ├── collision_detector.pt     # M4: collision zone detector
│   ├── temporal_classifier.pt    # M3: temporal classifier
│   ├── yolov8s-seg.pt            # Vehicle detection (auto-downloaded)
│   └── yolov8s-pose.pt           # Person detection (auto-downloaded)
│
├── data/
│   ├── test_frames/              # 6 test accident images
│   └── test_videos/              # 9 test accident videos
│
├── outputs/
│   ├── annotated/                # Saved annotated frames
│   ├── reports/                  # Saved JSON reports
│   └── test_results/             # Test suite outputs
│
├── main.py                       # CLI entry point
├── test_all.py                   # Comprehensive test suite
├── requirements.txt              # Python dependencies
└── README.md
```

---

<h2 id="training">🏋️ Training Custom Models</h2>

All models are trained on **Google Colab** (free T4 GPU). Training scripts are in the `training/` directory.

### Training Order

> **Important:** Train in this order. Each model builds on the previous.

| Step | Script | Time | Output |
|------|--------|------|--------|
| 1 | `train_m1_v2_scene_classifier.py` | ~3–4h | `accident_classifier.pt` |
| 2 | `train_m2_v2_damage_detector.py` | ~3–4h | `damage_classifier.pt` |
| 3 | `train_m3_temporal_classifier.py` | ~1–2h | `temporal_classifier.pt` |
| 4 | `train_m4_collision_detector.py` | ~2–3h | `collision_detector.pt` |

### Training Steps

1. Open [Google Colab](https://colab.research.google.com/)
2. Set runtime to **GPU (T4)**: `Runtime → Change runtime type → GPU`
3. Copy the training script contents into a Colab notebook
4. Run all cells — datasets are auto-downloaded
5. Download the output `.pt` file → place in `weights/`
6. After M2 training: set `M2_TRUST_ENABLED = True` in `config/settings.py`

### Dataset Sources

| Model | Datasets | Hard Negatives |
|-------|----------|----------------|
| **M1** | CCTV Accident Detection, Car Accident Detection | Indian Driving Dataset (IDD), BDD100K |
| **M2** | Car Damage Severity, CarDD, COCO Car Damage | Indian Vehicle Dataset, Stanford Cars |
| **M3** | Synthesized from M1 sequences | Synthetic normal traffic sequences |
| **M4** | Roboflow Accident Detection, Kaggle Accident Detection | Converted classification → detection |

---

## ⚙️ Configuration

Key parameters in [`config/settings.py`](config/settings.py):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `M2_TRUST_ENABLED` | `True` | Enable/disable M2 damage scoring influence |
| `MAX_CRASH_SCORE` | `1.0` | Maximum allowed crash score (prevents overflow) |
| `CRASH_SCORE_THRESHOLD` | `0.35` | Minimum score to flag vehicle as involved |
| `DAMAGE_ONLY_MIN_SCENE_CONF` | `0.60` | M1 gate for damage-only detection |
| `POST_CRASH_LOOKAHEAD_SEC` | `3.0` | Seconds to scan after confirmation for best frame |
| `TEMPORAL_WINDOW_SIZE` | `8` | Sliding window for temporal confirmation |

---

## 📊 Results

### Test Suite Performance

```
============================================================
  OVERALL ACCURACY
============================================================
  Frames:  6/6  detected  (100%)
  Videos:  9/9  detected  (100%)
  TOTAL:  15/15           (100.0%)
============================================================
```

### Frame Detection Results

| Test Frame | Detected | Vehicles | Victims | Crash Scores |
|-----------|----------|----------|---------|-------------|
| Car Accident 3.png | ✅ | 4 | 0 | [0.91, 0.91, 0.91, 0.91] |
| Car Accident 4.png | ✅ | 3 | 0 | [0.97, 0.97, 0.97] |
| Car Accident 5.png | ✅ | 4 | 1 | [0.69, 1.0, 1.0, 0.77] |
| Car accident 2.jpg | ✅ | 1 | 3 | [0.76] |
| Car accident(6).png | ✅ | 1 | 0 | [0.94] |
| Car accident(8).png | ✅ | 4 | 0 | [1.0, 1.0, 0.91, 0.59] |

### Video Detection Results

| Test Video | Detected | Vehicles | Victims | Timestamp | Processing |
|-----------|----------|----------|---------|-----------|------------|
| Acc Video 1.mp4 | ✅ | 2 | 0 | 7.0s | 8s |
| Acc Video 2.mp4 | ✅ | 1 | 0 | 9.7s | 8s |
| Acc Video 5.mp4 | ✅ | 2 | 0 | 6.7s | 9s |
| Acc Video 6.mp4 | ✅ | 2 | 0 | 7.7s | 10s |
| Acc Video 9.mp4 | ✅ | 4 | 0 | 7.7s | 12s |
| Acc video 3.mp4 | ✅ | 2 | 3 | 8.0s | 8s |
| Acc video 4.mp4 | ✅ | 3 | 0 | 6.7s | 8s |
| Acc video 7.mp4 | ✅ | 3 | 0 | 9.3s | 10s |
| Acc video 8.mp4 | ✅ | 2 | 2 | 10.2s | 12s |

---

## 📄 License

This project is for educational and research purposes. See individual model licenses for YOLO (AGPL-3.0) and dataset-specific terms.

---

<p align="center">
  Built with 🧠 YOLOv8 · OpenCV · PyTorch
  <br />
  <strong>RapidAid</strong> — Because every second counts.
</p>

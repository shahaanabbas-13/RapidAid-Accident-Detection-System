# RapidAid — Multi-Model Training Guide

## Overview

The RapidAid pipeline uses an **ensemble of 5 models** working together.
Three are pre-trained (M4, M5) and two need custom training (M1, M2, M3).

```
┌─────────────────────────────────────────────────────────┐
│  M1: Scene Classifier (YOLOv8n-cls) ← TRAIN THIS FIRST │
│  M2: Damage Detector  (YOLOv8n-det) ← TRAIN 2ND        │
│  M3: Temporal Classifier (LSTM)      ← TRAIN 3RD       │
│  M4: Vehicle Segmentation (YOLOv8s-seg)  ← PRE-TRAINED │
│  M5: Pose Estimation (YOLOv8s-pose)      ← PRE-TRAINED │
└─────────────────────────────────────────────────────────┘
```

## Priority Order

### M1: Scene Classifier ⭐ (HIGHEST PRIORITY)

**What it does**: Looks at the entire frame and says "accident" or "not accident"

**What it fixes**:
- Image 3: Cars crashing with smoke → classifier sees the smoke/impact
- Image 4: Taxi hitting person in busy traffic → classifier recognizes accident scene
- False positives: Normal traffic wrongly flagged → classifier rejects

**Training**: `training/train_m1_scene_classifier.py`
**Time**: ~2-3 hours on Colab T4 GPU
**Output**: `weights/accident_classifier.pt` (~5 MB)
**Integration**: Already built-in! Just drop the file in `weights/`

---

### M2: Damage Detector (MEDIUM PRIORITY)

**What it does**: Detects crash EVIDENCE objects in the frame:
- Smoke/dust clouds
- Scattered debris on road
- Vehicle damage (dents, crumpled metal)
- Fire
- Overturned vehicles

**What it fixes**:
- Image 7: Truck falling on car → detects the overturned truck
- Smoke-obscured crashes → detects the smoke cloud itself
- Post-crash scenes → detects debris on road

**Training**: `training/train_m2_damage_detector.py`
**Time**: ~3-4 hours on Colab T4 GPU
**Output**: `weights/damage_detector.pt` (~12 MB)
**Integration**: Needs new module `models/damage_detector.py` (see below)

---

### M3: Temporal Classifier (ADVANCED)

**What it does**: Analyzes 16-frame VIDEO SEQUENCES (not single frames) to detect temporal accident patterns:
- Normal → impact → debris (crash pattern)
- Moving → sudden stop → people running (aftermath pattern)
- Vehicle trajectories converging → collision

**What it fixes**:
- Video timing: Detects the EXACT moment of crash
- Busy traffic: Distinguishes "car passing person" from "car hitting person"
- Reduces video false positives by requiring temporal consistency

**Training**: `training/train_m3_temporal_classifier.py`
**Time**: ~4-6 hours on Colab T4 GPU (needs M1 trained first!)
**Output**: `weights/temporal_classifier.pt` (~2 MB)
**Integration**: Needs new module `models/temporal_classifier.py` (see below)

---

## How to Train (Google Colab)

### Step-by-Step for M1 (Start Here):

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Runtime → Change runtime type → **GPU (T4)**
3. Create a new notebook
4. Copy contents of `training/train_m1_scene_classifier.py` into a code cell
5. **Run the cell** — training is fully automated:
   - Downloads datasets from Kaggle
   - Organizes into train/val splits
   - Trains YOLOv8n-cls for 80 epochs
   - Evaluates and exports the model
6. Download `accident_classifier.pt` from `/content/`
7. Place in `AccidentDetection/weights/accident_classifier.pt`
8. Run `python run_all.py` — pipeline auto-detects the model!

### For M2:
Same process but with `train_m2_damage_detector.py`. Needs a Roboflow
account (free) for the damage detection dataset, OR manual upload of
annotated damage images.

### For M3:
Same process but with `train_m3_temporal_classifier.py`. **Train M1 first!**
M3 uses M1's feature extractor as its backbone. Needs accident video clips
uploaded to Colab folders.

---

## Integration Architecture

After training, the ensemble pipeline works like this:

```
Frame arrives
    │
    ├─→ M1: "Is this an accident scene?" → scene_score (0-1)
    ├─→ M2: "Any damage/debris/smoke?" → damage_score (0-1)
    ├─→ M4: Vehicle detection → vehicle bboxes
    └─→ M5: Pose estimation → person keypoints
         │
         ├─→ Geometric Analysis → crash_score (0-1)
         └─→ Victim Analysis → victim_score (0-1)
              │
              └─→ Fusion: weighted vote of all scores
                   │
                   ├─→ Video: M3 temporal consistency check
                   └─→ FINAL DECISION: accident / no accident
```

### Fusion Weights:
```
final = 0.35 × scene_score      (M1)
      + 0.25 × crash_score       (geometric)
      + 0.20 × damage_score      (M2)
      + 0.15 × victim_score      (pose analysis)
      + 0.05 × temporal_score    (M3, video only)
```

---

## Datasets Summary

| Dataset | Used By | Source | Size | Free? |
|---------|---------|--------|------|-------|
| Kaggle CCTV Accident | M1 | kaggle.com/ckay16/accident-detection-from-cctv-footage | ~1,500 | ✅ |
| Kaggle Car Accident | M1 | kaggle.com/meaborak/car-accident-detection | ~1,000 | ✅ |
| Roboflow Damage | M2 | universe.roboflow.com | varies | ✅ |
| DoTA | M3 | github.com/MoonBlvd/Detection-of-Traffic-Anomaly | 4,677 vids | ✅ |
| CADP | M1, M3 | ankitsha26/CarAccidentDetectionPrediction | 1,416 vids | ✅ |
| IITH Accident | M1 | IIT Hyderabad (Indian CCTV) | varies | Academic |
| Your own data | All | Upload to Colab | any | ✅ |

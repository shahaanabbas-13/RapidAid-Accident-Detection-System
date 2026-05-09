"""
=====================================================================
  RapidAid — M2: Damage & Debris Detector (Kaggle Edition)
  Classification-based: Damaged Vehicle vs Normal Vehicle
  Run this on Google Colab (GPU Runtime)
  Training time: ~2-3 hours on T4 GPU
=====================================================================

WHAT THIS TRAINS:
  A YOLOv8n-cls classifier that looks at CROPPED vehicle regions
  and decides: "Is this vehicle damaged or normal?"

  It learns to recognize:
  - Dents, scratches, crumpled metal
  - Broken glass/windshields
  - Missing parts (bumpers, doors)
  - Severely deformed vehicle shapes
  - Smoke/fire near vehicles

WHY THIS IS NEEDED:
  The existing pipeline detects vehicles using YOLO but can't tell
  if a vehicle is DAMAGED. M2 adds this capability:
  - Crop each detected vehicle bbox
  - Run M2 classifier on the crop
  - If "damaged" → boost crash score for that vehicle
  - Even a single damaged vehicle = evidence of accident

HOW TO USE:
  1. Run this on Google Colab with GPU
  2. Download damage_classifier.pt
  3. Place in AccidentDetection/weights/damage_classifier.pt
  4. Pipeline integration code provided at the end

DATASETS:
  Uses freely available Kaggle datasets (no API keys needed):
  - CarDD Dataset (Car Damage Detection)
  - Car Damage Severity Dataset

BEFORE RUNNING:
  1. Runtime -> Change runtime type -> GPU (T4)
  2. Run all cells in order
"""

# ============================================================
# CELL 1: Setup & GPU Check
# ============================================================
import torch
import os
import shutil
import random
import glob

print("=" * 60)
print("  M2: Damage Classifier — Kaggle Dataset Training")
print("=" * 60)

if torch.cuda.is_available():
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("WARNING: No GPU detected!")
    print("   Go to: Runtime -> Change runtime type -> GPU (T4)")

import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "ultralytics", "kagglehub", "Pillow", "tqdm"])
print("Dependencies installed\n")


# ============================================================
# CELL 2: Download & Organize Kaggle Datasets
# ============================================================
DATASET_DIR = "/content/damage_dataset"

for split in ["train", "val"]:
    for cls in ["damaged", "normal"]:
        os.makedirs(os.path.join(DATASET_DIR, split, cls), exist_ok=True)

print("Dataset directories created")

all_damaged = []
all_normal = []

# --- Dataset 1: Car Damage Severity ---
print("\n[Dataset 1/3] Downloading Car Damage Severity dataset...")
try:
    import kagglehub
    path1 = kagglehub.dataset_download("anujms/car-damage-detection")
    print(f"   Downloaded to: {path1}")

    for root, dirs, files in os.walk(path1):
        folder_name = os.path.basename(root).lower().replace(" ", "").replace("_", "").replace("-", "")
        images = [os.path.join(root, f) for f in files
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        if not images:
            continue

        # Damaged categories
        if any(x in folder_name for x in ["damage", "crash", "severe", "moderate", "minor",
                                            "front", "rear", "side", "rollover"]):
            all_damaged.extend(images)
            print(f"   -> Damaged: {os.path.basename(root)} ({len(images)} images)")
        # Normal/whole categories
        elif any(x in folder_name for x in ["normal", "whole", "undamage", "nodamage", "good"]):
            all_normal.extend(images)
            print(f"   -> Normal: {os.path.basename(root)} ({len(images)} images)")

except Exception as e:
    print(f"   Dataset 1 failed: {e}")


# --- Dataset 2: CarDD Dataset ---
print("\n[Dataset 2/3] Downloading CarDD dataset...")
try:
    import kagglehub
    path2 = kagglehub.dataset_download("xinkuangwang/cardd-dataset")
    print(f"   Downloaded to: {path2}")

    for root, dirs, files in os.walk(path2):
        folder_name = os.path.basename(root).lower().replace(" ", "").replace("_", "").replace("-", "")
        images = [os.path.join(root, f) for f in files
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        if not images:
            continue

        if any(x in folder_name for x in ["damage", "crash", "accident", "broken", "dent",
                                            "scratch", "severe", "shatter"]):
            all_damaged.extend(images)
            print(f"   -> Damaged: {os.path.basename(root)} ({len(images)} images)")
        elif any(x in folder_name for x in ["normal", "whole", "clean", "good", "undamage"]):
            all_normal.extend(images)
            print(f"   -> Normal: {os.path.basename(root)} ({len(images)} images)")

except Exception as e:
    print(f"   Dataset 2 failed: {e}")


# --- Dataset 3: Additional car damage dataset ---
print("\n[Dataset 3/3] Downloading additional damage dataset...")
try:
    import kagglehub
    path3 = kagglehub.dataset_download("lplenern/coco-car-damage-detection-dataset")
    print(f"   Downloaded to: {path3}")

    for root, dirs, files in os.walk(path3):
        folder_name = os.path.basename(root).lower().replace(" ", "").replace("_", "").replace("-", "")
        images = [os.path.join(root, f) for f in files
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        if not images:
            continue

        if any(x in folder_name for x in ["damage", "crash"]):
            all_damaged.extend(images)
            print(f"   -> Damaged: {os.path.basename(root)} ({len(images)} images)")
        elif any(x in folder_name for x in ["normal", "whole"]):
            all_normal.extend(images)
            print(f"   -> Normal: {os.path.basename(root)} ({len(images)} images)")

except Exception as e:
    print(f"   Dataset 3 not available: {e}")


# --- Fallback: if auto-detection missed folders, scan all ---
if not all_damaged or not all_normal:
    print("\n   Scanning all downloaded folders for images...")
    all_downloaded_paths = [p for p in [
        locals().get('path1', ''),
        locals().get('path2', ''),
        locals().get('path3', ''),
    ] if p and os.path.exists(p)]

    for base_path in all_downloaded_paths:
        for root, dirs, files in os.walk(base_path):
            images = [os.path.join(root, f) for f in files
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            if not images:
                continue

            full_path = root.lower()
            basename = os.path.basename(root).lower()

            # Try to classify by any folder in the path
            if any(x in full_path for x in ["/damage", "/crash", "/accident",
                                              "/severe", "/moderate", "/minor",
                                              "/broken", "/dent", "/scratch"]):
                all_damaged.extend(images)
            elif any(x in full_path for x in ["/normal", "/whole", "/clean",
                                                "/good", "/undamage", "/no_damage"]):
                all_normal.extend(images)
            # Try by folder index (0=normal, 1=damaged is common convention)
            elif basename == "1" or basename == "01":
                all_damaged.extend(images)
            elif basename == "0" or basename == "00":
                all_normal.extend(images)


# --- User data fallback ---
USER_DMG_DIR = "/content/user_damage/damaged"
USER_NORM_DIR = "/content/user_damage/normal"
os.makedirs(USER_DMG_DIR, exist_ok=True)
os.makedirs(USER_NORM_DIR, exist_ok=True)

user_dmg = glob.glob(os.path.join(USER_DMG_DIR, "*.*"))
user_norm = glob.glob(os.path.join(USER_NORM_DIR, "*.*"))
if user_dmg or user_norm:
    all_damaged.extend(user_dmg)
    all_normal.extend(user_norm)
    print(f"   User images: {len(user_dmg)} damaged + {len(user_norm)} normal")


# --- Summary ---
print(f"\nCombined dataset: {len(all_damaged)} damaged + {len(all_normal)} normal images")

if not all_damaged or not all_normal:
    print("""
=====================================================================
  NO IMAGES FOUND — Manual Setup Required

  Upload images to these Colab folders:

  /content/user_damage/damaged/    (200+ damaged vehicle images)
  /content/user_damage/normal/     (200+ normal vehicle images)

  Then re-run this cell.
=====================================================================
""")
    raise SystemExit("Upload images first, then re-run.")


# Balance classes
min_count = min(len(all_damaged), len(all_normal))
max_count = max(len(all_damaged), len(all_normal))

if max_count > min_count * 3:
    print(f"   Class imbalance detected ({max_count} vs {min_count})")
    print(f"   Downsampling majority to {min_count * 2}...")
    if len(all_damaged) > len(all_normal):
        random.shuffle(all_damaged)
        all_damaged = all_damaged[:min_count * 2]
    else:
        random.shuffle(all_normal)
        all_normal = all_normal[:min_count * 2]

# Shuffle and split (80/20)
random.shuffle(all_damaged)
random.shuffle(all_normal)

split_dmg = int(len(all_damaged) * 0.8)
split_norm = int(len(all_normal) * 0.8)

splits = {
    ("train", "damaged"): all_damaged[:split_dmg],
    ("val", "damaged"): all_damaged[split_dmg:],
    ("train", "normal"): all_normal[:split_norm],
    ("val", "normal"): all_normal[split_norm:],
}

for (split, cls), images in splits.items():
    dst_dir = os.path.join(DATASET_DIR, split, cls)
    for i, img_path in enumerate(images):
        ext = os.path.splitext(img_path)[1]
        shutil.copy2(img_path, os.path.join(dst_dir, f"{cls}_{split}_{i:05d}{ext}"))

print("\nDataset organized:")
for split in ["train", "val"]:
    for cls in ["damaged", "normal"]:
        d = os.path.join(DATASET_DIR, split, cls)
        n = len(os.listdir(d))
        print(f"   {split}/{cls}: {n} images")


# ============================================================
# CELL 3: Train YOLOv8n-cls Damage Classifier
# ============================================================
from ultralytics import YOLO

print("\n" + "=" * 60)
print("  TRAINING M2: DAMAGE CLASSIFIER")
print("  Estimated time: ~2-3 hours on T4 GPU")
print("=" * 60)

device = "0" if torch.cuda.is_available() else "cpu"
print(f"  Device: {'GPU (' + torch.cuda.get_device_name(0) + ')' if device == '0' else 'CPU'}")

model = YOLO("yolov8n-cls.pt")

results = model.train(
    data=DATASET_DIR,
    epochs=80,
    imgsz=224,
    batch=32 if device == "0" else 16,
    patience=15,
    device=device,
    project="/content/runs/classify",
    name="m2_damage_classifier",
    exist_ok=True,
    verbose=True,
    # Augmentation tuned for damage detection
    hsv_h=0.01,
    hsv_s=0.5,
    hsv_v=0.3,
    degrees=10,
    translate=0.1,
    scale=0.4,
    fliplr=0.5,
    mosaic=0.0,
    erasing=0.2,
    crop_fraction=0.85,
)

print("\nM2 Training complete!")


# ============================================================
# CELL 4: Evaluate
# ============================================================
metrics = model.val()
print(f"\n  Top-1 Accuracy: {metrics.top1:.3f}")
print(f"  Top-5 Accuracy: {metrics.top5:.3f}")

# Sample predictions
print("\n  Sample predictions:")
val_images = glob.glob(os.path.join(DATASET_DIR, "val", "**", "*.jpg"), recursive=True)
val_images += glob.glob(os.path.join(DATASET_DIR, "val", "**", "*.png"), recursive=True)
for img in val_images[:10]:
    res = model.predict(img, imgsz=224, verbose=False)
    pred_class = res[0].probs.top1
    pred_conf = float(res[0].probs.top1conf)
    true_class = "damaged" if "/damaged/" in img else "normal"
    pred_name = model.names[pred_class]
    status = "CORRECT" if pred_name == true_class else "WRONG"
    print(f"    [{status}] True: {true_class:10s} Pred: {pred_name:10s} ({pred_conf:.2f})")


# ============================================================
# CELL 5: Export & Download
# ============================================================
best_path = "/content/runs/classify/m2_damage_classifier/weights/best.pt"

if os.path.exists(best_path):
    download_path = "/content/damage_classifier.pt"
    shutil.copy2(best_path, download_path)
    size_mb = os.path.getsize(download_path) / (1024 * 1024)

    print(f"""
=====================================================================
  M2: DAMAGE CLASSIFIER READY — {size_mb:.1f} MB

  DOWNLOAD:
  1. Click the folder icon (left sidebar)
  2. Find 'damage_classifier.pt' in /content/
  3. Right-click -> Download

  INSTALL:
  Place at: AccidentDetection/weights/damage_classifier.pt

  INTEGRATION:
  After downloading, I'll add integration code to the pipeline.
  The damage classifier will check each detected vehicle crop
  and boost crash scores for visibly damaged vehicles.
=====================================================================
""")

    try:
        from google.colab import files
        files.download(download_path)
        print("Download triggered!")
    except:
        print("(Use the file browser to download manually)")
else:
    print("Model file not found. Check training logs above.")

print("\nM2 Done!")

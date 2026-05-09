"""
╔══════════════════════════════════════════════════════════════╗
║  RapidAid — M2 v2: Damage Detector (HARDENED)               ║
║  With Indian Vehicle Hard Negatives                          ║
║  Run this on Google Colab (GPU Runtime)                      ║
║  Training time: ~3-4 hours on T4 GPU                         ║
╚══════════════════════════════════════════════════════════════╝

KEY IMPROVEMENTS OVER v1:
  1. Added INDIAN VEHICLE CROPS as "normal" class (taxis, auto-rickshaws,
     worn trucks — the vehicles M2v1 misclassifies as damaged)
  2. Uses only CLEAR crash damage as "damaged" (not mild scratches)
  3. Higher resolution (320 vs 224) for better damage visibility
  4. Aggressive augmentation for lighting robustness
  5. User can upload misclassified vehicle crops as hard negatives

THE CORE PROBLEM M2v1 HAD:
  M2v1 was trained on Western luxury car photos. It learned:
  "old/worn vehicle = damaged". Indian taxis, auto-rickshaws, and
  trucks all look "old/worn" → M2v1 gives them 96% damage scores.
  
  M2v2 fixes this by adding many Indian vehicle crops as "normal".

AFTER TRAINING:
  Download damage_classifier.pt → place in weights/ folder
  Then set M2_TRUST_ENABLED = True in config/settings.py

BEFORE RUNNING:
  1. Runtime → Change runtime type → Select GPU (T4)
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
import cv2
import numpy as np

print("=" * 60)
print("  M2 v2: Damage Classifier — Indian Vehicle Hard Negatives")
print("=" * 60)

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ GPU: {gpu_name} ({vram:.1f} GB)")
else:
    print("⚠️  No GPU! Runtime → Change runtime type → GPU (T4)")

import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "ultralytics", "kagglehub", "Pillow", "tqdm",
                       "gdown", "opencv-python-headless"])
print("✅ Dependencies installed\n")


# ============================================================
# CELL 2: Download & Organize Datasets
# ============================================================
DATASET_DIR = "/content/m2v2_dataset"

for split in ["train", "val"]:
    for cls in ["damaged", "normal"]:
        os.makedirs(os.path.join(DATASET_DIR, split, cls), exist_ok=True)

all_damaged = []
all_normal = []


# --- Dataset 1: Car Damage Severity ---
print("\n📥 [Dataset 1/5] Car Damage Severity dataset...")
try:
    import kagglehub
    path1 = kagglehub.dataset_download("anujms/car-damage-detection")
    print(f"   ✅ Downloaded to: {path1}")

    for root, dirs, files in os.walk(path1):
        folder = os.path.basename(root).lower().replace(" ", "").replace("_", "")
        images = [os.path.join(root, f) for f in files
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        if not images:
            continue
        if any(x in folder for x in ["damage", "crash", "severe", "moderate",
                                       "front", "rear", "side", "rollover"]):
            all_damaged.extend(images)
            print(f"   → Damaged: {os.path.basename(root)} ({len(images)})")
        elif any(x in folder for x in ["normal", "whole", "undamage", "good"]):
            all_normal.extend(images)
            print(f"   → Normal: {os.path.basename(root)} ({len(images)})")
except Exception as e:
    print(f"   ❌ Failed: {e}")


# --- Dataset 2: CarDD Dataset ---
print("\n📥 [Dataset 2/5] CarDD dataset...")
try:
    import kagglehub
    path2 = kagglehub.dataset_download("xinkuangwang/cardd-dataset")
    print(f"   ✅ Downloaded to: {path2}")

    for root, dirs, files in os.walk(path2):
        folder = os.path.basename(root).lower().replace(" ", "").replace("_", "")
        images = [os.path.join(root, f) for f in files
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        if not images:
            continue
        if any(x in folder for x in ["damage", "crash", "accident", "broken",
                                       "dent", "scratch", "severe", "shatter"]):
            all_damaged.extend(images)
            print(f"   → Damaged: {os.path.basename(root)} ({len(images)})")
        elif any(x in folder for x in ["normal", "whole", "clean", "good"]):
            all_normal.extend(images)
            print(f"   → Normal: {os.path.basename(root)} ({len(images)})")
except Exception as e:
    print(f"   ❌ Failed: {e}")


# --- Dataset 3: COCO Car Damage ---
print("\n📥 [Dataset 3/5] COCO Car Damage dataset...")
try:
    import kagglehub
    path3 = kagglehub.dataset_download("lplenern/coco-car-damage-detection-dataset")
    print(f"   ✅ Downloaded to: {path3}")

    for root, dirs, files in os.walk(path3):
        folder = os.path.basename(root).lower().replace(" ", "").replace("_", "")
        images = [os.path.join(root, f) for f in files
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        if not images:
            continue
        if any(x in folder for x in ["damage", "crash"]):
            all_damaged.extend(images)
        elif any(x in folder for x in ["normal", "whole"]):
            all_normal.extend(images)
except Exception as e:
    print(f"   ⚠️ Not available: {e}")


# --- Dataset 4: HARD NEGATIVES — Indian vehicles ---
print("\n📥 [Dataset 4/5] Indian vehicle hard negatives...")
print("   ℹ️  These are the vehicles M2v1 misclassifies as 'damaged':")
print("   - Yellow Kolkata taxis (old Ambassadors)")
print("   - Auto-rickshaws")
print("   - Worn-looking trucks/lorries")
print("   - Indian buses")

try:
    import kagglehub
    # Indian vehicle images — diverse types
    idd_path = kagglehub.dataset_download("dataclusterlabs/indian-vehicle-dataset")
    print(f"   ✅ Downloaded to: {idd_path}")

    idd_images = []
    for root, dirs, files in os.walk(idd_path):
        images = [os.path.join(root, f) for f in files
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        idd_images.extend(images)

    random.shuffle(idd_images)
    hard_negatives = idd_images[:600]
    all_normal.extend(hard_negatives)
    print(f"   → Added {len(hard_negatives)} Indian vehicle crops as NORMAL")

except Exception as e:
    print(f"   ⚠️ Indian vehicle dataset not available: {e}")

# Try additional vehicle dataset
try:
    import kagglehub
    veh_path = kagglehub.dataset_download("jessicali9530/stanford-cars-dataset")
    print(f"   ✅ Stanford Cars: {veh_path}")

    veh_images = []
    for root, dirs, files in os.walk(veh_path):
        images = [os.path.join(root, f) for f in files
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        veh_images.extend(images)

    random.shuffle(veh_images)
    all_normal.extend(veh_images[:400])
    print(f"   → Added {min(len(veh_images), 400)} clean vehicle crops as NORMAL")

except Exception as e:
    print(f"   ⚠️ Stanford Cars not available: {e}")


# --- Dataset 5: User vehicle crops ---
print("\n📥 [Dataset 5/5] User-uploaded vehicle crops...")
USER_DMG = "/content/user_damage/damaged"
USER_NORM = "/content/user_damage/normal"
USER_HARD = "/content/user_damage/hard_negatives"

os.makedirs(USER_DMG, exist_ok=True)
os.makedirs(USER_NORM, exist_ok=True)
os.makedirs(USER_HARD, exist_ok=True)

user_dmg = glob.glob(os.path.join(USER_DMG, "*.*"))
user_norm = glob.glob(os.path.join(USER_NORM, "*.*"))
user_hard = glob.glob(os.path.join(USER_HARD, "*.*"))

if user_dmg or user_norm or user_hard:
    all_damaged.extend(user_dmg)
    all_normal.extend(user_norm)
    all_normal.extend(user_hard)
    print(f"   ✅ {len(user_dmg)} damaged + {len(user_norm)} normal "
          f"+ {len(user_hard)} hard negatives")
else:
    print("   ℹ️  No user crops found.")
    print(f"   To add false-positive vehicles, crop them and upload to:")
    print(f"   {USER_HARD}/")


# --- Fallback folder scanning ---
if not all_damaged or not all_normal:
    print("\n   Scanning all downloads for undetected images...")
    for base_path in [locals().get(f'path{i}', '') for i in range(1, 4)]:
        if not base_path or not os.path.exists(base_path):
            continue
        for root, dirs, files in os.walk(base_path):
            images = [os.path.join(root, f) for f in files
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            if not images:
                continue
            full_path = root.lower()
            if any(x in full_path for x in ["/damage", "/crash", "/accident",
                                              "/severe", "/broken", "/dent"]):
                all_damaged.extend(images)
            elif any(x in full_path for x in ["/normal", "/whole", "/clean",
                                                "/good", "/undamage"]):
                all_normal.extend(images)
            elif os.path.basename(root) in ("1", "01"):
                all_damaged.extend(images)
            elif os.path.basename(root) in ("0", "00"):
                all_normal.extend(images)

# --- Summary ---
print(f"\n📊 Combined: {len(all_damaged)} damaged + {len(all_normal)} normal")

if not all_damaged or not all_normal:
    print("❌ Upload images to the folders above, then re-run.")
    raise SystemExit("Need data.")


# Balance classes
max_per_class = 2000
if len(all_damaged) > max_per_class:
    random.shuffle(all_damaged)
    all_damaged = all_damaged[:max_per_class]
if len(all_normal) > max_per_class:
    random.shuffle(all_normal)
    all_normal = all_normal[:max_per_class]

min_count = min(len(all_damaged), len(all_normal))
max_count = max(len(all_damaged), len(all_normal))
if max_count > min_count * 2:
    if len(all_damaged) > len(all_normal):
        random.shuffle(all_damaged)
        all_damaged = all_damaged[:min_count * 2]
    else:
        random.shuffle(all_normal)
        all_normal = all_normal[:min_count * 2]

# Split (85/15)
random.shuffle(all_damaged)
random.shuffle(all_normal)

split_dmg = int(len(all_damaged) * 0.85)
split_norm = int(len(all_normal) * 0.85)

splits = {
    ("train", "damaged"): all_damaged[:split_dmg],
    ("val", "damaged"): all_damaged[split_dmg:],
    ("train", "normal"): all_normal[:split_norm],
    ("val", "normal"): all_normal[split_norm:],
}

for (split, cls), images in splits.items():
    dst = os.path.join(DATASET_DIR, split, cls)
    for i, img_path in enumerate(images):
        ext = os.path.splitext(img_path)[1]
        shutil.copy2(img_path, os.path.join(dst, f"{cls}_{split}_{i:05d}{ext}"))

print("\n✅ Dataset organized:")
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
print("  TRAINING M2 v2: DAMAGE CLASSIFIER (HARDENED)")
print("  With Indian vehicle hard negatives")
print("  Estimated time: ~3-4 hours on T4 GPU")
print("=" * 60)

device = "0" if torch.cuda.is_available() else "cpu"

model = YOLO("yolov8n-cls.pt")

results = model.train(
    data=DATASET_DIR,
    epochs=100,
    imgsz=320,               # Higher res (was 224)
    batch=32 if device == "0" else 8,
    patience=20,
    device=device,
    project="/content/runs/classify",
    name="m2v2_damage_classifier",
    exist_ok=True,
    verbose=True,
    # Augmentation for damage detection
    hsv_h=0.01,
    hsv_s=0.5,
    hsv_v=0.4,
    degrees=10,
    translate=0.1,
    scale=0.4,
    fliplr=0.5,
    mosaic=0.0,
    erasing=0.25,
    crop_fraction=0.9,       # Less aggressive cropping for vehicle damage
)

print("\n✅ M2 v2 Training complete!")


# ============================================================
# CELL 4: Evaluate
# ============================================================
metrics = model.val()
print(f"\n  Top-1 Accuracy: {metrics.top1:.3f}")
print(f"  Top-5 Accuracy: {metrics.top5:.3f}")

print("\n  Sample predictions:")
val_images = glob.glob(os.path.join(DATASET_DIR, "val", "**", "*.jpg"), recursive=True)
val_images += glob.glob(os.path.join(DATASET_DIR, "val", "**", "*.png"), recursive=True)

correct = 0
total = 0
for img in val_images[:20]:
    res = model.predict(img, imgsz=320, verbose=False)
    pred_class = res[0].probs.top1
    pred_conf = float(res[0].probs.top1conf)
    true_class = "damaged" if "/damaged/" in img else "normal"
    pred_name = model.names[pred_class]
    is_correct = pred_name == true_class
    if is_correct:
        correct += 1
    total += 1
    status = "✅" if is_correct else "❌"
    print(f"    {status} True: {true_class:10s} Pred: {pred_name:10s} ({pred_conf:.2f})")

if total > 0:
    print(f"\n  Sample accuracy: {correct}/{total} ({correct/total*100:.0f}%)")


# ============================================================
# CELL 5: Export & Download
# ============================================================
best_path = "/content/runs/classify/m2v2_damage_classifier/weights/best.pt"

if os.path.exists(best_path):
    download_path = "/content/damage_classifier.pt"
    shutil.copy2(best_path, download_path)
    size_mb = os.path.getsize(download_path) / (1024 * 1024)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  ✅ M2 v2: DAMAGE CLASSIFIER READY — {size_mb:.1f} MB               ║
╠══════════════════════════════════════════════════════════════╣
║                                                             ║
║  DOWNLOAD:                                                  ║
║  1. Click 📁 folder icon (left sidebar)                     ║
║  2. Find 'damage_classifier.pt' in /content/                ║
║  3. Right-click → Download                                  ║
║                                                             ║
║  INSTALL:                                                   ║
║  Place at: AccidentDetection/weights/damage_classifier.pt   ║
║  (Replace the old file)                                     ║
║                                                             ║
║  IMPORTANT: After installing both M1v2 and M2v2, set:       ║
║  M2_TRUST_ENABLED = True in config/settings.py              ║
╚══════════════════════════════════════════════════════════════╝
""")

    try:
        from google.colab import files
        files.download(download_path)
    except:
        print("(Use the file browser to download manually)")
else:
    print("❌ Model file not found. Check training logs.")

print("\n🎉 M2 v2 Done!")
"""

╔══════════════════════════════════════════════════════════════╗
║  After training M2v2:                                       ║
║  1. Download damage_classifier.pt                           ║
║  2. Place in AccidentDetection/weights/                     ║
║  3. Set M2_TRUST_ENABLED = True in config/settings.py       ║
║  4. Re-test all your test videos/frames                     ║
║  5. Then train M4 (collision detector) for Phase 3          ║
╚══════════════════════════════════════════════════════════════╝
"""

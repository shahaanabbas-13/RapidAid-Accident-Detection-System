"""
╔══════════════════════════════════════════════════════════════╗
║  RapidAid — M1 v2: Scene Classifier (HARDENED)              ║
║  Multi-Dataset + Hard Negative Mining                        ║
║  Run this on Google Colab (GPU Runtime)                      ║
║  Training time: ~3-4 hours on T4 GPU                         ║
╚══════════════════════════════════════════════════════════════╝

KEY IMPROVEMENTS OVER v1:
  1. Added Indian traffic scenes as HARD NEGATIVES (busy roads,
     auto-rickshaws, worn trucks — the scenes M1v1 gets wrong)
  2. 3 additional datasets for more diverse accident examples
  3. Higher resolution training (320 vs 224) for better feature extraction
  4. Longer training with cosine annealing scheduler
  5. Explicit hard negative folder for user-provided failures

DATASETS:
  - Kaggle CCTV Accident Detection (~1,500 frames)
  - Kaggle Car Accident Detection (~1,000 frames)
  - Kaggle Indian Driving Dataset (IDD) — hard negatives
  - BDD100K Diverse Driving — hard negatives
  - User test frames that were misclassified (uploaded)

AFTER TRAINING:
  Download accident_classifier.pt → place in weights/ folder
  Then set M2_TRUST_ENABLED = False in settings.py (keep it disabled
  until M2 is also retrained)

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
from pathlib import Path

print("=" * 60)
print("  M1 v2: Scene Classifier — Hard Negative Mining")
print("=" * 60)

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ GPU detected: {gpu_name} ({vram:.1f} GB VRAM)")
else:
    print("⚠️  WARNING: No GPU detected!")
    print("   Go to: Runtime → Change runtime type → GPU (T4)")

import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "ultralytics", "kagglehub", "Pillow", "tqdm",
                       "gdown"])
print("✅ Dependencies installed\n")


# ============================================================
# CELL 2: Download & Merge Multiple Datasets
# ============================================================
DATASET_DIR = "/content/m1v2_dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")

for split in ["train", "val"]:
    for cls in ["accident", "no_accident"]:
        os.makedirs(os.path.join(DATASET_DIR, split, cls), exist_ok=True)

print("📁 Dataset directories created")

all_accident_images = []
all_normal_images = []


# --- Dataset 1: Kaggle CCTV Accident Detection ---
print("\n📥 [Dataset 1/5] Kaggle CCTV Accident dataset...")
try:
    import kagglehub
    kaggle_path = kagglehub.dataset_download(
        "ckay16/accident-detection-from-cctv-footage"
    )
    print(f"   ✅ Downloaded to: {kaggle_path}")

    for root, dirs, files in os.walk(kaggle_path):
        folder = os.path.basename(root).lower().replace(" ", "").replace("_", "").replace("-", "")
        images = [os.path.join(root, f) for f in files
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        if not images:
            continue
        if any(x in folder for x in ["nonaccident", "noaccident", "normal", "negative"]):
            all_normal_images.extend(images)
            print(f"   → Normal: {os.path.basename(root)} ({len(images)})")
        elif any(x in folder for x in ["accident", "positive", "crash"]):
            all_accident_images.extend(images)
            print(f"   → Accident: {os.path.basename(root)} ({len(images)})")

    # Fallback folder detection
    if not all_accident_images and not all_normal_images:
        for root, dirs, files in os.walk(kaggle_path):
            images = [os.path.join(root, f) for f in files
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            if images:
                base = os.path.basename(root).lower()
                if base in ("1", "pos"):
                    all_accident_images.extend(images)
                elif base in ("0", "neg"):
                    all_normal_images.extend(images)

except Exception as e:
    print(f"   ❌ Failed: {e}")


# --- Dataset 2: Car Accident Detection ---
print("\n📥 [Dataset 2/5] Car Accident Detection dataset...")
try:
    import kagglehub
    kaggle_path2 = kagglehub.dataset_download("meaborak/car-accident-detection")
    print(f"   ✅ Downloaded to: {kaggle_path2}")

    for root, dirs, files in os.walk(kaggle_path2):
        folder = os.path.basename(root).lower().replace(" ", "").replace("_", "").replace("-", "")
        images = [os.path.join(root, f) for f in files
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        if not images:
            continue
        if any(x in folder for x in ["nonaccident", "noaccident", "normal", "negative", "non"]):
            all_normal_images.extend(images)
            print(f"   → Normal: {os.path.basename(root)} ({len(images)})")
        elif any(x in folder for x in ["accident", "positive", "crash"]):
            all_accident_images.extend(images)
            print(f"   → Accident: {os.path.basename(root)} ({len(images)})")
except Exception as e:
    print(f"   ⚠️ Not available: {e}")


# --- Dataset 3: HARD NEGATIVES — Indian traffic scenes ---
print("\n📥 [Dataset 3/5] Indian traffic hard negatives...")
print("   ℹ️  Downloading Indian traffic / dashcam scenes")
print("   These are the scenes M1v1 incorrectly classified as accidents:")
print("   - Busy Indian intersections with many vehicles")
print("   - Auto-rickshaws, old taxis, worn trucks")
print("   - Night traffic, rainy conditions")
try:
    import kagglehub
    # Indian Driving Dataset — diverse Indian road scenes
    idd_path = kagglehub.dataset_download("sakshamjn/idd-clean")
    print(f"   ✅ IDD downloaded to: {idd_path}")

    idd_images = []
    for root, dirs, files in os.walk(idd_path):
        images = [os.path.join(root, f) for f in files
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        idd_images.extend(images)

    # Take up to 800 Indian driving images as hard negatives
    random.shuffle(idd_images)
    idd_hard_negatives = idd_images[:800]
    all_normal_images.extend(idd_hard_negatives)
    print(f"   → Added {len(idd_hard_negatives)} Indian driving scenes as negatives")

except Exception as e:
    print(f"   ⚠️ IDD not available: {e}")
    print("   Trying alternative Indian traffic dataset...")
    try:
        import kagglehub
        alt_path = kagglehub.dataset_download("dataclusterlabs/indian-vehicle-dataset")
        print(f"   ✅ Downloaded to: {alt_path}")
        alt_images = []
        for root, dirs, files in os.walk(alt_path):
            images = [os.path.join(root, f) for f in files
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            alt_images.extend(images)
        random.shuffle(alt_images)
        all_normal_images.extend(alt_images[:600])
        print(f"   → Added {min(len(alt_images), 600)} Indian vehicle images as negatives")
    except Exception as e2:
        print(f"   ⚠️ Alternative also failed: {e2}")


# --- Dataset 4: BDD100K Diverse Driving (additional negatives) ---
print("\n📥 [Dataset 4/5] Diverse driving scenes (BDD-style)...")
try:
    import kagglehub
    bdd_path = kagglehub.dataset_download("solesensei/solesensei_bdd100k")
    print(f"   ✅ Downloaded to: {bdd_path}")

    bdd_images = []
    for root, dirs, files in os.walk(bdd_path):
        images = [os.path.join(root, f) for f in files
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        bdd_images.extend(images)

    random.shuffle(bdd_images)
    bdd_negatives = bdd_images[:500]
    all_normal_images.extend(bdd_negatives)
    print(f"   → Added {len(bdd_negatives)} diverse driving scenes as negatives")

except Exception as e:
    print(f"   ⚠️ Not available: {e}")


# --- Dataset 5: User-provided images (hard negatives + accidents) ---
print("\n📥 [Dataset 5/5] User-uploaded images...")
USER_ACC_DIR = "/content/user_data/accident"
USER_NORM_DIR = "/content/user_data/no_accident"
USER_HARD_NEG_DIR = "/content/user_data/hard_negatives"

os.makedirs(USER_ACC_DIR, exist_ok=True)
os.makedirs(USER_NORM_DIR, exist_ok=True)
os.makedirs(USER_HARD_NEG_DIR, exist_ok=True)

user_acc = glob.glob(os.path.join(USER_ACC_DIR, "*.*"))
user_norm = glob.glob(os.path.join(USER_NORM_DIR, "*.*"))
user_hard = glob.glob(os.path.join(USER_HARD_NEG_DIR, "*.*"))

if user_acc or user_norm or user_hard:
    all_accident_images.extend(user_acc)
    all_normal_images.extend(user_norm)
    all_normal_images.extend(user_hard)
    print(f"   ✅ {len(user_acc)} accident + {len(user_norm)} normal "
          f"+ {len(user_hard)} hard negatives")
else:
    print(f"   ℹ️  No user images found.")
    print(f"   To add hard negatives (frames M1 gets wrong), upload to:")
    print(f"   {USER_HARD_NEG_DIR}/")
    print(f"   These should be busy traffic scenes WITHOUT accidents")


# --- Organize combined dataset ---
print(f"\n📊 Combined: {len(all_accident_images)} accident + "
      f"{len(all_normal_images)} normal images")

if not all_accident_images or not all_normal_images:
    print("""
╔══════════════════════════════════════════════════════════════╗
║  NO IMAGES FOUND — Manual Setup Required                    ║
║  Upload images to the folders printed above.                ║
╚══════════════════════════════════════════════════════════════╝
""")
    raise SystemExit("Upload images first, then re-run.")

# Balance classes (target 1:1 ratio, up to 2500 each)
max_per_class = 2500
if len(all_accident_images) > max_per_class:
    random.shuffle(all_accident_images)
    all_accident_images = all_accident_images[:max_per_class]
if len(all_normal_images) > max_per_class:
    random.shuffle(all_normal_images)
    all_normal_images = all_normal_images[:max_per_class]

# If one class has >2x the other, downsample
min_count = min(len(all_accident_images), len(all_normal_images))
max_count = max(len(all_accident_images), len(all_normal_images))
if max_count > min_count * 2:
    if len(all_accident_images) > len(all_normal_images):
        random.shuffle(all_accident_images)
        all_accident_images = all_accident_images[:min_count * 2]
    else:
        random.shuffle(all_normal_images)
        all_normal_images = all_normal_images[:min_count * 2]

# Shuffle and split (85/15 for more training data)
random.shuffle(all_accident_images)
random.shuffle(all_normal_images)

split_acc = int(len(all_accident_images) * 0.85)
split_norm = int(len(all_normal_images) * 0.85)

splits = {
    ("train", "accident"): all_accident_images[:split_acc],
    ("val", "accident"): all_accident_images[split_acc:],
    ("train", "no_accident"): all_normal_images[:split_norm],
    ("val", "no_accident"): all_normal_images[split_norm:],
}

for (split, cls), images in splits.items():
    dst_dir = os.path.join(DATASET_DIR, split, cls)
    for i, img_path in enumerate(images):
        ext = os.path.splitext(img_path)[1]
        shutil.copy2(img_path, os.path.join(dst_dir, f"{cls}_{split}_{i:05d}{ext}"))

print("\n✅ Combined dataset organized:")
for split in ["train", "val"]:
    for cls in ["accident", "no_accident"]:
        d = os.path.join(DATASET_DIR, split, cls)
        n = len(os.listdir(d))
        print(f"   {split}/{cls}: {n} images")


# ============================================================
# CELL 3: Train YOLOv8n-cls (Enhanced)
# ============================================================
from ultralytics import YOLO

print("\n" + "=" * 60)
print("  TRAINING M1 v2: SCENE CLASSIFIER (HARDENED)")
print("  Higher resolution + more epochs + hard negatives")
print("  Estimated time: ~3-4 hours on T4 GPU")
print("=" * 60)

device = "0" if torch.cuda.is_available() else "cpu"
print(f"  Device: {'GPU (' + torch.cuda.get_device_name(0) + ')' if device == '0' else 'CPU'}")

model = YOLO("yolov8n-cls.pt")

results = model.train(
    data=DATASET_DIR,
    epochs=100,              # More epochs for harder dataset
    imgsz=320,               # Higher resolution (was 224)
    batch=32 if device == "0" else 8,
    patience=20,             # More patience for convergence
    device=device,
    project="/content/runs/classify",
    name="m1v2_scene_classifier",
    exist_ok=True,
    verbose=True,
    # Aggressive augmentation for CCTV robustness
    hsv_h=0.02,
    hsv_s=0.7,
    hsv_v=0.5,              # Strong brightness variation (CCTV quality)
    degrees=15,
    translate=0.15,
    scale=0.5,
    shear=5,
    fliplr=0.5,
    flipud=0.0,
    mosaic=0.0,              # No mosaic for full scene context
    erasing=0.35,            # Stronger erasing for occlusion robustness
    crop_fraction=0.85,
)

print("\n✅ M1 v2 Training complete!")


# ============================================================
# CELL 4: Evaluate
# ============================================================
metrics = model.val()
print(f"\n  Top-1 Accuracy: {metrics.top1:.3f}")
print(f"  Top-5 Accuracy: {metrics.top5:.3f}")

# Test on samples
print("\n  Sample predictions:")
val_images = glob.glob(os.path.join(VAL_DIR, "**", "*.jpg"), recursive=True)
val_images += glob.glob(os.path.join(VAL_DIR, "**", "*.png"), recursive=True)

correct = 0
total = 0
for img in val_images[:20]:
    res = model.predict(img, imgsz=320, verbose=False)
    pred_class = res[0].probs.top1
    pred_conf = float(res[0].probs.top1conf)
    true_class = "accident" if "/accident/" in img else "no_accident"
    pred_name = model.names[pred_class]
    is_correct = pred_name == true_class
    if is_correct:
        correct += 1
    total += 1
    status = "✅" if is_correct else "❌"
    print(f"    {status} True: {true_class:15s} → Pred: {pred_name:15s} ({pred_conf:.2f})")

if total > 0:
    print(f"\n  Sample accuracy: {correct}/{total} ({correct/total*100:.0f}%)")


# ============================================================
# CELL 5: Export & Download
# ============================================================
best_path = "/content/runs/classify/m1v2_scene_classifier/weights/best.pt"

if os.path.exists(best_path):
    download_path = "/content/accident_classifier.pt"
    shutil.copy2(best_path, download_path)
    size_mb = os.path.getsize(download_path) / (1024 * 1024)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  ✅ M1 v2: SCENE CLASSIFIER READY — {size_mb:.1f} MB                ║
╠══════════════════════════════════════════════════════════════╣
║                                                             ║
║  DOWNLOAD:                                                  ║
║  1. Click 📁 folder icon (left sidebar)                     ║
║  2. Find 'accident_classifier.pt' in /content/              ║
║  3. Right-click → Download                                  ║
║                                                             ║
║  INSTALL:                                                   ║
║  Place at: AccidentDetection/weights/accident_classifier.pt ║
║  (Replace the old file)                                     ║
║                                                             ║
║  IMPORTANT: After replacing, keep M2_TRUST_ENABLED=False    ║
║  in settings.py until you also retrain M2!                  ║
╚══════════════════════════════════════════════════════════════╝
""")

    try:
        from google.colab import files
        files.download(download_path)
        print("📥 Download triggered!")
    except:
        print("(Use the file browser to download manually)")
else:
    print("❌ Model file not found. Check training logs above.")

print("\n🎉 M1 v2 Done!")
"""

╔══════════════════════════════════════════════════════════════╗
║  COPY ALL THE CODE ABOVE INTO A SINGLE GOOGLE COLAB CELL    ║
║  OR separate into cells at each # CELL comment              ║
║                                                             ║
║  After training:                                            ║
║  1. Download accident_classifier.pt                         ║
║  2. Place in AccidentDetection/weights/                     ║
║  3. Then run train_m2_v2_damage_detector.py in Colab        ║
╚══════════════════════════════════════════════════════════════╝
"""

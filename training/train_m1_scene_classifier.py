"""
╔══════════════════════════════════════════════════════════════╗
║  RapidAid — M1: Scene Classifier (ENHANCED)                 ║
║  Multi-Dataset Training for Accident vs Non-Accident         ║
║  Run this on Google Colab (GPU Runtime)                      ║
║  Training time: ~2-3 hours on T4 GPU                         ║
╚══════════════════════════════════════════════════════════════╝

WHAT THIS TRAINS:
  A YOLOv8n-cls binary classifier that looks at the ENTIRE FRAME
  and decides: "Is this an accident scene?"
  
  It learns to recognize:
  - Smoke, debris, scattered parts
  - Vehicles at unusual angles
  - Damaged/crumpled vehicles
  - People lying on roads near vehicles
  - Emergency vehicle presence

DATASETS USED (Combined for best results):
  1. Kaggle CCTV Accident Detection (~1,500 frames)
  2. Roboflow Accident Detection (community contributed)
  3. Your own test frames (if uploaded)

AFTER TRAINING:
  Download accident_classifier.pt → place in weights/ folder
  Pipeline auto-detects and uses it!

BEFORE RUNNING:
  1. Go to Runtime → Change runtime type → Select GPU (T4)
  2. Then run all cells in order
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
print("  M1: Scene Classifier — Multi-Dataset Training")
print("=" * 60)

if torch.cuda.is_available():
    print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️  WARNING: No GPU detected!")
    print("   Go to: Runtime → Change runtime type → GPU (T4)")
    print("   Training on CPU will take 10+ hours instead of 2-3 hours.\n")

# Install dependencies
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "ultralytics", "kagglehub", "Pillow", "tqdm"])
print("✅ Dependencies installed\n")


# ============================================================
# CELL 2: Download & Merge Multiple Datasets
# ============================================================
DATASET_DIR = "/content/accident_dataset_v2"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")

# Create directory structure
for split in ["train", "val"]:
    for cls in ["accident", "no_accident"]:
        os.makedirs(os.path.join(DATASET_DIR, split, cls), exist_ok=True)

print("📁 Dataset directories created")

all_accident_images = []
all_normal_images = []


# --- Dataset 1: Kaggle CCTV Accident Detection ---
print("\n📥 [Dataset 1/3] Downloading Kaggle CCTV Accident dataset...")
try:
    import kagglehub
    kaggle_path = kagglehub.dataset_download("ckay16/accident-detection-from-cctv-footage")
    print(f"   ✅ Downloaded to: {kaggle_path}")

    # Auto-detect folder structure
    for root, dirs, files in os.walk(kaggle_path):
        folder_name = os.path.basename(root).lower().replace(" ", "").replace("_", "").replace("-", "")
        images = [os.path.join(root, f) for f in files
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        if not images:
            continue
        if any(x in folder_name for x in ["nonaccident", "noaccident", "normal", "negative"]):
            all_normal_images.extend(images)
            print(f"   → Normal: {os.path.basename(root)} ({len(images)} images)")
        elif any(x in folder_name for x in ["accident", "positive", "crash"]):
            all_accident_images.extend(images)
            print(f"   → Accident: {os.path.basename(root)} ({len(images)} images)")

    # Fallback folder detection by index
    if not all_accident_images and not all_normal_images:
        for root, dirs, files in os.walk(kaggle_path):
            images = [os.path.join(root, f) for f in files
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            if images:
                base = os.path.basename(root).lower()
                if "1" == base or "pos" in root.lower():
                    all_accident_images.extend(images)
                elif "0" == base or "neg" in root.lower():
                    all_normal_images.extend(images)

except Exception as e:
    print(f"   ❌ Kaggle download failed: {e}")
    print("   You can manually download from:")
    print("   https://www.kaggle.com/datasets/ckay16/accident-detection-from-cctv-footage")


# --- Dataset 2: Additional Kaggle Accident Dataset ---
print("\n📥 [Dataset 2/3] Downloading additional accident dataset...")
try:
    import kagglehub
    kaggle_path2 = kagglehub.dataset_download("meaborak/car-accident-detection")
    print(f"   ✅ Downloaded to: {kaggle_path2}")

    for root, dirs, files in os.walk(kaggle_path2):
        folder_name = os.path.basename(root).lower().replace(" ", "").replace("_", "").replace("-", "")
        images = [os.path.join(root, f) for f in files
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        if not images:
            continue
        if any(x in folder_name for x in ["nonaccident", "noaccident", "normal", "negative", "non"]):
            all_normal_images.extend(images)
            print(f"   → Normal: {os.path.basename(root)} ({len(images)} images)")
        elif any(x in folder_name for x in ["accident", "positive", "crash"]):
            all_accident_images.extend(images)
            print(f"   → Accident: {os.path.basename(root)} ({len(images)} images)")
except Exception as e:
    print(f"   ⚠️  Secondary dataset not available: {e}")


# --- Dataset 3: User-provided images ---
print("\n📥 [Dataset 3/3] Checking for user-uploaded images...")
USER_ACC_DIR = "/content/user_data/accident"
USER_NORM_DIR = "/content/user_data/no_accident"

os.makedirs(USER_ACC_DIR, exist_ok=True)
os.makedirs(USER_NORM_DIR, exist_ok=True)

user_acc = glob.glob(os.path.join(USER_ACC_DIR, "*.*"))
user_norm = glob.glob(os.path.join(USER_NORM_DIR, "*.*"))

if user_acc or user_norm:
    all_accident_images.extend(user_acc)
    all_normal_images.extend(user_norm)
    print(f"   ✅ Found {len(user_acc)} accident + {len(user_norm)} normal user images")
else:
    print(f"   ℹ️  No user images found.")
    print(f"   To add your own training data, upload images to:")
    print(f"   {USER_ACC_DIR}/")
    print(f"   {USER_NORM_DIR}/")
    print(f"   Then re-run this cell.")


# --- Organize combined dataset ---
print(f"\n📊 Combined dataset: {len(all_accident_images)} accident + {len(all_normal_images)} normal images")

if not all_accident_images or not all_normal_images:
    print("""
╔══════════════════════════════════════════════════════════════╗
║  NO IMAGES FOUND — Manual Setup Required                    ║
╠══════════════════════════════════════════════════════════════╣
║  Upload images to these Colab folders:                      ║
║                                                             ║
║  /content/user_data/accident/       (200+ accident images)  ║
║  /content/user_data/no_accident/    (200+ normal images)    ║
║                                                             ║
║  Then re-run this cell.                                     ║
╚══════════════════════════════════════════════════════════════╝
""")
    raise SystemExit("Upload images first, then re-run.")

# Balance classes (downsample majority if ratio > 3:1)
min_count = min(len(all_accident_images), len(all_normal_images))
max_count = max(len(all_accident_images), len(all_normal_images))

if max_count > min_count * 3:
    print(f"   ⚖️  Class imbalance detected ({max_count} vs {min_count})")
    print(f"   Downsampling majority class to {min_count * 2}...")
    if len(all_accident_images) > len(all_normal_images):
        random.shuffle(all_accident_images)
        all_accident_images = all_accident_images[:min_count * 2]
    else:
        random.shuffle(all_normal_images)
        all_normal_images = all_normal_images[:min_count * 2]

# Shuffle and split (80/20)
random.shuffle(all_accident_images)
random.shuffle(all_normal_images)

split_acc = int(len(all_accident_images) * 0.8)
split_norm = int(len(all_normal_images) * 0.8)

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
        # Rename to avoid conflicts between datasets
        shutil.copy2(img_path, os.path.join(dst_dir, f"{cls}_{split}_{i:05d}{ext}"))

print("\n✅ Combined dataset organized:")
for split in ["train", "val"]:
    for cls in ["accident", "no_accident"]:
        d = os.path.join(DATASET_DIR, split, cls)
        n = len(os.listdir(d))
        print(f"   {split}/{cls}: {n} images")


# ============================================================
# CELL 3: Train YOLOv8n-cls
# ============================================================
from ultralytics import YOLO

print("\n" + "=" * 60)
print("  TRAINING M1: SCENE CLASSIFIER")
print("  Estimated time: ~2-3 hours on T4 GPU")
print("=" * 60)

device = "0" if torch.cuda.is_available() else "cpu"
print(f"  Device: {'GPU (' + torch.cuda.get_device_name(0) + ')' if device == '0' else 'CPU'}")

# Load pretrained YOLOv8n-cls
model = YOLO("yolov8n-cls.pt")

# Train with aggressive data augmentation for robustness
results = model.train(
    data=DATASET_DIR,
    epochs=80,              # More epochs for better convergence
    imgsz=224,
    batch=32 if device == "0" else 16,
    patience=15,            # Early stopping patience
    device=device,
    project="/content/runs/classify",
    name="m1_scene_classifier",
    exist_ok=True,
    verbose=True,
    # Data augmentation — critical for CCTV robustness
    hsv_h=0.015,            # Hue variation (lighting changes)
    hsv_s=0.7,              # Saturation variation (weather)
    hsv_v=0.4,              # Value/brightness variation (day/night)
    degrees=15,             # Rotation (different camera angles)
    translate=0.15,         # Translation (different camera positions)
    scale=0.5,              # Scale variation (zoom levels)
    shear=5,                # Shear (perspective distortion)
    fliplr=0.5,             # Horizontal flip
    flipud=0.0,             # No vertical flip (unrealistic for CCTV)
    mosaic=0.0,             # No mosaic (we want full scene context)
    erasing=0.3,            # Random erasing (occlusion robustness)
    crop_fraction=0.8,      # Crop fraction for classification
)

print("\n✅ M1 Training complete!")


# ============================================================
# CELL 4: Evaluate
# ============================================================
metrics = model.val()
print(f"\n  Top-1 Accuracy: {metrics.top1:.3f}")
print(f"  Top-5 Accuracy: {metrics.top5:.3f}")

# Test on a few samples
print("\n  Sample predictions:")
val_images = glob.glob(os.path.join(VAL_DIR, "**", "*.jpg"), recursive=True)
val_images += glob.glob(os.path.join(VAL_DIR, "**", "*.png"), recursive=True)
for img in val_images[:10]:
    res = model.predict(img, imgsz=224, verbose=False)
    pred_class = res[0].probs.top1
    pred_conf = float(res[0].probs.top1conf)
    true_class = "accident" if "/accident/" in img else "no_accident"
    pred_name = model.names[pred_class]
    status = "✅" if pred_name == true_class else "❌"
    print(f"    {status} True: {true_class:15s} → Pred: {pred_name:15s} ({pred_conf:.2f})")


# ============================================================
# CELL 5: Export & Download
# ============================================================
best_path = "/content/runs/classify/m1_scene_classifier/weights/best.pt"

if os.path.exists(best_path):
    download_path = "/content/accident_classifier.pt"
    shutil.copy2(best_path, download_path)
    size_mb = os.path.getsize(download_path) / (1024 * 1024)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  ✅ M1: SCENE CLASSIFIER READY — {size_mb:.1f} MB                  ║
╠══════════════════════════════════════════════════════════════╣
║                                                             ║
║  DOWNLOAD:                                                  ║
║  1. Click 📁 folder icon (left sidebar)                     ║
║  2. Find 'accident_classifier.pt' in /content/              ║
║  3. Right-click → Download                                  ║
║                                                             ║
║  INSTALL:                                                   ║
║  Place the file in your local project at:                   ║
║  AccidentDetection/weights/accident_classifier.pt           ║
║                                                             ║
║  The pipeline will auto-detect and use it!                  ║
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

print("\n🎉 M1 Done!")

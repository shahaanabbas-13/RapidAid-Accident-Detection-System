"""
╔══════════════════════════════════════════════════════════════╗
║  RapidAid — Accident Classifier Training Script             ║
║  Run this on Google Colab (GPU Runtime)                     ║
║  Training time: ~1-2 hours on T4 GPU                        ║
╚══════════════════════════════════════════════════════════════╝

BEFORE RUNNING:
  1. Go to Runtime → Change runtime type → Select GPU (T4)
  2. Then run this cell
"""

# ============================================================
# STEP 0: Check GPU availability
# ============================================================
import torch
import os
import shutil
import glob

if torch.cuda.is_available():
    print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️  WARNING: No GPU detected!")
    print("   Go to: Runtime → Change runtime type → GPU (T4)")
    print("   Then restart and re-run this cell.")
    print("   Training on CPU will take 10+ hours instead of 1-2 hours.\n")

# ============================================================
# STEP 1: Install Dependencies
# ============================================================
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "kagglehub"])
print("✅ Dependencies installed")

# ============================================================
# STEP 2: Download & Organize Dataset
# ============================================================
DATASET_DIR = "/content/accident_dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")

# Create directory structure
for split in ["train", "val"]:
    for cls in ["accident", "no_accident"]:
        os.makedirs(os.path.join(DATASET_DIR, split, cls), exist_ok=True)

print("📁 Dataset directories created")

# Download Kaggle dataset
print("\n📥 Downloading Kaggle accident dataset...")
try:
    import kagglehub
    kaggle_path = kagglehub.dataset_download("ckay16/accident-detection-from-cctv-footage")
    print(f"✅ Downloaded to: {kaggle_path}")
except Exception as e:
    print(f"❌ Kaggle download failed: {e}")
    kaggle_path = None

# Explore the downloaded dataset structure
if kaggle_path and os.path.exists(kaggle_path):
    print("\n📂 Exploring dataset structure:")
    for root, dirs, files in os.walk(kaggle_path):
        level = root.replace(kaggle_path, '').count(os.sep)
        indent = ' ' * 2 * level
        img_count = len([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        if img_count > 0 or dirs:
            print(f"{indent}{os.path.basename(root)}/ ({img_count} images)")
        if level > 3:
            break

    # Find all image folders and classify them
    accident_images = []
    normal_images = []

    for root, dirs, files in os.walk(kaggle_path):
        folder_name = os.path.basename(root).lower().replace(" ", "").replace("_", "").replace("-", "")
        images = [os.path.join(root, f) for f in files
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        if not images:
            continue

        # Classify folder based on name
        if "nonaccident" in folder_name or "noaccident" in folder_name or "normal" in folder_name or "negative" in folder_name:
            normal_images.extend(images)
            print(f"  → Normal traffic: {os.path.basename(root)} ({len(images)} images)")
        elif "accident" in folder_name or "positive" in folder_name or "crash" in folder_name:
            accident_images.extend(images)
            print(f"  → Accident: {os.path.basename(root)} ({len(images)} images)")

    # If no classification by folder name, try parent folders
    if not accident_images and not normal_images:
        print("\n  Trying alternate folder detection...")
        for root, dirs, files in os.walk(kaggle_path):
            images = [os.path.join(root, f) for f in files
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            if images:
                # Check if any parent folder has accident/normal in name
                full_path = root.lower()
                if "1" in os.path.basename(root) or "pos" in full_path:
                    accident_images.extend(images)
                elif "0" in os.path.basename(root) or "neg" in full_path:
                    normal_images.extend(images)

    print(f"\n📊 Found: {len(accident_images)} accident, {len(normal_images)} normal images")

    if accident_images and normal_images:
        # Split into train (80%) and val (20%)
        import random
        random.shuffle(accident_images)
        random.shuffle(normal_images)

        split_acc = int(len(accident_images) * 0.8)
        split_norm = int(len(normal_images) * 0.8)

        splits = {
            ("train", "accident"): accident_images[:split_acc],
            ("val", "accident"): accident_images[split_acc:],
            ("train", "no_accident"): normal_images[:split_norm],
            ("val", "no_accident"): normal_images[split_norm:],
        }

        for (split, cls), images in splits.items():
            dst_dir = os.path.join(DATASET_DIR, split, cls)
            for img_path in images:
                shutil.copy2(img_path, dst_dir)

        print("✅ Dataset organized into train/val splits")
    else:
        print("❌ Could not auto-detect accident vs normal folders.")
        print("   Please manually organize images (see instructions below).")

# Count and verify dataset
print("\n📊 Dataset summary:")
total = 0
for split in ["train", "val"]:
    for cls in ["accident", "no_accident"]:
        d = os.path.join(DATASET_DIR, split, cls)
        n = len(os.listdir(d)) if os.path.exists(d) else 0
        total += n
        print(f"  {split}/{cls}: {n} images")

if total == 0:
    print("""
╔══════════════════════════════════════════════════════════════╗
║  NO IMAGES FOUND — Manual Setup Required                    ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Upload images to these Colab folders:                       ║
║                                                              ║
║  /content/accident_dataset/train/accident/     (200+ images) ║
║  /content/accident_dataset/train/no_accident/  (200+ images) ║
║  /content/accident_dataset/val/accident/       (50+ images)  ║
║  /content/accident_dataset/val/no_accident/    (50+ images)  ║
║                                                              ║
║  Then re-run this cell.                                      ║
╚══════════════════════════════════════════════════════════════╝
""")
    raise SystemExit("Upload images first, then re-run.")

# ============================================================
# STEP 3: Train YOLOv8n-cls Accident Classifier
# ============================================================
from ultralytics import YOLO

print("\n" + "=" * 60)
print("  TRAINING ACCIDENT CLASSIFIER")
print("  This will take ~1-2 hours on GPU, ~10+ hours on CPU")
print("=" * 60)

# Use GPU if available
device = "0" if torch.cuda.is_available() else "cpu"
print(f"  Device: {'GPU (' + torch.cuda.get_device_name(0) + ')' if device == '0' else 'CPU'}")

model = YOLO("yolov8n-cls.pt")

results = model.train(
    data=DATASET_DIR,
    epochs=50,
    imgsz=224,
    batch=32 if device == "0" else 16,
    patience=10,
    device=device,
    project="/content/runs/classify",
    name="accident_classifier",
    exist_ok=True,
    verbose=True,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    mosaic=0.0,
)

print("\n✅ Training complete!")

# ============================================================
# STEP 4: Evaluate
# ============================================================
metrics = model.val()
print(f"\n  Top-1 Accuracy: {metrics.top1:.3f}")
print(f"  Top-5 Accuracy: {metrics.top5:.3f}")

# ============================================================
# STEP 5: Export & Download
# ============================================================
best_path = "/content/runs/classify/accident_classifier/weights/best.pt"

if os.path.exists(best_path):
    download_path = "/content/accident_classifier.pt"
    shutil.copy2(best_path, download_path)
    size_mb = os.path.getsize(download_path) / (1024 * 1024)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  ✅ MODEL READY — {size_mb:.1f} MB                               ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  DOWNLOAD:                                                   ║
║  1. Click 📁 folder icon (left sidebar)                      ║
║  2. Find 'accident_classifier.pt' in /content/               ║
║  3. Right-click → Download                                   ║
║                                                              ║
║  INSTALL:                                                    ║
║  Place the file in your local project at:                    ║
║  AccidentDetection/weights/accident_classifier.pt            ║
║                                                              ║
║  The pipeline will auto-detect and use it!                   ║
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

print("\n🎉 Done!")

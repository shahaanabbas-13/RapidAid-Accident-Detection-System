"""
╔══════════════════════════════════════════════════════════════╗
║  RapidAid — M4: Collision Zone Detector                      ║
║  YOLOv8n Object Detection for Collision Bounding Boxes       ║
║  Run this on Google Colab (GPU Runtime)                      ║
║  Training time: ~2-3 hours on T4 GPU                         ║
╚══════════════════════════════════════════════════════════════╝

PARADIGM SHIFT:
  Instead of: Detect vehicles → Guess which ones collided (fragile)
  Now:        Detect the collision DIRECTLY → Find vehicles inside (robust)

WHAT THIS TRAINS:
  A YOLOv8n object detector that finds COLLISION ZONES in frames:
  - Class "accident": Bounding box around the collision area
    (crumpled vehicles, debris, smoke, impact point)

  This gives us a direct signal: "HERE is where the accident happened"
  Then we simply check which detected vehicles overlap with this zone.

DATASETS:
  Uses Roboflow accident detection datasets with bounding box annotations.
  These datasets contain CCTV/dashcam frames with accident zones labeled.

AFTER TRAINING:
  Download collision_detector.pt → place in weights/ folder
  Pipeline auto-detects and uses M4 as the PRIMARY detection signal!

BEFORE RUNNING:
  1. Runtime → Change runtime type → GPU (T4)
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
import json
import yaml
from pathlib import Path

print("=" * 60)
print("  M4: Collision Zone Detector — Direct Accident Detection")
print("=" * 60)

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ GPU: {gpu_name} ({vram:.1f} GB)")
else:
    print("⚠️  No GPU! Runtime → Change runtime type → GPU (T4)")

import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "ultralytics", "roboflow", "Pillow", "tqdm",
                       "kagglehub", "gdown"])
print("✅ Dependencies installed\n")


# ============================================================
# CELL 2: Download Accident Detection Datasets (Roboflow)
# ============================================================
DATASET_DIR = "/content/m4_dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

print("📥 Downloading accident detection datasets with bounding box annotations...")
print("   These datasets have collision zones labeled as bounding boxes.\n")

dataset_downloaded = False

# --- Strategy 1: Roboflow Universe datasets ---
print("📥 [Strategy 1] Roboflow Accident Detection datasets...")
try:
    from roboflow import Roboflow

    # Multiple Roboflow accident detection datasets
    rf_datasets = [
        # Accident detection from CCTV - bounding box annotations
        ("accident-detection-q0nnb", 1, "accident-detection"),
        # Road accident detection
        ("road-accident-detection-and-severity", 1, "road-accident"),
        # Vehicle crash detection
        ("accident-detection-8dvwn", 1, "accident-detection-v2"),
    ]

    for project_id, version, name in rf_datasets:
        try:
            # Using public Roboflow Universe (no API key needed for public datasets)
            rf = Roboflow(api_key="")  # Public access
            project = rf.workspace().project(project_id)
            dataset = project.version(version).download(
                "yolov8",
                location=os.path.join(DATASET_DIR, name)
            )
            dataset_downloaded = True
            print(f"   ✅ Downloaded: {name}")
        except Exception as e:
            print(f"   ⚠️ {name}: {e}")

except Exception as e:
    print(f"   Roboflow not available: {e}")


# --- Strategy 2: Kaggle accident detection datasets ---
if not dataset_downloaded:
    print("\n📥 [Strategy 2] Kaggle accident detection datasets...")
    try:
        import kagglehub

        # Try multiple Kaggle accident detection datasets
        kaggle_datasets = [
            "tabormeister/accident-detection-using-yolo",
            "saurabhshahane/accident-detection",
        ]

        for ds_name in kaggle_datasets:
            try:
                path = kagglehub.dataset_download(ds_name)
                print(f"   ✅ Downloaded: {ds_name} → {path}")

                # Check if it has YOLO format annotations
                for root, dirs, files in os.walk(path):
                    has_images = any(f.endswith(('.jpg', '.png')) for f in files)
                    has_labels = any(f.endswith('.txt') for f in files)
                    if has_images and has_labels:
                        print(f"   → Found YOLO annotations in {root}")
                        dataset_downloaded = True
                        # Copy to dataset dir
                        dst = os.path.join(DATASET_DIR, "kaggle_dataset")
                        if not os.path.exists(dst):
                            shutil.copytree(root, dst)
                        break
            except Exception as e:
                print(f"   ⚠️ {ds_name}: {e}")

    except Exception as e:
        print(f"   Kaggle not available: {e}")


# --- Strategy 3: Create from CCTV accident classification datasets ---
# If no pre-annotated detection datasets available, we convert the
# classification datasets into detection format by treating the entire
# accident frame as the bounding box (crude but effective for training)
if not dataset_downloaded:
    print("\n📥 [Strategy 3] Converting classification datasets to detection format...")
    print("   Will use accident frames with full-frame bounding boxes as labels.")
    print("   This teaches the model 'what an accident scene looks like'\n")

    try:
        import kagglehub

        # Download accident classification dataset
        cls_path = kagglehub.dataset_download(
            "ckay16/accident-detection-from-cctv-footage"
        )
        print(f"   ✅ Downloaded: {cls_path}")

        # Find accident and non-accident images
        accident_imgs = []
        normal_imgs = []

        for root, dirs, files in os.walk(cls_path):
            folder = os.path.basename(root).lower().replace(" ", "").replace("_", "")
            images = [os.path.join(root, f) for f in files
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if not images:
                continue
            if any(x in folder for x in ["accident", "positive", "crash"]):
                accident_imgs.extend(images)
            elif any(x in folder for x in ["nonaccident", "noaccident", "normal"]):
                normal_imgs.extend(images)

        # Fallback
        if not accident_imgs:
            for root, dirs, files in os.walk(cls_path):
                images = [os.path.join(root, f) for f in files
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                base = os.path.basename(root).lower()
                if base in ("1",):
                    accident_imgs.extend(images)
                elif base in ("0",):
                    normal_imgs.extend(images)

        print(f"   Found {len(accident_imgs)} accident + {len(normal_imgs)} normal")

        if accident_imgs:
            # Create YOLO detection dataset
            # For accident images: label = centered bbox covering 60-90% of frame
            # For normal images: no labels (background)
            random.shuffle(accident_imgs)
            random.shuffle(normal_imgs)

            # Limit to manageable size
            accident_imgs = accident_imgs[:1500]
            normal_imgs = normal_imgs[:800]

            # Split
            split_acc = int(len(accident_imgs) * 0.85)
            split_norm = int(len(normal_imgs) * 0.85)

            for split_name, acc_list, norm_list in [
                ("train", accident_imgs[:split_acc], normal_imgs[:split_norm]),
                ("val", accident_imgs[split_acc:], normal_imgs[split_norm:]),
            ]:
                img_dir = os.path.join(DATASET_DIR, "converted", split_name, "images")
                lbl_dir = os.path.join(DATASET_DIR, "converted", split_name, "labels")
                os.makedirs(img_dir, exist_ok=True)
                os.makedirs(lbl_dir, exist_ok=True)

                # Accident images with labels
                for i, img_path in enumerate(acc_list):
                    ext = os.path.splitext(img_path)[1]
                    fname = f"accident_{i:05d}"
                    shutil.copy2(img_path, os.path.join(img_dir, fname + ext))

                    # YOLO label: class 0 (accident), centered bbox
                    # Covering 70% of frame to approximate collision zone
                    with open(os.path.join(lbl_dir, fname + ".txt"), "w") as f:
                        # Format: class x_center y_center width height (normalized)
                        f.write("0 0.5 0.5 0.7 0.7\n")

                # Normal images (no labels = background)
                for i, img_path in enumerate(norm_list):
                    ext = os.path.splitext(img_path)[1]
                    fname = f"normal_{i:05d}"
                    shutil.copy2(img_path, os.path.join(img_dir, fname + ext))
                    # Create empty label file (no objects)
                    with open(os.path.join(lbl_dir, fname + ".txt"), "w") as f:
                        pass  # Empty = no detections

            # Create YAML config
            yaml_path = os.path.join(DATASET_DIR, "converted", "data.yaml")
            yaml_content = {
                "path": os.path.join(DATASET_DIR, "converted"),
                "train": "train/images",
                "val": "val/images",
                "nc": 1,
                "names": ["accident"],
            }
            with open(yaml_path, "w") as f:
                yaml.dump(yaml_content, f, default_flow_style=False)

            dataset_downloaded = True
            print(f"   ✅ Converted to YOLO detection format")
            print(f"   Train: {len(acc_list)} accident + {len(norm_list)} background")

    except Exception as e:
        print(f"   ❌ Conversion failed: {e}")


# --- Strategy 4: Download additional Kaggle dataset ---
if not dataset_downloaded:
    print("\n📥 [Strategy 4] Alternative Kaggle dataset...")
    try:
        import kagglehub
        path = kagglehub.dataset_download("meaborak/car-accident-detection")
        print(f"   Downloaded: {path}")

        accident_imgs = []
        normal_imgs = []
        for root, dirs, files in os.walk(path):
            folder = os.path.basename(root).lower().replace(" ", "").replace("_", "")
            images = [os.path.join(root, f) for f in files
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if any(x in folder for x in ["accident", "positive"]):
                accident_imgs.extend(images)
            elif any(x in folder for x in ["nonaccident", "normal", "non"]):
                normal_imgs.extend(images)

        if accident_imgs:
            random.shuffle(accident_imgs)
            random.shuffle(normal_imgs)
            accident_imgs = accident_imgs[:1500]
            normal_imgs = normal_imgs[:800]

            split_acc = int(len(accident_imgs) * 0.85)
            split_norm = int(len(normal_imgs) * 0.85)

            for split_name, acc_list, norm_list in [
                ("train", accident_imgs[:split_acc], normal_imgs[:split_norm]),
                ("val", accident_imgs[split_acc:], normal_imgs[split_norm:]),
            ]:
                img_dir = os.path.join(DATASET_DIR, "converted", split_name, "images")
                lbl_dir = os.path.join(DATASET_DIR, "converted", split_name, "labels")
                os.makedirs(img_dir, exist_ok=True)
                os.makedirs(lbl_dir, exist_ok=True)

                for i, img_path in enumerate(acc_list):
                    ext = os.path.splitext(img_path)[1]
                    fname = f"accident_{i:05d}"
                    shutil.copy2(img_path, os.path.join(img_dir, fname + ext))
                    with open(os.path.join(lbl_dir, fname + ".txt"), "w") as f:
                        f.write("0 0.5 0.5 0.7 0.7\n")

                for i, img_path in enumerate(norm_list):
                    ext = os.path.splitext(img_path)[1]
                    fname = f"normal_{i:05d}"
                    shutil.copy2(img_path, os.path.join(img_dir, fname + ext))
                    with open(os.path.join(lbl_dir, fname + ".txt"), "w") as f:
                        pass

            yaml_path = os.path.join(DATASET_DIR, "converted", "data.yaml")
            yaml_content = {
                "path": os.path.join(DATASET_DIR, "converted"),
                "train": "train/images",
                "val": "val/images",
                "nc": 1,
                "names": ["accident"],
            }
            with open(yaml_path, "w") as f:
                yaml.dump(yaml_content, f, default_flow_style=False)

            dataset_downloaded = True
            print(f"   ✅ Converted {len(accident_imgs)} accident + "
                  f"{len(normal_imgs)} normal images")

    except Exception as e:
        print(f"   ❌ Failed: {e}")


# --- Allow user to upload annotated data ---
USER_DATA = "/content/user_m4_data"
os.makedirs(os.path.join(USER_DATA, "images"), exist_ok=True)
os.makedirs(os.path.join(USER_DATA, "labels"), exist_ok=True)

if not dataset_downloaded:
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  NO DATASETS DOWNLOADED — Please upload data manually       ║
║                                                             ║
║  Upload annotated images to:                                ║
║  {USER_DATA}/images/                                        ║
║  {USER_DATA}/labels/                                        ║
║                                                             ║
║  YOLO format: class x_center y_center width height          ║
║  Class 0 = accident zone                                    ║
╚══════════════════════════════════════════════════════════════╝
""")
    raise SystemExit("Upload data, then re-run.")


# ============================================================
# CELL 3: Find or create YAML config
# ============================================================
# Look for existing YAML files from downloaded datasets
yaml_path = None

# Check for Roboflow-style datasets
for root, dirs, files in os.walk(DATASET_DIR):
    for f in files:
        if f.endswith('.yaml') or f.endswith('.yml'):
            yaml_path = os.path.join(root, f)
            print(f"   Found YAML config: {yaml_path}")
            break
    if yaml_path:
        break

if yaml_path is None:
    yaml_path = os.path.join(DATASET_DIR, "converted", "data.yaml")

print(f"\n📋 Using config: {yaml_path}")

# Read and display config
with open(yaml_path) as f:
    config = yaml.safe_load(f)
    print(f"   Classes: {config.get('names', 'unknown')}")
    print(f"   NC: {config.get('nc', 'unknown')}")


# ============================================================
# CELL 4: Train YOLOv8n Collision Zone Detector
# ============================================================
from ultralytics import YOLO

print("\n" + "=" * 60)
print("  TRAINING M4: COLLISION ZONE DETECTOR")
print("  This model detects WHERE accidents happen in the frame")
print("  Estimated time: ~2-3 hours on T4 GPU")
print("=" * 60)

device = "0" if torch.cuda.is_available() else "cpu"

# Use YOLOv8n (nano) for speed — we need this to run on CPU locally
model = YOLO("yolov8n.pt")

results = model.train(
    data=yaml_path,
    epochs=120,
    imgsz=640,               # Standard YOLO detection resolution
    batch=16 if device == "0" else 4,
    patience=25,
    device=device,
    project="/content/runs/detect",
    name="m4_collision_detector",
    exist_ok=True,
    verbose=True,
    # Augmentation for CCTV robustness
    hsv_h=0.02,
    hsv_s=0.7,
    hsv_v=0.5,
    degrees=10,
    translate=0.15,
    scale=0.5,
    shear=5,
    fliplr=0.5,
    flipud=0.0,
    mosaic=1.0,              # Mosaic helps for detection tasks
    erasing=0.3,
    # Detection-specific settings
    iou=0.5,                 # IoU threshold for NMS
    conf=0.25,               # Confidence threshold
)

print("\n✅ M4 Training complete!")


# ============================================================
# CELL 5: Evaluate
# ============================================================
metrics = model.val()
print(f"\n  mAP50: {metrics.box.map50:.3f}")
print(f"  mAP50-95: {metrics.box.map:.3f}")
print(f"  Precision: {metrics.box.mp:.3f}")
print(f"  Recall: {metrics.box.mr:.3f}")

# Test on some validation images
print("\n  Sample predictions:")
val_img_dir = config.get("val", "val/images")
if not os.path.isabs(val_img_dir):
    val_img_dir = os.path.join(config.get("path", DATASET_DIR), val_img_dir)

val_images = glob.glob(os.path.join(val_img_dir, "*.jpg"))
val_images += glob.glob(os.path.join(val_img_dir, "*.png"))

for img_path in val_images[:5]:
    res = model.predict(img_path, imgsz=640, verbose=False, conf=0.25)
    n_boxes = len(res[0].boxes) if res[0].boxes is not None else 0
    print(f"    {os.path.basename(img_path)}: {n_boxes} collision zone(s) detected")
    if n_boxes > 0:
        for box in res[0].boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            cls_name = model.names[cls]
            print(f"      → {cls_name} (conf={conf:.2f})")


# ============================================================
# CELL 6: Export & Download
# ============================================================
best_path = "/content/runs/detect/m4_collision_detector/weights/best.pt"

if os.path.exists(best_path):
    download_path = "/content/collision_detector.pt"
    shutil.copy2(best_path, download_path)
    size_mb = os.path.getsize(download_path) / (1024 * 1024)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  ✅ M4: COLLISION ZONE DETECTOR READY — {size_mb:.1f} MB             ║
╠══════════════════════════════════════════════════════════════╣
║                                                             ║
║  DOWNLOAD:                                                  ║
║  1. Click 📁 folder icon (left sidebar)                     ║
║  2. Find 'collision_detector.pt' in /content/               ║
║  3. Right-click → Download                                  ║
║                                                             ║
║  INSTALL:                                                   ║
║  Place at: AccidentDetection/weights/collision_detector.pt  ║
║                                                             ║
║  The pipeline auto-detects and uses M4 as PRIMARY signal!   ║
║  M4 detects WHERE the accident happened, then checks which  ║
║  vehicles are inside that zone.                             ║
╚══════════════════════════════════════════════════════════════╝
""")

    try:
        from google.colab import files
        files.download(download_path)
    except:
        print("(Use file browser to download)")
else:
    print("❌ Model not found. Check training logs.")

print("\n🎉 M4 Done! Place collision_detector.pt in weights/ folder.")

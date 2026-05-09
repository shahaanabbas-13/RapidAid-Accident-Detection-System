"""
╔══════════════════════════════════════════════════════════════╗
║  RapidAid — M3 Retraining with CCD Dataset                  ║
║  Uses 1,500 crash + 3,000 normal dashcam MP4 videos          ║
║  Run on Google Colab (GPU Runtime: T4)                       ║
║  Training time: ~1-2 hours on T4                             ║
╚══════════════════════════════════════════════════════════════╝

STEPS BEFORE RUNNING:
  1. Go to Runtime → Change runtime type → GPU (T4)
  2. Add CCD dataset to your Google Drive (instructions below)
  3. Upload your trained accident_classifier.pt (M1) when prompted
  4. Run all cells in order

CCD DATASET SETUP (do this ONCE, before running):
  1. Open: https://drive.google.com/drive/folders/1NUwC-bkka0-iPqhEhIgsXWtj0DA2MR-F
  2. In Google Drive, right-click the "CarCrash" folder
  3. Click "Organize" → "Add shortcut" → "My Drive"
  4. Now the CCD dataset is accessible when you mount Drive in Colab
"""

# ============================================================
# CELL 1: Setup & GPU Check
# ============================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import shutil
import random
import glob
import json
import cv2
import subprocess
import sys
import numpy as np
from tqdm import tqdm

print("=" * 60)
print("  M3: Temporal Classifier — Retraining with CCD Dataset")
print("=" * 60)

if torch.cuda.is_available():
    print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda")
else:
    print("⚠️  No GPU — training will be slow")
    device = torch.device("cpu")

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "ultralytics", "tqdm", "opencv-python-headless"])
print("✅ Dependencies installed\n")


# ============================================================
# CELL 2: Upload M1 Model
# ============================================================
from ultralytics import YOLO

M1_PATH = "/content/accident_classifier.pt"

if not os.path.exists(M1_PATH):
    print("=" * 60)
    print("  Upload your trained accident_classifier.pt")
    print("=" * 60)
    try:
        from google.colab import files as colab_files
        uploaded = colab_files.upload()
        for name in uploaded.keys():
            if name != M1_PATH:
                os.rename(name, M1_PATH)
            print(f"  ✅ Uploaded: {name}")
    except Exception:
        print("  ⚠️  Upload failed. Using pretrained YOLOv8n-cls instead.")

if not os.path.exists(M1_PATH):
    print("  Falling back to pretrained YOLOv8n-cls")
    M1_PATH = "yolov8n-cls.pt"

print(f"\n📦 Loading feature extractor: {M1_PATH}")
backbone_model = YOLO(M1_PATH, task="classify")
print("✅ Feature extractor loaded")


def extract_features(frame, model=backbone_model):
    """Extract probability features from a frame using M1."""
    results = model.predict(frame, imgsz=224, verbose=False, device="cpu")
    if results and results[0].probs is not None:
        return results[0].probs.data.cpu().numpy()
    return np.zeros(2)


# ============================================================
# CELL 3: Mount Google Drive & Access CCD Dataset
# ============================================================
from google.colab import drive

print("\n" + "=" * 60)
print("  MOUNTING GOOGLE DRIVE")
print("=" * 60)
print("""
  PREREQUISITE — Do this once BEFORE running this cell:
  
  1. Open in browser:
     https://drive.google.com/drive/folders/1NUwC-bkka0-iPqhEhIgsXWtj0DA2MR-F
  
  2. You'll see a folder called "CarCrash" (or similar)
  
  3. Right-click the folder → "Organize" → "Add shortcut"
     → Select "My Drive" → Click "Add"
  
  4. Now the CCD data is accessible from your Google Drive!
""")

drive.mount('/content/drive')

# Find the CCD videos folder
# Try common locations where the shortcut might be
CCD_SEARCH_PATHS = [
    "/content/drive/MyDrive/CarCrash",
    "/content/drive/MyDrive/CCD",
    "/content/drive/MyDrive/CarCrashDataset",
    "/content/drive/MyDrive/Car Crash Dataset",
]

ccd_root = None
for path in CCD_SEARCH_PATHS:
    if os.path.exists(path):
        ccd_root = path
        print(f"✅ Found CCD at: {ccd_root}")
        break

if ccd_root is None:
    # Search for it
    print("🔍 Searching for CCD in Google Drive...")
    for root, dirs, files in os.walk("/content/drive/MyDrive"):
        for d in dirs:
            if "crash" in d.lower() or "carcrash" in d.lower():
                candidate = os.path.join(root, d)
                # Check if it has a videos subfolder
                if (os.path.exists(os.path.join(candidate, "videos")) or
                        glob.glob(os.path.join(candidate, "**/*.mp4"), recursive=True)):
                    ccd_root = candidate
                    print(f"✅ Found CCD at: {ccd_root}")
                    break
        if ccd_root:
            break
        # Don't recurse too deep
        if root.count(os.sep) - "/content/drive/MyDrive".count(os.sep) > 2:
            dirs.clear()

if ccd_root is None:
    print("""
╔══════════════════════════════════════════════════════════════╗
║  ❌ CCD DATASET NOT FOUND IN GOOGLE DRIVE                   ║
╠══════════════════════════════════════════════════════════════╣
║                                                             ║
║  Please add the CCD shortcut to your Drive first:           ║
║                                                             ║
║  1. Open this link in your browser:                         ║
║     https://drive.google.com/drive/folders/                 ║
║     1NUwC-bkka0-iPqhEhIgsXWtj0DA2MR-F                      ║
║                                                             ║
║  2. Right-click "CarCrash" → Organize → Add shortcut       ║
║     → My Drive → Add                                        ║
║                                                             ║
║  3. Re-run this cell after adding the shortcut              ║
╚══════════════════════════════════════════════════════════════╝
""")
    raise SystemExit("Add CCD shortcut to Google Drive first")


# ============================================================
# CELL 4: Unzip & Organize CCD Videos
# ============================================================
#
# CCD stores videos as ZIP archives:
#   CarCrash/videos/Crash-1500.zip  (756 MB → 1,500 crash MP4s)
#   CarCrash/videos/Normal.zip      (6 GB  → 3,000 normal MP4s)
#
# We unzip to /content/ (Colab local storage) for fast access.
# ============================================================
import zipfile

VIDEO_DATASET_DIR = "/content/video_dataset"
EXTRACT_DIR = "/content/ccd_extracted"
SEQUENCE_LENGTH = 16
FRAME_SKIP = 2
MAX_VIDEOS_PER_CLASS = 300  # Use 300 per class (~1 hour training)

os.makedirs(f"{VIDEO_DATASET_DIR}/accident", exist_ok=True)
os.makedirs(f"{VIDEO_DATASET_DIR}/no_accident", exist_ok=True)
os.makedirs(EXTRACT_DIR, exist_ok=True)

print("\n" + "=" * 60)
print("  UNZIPPING & ORGANIZING CCD VIDEOS")
print("=" * 60)

videos_dir = os.path.join(ccd_root, "videos")

# --- Unzip Crash-1500.zip ---
crash_zip = os.path.join(videos_dir, "Crash-1500.zip")
crash_extract = os.path.join(EXTRACT_DIR, "Crash-1500")

if os.path.exists(crash_extract) and glob.glob(os.path.join(crash_extract, "*.mp4")):
    print(f"  ✅ Crash videos already extracted: {crash_extract}")
else:
    if os.path.exists(crash_zip):
        print(f"  📦 Unzipping Crash-1500.zip (756 MB)...")
        os.makedirs(crash_extract, exist_ok=True)
        with zipfile.ZipFile(crash_zip, 'r') as zf:
            # Extract only MP4 files, limit to MAX_VIDEOS_PER_CLASS
            mp4_members = [m for m in zf.namelist() if m.lower().endswith('.mp4')]
            print(f"     Found {len(mp4_members)} MP4 files in archive")
            for i, member in enumerate(mp4_members[:MAX_VIDEOS_PER_CLASS]):
                zf.extract(member, crash_extract)
                if (i + 1) % 100 == 0:
                    print(f"     Extracted {i+1}/{min(len(mp4_members), MAX_VIDEOS_PER_CLASS)}")
        print(f"  ✅ Crash videos extracted!")
    else:
        print(f"  ⚠️  Crash-1500.zip not found at: {crash_zip}")

# --- Unzip Normal.zip ---
normal_zip = os.path.join(videos_dir, "Normal.zip")
normal_extract = os.path.join(EXTRACT_DIR, "Normal")

if os.path.exists(normal_extract) and glob.glob(os.path.join(normal_extract, "*.mp4")):
    print(f"  ✅ Normal videos already extracted: {normal_extract}")
else:
    if os.path.exists(normal_zip):
        print(f"  📦 Unzipping Normal.zip (6 GB) — extracting {MAX_VIDEOS_PER_CLASS} videos...")
        os.makedirs(normal_extract, exist_ok=True)
        with zipfile.ZipFile(normal_zip, 'r') as zf:
            mp4_members = [m for m in zf.namelist() if m.lower().endswith('.mp4')]
            print(f"     Found {len(mp4_members)} MP4 files in archive")
            # Only extract MAX_VIDEOS_PER_CLASS to save time and disk space
            for i, member in enumerate(mp4_members[:MAX_VIDEOS_PER_CLASS]):
                zf.extract(member, normal_extract)
                if (i + 1) % 100 == 0:
                    print(f"     Extracted {i+1}/{min(len(mp4_members), MAX_VIDEOS_PER_CLASS)}")
        print(f"  ✅ Normal videos extracted!")
    else:
        print(f"  ⚠️  Normal.zip not found at: {normal_zip}")

# --- Find extracted MP4s ---
crash_vids = sorted(glob.glob(os.path.join(crash_extract, "**", "*.mp4"), recursive=True))
normal_vids = sorted(glob.glob(os.path.join(normal_extract, "**", "*.mp4"), recursive=True))

print(f"\n  Extracted: {len(crash_vids)} crash + {len(normal_vids)} normal videos")

if not crash_vids or not normal_vids:
    print("❌ Could not find extracted videos")
    print(f"   Crash dir: {os.listdir(crash_extract) if os.path.exists(crash_extract) else 'N/A'}")
    print(f"   Normal dir: {os.listdir(normal_extract) if os.path.exists(normal_extract) else 'N/A'}")
    raise SystemExit("ZIP extraction failed — check Drive permissions")

# Randomly sample MAX_VIDEOS_PER_CLASS from each
random.seed(42)
crash_sample = random.sample(crash_vids, min(len(crash_vids), MAX_VIDEOS_PER_CLASS))
normal_sample = random.sample(normal_vids, min(len(normal_vids), MAX_VIDEOS_PER_CLASS))

print(f"  Using: {len(crash_sample)} crash + {len(normal_sample)} normal")
print(f"  (Capped at {MAX_VIDEOS_PER_CLASS} per class for ~1hr training)\n")

# Create symlinks instead of copying (faster, saves disk space)
print("  Creating symlinks to CCD videos...")
for i, v in enumerate(crash_sample):
    link = os.path.join(VIDEO_DATASET_DIR, "accident", f"crash_{i:04d}.mp4")
    if not os.path.exists(link):
        os.symlink(v, link)

for i, v in enumerate(normal_sample):
    link = os.path.join(VIDEO_DATASET_DIR, "no_accident", f"normal_{i:04d}.mp4")
    if not os.path.exists(link):
        os.symlink(v, link)

acc_count = len(glob.glob(f"{VIDEO_DATASET_DIR}/accident/*.mp4"))
norm_count = len(glob.glob(f"{VIDEO_DATASET_DIR}/no_accident/*.mp4"))
print(f"✅ Dataset ready: {acc_count} accident + {norm_count} normal videos\n")


# ============================================================
# CELL 5: Extract Features from Video Clips
# ============================================================
print("=" * 60)
print("  EXTRACTING FEATURES FROM VIDEO CLIPS")
print(f"  Sequence length: {SEQUENCE_LENGTH} frames")
print(f"  Frame skip: every {FRAME_SKIP}th frame")
print("=" * 60)

all_sequences = []

for label, folder in [(1, "accident"), (0, "no_accident")]:
    video_dir = f"{VIDEO_DATASET_DIR}/{folder}"
    videos = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    print(f"\n  Processing {len(videos)} {folder} videos...")

    for vid_path in tqdm(videos, desc=f"  {folder}"):
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames < SEQUENCE_LENGTH * FRAME_SKIP:
            skip = max(1, total_frames // SEQUENCE_LENGTH)
        else:
            skip = FRAME_SKIP

        frame_features = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % skip == 0:
                features = extract_features(frame)
                frame_features.append(features)
            frame_idx += 1

        cap.release()

        if len(frame_features) < SEQUENCE_LENGTH:
            continue

        # Create overlapping sequences (50% overlap)
        stride = SEQUENCE_LENGTH // 2
        for start in range(0, len(frame_features) - SEQUENCE_LENGTH + 1, stride):
            seq = frame_features[start:start + SEQUENCE_LENGTH]
            all_sequences.append((np.array(seq), label))

print(f"\n✅ Extracted {len(all_sequences)} sequences total")
accident_seqs = sum(1 for _, l in all_sequences if l == 1)
normal_seqs = sum(1 for _, l in all_sequences if l == 0)
print(f"   Accident: {accident_seqs}, Normal: {normal_seqs}")


# ============================================================
# CELL 6: Define LSTM Model
# ============================================================
class TemporalAccidentClassifier(nn.Module):
    """Bidirectional LSTM with attention for temporal accident detection."""

    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64), nn.Tanh(), nn.Linear(64, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(lstm_out * attn_weights, dim=1)
        return self.classifier(context).squeeze(-1)


class VideoSequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        features, label = self.sequences[idx]
        return torch.FloatTensor(features), torch.FloatTensor([label])


# ============================================================
# CELL 7: Train the LSTM Model
# ============================================================
if not all_sequences:
    print("❌ No sequences extracted.")
    raise SystemExit("Need video data")

# Split train/val (80/20)
random.shuffle(all_sequences)
split_idx = int(len(all_sequences) * 0.8)
train_seqs = all_sequences[:split_idx]
val_seqs = all_sequences[split_idx:]

feature_dim = all_sequences[0][0].shape[1]
print(f"\n  Feature dimension: {feature_dim}")
print(f"  Train sequences: {len(train_seqs)}")
print(f"  Val sequences: {len(val_seqs)}")

train_loader = DataLoader(VideoSequenceDataset(train_seqs), batch_size=32, shuffle=True)
val_loader = DataLoader(VideoSequenceDataset(val_seqs), batch_size=32, shuffle=False)

model_lstm = TemporalAccidentClassifier(
    input_dim=feature_dim, hidden_dim=128, num_layers=2, dropout=0.3
).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model_lstm.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

print(f"  Model parameters: {sum(p.numel() for p in model_lstm.parameters()):,}")
print(f"  Device: {device}")

print("\n" + "=" * 60)
print("  TRAINING M3: TEMPORAL CLASSIFIER (CCD Dataset)")
print("=" * 60)

best_val_acc = 0
best_epoch = 0
patience_counter = 0
MAX_PATIENCE = 15
NUM_EPOCHS = 100

for epoch in range(NUM_EPOCHS):
    # Training
    model_lstm.train()
    train_loss = train_correct = train_total = 0

    for features, labels in train_loader:
        features, labels = features.to(device), labels.squeeze().to(device)
        optimizer.zero_grad()
        outputs = model_lstm(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_correct += ((outputs > 0.5).float() == labels).sum().item()
        train_total += labels.size(0)

    train_acc = train_correct / max(train_total, 1)

    # Validation
    model_lstm.eval()
    val_loss = val_correct = val_total = 0

    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.squeeze().to(device)
            outputs = model_lstm(features)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_correct += ((outputs > 0.5).float() == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / max(val_total, 1)
    scheduler.step(val_loss)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1:3d}: "
              f"train_loss={train_loss/len(train_loader):.4f} "
              f"train_acc={train_acc:.3f} "
              f"val_loss={val_loss/len(val_loader):.4f} "
              f"val_acc={val_acc:.3f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        patience_counter = 0
        torch.save({
            "model_state_dict": model_lstm.state_dict(),
            "feature_dim": feature_dim,
            "sequence_length": SEQUENCE_LENGTH,
            "hidden_dim": 128,
            "num_layers": 2,
        }, "/content/temporal_classifier.pt")
    else:
        patience_counter += 1
        if patience_counter >= MAX_PATIENCE:
            print(f"\n  Early stopping at epoch {epoch+1}")
            break

print(f"\n✅ M3 Training complete!")
print(f"   Best val accuracy: {best_val_acc:.3f} (epoch {best_epoch})")


# ============================================================
# CELL 8: Export & Download
# ============================================================
model_path = "/content/temporal_classifier.pt"

if os.path.exists(model_path):
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  ✅ M3 RETRAINED WITH CCD — {size_mb:.1f} MB                      ║
╠══════════════════════════════════════════════════════════════╣
║                                                             ║
║  Best validation accuracy: {best_val_acc:.1%}                      ║
║  Training data: {len(crash_sample)} crash + {len(normal_sample)} normal videos       ║
║  Sequences: {len(all_sequences)} total                             ║
║                                                             ║
║  DOWNLOAD & INSTALL:                                        ║
║  1. Download temporal_classifier.pt from /content/           ║
║  2. Replace: AccidentDetection/weights/temporal_classifier.pt║
║  3. Run: python run_all.py                                  ║
╚══════════════════════════════════════════════════════════════╝
""")
    try:
        from google.colab import files
        files.download(model_path)
    except Exception:
        print("(Use file browser to download manually)")

print("\n🎉 Done! Replace your weights/temporal_classifier.pt and re-test.")

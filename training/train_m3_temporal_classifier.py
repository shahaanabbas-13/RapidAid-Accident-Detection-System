"""
╔══════════════════════════════════════════════════════════════╗
║  RapidAid — M3: Temporal Video Classifier                    ║
║  Video-Level Accident Detection Using Frame Sequences         ║
║  Run this on Google Colab (GPU Runtime)                       ║
║  Training time: ~4-6 hours on T4 GPU                          ║
╚══════════════════════════════════════════════════════════════╝

WHAT THIS TRAINS:
  An LSTM-based temporal classifier that analyzes SEQUENCES of frames
  (not individual frames) to detect accidents in video.

  It learns temporal patterns like:
  - Sudden deceleration → impact → debris scatter
  - Normal flow → vehicles converging → collision
  - Moving traffic → sudden stop → people running
  - Vehicle trajectory → unexpected angle change

WHY THIS MODEL IS NEEDED:
  M1 (Scene Classifier) looks at individual frames — it can miss:
  - The MOMENT of impact (which may last only 2-3 frames)
  - Pre-crash patterns (vehicles converging rapidly)
  - Post-crash patterns (sudden traffic stop, people rushing)
  
  M3 sees the TEMPORAL CONTEXT — the before/during/after sequence.
  This is especially powerful for:
  - Image 4 failure: "yellow taxi hitting person in busy traffic"
    → M3 would see the person's trajectory suddenly stopping
  - Video timing accuracy: knowing WHEN the crash happens
  - Distinguishing "car passing person" from "car hitting person"

HOW IT WORKS:
  1. Extract features from each frame using M1's backbone (YOLOv8n-cls)
  2. Feed feature sequences into an LSTM/GRU
  3. LSTM learns to detect temporal anomalies (crash patterns)
  4. Output: per-clip accident probability

DATASET:
  Uses DoTA (Detection of Traffic Anomaly) — 4,677 dashcam videos
  with temporal accident annotations. Falls back to CADP or custom
  video datasets if DoTA isn't available.

ARCHITECTURE:
  Frame → YOLOv8n-cls backbone → 512-dim features
  16-frame sequence → LSTM (256 hidden) → FC → Sigmoid
  Output: P(accident in this 16-frame clip)

BEFORE RUNNING:
  1. Go to Runtime → Change runtime type → Select GPU (T4)
  2. Train M1 first (need the scene classifier backbone)
  3. Then run all cells in order
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
import numpy as np
from tqdm import tqdm

print("=" * 60)
print("  M3: Temporal Video Classifier — Training")
print("=" * 60)

if torch.cuda.is_available():
    print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda")
else:
    print("⚠️  No GPU detected — training will be very slow")
    device = torch.device("cpu")

import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "ultralytics", "kagglehub", "Pillow", "tqdm", "opencv-python-headless"])
print("✅ Dependencies installed\n")


# ============================================================
# CELL 2: Upload M1 Model & Load Feature Extractor
# ============================================================
from ultralytics import YOLO

M1_PATH = "/content/accident_classifier.pt"

# Check if M1 exists — prompt upload if not
if not os.path.exists(M1_PATH):
    print("=" * 60)
    print("  M1 MODEL REQUIRED FOR BEST RESULTS")
    print("=" * 60)
    print("  Upload your trained accident_classifier.pt file.")
    print("  (If you skip this, a generic pretrained model is used,")
    print("   which will produce lower-quality temporal features.)\n")

    try:
        from google.colab import files as colab_files
        uploaded = colab_files.upload()
        for name in uploaded.keys():
            target = "/content/accident_classifier.pt"
            if name != target:
                os.rename(name, target)
            print(f"  ✅ Uploaded: {name} → {target}")
    except Exception:
        print("  ⚠️  Upload skipped or not in Colab environment.")

if not os.path.exists(M1_PATH):
    print("  Using pretrained YOLOv8n-cls as feature extractor instead.")
    M1_PATH = "yolov8n-cls.pt"

print(f"\n📦 Loading feature extractor from: {M1_PATH}")
backbone_model = YOLO(M1_PATH, task="classify")
print("✅ Feature extractor loaded")


def extract_features(frame, model=backbone_model):
    """
    Extract feature vector from a single frame using the M1 backbone.
    Returns a 1D numpy array of features.
    """
    results = model.predict(frame, imgsz=224, verbose=False, device="cpu")
    if results and results[0].probs is not None:
        probs = results[0].probs.data.cpu().numpy()
        return probs
    return np.zeros(2)


# ============================================================
# CELL 3: Video Dataset Preparation
# ============================================================
#
# Uses the CCD (Car Crash Dataset) — 1,500 crash + 3,000 normal
# dashcam MP4 videos from Google Drive.
#
# Paper: "Uncertainty-based Traffic Accident Anticipation with
#         Spatio-Temporal Relational Learning" (ACM MM 2020)
# Repo:  https://github.com/Cogito2012/CarCrashDataset
# ============================================================

VIDEO_DATASET_DIR = "/content/video_dataset"
FEATURES_DIR = "/content/video_features"
SEQUENCE_LENGTH = 16   # Frames per sequence
FRAME_SKIP = 2         # Sample every Nth frame
MAX_VIDEOS_PER_CLASS = 300  # Limit to keep training time ~1 hour

os.makedirs(f"{VIDEO_DATASET_DIR}/accident", exist_ok=True)
os.makedirs(f"{VIDEO_DATASET_DIR}/no_accident", exist_ok=True)

# Check if videos already exist (from a previous run)
acc_vids = glob.glob(f"{VIDEO_DATASET_DIR}/accident/*.*")
norm_vids = glob.glob(f"{VIDEO_DATASET_DIR}/no_accident/*.*")

if acc_vids and norm_vids:
    print(f"✅ Videos already present: {len(acc_vids)} accident + {len(norm_vids)} normal")
else:
    print("""
╔══════════════════════════════════════════════════════════════╗
║  VIDEO DATASET DOWNLOAD                                     ║
╠══════════════════════════════════════════════════════════════╣
║                                                             ║
║  Attempting to download CCD (Car Crash Dataset):            ║
║  • 1,500 crash videos (dashcam MP4)                         ║
║  • 3,000 normal driving videos (dashcam MP4)                ║
║  • Source: Google Drive (no auth required)                   ║
║                                                             ║
║  If auto-download fails, use one of the manual options      ║
║  described below.                                           ║
╚══════════════════════════════════════════════════════════════╝
""")

    # ---- METHOD 1: CCD from Google Drive via gdown ----
    download_success = False
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gdown"])
        import gdown

        # CCD Google Drive folder:
        # https://drive.google.com/drive/folders/1NUwC-bkka0-iPqhEhIgsXWtj0DA2MR-F
        CCD_DIR = "/content/CCD"
        os.makedirs(CCD_DIR, exist_ok=True)

        print("📥 Downloading CCD dataset from Google Drive...")
        print("   (This downloads ~2-4 GB, may take 5-15 minutes)\n")

        # Download the videos folder from CCD
        # The CCD folder structure is:
        #   CarCrash/videos/Crash-1500/  (accident MP4s: 000001.mp4 ... 001500.mp4)
        #   CarCrash/videos/Normal/      (normal MP4s:   000001.mp4 ... 003000.mp4)
        gdown.download_folder(
            url="https://drive.google.com/drive/folders/1NUwC-bkka0-iPqhEhIgsXWtj0DA2MR-F",
            output=CCD_DIR,
            quiet=False,
            use_cookies=False,
        )

        # Find and copy videos to our standard folders
        crash_vids = sorted(glob.glob(f"{CCD_DIR}/**/Crash*/**/*.mp4", recursive=True))
        normal_vids_found = sorted(glob.glob(f"{CCD_DIR}/**/Normal/**/*.mp4", recursive=True))

        # Also check for flat structure
        if not crash_vids:
            crash_vids = sorted(glob.glob(f"{CCD_DIR}/**/Crash*/*.mp4", recursive=True))
        if not normal_vids_found:
            normal_vids_found = sorted(glob.glob(f"{CCD_DIR}/**/Normal/*.mp4", recursive=True))

        # Fallback: check any .mp4 in CCD dir
        if not crash_vids and not normal_vids_found:
            all_mp4s = sorted(glob.glob(f"{CCD_DIR}/**/*.mp4", recursive=True))
            print(f"   Found {len(all_mp4s)} total MP4 files in download")
            # Try to categorize by path keywords
            for mp4 in all_mp4s:
                path_lower = mp4.lower()
                if "crash" in path_lower or "positive" in path_lower or "accident" in path_lower:
                    crash_vids.append(mp4)
                elif "normal" in path_lower or "negative" in path_lower:
                    normal_vids_found.append(mp4)

        if crash_vids and normal_vids_found:
            # Copy up to MAX_VIDEOS_PER_CLASS videos to keep training manageable
            print(f"\n   Found: {len(crash_vids)} crash + {len(normal_vids_found)} normal")
            n_crash = min(len(crash_vids), MAX_VIDEOS_PER_CLASS)
            n_normal = min(len(normal_vids_found), MAX_VIDEOS_PER_CLASS)

            print(f"   Using: {n_crash} crash + {n_normal} normal (capped at {MAX_VIDEOS_PER_CLASS})")

            for v in crash_vids[:n_crash]:
                shutil.copy2(v, f"{VIDEO_DATASET_DIR}/accident/")
            for v in normal_vids_found[:n_normal]:
                shutil.copy2(v, f"{VIDEO_DATASET_DIR}/no_accident/")

            download_success = True
            print("   ✅ CCD videos organized!")
        else:
            print(f"   ⚠️  Could not find crash/normal folders in CCD download")
            print(f"   Downloaded content: {os.listdir(CCD_DIR)}")

    except Exception as e:
        print(f"   ⚠️  CCD download failed: {e}")

    # ---- METHOD 2: Kaggle (if CCD failed) ----
    if not download_success:
        print("\n📥 Trying Kaggle video dataset...")
        try:
            import kagglehub
            kaggle_vid_path = kagglehub.dataset_download(
                "maborak/road-accident-video-dataset"
            )
            print(f"   ✅ Downloaded to: {kaggle_vid_path}")

            for root, dirs, files_list in os.walk(kaggle_vid_path):
                folder_name = os.path.basename(root).lower()
                videos = [os.path.join(root, f) for f in files_list
                          if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
                if not videos:
                    continue
                if any(x in folder_name for x in ["accident", "crash", "positive"]):
                    for v in videos[:MAX_VIDEOS_PER_CLASS]:
                        shutil.copy2(v, f"{VIDEO_DATASET_DIR}/accident/")
                    print(f"   → Accident videos: {min(len(videos), MAX_VIDEOS_PER_CLASS)}")
                elif any(x in folder_name for x in ["normal", "negative", "noaccident"]):
                    for v in videos[:MAX_VIDEOS_PER_CLASS]:
                        shutil.copy2(v, f"{VIDEO_DATASET_DIR}/no_accident/")
                    print(f"   → Normal videos: {min(len(videos), MAX_VIDEOS_PER_CLASS)}")
            download_success = True
        except Exception as e:
            print(f"   ⚠️  Kaggle download failed: {e}")

    # ---- METHOD 3: Manual upload (last resort) ----
    if not download_success:
        print("""
╔══════════════════════════════════════════════════════════════╗
║  ⚠️  AUTO-DOWNLOAD FAILED — MANUAL UPLOAD NEEDED           ║
╠══════════════════════════════════════════════════════════════╣
║                                                             ║
║  Option A: Upload via Google Drive (recommended)            ║
║  1. Mount Drive: from google.colab import drive             ║
║                  drive.mount('/content/drive')               ║
║  2. Copy videos from your Drive to:                         ║
║     /content/video_dataset/accident/                        ║
║     /content/video_dataset/no_accident/                     ║
║                                                             ║
║  Option B: Upload directly                                  ║
║  Click the upload button below to upload MP4 files.         ║
║                                                             ║
║  Option C: Download CCD manually                            ║
║  1. Go to: https://github.com/Cogito2012/CarCrashDataset   ║
║  2. Download from Google Drive link in README               ║
║  3. Upload the videos folder to Colab                       ║
╚══════════════════════════════════════════════════════════════╝
""")
        try:
            from google.colab import files as colab_files

            print("📁 Upload ACCIDENT videos (MP4):")
            acc_uploaded = colab_files.upload()
            for name in acc_uploaded.keys():
                shutil.move(name, f"{VIDEO_DATASET_DIR}/accident/{name}")
            print(f"   ✅ {len(acc_uploaded)} accident videos uploaded\n")

            print("📁 Upload NORMAL traffic videos (MP4):")
            norm_uploaded = colab_files.upload()
            for name in norm_uploaded.keys():
                shutil.move(name, f"{VIDEO_DATASET_DIR}/no_accident/{name}")
            print(f"   ✅ {len(norm_uploaded)} normal videos uploaded")
        except Exception as e:
            print(f"   Upload error: {e}")

    # Final count
    acc_vids = glob.glob(f"{VIDEO_DATASET_DIR}/accident/*.*")
    norm_vids = glob.glob(f"{VIDEO_DATASET_DIR}/no_accident/*.*")
    print(f"\n📊 Final count: {len(acc_vids)} accident + {len(norm_vids)} normal videos")

    if not acc_vids or not norm_vids:
        print("❌ Need videos in BOTH folders to proceed.")
        print("   Re-run this cell after uploading videos.")


# ============================================================
# CELL 4: Extract Features from Video Clips
# ============================================================
print("\n" + "=" * 60)
print("  EXTRACTING FEATURES FROM VIDEO CLIPS")
print(f"  Sequence length: {SEQUENCE_LENGTH} frames")
print(f"  Frame skip: every {FRAME_SKIP}th frame")
print("=" * 60)

all_sequences = []  # List of (feature_sequence, label)

for label, folder in [(1, "accident"), (0, "no_accident")]:
    video_dir = f"{VIDEO_DATASET_DIR}/{folder}"
    videos = glob.glob(os.path.join(video_dir, "*.*"))
    print(f"\n  Processing {len(videos)} {folder} videos...")

    for vid_path in tqdm(videos, desc=f"  {folder}"):
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames < SEQUENCE_LENGTH * FRAME_SKIP:
            # Video too short — use all frames
            skip = max(1, total_frames // SEQUENCE_LENGTH)
        else:
            skip = FRAME_SKIP

        # Extract features from sampled frames
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

        # Create overlapping sequences from the video
        stride = SEQUENCE_LENGTH // 2  # 50% overlap
        for start in range(0, len(frame_features) - SEQUENCE_LENGTH + 1, stride):
            seq = frame_features[start:start + SEQUENCE_LENGTH]
            all_sequences.append((np.array(seq), label))

print(f"\n✅ Extracted {len(all_sequences)} sequences total")
accident_seqs = sum(1 for _, l in all_sequences if l == 1)
normal_seqs = sum(1 for _, l in all_sequences if l == 0)
print(f"   Accident: {accident_seqs}, Normal: {normal_seqs}")


# ============================================================
# CELL 5: Define LSTM Model
# ============================================================
class TemporalAccidentClassifier(nn.Module):
    """
    LSTM-based temporal classifier for accident detection.

    Input: sequence of M1 feature vectors (SEQUENCE_LENGTH × feature_dim)
    Output: probability of accident in the sequence
    """

    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        # Attention mechanism — focus on the most important frames
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x shape: (batch, seq_len, feature_dim)
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch, seq_len, hidden_dim * 2)

        # Attention weights
        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)

        # Weighted sum of LSTM outputs
        context = torch.sum(lstm_out * attn_weights, dim=1)
        # context shape: (batch, hidden_dim * 2)

        # Classify
        output = self.classifier(context)
        return output.squeeze(-1)


class VideoSequenceDataset(Dataset):
    """PyTorch dataset for video feature sequences."""

    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        features, label = self.sequences[idx]
        return torch.FloatTensor(features), torch.FloatTensor([label])


# ============================================================
# CELL 6: Train the LSTM Model
# ============================================================
if not all_sequences:
    print("❌ No sequences extracted. Upload videos first.")
    raise SystemExit("Need video data")

# Split into train/val (80/20)
random.shuffle(all_sequences)
split_idx = int(len(all_sequences) * 0.8)
train_seqs = all_sequences[:split_idx]
val_seqs = all_sequences[split_idx:]

# Get feature dimension from first sequence
feature_dim = all_sequences[0][0].shape[1]
print(f"\n  Feature dimension: {feature_dim}")
print(f"  Train sequences: {len(train_seqs)}")
print(f"  Val sequences: {len(val_seqs)}")

# Create datasets and loaders
train_dataset = VideoSequenceDataset(train_seqs)
val_dataset = VideoSequenceDataset(val_seqs)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize model
model_lstm = TemporalAccidentClassifier(
    input_dim=feature_dim,
    hidden_dim=128,
    num_layers=2,
    dropout=0.3
).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model_lstm.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

print(f"\n  Model parameters: {sum(p.numel() for p in model_lstm.parameters()):,}")
print(f"  Device: {device}")

print("\n" + "=" * 60)
print("  TRAINING M3: TEMPORAL CLASSIFIER")
print("  Estimated time: ~30-60 minutes")
print("=" * 60)

best_val_acc = 0
best_epoch = 0
patience_counter = 0
MAX_PATIENCE = 15
NUM_EPOCHS = 100

for epoch in range(NUM_EPOCHS):
    # --- Training ---
    model_lstm.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.squeeze().to(device)

        optimizer.zero_grad()
        outputs = model_lstm(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        predicted = (outputs > 0.5).float()
        train_correct += (predicted == labels).sum().item()
        train_total += labels.size(0)

    train_acc = train_correct / max(train_total, 1)

    # --- Validation ---
    model_lstm.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            labels = labels.squeeze().to(device)

            outputs = model_lstm(features)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            predicted = (outputs > 0.5).float()
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / max(val_total, 1)
    scheduler.step(val_loss)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1:3d}: "
              f"train_loss={train_loss/len(train_loader):.4f} "
              f"train_acc={train_acc:.3f} "
              f"val_loss={val_loss/len(val_loader):.4f} "
              f"val_acc={val_acc:.3f}")

    # Save best model
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
# CELL 7: Export & Download
# ============================================================
model_path = "/content/temporal_classifier.pt"

if os.path.exists(model_path):
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  ✅ M3: TEMPORAL CLASSIFIER READY — {size_mb:.1f} MB               ║
╠══════════════════════════════════════════════════════════════╣
║                                                             ║
║  DOWNLOAD:                                                  ║
║  1. Find 'temporal_classifier.pt' in /content/              ║
║  2. Right-click → Download                                  ║
║                                                             ║
║  INSTALL:                                                   ║
║  Place at: AccidentDetection/weights/temporal_classifier.pt ║
║                                                             ║
║  NOTE: Integration code for M3 needs to be added to         ║
║  the video_processor.py pipeline.                           ║
╚══════════════════════════════════════════════════════════════╝
""")

    try:
        from google.colab import files
        files.download(model_path)
    except:
        print("(Use the file browser to download manually)")

print("\n🎉 M3 Done!")

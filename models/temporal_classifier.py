"""
RapidAid — Temporal Classifier Module (M3)

Uses an LSTM-based model to classify sequences of video frames
as 'accident' or 'normal' based on temporal patterns.

The model was trained on frame-level features extracted by M1
(scene classifier) fed into a bidirectional LSTM with attention.

Architecture:
    Frame → M1 backbone → feature vector (2-dim probs)
    16-frame sequence → BiLSTM (128 hidden, 2 layers)
    → Attention → FC → Sigmoid → P(accident)

Usage:
    classifier = TemporalClassifier()
    if classifier.is_available():
        classifier.add_frame(frame)  # call per sampled frame
        if classifier.has_enough_frames():
            score = classifier.classify_sequence()
            # score: 0.0 (normal) to 1.0 (accident sequence)
"""
import os
import torch
import torch.nn as nn
import numpy as np
from config import settings


# LSTM model architecture — must match training script exactly
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


class TemporalClassifier:
    """
    Wrapper for the temporal accident classifier.

    Maintains a sliding window of frame features and classifies
    the sequence when enough frames are accumulated.
    """

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = getattr(
                settings, 'TEMPORAL_CLASSIFIER_MODEL',
                os.path.join(settings.WEIGHTS_DIR, "temporal_classifier.pt")
            )

        self.model_path = model_path
        self.model = None
        self.feature_extractor = None
        self.available = False
        self.sequence_length = 16  # Default, overridden by checkpoint
        self.feature_dim = 2       # Default, overridden by checkpoint
        self.feature_buffer = []   # Sliding window of features
        self.device = torch.device("cpu")

        self._load_model()

    def _load_model(self):
        """Load the LSTM model from checkpoint."""
        if not os.path.exists(self.model_path):
            print(f"[TemporalClassifier] Model not found: {self.model_path}")
            return

        try:
            checkpoint = torch.load(
                self.model_path, map_location=self.device, weights_only=False
            )

            # Extract model config from checkpoint
            self.feature_dim = checkpoint.get("feature_dim", 2)
            self.sequence_length = checkpoint.get("sequence_length", 16)
            hidden_dim = checkpoint.get("hidden_dim", 128)
            num_layers = checkpoint.get("num_layers", 2)

            # Build model
            self.model = TemporalAccidentClassifier(
                input_dim=self.feature_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
            )

            # Load weights
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            self.available = True

            print(f"[TemporalClassifier] Model loaded: {self.model_path}")
            print(f"[TemporalClassifier] Sequence length: {self.sequence_length}, "
                  f"Feature dim: {self.feature_dim}")

        except Exception as e:
            print(f"[TemporalClassifier] Failed to load model: {e}")
            self.available = False

    def is_available(self):
        """Check if M3 model is loaded and ready."""
        return self.available

    def reset(self):
        """Clear the feature buffer (call at start of each video)."""
        self.feature_buffer = []

    def add_frame_features(self, features):
        """
        Add pre-extracted features for one frame to the sliding window.

        Args:
            features: numpy array of M1 probability features
        """
        if not self.available:
            return

        self.feature_buffer.append(features)

        # Keep only enough for one sequence + some overlap
        max_buffer = self.sequence_length * 2
        if len(self.feature_buffer) > max_buffer:
            self.feature_buffer = self.feature_buffer[-max_buffer:]

    def extract_and_add_frame(self, frame, frame_classifier):
        """
        Extract features from a frame using M1 and add to buffer.

        Args:
            frame: BGR numpy array
            frame_classifier: FrameClassifier instance (M1)
        """
        if not self.available or not frame_classifier.is_available():
            return

        # Use M1 to get probability features
        _, cls_conf = frame_classifier.classify(frame)

        # Build feature vector from the raw probabilities
        # We access the last prediction's probs directly
        try:
            results = frame_classifier.model.predict(
                frame, imgsz=224, verbose=False
            )
            if results and results[0].probs is not None:
                features = results[0].probs.data.cpu().numpy()
                self.add_frame_features(features)
                return
        except Exception:
            pass

        # Fallback: use cls_conf as a 2-dim feature
        features = np.array([1.0 - cls_conf, cls_conf])
        self.add_frame_features(features)

    def has_enough_frames(self):
        """Check if we have enough frames for a sequence classification."""
        return len(self.feature_buffer) >= self.sequence_length

    def classify_sequence(self):
        """
        Classify the most recent sequence of frames.

        Returns:
            float: accident probability (0.0 to 1.0), or 0.0 if unavailable
        """
        if not self.available or not self.has_enough_frames():
            return 0.0

        try:
            # Take the most recent sequence_length frames
            seq = self.feature_buffer[-self.sequence_length:]
            seq_array = np.array(seq, dtype=np.float32)

            # Ensure correct feature dimension
            if seq_array.shape[1] != self.feature_dim:
                # Pad or truncate feature dimension
                if seq_array.shape[1] < self.feature_dim:
                    pad = np.zeros(
                        (seq_array.shape[0], self.feature_dim - seq_array.shape[1]),
                        dtype=np.float32
                    )
                    seq_array = np.concatenate([seq_array, pad], axis=1)
                else:
                    seq_array = seq_array[:, :self.feature_dim]

            # Convert to tensor: (1, seq_len, feature_dim)
            tensor = torch.FloatTensor(seq_array).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(tensor)
                score = float(output.item())

            return round(score, 3)

        except Exception as e:
            print(f"[TemporalClassifier] Classification error: {e}")
            return 0.0

    def get_buffer_size(self):
        """Return current number of frames in the buffer."""
        return len(self.feature_buffer)

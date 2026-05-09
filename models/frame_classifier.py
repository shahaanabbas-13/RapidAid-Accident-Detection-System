"""
RapidAid — Frame Classifier Module

Uses a fine-tuned YOLOv8n-cls model to classify frames as
'accident' or 'no_accident' BEFORE running the expensive
geometric analysis pipeline.

This acts as a fast pre-filter that eliminates false positives
from normal traffic scenes.

Usage:
    classifier = FrameClassifier()
    if classifier.is_available():
        is_accident, confidence = classifier.classify(frame)
"""
import os
import cv2
from config import settings


class FrameClassifier:
    """
    Binary accident/no-accident classifier using fine-tuned YOLOv8n-cls.

    If the model file doesn't exist, this module gracefully degrades
    and the pipeline falls back to geometric-only analysis.
    """

    def __init__(self, model_path=None):
        """
        Initialize the frame classifier.

        Args:
            model_path: path to trained accident_classifier.pt
                        (defaults to settings.ACCIDENT_CLASSIFIER_MODEL)
        """
        self.model_path = model_path or settings.ACCIDENT_CLASSIFIER_MODEL
        self.model = None
        self.available = False
        self.class_names = {}

        if os.path.exists(self.model_path):
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.model_path, task="classify")
                self.class_names = self.model.names
                self.available = True
                self._accident_idx = self._find_accident_class()
                print(f"[FrameClassifier] Model loaded: {self.model_path}")
                print(f"[FrameClassifier] Classes: {self.class_names}")
                print(f"[FrameClassifier] Accident class index: {self._accident_idx}")
            except Exception as e:
                print(f"[FrameClassifier] Failed to load model: {e}")
                self.available = False
        else:
            print(f"[FrameClassifier] Model not found at: {self.model_path}")
            print(f"[FrameClassifier] Running without pre-filter (geometric-only mode)")

    def is_available(self):
        """Check if the classifier model is loaded and ready."""
        return self.available

    def classify(self, frame, confidence_threshold=0.35):
        """
        Classify a frame as accident or no_accident.

        Args:
            frame: BGR numpy array (or path to image)
            confidence_threshold: minimum confidence to classify as accident

        Returns:
            tuple: (is_accident: bool, confidence: float)
                   is_accident is True if frame classified as accident
                   confidence is the model's confidence (0.0 to 1.0)
        """
        if not self.available:
            # Fallback: assume potential accident (let geometric analysis decide)
            return True, 0.0

        # Load frame if path
        if isinstance(frame, str):
            frame = cv2.imread(frame)

        # Run classification
        results = self.model.predict(
            frame,
            imgsz=224,
            verbose=False,
            device="cpu",
        )

        if not results or len(results) == 0:
            return True, 0.0  # Fallback

        result = results[0]
        probs = result.probs

        if probs is None:
            return True, 0.0

        # Use cached accident class index
        accident_idx = getattr(self, '_accident_idx', None)
        if accident_idx is None:
            accident_idx = self._find_accident_class()

        if accident_idx is not None:
            accident_conf = float(probs.data[accident_idx])
            is_accident = accident_conf >= confidence_threshold
            return is_accident, accident_conf
        else:
            # Can't find accident class — assume accident (safe fallback)
            return True, 0.0

    def _find_accident_class(self):
        """
        Find the index of the 'accident' class in model's class names.

        Handles various label formats from different training setups:
          - 'Accident' / 'accident'
          - 'crash' / 'Crash'
          - 'positive' / 'Positive'
          - '1' (numeric label)
          - 'damaged' (from damage-style training)
        """
        positive_keywords = ["accident", "crash", "positive", "damaged", "collision"]
        negative_keywords = ["no_accident", "no accident", "non_accident",
                             "normal", "negative", "safe", "non accident"]

        # First pass: find explicit accident class
        for idx, name in self.class_names.items():
            name_lower = name.lower().strip()
            # Skip if it's a negative class
            if any(neg in name_lower for neg in negative_keywords):
                continue
            if any(pos in name_lower for pos in positive_keywords):
                return idx

        # Second pass: if binary classifier with just 2 classes,
        # find the non-negative class
        if len(self.class_names) == 2:
            for idx, name in self.class_names.items():
                name_lower = name.lower().strip()
                if any(neg in name_lower for neg in negative_keywords):
                    # Return the OTHER index
                    other_idx = 1 - idx
                    if other_idx in self.class_names:
                        return other_idx

        # Fallback: index 0 (alphabetical order: accident=0, normal=1)
        return 0 if len(self.class_names) > 0 else None

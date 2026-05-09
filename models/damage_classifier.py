"""
RapidAid — Damage Classifier Module (M2)

Uses a fine-tuned YOLOv8n-cls model to classify cropped vehicle
regions as 'damaged' or 'normal'.

This adds a critical signal to the pipeline: even if geometric
analysis doesn't detect overlap between vehicles, visible DAMAGE
on a vehicle (dents, crumpled metal, broken glass) is strong
evidence of an accident.

Usage:
    classifier = DamageClassifier()
    if classifier.is_available():
        score = classifier.classify_vehicle(frame, vehicle_bbox)
        # score: 0.0 (normal) to 1.0 (clearly damaged)
"""
import os
import cv2
import numpy as np
from config import settings


# Path to the damage classifier model
DAMAGE_MODEL_PATH = getattr(settings, 'DAMAGE_CLASSIFIER_MODEL',
                            os.path.join(settings.WEIGHTS_DIR, "damage_classifier.pt"))

# Minimum crop size (pixels) — don't classify tiny vehicles
MIN_CROP_SIZE = 32

# Padding around vehicle bbox for cropping (fraction of bbox size)
CROP_PADDING = 0.05


class DamageClassifier:
    """
    Classifies cropped vehicle images as damaged or normal.

    If the model file doesn't exist, this module gracefully degrades
    and returns 0.0 for all vehicles (no damage signal).
    """

    def __init__(self, model_path=None):
        """
        Initialize the damage classifier.

        Args:
            model_path: path to trained damage_classifier.pt
        """
        self.model_path = model_path or DAMAGE_MODEL_PATH
        self.model = None
        self.available = False
        self.class_names = {}
        self.damaged_idx = None

        if os.path.exists(self.model_path):
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.model_path, task="classify")
                self.class_names = self.model.names
                self.available = True
                self.damaged_idx = self._find_damaged_class()
                print(f"[DamageClassifier] Model loaded: {self.model_path}")
                print(f"[DamageClassifier] Classes: {self.class_names}")
                print(f"[DamageClassifier] Damaged class index: {self.damaged_idx}")
            except Exception as e:
                print(f"[DamageClassifier] Failed to load model: {e}")
                self.available = False
        else:
            print(f"[DamageClassifier] Model not found at: {self.model_path}")
            print(f"[DamageClassifier] Running without damage detection")

    def is_available(self):
        """Check if the damage classifier model is loaded and ready."""
        return self.available

    def classify_vehicle(self, frame, bbox):
        """
        Classify a single vehicle crop as damaged or normal.

        Args:
            frame: full BGR frame (numpy array)
            bbox: vehicle bounding box [x1, y1, x2, y2]

        Returns:
            float: damage confidence (0.0 = normal, 1.0 = clearly damaged)
        """
        if not self.available:
            return 0.0

        # Crop vehicle region with padding
        crop = self._crop_vehicle(frame, bbox)
        if crop is None:
            return 0.0

        # Run classification
        try:
            results = self.model.predict(
                crop,
                imgsz=224,
                verbose=False,
                device="cpu",
            )

            if not results or len(results) == 0:
                return 0.0

            probs = results[0].probs
            if probs is None:
                return 0.0

            if self.damaged_idx is not None:
                return float(probs.data[self.damaged_idx])
            else:
                return 0.0

        except Exception as e:
            return 0.0

    def classify_all_vehicles(self, frame, vehicles):
        """
        Classify all detected vehicles and return damage scores.

        Args:
            frame: full BGR frame (numpy array)
            vehicles: list of vehicle dicts with 'bbox' key

        Returns:
            dict: mapping vehicle bbox tuple → damage score (0-1)
        """
        if not self.available:
            return {}

        damage_scores = {}
        for vehicle in vehicles:
            bbox = vehicle["bbox"]
            score = self.classify_vehicle(frame, bbox)
            damage_scores[tuple(bbox)] = score

        return damage_scores

    def _crop_vehicle(self, frame, bbox):
        """
        Crop a vehicle region from the frame with padding.

        Returns None if the crop is too small.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox

        # Add padding
        bw = x2 - x1
        bh = y2 - y1
        pad_x = int(bw * CROP_PADDING)
        pad_y = int(bh * CROP_PADDING)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        # Check minimum size
        if (x2 - x1) < MIN_CROP_SIZE or (y2 - y1) < MIN_CROP_SIZE:
            return None

        return frame[y1:y2, x1:x2]

    def _find_damaged_class(self):
        """
        Find the index of the 'damaged' class in model's class names.

        Handles various label formats:
          - 'damaged' / 'Damaged'
          - 'crash' / 'Crash'
          - 'broken' / 'dent'
          - 'accident' / 'Accident'
          - 'positive'
        """
        positive_keywords = ["damage", "crash", "broken", "dent",
                             "accident", "positive", "wreck"]
        negative_keywords = ["normal", "undamaged", "intact", "clean",
                             "negative", "safe", "no_damage", "whole"]

        # First pass: find explicit damaged class
        for idx, name in self.class_names.items():
            name_lower = name.lower().strip()
            if any(neg in name_lower for neg in negative_keywords):
                continue
            if any(pos in name_lower for pos in positive_keywords):
                return idx

        # Second pass: binary classifier — find non-negative class
        if len(self.class_names) == 2:
            for idx, name in self.class_names.items():
                name_lower = name.lower().strip()
                if any(neg in name_lower for neg in negative_keywords):
                    other_idx = 1 - idx
                    if other_idx in self.class_names:
                        return other_idx
            # Fallback: alphabetical order → damaged=0, normal=1
            return 0

        return None

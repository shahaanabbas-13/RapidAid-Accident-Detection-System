"""
RapidAid — Collision Zone Detector Module (M4)

Uses a fine-tuned YOLOv8n object detector to directly detect
collision zones in frames. This is the PRIMARY accident detection
signal — it replaces geometric heuristic guesswork.

The model detects bounding boxes around accident areas (crumpled
vehicles, debris, impact points). Vehicles CLOSEST to the zone
center are flagged as "involved" — not ALL vehicles inside.

Usage:
    detector = CollisionDetector()
    if detector.is_available():
        zones = detector.detect(frame)
        involved = detector.get_involved_vehicles(zones, vehicles)
"""
import os
import math
import numpy as np
from config import settings
from utils.geometry import compute_iou, compute_overlap_ratio, compute_box_center


# Path to the collision detector model
COLLISION_MODEL_PATH = os.path.join(settings.WEIGHTS_DIR, "collision_detector.pt")

# Minimum confidence for collision zone detection
COLLISION_CONFIDENCE = 0.25

# Minimum overlap between vehicle bbox and collision zone to flag as involved
COLLISION_OVERLAP_THRESHOLD = 0.10

# Minimum collision zone area ratio (fraction of frame) — filter tiny false positives
MIN_ZONE_AREA_RATIO = 0.01

# Maximum number of collision zones to return (prevent over-detection)
MAX_ZONES = 5

# When collision zone covers >50% of the frame, it's too broad to use
# "center_inside" as a criterion — use proximity ranking instead
LARGE_ZONE_AREA_RATIO = 0.40

# Maximum number of vehicles to flag from a single large zone
# (prevents flagging 7 vehicles when only 2 are actually involved)
MAX_VEHICLES_PER_LARGE_ZONE = 3


class CollisionDetector:
    """
    Detects collision zones directly in frames using YOLOv8n.

    If the model file doesn't exist, this module gracefully degrades
    and the pipeline falls back to geometric analysis.
    """

    def __init__(self, model_path=None):
        self.model_path = model_path or COLLISION_MODEL_PATH
        self.model = None
        self.available = False
        self.class_names = {}

        if os.path.exists(self.model_path):
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.model_path)
                self.class_names = self.model.names
                self.available = True
                print(f"[CollisionDetector] Model loaded: {self.model_path}")
                print(f"[CollisionDetector] Classes: {self.class_names}")
            except Exception as e:
                print(f"[CollisionDetector] Failed to load: {e}")
                self.available = False
        else:
            print(f"[CollisionDetector] Model not found: {self.model_path}")
            print(f"[CollisionDetector] Using geometric fallback")

    def is_available(self):
        """Check if M4 model is loaded and ready."""
        return self.available

    def detect(self, frame, confidence=None):
        """
        Detect collision zones in a frame.

        Args:
            frame: BGR numpy array
            confidence: minimum confidence (uses COLLISION_CONFIDENCE if None)

        Returns:
            list of zone dicts, each containing:
                - 'bbox': [x1, y1, x2, y2]
                - 'confidence': detection confidence (0-1)
                - 'class': class name
                - 'area_ratio': zone area / frame area
        """
        if not self.available:
            return []

        conf = confidence or COLLISION_CONFIDENCE

        try:
            results = self.model(
                frame,
                verbose=False,
                conf=conf,
                device="cpu",
            )[0]

            if results.boxes is None or len(results.boxes) == 0:
                return []

            frame_h, frame_w = frame.shape[:2]
            frame_area = frame_h * frame_w

            zones = []
            for box in results.boxes:
                bbox = [int(c) for c in box.xyxy[0].tolist()]
                conf_val = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.class_names.get(cls_id, f"class_{cls_id}")

                # Compute area ratio
                zone_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                area_ratio = zone_area / frame_area if frame_area > 0 else 0

                # Filter tiny zones (noise)
                if area_ratio < MIN_ZONE_AREA_RATIO:
                    continue

                zones.append({
                    "bbox": bbox,
                    "confidence": round(conf_val, 3),
                    "class": cls_name,
                    "area_ratio": round(area_ratio, 4),
                    "center": compute_box_center(bbox),
                    "is_large": area_ratio > LARGE_ZONE_AREA_RATIO,
                })

            # Sort by confidence and limit
            zones.sort(key=lambda z: z["confidence"], reverse=True)
            return zones[:MAX_ZONES]

        except Exception as e:
            print(f"[CollisionDetector] Detection error: {e}")
            return []

    def _distance_to_zone_center(self, vehicle_bbox, zone_center):
        """Compute distance from vehicle center to zone center."""
        v_center = compute_box_center(vehicle_bbox)
        dx = v_center[0] - zone_center[0]
        dy = v_center[1] - zone_center[1]
        return math.sqrt(dx * dx + dy * dy)

    def get_involved_vehicles(self, zones, vehicles, overlap_threshold=None):
        """
        Find vehicles that overlap with detected collision zones.

        When the collision zone is LARGE (>40% of frame), uses proximity-
        based ranking instead of simple overlap to avoid flagging every
        vehicle in the scene. Only the closest vehicles to the zone
        center are selected (max 3 per large zone).

        Args:
            zones: list of zone dicts from detect()
            vehicles: list of vehicle dicts from VehicleDetector
            overlap_threshold: minimum IoU/overlap to flag as involved

        Returns:
            list of vehicle dicts with added 'crash_score' and 'collision_zone'
        """
        if not zones or not vehicles:
            return []

        threshold = overlap_threshold or COLLISION_OVERLAP_THRESHOLD
        involved = []
        involved_bbox_keys = set()

        for zone in zones:
            z_bbox = zone["bbox"]
            z_center = zone["center"]
            is_large_zone = zone.get("is_large", False)

            # For LARGE zones: rank all overlapping vehicles by proximity
            # to zone center, then pick the top N closest
            if is_large_zone:
                candidates = []
                for vehicle in vehicles:
                    v_bbox = vehicle["bbox"]
                    v_key = tuple(v_bbox)
                    if v_key in involved_bbox_keys:
                        continue

                    iou = compute_iou(v_bbox, z_bbox)
                    overlap = compute_overlap_ratio(v_bbox, z_bbox)
                    center = vehicle.get("center", compute_box_center(v_bbox))
                    center_inside = (z_bbox[0] <= center[0] <= z_bbox[2] and
                                     z_bbox[1] <= center[1] <= z_bbox[3])

                    if iou >= threshold or overlap >= threshold or center_inside:
                        dist = self._distance_to_zone_center(v_bbox, z_center)
                        # Also factor in damage score if available
                        dmg = vehicle.get("damage_score", 0)
                        # Lower distance = closer to crash point = more likely involved
                        # Higher damage = more likely involved
                        # Combined ranking: distance penalty - damage bonus
                        rank_score = dist - (dmg * 200)
                        candidates.append((vehicle, iou, overlap, dist, rank_score))

                # Sort by rank_score (lower = more likely involved)
                candidates.sort(key=lambda c: c[4])

                # Take top N closest vehicles
                max_vehs = MAX_VEHICLES_PER_LARGE_ZONE
                for vehicle, iou, overlap, dist, _ in candidates[:max_vehs]:
                    v_copy = vehicle.copy()
                    score = max(iou, overlap) * 0.6 + zone["confidence"] * 0.4
                    v_copy["crash_score"] = round(
                        min(settings.MAX_CRASH_SCORE, score), 3
                    )
                    v_copy["collision_zone"] = zone["bbox"]
                    involved.append(v_copy)
                    involved_bbox_keys.add(tuple(vehicle["bbox"]))

            else:
                # For SMALL zones: original logic — overlap or center inside
                for vehicle in vehicles:
                    v_bbox = vehicle["bbox"]
                    v_key = tuple(v_bbox)
                    if v_key in involved_bbox_keys:
                        continue

                    iou = compute_iou(v_bbox, z_bbox)
                    overlap = compute_overlap_ratio(v_bbox, z_bbox)
                    center = vehicle.get("center", compute_box_center(v_bbox))
                    center_inside = (z_bbox[0] <= center[0] <= z_bbox[2] and
                                     z_bbox[1] <= center[1] <= z_bbox[3])

                    if iou >= threshold or overlap >= threshold or center_inside:
                        v_copy = vehicle.copy()
                        score = max(iou, overlap) * 0.6 + zone["confidence"] * 0.4
                        v_copy["crash_score"] = round(
                            min(settings.MAX_CRASH_SCORE, score), 3
                        )
                        v_copy["collision_zone"] = zone["bbox"]
                        involved.append(v_copy)
                        involved_bbox_keys.add(v_key)

        return involved

    def get_best_zone(self, zones):
        """Return the highest-confidence collision zone, or None."""
        if not zones:
            return None
        return zones[0]  # Already sorted by confidence

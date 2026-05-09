"""
RapidAid — Vehicle Detection Engine

Uses YOLOv8s-seg to detect all vehicles in a frame with segmentation masks.
Filters out background/distant vehicles using area and position heuristics.
"""
import numpy as np
from ultralytics import YOLO
from utils.geometry import compute_iou
from config import settings
from config.vehicle_classes import (
    VEHICLE_CLASS_IDS, get_rapidaid_label, get_display_name,
    VEHICLE_SIZE_RANGE, is_vehicle_class
)
from utils.geometry import compute_box_area, compute_box_center


class VehicleDetector:
    """
    Detects all vehicles in a frame using YOLOv8 segmentation model.
    Applies filtering to remove background/distant vehicles.
    """

    def __init__(self, model_path=None):
        """
        Initialize the vehicle detector.

        Args:
            model_path: path to YOLOv8-seg weights (uses default if None)
        """
        path = model_path or settings.VEHICLE_MODEL
        print(f"[VehicleDetector] Loading model: {path}")
        self.model = YOLO(path)
        print("[VehicleDetector] Model loaded successfully.")

    def detect(self, frame, filter_background=True):
        """
        Detect all vehicles in a frame.

        Args:
            frame: numpy array (BGR image) or path to image
            filter_background: if True, remove distant/background vehicles

        Returns:
            list of vehicle dicts, each containing:
                - 'type': RapidAid vehicle label
                - 'display_name': human-readable name
                - 'confidence': detection confidence (0-100)
                - 'bbox': [x1, y1, x2, y2]
                - 'polygon': segmentation polygon points
                - 'area_ratio': bbox area / frame area
                - 'center': (cx, cy) center point
                - 'coco_class_id': original COCO class ID
        """
        # Run inference
        results = self.model(frame, verbose=False)[0]

        # Get frame dimensions
        if isinstance(frame, str):
            import cv2
            img = cv2.imread(frame)
            frame_h, frame_w = img.shape[:2]
        else:
            frame_h, frame_w = frame.shape[:2]

        frame_area = frame_h * frame_w

        vehicles = []

        if results.masks is None:
            return vehicles

        for i, box in enumerate(results.boxes):
            coco_id = int(box.cls[0])
            conf = float(box.conf[0])

            # Skip non-vehicle classes
            if not is_vehicle_class(coco_id):
                continue

            # Skip low confidence detections
            if conf < settings.VEHICLE_CONFIDENCE_THRESHOLD:
                continue

            bbox = [int(c) for c in box.xyxy[0].tolist()]
            area = compute_box_area(bbox)
            area_ratio = area / frame_area

            vehicle_type = get_rapidaid_label(coco_id)

            # Get segmentation polygon
            polygon = results.masks.xy[i] if i < len(results.masks.xy) else np.array([])

            vehicle = {
                "type": vehicle_type,
                "display_name": get_display_name(vehicle_type),
                "confidence": round(conf * 100, 1),
                "bbox": bbox,
                "polygon": polygon,
                "area_ratio": area_ratio,
                "center": compute_box_center(bbox),
                "coco_class_id": coco_id,
            }

            vehicles.append(vehicle)

        # Cross-class NMS: remove duplicate detections of the same object
        # (e.g., YOLO detecting a motorcycle as both 'Motorcycle' and 'Car')
        vehicles = self._cross_class_nms(vehicles)

        # Filter background vehicles
        if filter_background:
            vehicles = self._filter_background(vehicles, frame_h, frame_w, frame_area)

        return vehicles

    def _filter_background(self, vehicles, frame_h, frame_w, frame_area):
        """
        Remove vehicles that are likely background/parked/distant.

        Filtering criteria:
        1. Too small (distant)
        2. Too large (partially visible, too close)
        3. In the far background (top of frame)
        """
        filtered = []

        for v in vehicles:
            area_ratio = v["area_ratio"]
            vehicle_type = v["type"]
            center = v["center"]

            # Check minimum area
            min_area = VEHICLE_SIZE_RANGE.get(vehicle_type, {}).get("min", settings.MIN_VEHICLE_AREA_RATIO)
            if area_ratio < min_area:
                continue

            # Check maximum area
            max_area = VEHICLE_SIZE_RANGE.get(vehicle_type, {}).get("max", settings.MAX_VEHICLE_AREA_RATIO)
            if area_ratio > max_area:
                continue

            # Check if vehicle is in far background (top portion of frame)
            # Only filter if the vehicle is also very small
            y_ratio = center[1] / frame_h
            if y_ratio < settings.BACKGROUND_Y_THRESHOLD_RATIO and area_ratio < 0.02:
                continue

            filtered.append(v)

        return filtered

    def _cross_class_nms(self, vehicles, iou_threshold=0.50):
        """
        Cross-class Non-Maximum Suppression.

        Removes duplicate detections of the same physical object across
        different YOLO classes. Keeps the detection with highest confidence.

        Example: If YOLO detects the same motorcycle as both 'Motorcycle'
        (35%) and 'Car' (28%), this removes the 'Car' duplicate.
        """
        if len(vehicles) <= 1:
            return vehicles

        # Sort by confidence (highest first)
        vehicles = sorted(vehicles,
                          key=lambda v: v["confidence"], reverse=True)

        keep = []
        for v in vehicles:
            is_duplicate = False
            for kept in keep:
                iou = compute_iou(v["bbox"], kept["bbox"])
                if iou > iou_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                keep.append(v)

        return keep

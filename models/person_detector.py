"""
RapidAid — Person Detection Engine

Uses YOLOv8s-pose to detect all persons in a frame with skeleton keypoints.
Provides raw detections for the Victim Classifier to process.
"""
import numpy as np
from ultralytics import YOLO
from config import settings
from utils.geometry import compute_box_area, compute_box_center, compute_aspect_ratio


# YOLOv8 Pose keypoint indices
KEYPOINT_NAMES = {
    0: "nose",
    1: "left_eye",     2: "right_eye",
    3: "left_ear",     4: "right_ear",
    5: "left_shoulder", 6: "right_shoulder",
    7: "left_elbow",   8: "right_elbow",
    9: "left_wrist",   10: "right_wrist",
    11: "left_hip",    12: "right_hip",
    13: "left_knee",   14: "right_knee",
    15: "left_ankle",  16: "right_ankle",
}


class PersonDetector:
    """
    Detects all persons in a frame using YOLOv8-pose model.
    Returns raw person detections with keypoint data.
    """

    def __init__(self, model_path=None):
        """
        Initialize the person detector.

        Args:
            model_path: path to YOLOv8-pose weights (uses default if None)
        """
        path = model_path or settings.POSE_MODEL
        print(f"[PersonDetector] Loading model: {path}")
        self.model = YOLO(path)
        print("[PersonDetector] Model loaded successfully.")

    def detect(self, frame):
        """
        Detect all persons in a frame with keypoints.

        Args:
            frame: numpy array (BGR image) or path to image

        Returns:
            list of person dicts, each containing:
                - 'confidence': detection confidence (0-100)
                - 'bbox': [x1, y1, x2, y2]
                - 'center': (cx, cy)
                - 'aspect_ratio': width/height
                - 'keypoints': dict of {name: (x, y, conf)} for each keypoint
                - 'bbox_width': width of bbox
                - 'bbox_height': height of bbox
        """
        results = self.model(frame, verbose=False)[0]

        if isinstance(frame, str):
            import cv2
            img = cv2.imread(frame)
            frame_h, frame_w = img.shape[:2]
        else:
            frame_h, frame_w = frame.shape[:2]

        persons = []

        for i, box in enumerate(results.boxes):
            coco_id = int(box.cls[0])
            conf = float(box.conf[0])

            # Only process person class (COCO id = 0)
            if coco_id != 0:
                continue

            # Apply confidence threshold
            if conf < settings.PERSON_CONFIDENCE_THRESHOLD:
                continue

            bbox = [int(c) for c in box.xyxy[0].tolist()]
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]

            # Extract keypoints if available
            keypoints = {}
            if results.keypoints is not None and i < len(results.keypoints):
                kp_data = results.keypoints[i].data[0]  # shape: (17, 3) -> x, y, conf
                for kp_idx, kp_name in KEYPOINT_NAMES.items():
                    if kp_idx < len(kp_data):
                        kx, ky, kconf = kp_data[kp_idx].tolist()
                        keypoints[kp_name] = (kx, ky, kconf)

            person = {
                "confidence": round(conf * 100, 1),
                "bbox": bbox,
                "center": compute_box_center(bbox),
                "aspect_ratio": compute_aspect_ratio(bbox),
                "keypoints": keypoints,
                "bbox_width": w,
                "bbox_height": h,
            }

            persons.append(person)

        return persons

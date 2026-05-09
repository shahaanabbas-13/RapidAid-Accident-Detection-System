"""
RapidAid — Vehicle-Pedestrian Collision Detector

Detects collisions between vehicles and pedestrians using:
1. Person-vehicle bbox overlap / proximity
2. Person posture analysis (non-standing near vehicle = impact)
3. Person position relative to vehicle (front/side = impact zone)
4. Size ratio analysis (large vehicle near small person = dangerous)

This module runs INDEPENDENTLY of the frame classifier — it does NOT
require a trained accident_classifier.pt to detect car-hitting-person
scenarios. It uses purely geometric + postural signals.

IMPORTANT: This detector is CONSERVATIVE — it requires strong evidence
of actual contact (high overlap or person clearly not standing while
in contact with vehicle). It does NOT flag standing bystanders near
vehicles, even if they are close.
"""
import math
from config import settings
from utils.geometry import (
    compute_overlap_ratio, compute_edge_distance, compute_box_area,
    compute_box_center, compute_diagonal
)


# Thresholds for vehicle-pedestrian collision scoring
VP_OVERLAP_WEIGHT = 0.40
VP_PROXIMITY_WEIGHT = 0.15
VP_POSTURE_WEIGHT = 0.30
VP_IMPACT_ZONE_WEIGHT = 0.05
VP_SIZE_RATIO_WEIGHT = 0.10

# Minimum score to flag a vehicle-pedestrian collision
# Raised from 0.30 → 0.40 to reduce false positives on standing bystanders
VP_COLLISION_THRESHOLD = 0.40

# Maximum edge distance (pixels) to consider a person "near" a vehicle
# Reduced from 120 → 60 to avoid flagging distant bystanders
VP_MAX_EDGE_DISTANCE = 60

# Overlap ratio above which a collision is almost certain
VP_HIGH_OVERLAP = 0.20

# Minimum overlap required for ANY detection (gate)
# Person must have at least some physical contact with vehicle
VP_MIN_OVERLAP_GATE = 0.03


class VehiclePedestrianDetector:
    """
    Detects vehicle-pedestrian collisions using geometric and postural analysis.

    Unlike the AccidentClassifier (which scores vehicle-vehicle pairs), this
    module specifically handles the car-hits-person scenario that traditional
    vehicle overlap detection cannot catch.

    CONSERVATIVE DESIGN: requires strong overlap evidence + non-standing
    posture to reduce false positives on standing bystanders near vehicles.
    """

    def detect(self, vehicles, persons, frame_shape):
        """
        Detect potential vehicle-pedestrian collisions.

        Args:
            vehicles: list of vehicle dicts from VehicleDetector
            persons: list of person dicts from PersonDetector
            frame_shape: (height, width, channels) of the frame

        Returns:
            list of collision dicts, each containing:
                - 'vehicle': the involved vehicle dict (with crash_score added)
                - 'person': the victim dict (with status set)
                - 'score': collision confidence score (0-1)
        """
        if not vehicles or not persons:
            return []

        frame_h, frame_w = frame_shape[:2]
        frame_diag = math.hypot(frame_w, frame_h)

        collisions = []

        for person in persons:
            # CRITICAL: Skip clearly standing persons immediately.
            # Standing people near vehicles are bystanders/helpers, not victims.
            if self._is_clearly_standing(person):
                continue

            best_collision = None
            best_score = 0.0

            for vehicle in vehicles:
                score, signals = self._score_collision(
                    vehicle, person, frame_h, frame_w, frame_diag
                )

                if score >= VP_COLLISION_THRESHOLD and score > best_score:
                    best_score = score
                    best_collision = {
                        "vehicle": vehicle,
                        "person": person,
                        "score": score,
                        "signals": signals,
                    }

            if best_collision is not None:
                collisions.append(best_collision)

        return collisions

    def _is_clearly_standing(self, person):
        """
        Check if a person is clearly standing upright — if so, they should
        NOT be flagged as a collision victim regardless of proximity.

        Uses keypoints first (most reliable), then bbox aspect ratio.
        """
        # Keypoint check
        kps = person.get("keypoints", {})
        min_c = settings.KEYPOINT_CONFIDENCE_THRESHOLD
        standing_kp = self._check_standing_kps(kps, min_c)
        if standing_kp is True:
            return True

        # Bbox check: person with h > 1.5 * w is very likely standing
        h = person.get("bbox_height", 0)
        w = person.get("bbox_width", 0)
        if h > 0 and w > 0 and h > 1.5 * w:
            return True

        return False

    def _score_collision(self, vehicle, person, frame_h, frame_w, frame_diag):
        """
        Score the likelihood of a vehicle-pedestrian collision.

        Returns:
            tuple: (total_score, signal_dict)
        """
        signals = {}

        v_bbox = vehicle["bbox"]
        p_bbox = person["bbox"]

        # === Signal 1: Person-Vehicle bbox overlap ===
        overlap = compute_overlap_ratio(p_bbox, v_bbox)
        if overlap > 0:
            overlap_score = min(1.0, overlap / VP_HIGH_OVERLAP)
        else:
            overlap_score = 0.0
        signals["overlap"] = overlap_score

        # === Signal 2: Edge proximity ===
        edge_dist = compute_edge_distance(p_bbox, v_bbox)
        if edge_dist <= 0:
            prox_score = 1.0
        elif edge_dist < VP_MAX_EDGE_DISTANCE:
            prox_score = 1.0 - (edge_dist / VP_MAX_EDGE_DISTANCE)
        else:
            prox_score = 0.0
        signals["proximity"] = prox_score

        # === Signal 3: Person posture ===
        posture_score = self._compute_posture_score(person)
        signals["posture"] = posture_score

        # === Signal 4: Impact zone ===
        impact_score = self._compute_impact_zone_score(vehicle, person)
        signals["impact_zone"] = impact_score

        # === Signal 5: Size ratio ===
        v_area = compute_box_area(v_bbox)
        p_area = compute_box_area(p_bbox)
        if p_area > 0:
            size_ratio = v_area / p_area
            size_score = min(1.0, size_ratio / 5.0)
        else:
            size_score = 0.0
        signals["size_ratio"] = size_score

        # === Weighted total ===
        total = (
            VP_OVERLAP_WEIGHT * overlap_score +
            VP_PROXIMITY_WEIGHT * prox_score +
            VP_POSTURE_WEIGHT * posture_score +
            VP_IMPACT_ZONE_WEIGHT * impact_score +
            VP_SIZE_RATIO_WEIGHT * size_score
        )

        # === Bonus: High overlap is almost certain collision ===
        if overlap > VP_HIGH_OVERLAP:
            total = max(total, 0.65)

        # === STRICT GATE ===
        # Must have BOTH:
        #   1. Physical contact evidence (overlap > 3% OR touching/overlapping edges)
        #   2. Non-standing posture (posture_score > 0.3)
        # Without both, there's insufficient evidence of an actual collision.
        has_contact = overlap >= VP_MIN_OVERLAP_GATE or edge_dist <= 0
        has_posture = posture_score > 0.3

        if not has_contact:
            total = 0.0
        elif not has_posture and overlap < VP_HIGH_OVERLAP:
            # Standing person with moderate overlap — could be just standing
            # next to vehicle. Only flag if overlap is very high (trapped).
            total = 0.0

        return total, signals

    def _compute_posture_score(self, person):
        """
        Score how much the person's posture suggests impact.

        0.0 = clearly standing (bystander)
        0.5 = ambiguous posture
        1.0 = clearly not standing (fallen/crouched/hit)
        """
        # Check keypoints first (most reliable)
        kps = person.get("keypoints", {})
        min_c = settings.KEYPOINT_CONFIDENCE_THRESHOLD

        standing_kp = self._check_standing_kps(kps, min_c)
        if standing_kp is True:
            return 0.0
        elif standing_kp is False:
            return 0.8

        # Fallback: bbox aspect ratio
        h = person.get("bbox_height", 0)
        w = person.get("bbox_width", 0)

        if h <= 0 or w <= 0:
            return 0.3

        ratio = h / w  # height / width

        if ratio > 1.8:
            return 0.0  # Very tall and narrow — clearly standing
        elif ratio > 1.4:
            return 0.15  # Likely standing
        elif ratio > 1.0:
            return 0.35  # Ambiguous — could be crouching
        elif ratio > 0.7:
            return 0.6   # Likely fallen or crouching
        else:
            return 0.9   # Wide and short — clearly fallen

    def _check_standing_kps(self, kps, min_c):
        """Check standing posture from shoulder/hip keypoints."""
        shoulder = self._avg_kp(kps.get("left_shoulder"), kps.get("right_shoulder"), min_c)
        hip = self._avg_kp(kps.get("left_hip"), kps.get("right_hip"), min_c)
        if shoulder is None or hip is None:
            return None
        dx = hip[0] - shoulder[0]
        dy = hip[1] - shoulder[1]
        angle = abs(math.degrees(math.atan2(dx, dy)))
        return angle < 35 and shoulder[1] < hip[1]

    def _avg_kp(self, kp1, kp2, min_c):
        """Average two keypoints if confidence is sufficient."""
        valid = [k for k in [kp1, kp2] if k and len(k) >= 3 and k[2] >= min_c]
        if not valid:
            return None
        return (sum(k[0] for k in valid) / len(valid),
                sum(k[1] for k in valid) / len(valid))

    def _compute_impact_zone_score(self, vehicle, person):
        """
        Score whether the person is in the vehicle's likely impact zone.
        """
        v_center = compute_box_center(vehicle["bbox"])
        p_center = compute_box_center(person["bbox"])

        v_bbox = vehicle["bbox"]
        v_w = v_bbox[2] - v_bbox[0]
        v_h = v_bbox[3] - v_bbox[1]

        dx = p_center[0] - v_center[0]
        dy = p_center[1] - v_center[1]

        if v_w > 0 and v_h > 0:
            norm_dx = abs(dx) / v_w
            norm_dy = abs(dy) / v_h
        else:
            return 0.5

        if norm_dx < 0.7 and norm_dy < 0.7:
            return 1.0
        elif norm_dx < 1.2 or norm_dy < 1.2:
            return 0.7
        elif norm_dx < 2.0 or norm_dy < 2.0:
            return 0.4
        else:
            return 0.1

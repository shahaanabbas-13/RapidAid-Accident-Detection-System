"""
RapidAid — Victim Classifier Module

Classifies detected persons as victims or bystanders using:
1. Skeleton keypoint analysis (standing vs lying)
2. Spatial relationship to involved vehicles
3. Position relative to the accident zone
4. Standalone victim detection (persons lying without vehicle involvement)
5. Lying-on-road detection (persons clearly on the ground near any vehicle)
"""
import math
from config import settings
from utils.geometry import (
    compute_overlap_ratio, compute_edge_distance,
    compute_diagonal, point_in_box
)


class VictimClassifier:
    """Classifies each detected person as victim or bystander."""

    def classify(self, persons, involved_vehicles, accident_zone):
        """
        Classify all detected persons.
        Returns list of victim dicts with 'status' field.
        """
        victims = []
        for person in persons:
            status = self._classify_person(person, involved_vehicles, accident_zone)
            if status is not None:
                victim = {
                    "confidence": person["confidence"],
                    "bbox": person["bbox"],
                    "center": person["center"],
                    "status": status,
                }
                victims.append(victim)
        return victims

    def detect_standalone_victims(self, persons):
        """
        Detect victims lying on the ground even without vehicle involvement.
        Used for hit-and-run or aftermath scenes.

        Uses a relaxed check: if person is NOT standing and NOT clearly
        upright, treat them as a potential victim. From elevated CCTV
        angles, lying persons can have bbox height > width due to
        perspective compression.

        Returns:
            list of victim dicts with status 'fallen'
        """
        victims = []
        for person in persons:
            # Skip anyone who is clearly standing
            if self._is_standing(person):
                continue

            # Primary: person is clearly lying down
            if self._is_lying(person):
                victim = {
                    "confidence": person["confidence"],
                    "bbox": person["bbox"],
                    "center": person["center"],
                    "status": "fallen",
                }
                victims.append(victim)
            else:
                # Relaxed: person is NOT standing but not clearly lying
                # From CCTV angles, lying persons often have height > width
                # Use bbox aspect ratio as hint: ratio > 0.40 is ambiguous
                w = person.get("bbox_width", 0)
                h = person.get("bbox_height", 0)
                if h > 0 and w / h > 0.40:
                    victim = {
                        "confidence": person["confidence"],
                        "bbox": person["bbox"],
                        "center": person["center"],
                        "status": "fallen",
                    }
                    victims.append(victim)

        return victims

    def _classify_person(self, person, involved_vehicles, zone):
        """
        Returns status string or None if bystander.

        Classification rules (in priority order):
        1. Standing → bystander (skip)
        2. Overlapping involved vehicle → trapped
        3. Lying inside accident zone → fallen
        4. Lying near accident zone → nearby
        5. Lying near any involved vehicle (even outside zone) → fallen
        6. Non-lying but very close to vehicle → nearby
        """
        # Rule 1: Standing detection — must be checked thoroughly
        if self._is_standing(person):
            return None  # Bystander / helper

        # Rule 2: Trapped — person overlaps an involved vehicle
        for vehicle in involved_vehicles:
            overlap = compute_overlap_ratio(person["bbox"], vehicle["bbox"])
            if overlap > settings.TRAPPED_OVERLAP_THRESHOLD:
                return "trapped"

        # Rule 3: Lying person inside accident zone → fallen victim
        if zone is not None and point_in_box(person["center"], zone):
            if self._is_lying(person):
                return "fallen"

        # Rule 4: Lying person near accident zone → nearby victim
        if zone is not None:
            zone_diag = compute_diagonal(zone)
            edge_dist = compute_edge_distance(person["bbox"], zone)
            if zone_diag > 0 and (edge_dist / zone_diag) < 0.3:
                if self._is_lying(person):
                    return "nearby"

        # Rule 5 (NEW): Lying person near ANY involved vehicle → fallen
        # This catches victims who are on the ground near vehicles but
        # outside the computed accident zone (which may be too tight).
        # This fixes cases where persons are lying right beside the car
        # but the zone is centered elsewhere.
        if self._is_lying(person):
            for vehicle in involved_vehicles:
                edge_dist = compute_edge_distance(person["bbox"], vehicle["bbox"])
                v_diag = compute_diagonal(vehicle["bbox"])
                # Within 1.5x the vehicle's diagonal — close enough
                if v_diag > 0 and edge_dist < v_diag * 1.5:
                    return "fallen"

        # Rule 6: Non-lying person very close to involved vehicle
        # (driver who exited, or person hit but still upright)
        # STRICT: only flag if very close (edge_dist < 20) to avoid
        # flagging standing bystanders
        for vehicle in involved_vehicles:
            edge_dist = compute_edge_distance(person["bbox"], vehicle["bbox"])
            if edge_dist < 20:  # Very close — likely involved
                return "nearby"

        return None

    def _is_standing(self, person):
        """
        Check if person is standing using keypoints + bbox fallback.
        More conservative: use keypoints first, bbox ratio as fallback.
        """
        # Check keypoints FIRST — more reliable than bbox
        kps = person.get("keypoints", {})
        min_c = settings.KEYPOINT_CONFIDENCE_THRESHOLD
        result = self._check_standing_kps(kps, min_c)
        if result is not None:
            return result

        # Fallback: use bbox aspect ratio only when no keypoints available
        h = person.get("bbox_height", 0)
        w = person.get("bbox_width", 0)

        # Only classify as standing if clearly upright (tall and narrow)
        if w > 0 and h > settings.STANDING_RATIO * w:
            return True

        return False

    def _is_lying(self, person):
        """
        Check if person is lying using keypoints + bbox fallback.
        Uses a relaxed ratio (0.55) compared to the strict LYING_RATIO
        to better handle CCTV camera angles where perspective makes
        lying persons appear taller than wide.
        """
        # Check keypoints first — most reliable
        kps = person.get("keypoints", {})
        min_c = settings.KEYPOINT_CONFIDENCE_THRESHOLD
        result = self._check_lying_kps(kps, min_c)
        if result is not None:
            return result

        # Fallback: use relaxed ratio for CCTV angles
        # Standard LYING_RATIO (0.75) is too strict for elevated cameras
        h = person.get("bbox_height", 0)
        w = person.get("bbox_width", 0)
        if h > 0:
            return (w / h) > 0.55

        return False

    def _check_standing_kps(self, kps, min_c):
        """Use shoulder/hip keypoints to detect standing posture."""
        shoulder = self._avg_kp(kps.get("left_shoulder"), kps.get("right_shoulder"), min_c)
        hip = self._avg_kp(kps.get("left_hip"), kps.get("right_hip"), min_c)
        if shoulder is None or hip is None:
            return None
        dx = hip[0] - shoulder[0]
        dy = hip[1] - shoulder[1]
        angle = abs(math.degrees(math.atan2(dx, dy)))
        # Standing: torso is roughly vertical (angle < 35° from vertical)
        # AND shoulders are above hips in the image
        return angle < 35 and shoulder[1] < hip[1]

    def _check_lying_kps(self, kps, min_c):
        """Use shoulder/hip keypoints to detect lying posture."""
        shoulder = self._avg_kp(kps.get("left_shoulder"), kps.get("right_shoulder"), min_c)
        hip = self._avg_kp(kps.get("left_hip"), kps.get("right_hip"), min_c)
        if shoulder is None or hip is None:
            return None
        dx = hip[0] - shoulder[0]
        dy = hip[1] - shoulder[1]
        if abs(dx) < 1:
            return False
        angle = abs(math.degrees(math.atan2(dy, dx)))
        thresh = settings.LYING_ANGLE_THRESHOLD
        return angle < thresh or abs(180 - angle) < thresh

    def _avg_kp(self, kp1, kp2, min_c):
        """Average two keypoints if confidence is sufficient."""
        valid = [k for k in [kp1, kp2] if k and len(k) >= 3 and k[2] >= min_c]
        if not valid:
            return None
        return (sum(k[0] for k in valid) / len(valid),
                sum(k[1] for k in valid) / len(valid))

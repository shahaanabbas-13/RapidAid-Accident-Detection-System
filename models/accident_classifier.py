"""
RapidAid — Accident Classifier Module

Determines which vehicles are involved in an accident using a multi-signal
crash scoring system. Each pair of vehicles is scored on multiple indicators.

Supports:
  - Multi-vehicle collision (pairwise scoring)
  - Single-vehicle crash (deformation + classifier confidence)
  - Classifier-geometric fusion (adaptive thresholds)
  - Transitive closure (A→B→C all involved)
  - Cluster penalty (crowded traffic de-prioritization)
"""
import math
import numpy as np
from config import settings
from config.vehicle_classes import NORMAL_ASPECT_RATIOS
from utils.geometry import (
    compute_iou, compute_edge_distance, compute_diagonal,
    check_pixel_collision, compute_relative_angle, compute_aspect_ratio,
    compute_box_area, compute_box_center
)


class AccidentClassifier:
    """
    Classifies which vehicles in a scene are involved in an accident.
    Uses a multi-signal weighted scoring system per vehicle pair.
    Supports adaptive thresholds based on classifier confidence.
    """

    def __init__(self):
        self.weights = settings.CRASH_SCORE_WEIGHTS
        self.base_threshold = settings.CRASH_SCORE_THRESHOLD

    def classify(self, vehicles, frame_shape, classifier_confidence=None):
        """
        Analyze all vehicle pairs and determine which are involved in accident.

        Args:
            vehicles: list of vehicle dicts from VehicleDetector
            frame_shape: (height, width, channels) of the frame
            classifier_confidence: float (0-1) from FrameClassifier, or None.
                When provided, modulates the crash score threshold:
                - High confidence → lower threshold (classifier agrees)
                - Low confidence → higher threshold (need more geometric proof)

        Returns:
            list of vehicle dicts that are involved in the accident,
            each with an added 'crash_score' field
        """
        # Compute adaptive threshold based on classifier confidence
        threshold = self._compute_adaptive_threshold(classifier_confidence)

        if len(vehicles) < 2:
            # Need at least 2 vehicles for a collision
            # Single deformed vehicle could indicate hit-and-run
            return self._check_single_vehicle_accident(
                vehicles, frame_shape, classifier_confidence
            )

        # === CLUSTER PENALTY: detect crowded traffic ===
        # Count how many vehicles are in a tight proximity cluster.
        # If many vehicles are all close together, it's likely normal
        # congestion — not a crash. Apply penalty to crash scores.
        cluster_penalties = self._compute_cluster_penalties(vehicles)

        involved_indices = set()
        crash_scores = {}
        pair_scores = {}  # Store (i,j) → score for transitive closure

        # Check every pair
        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                # Both vehicles must have minimum detection confidence
                conf_a = vehicles[i].get("confidence", 0)
                conf_b = vehicles[j].get("confidence", 0)
                if min(conf_a, conf_b) < settings.MIN_PAIR_CONFIDENCE:
                    continue

                score, signals = self._compute_crash_score(
                    vehicles[i], vehicles[j], frame_shape
                )

                # Size/confidence multiplier — prevent small, low-confidence
                # duplicate-like detections from dominating the scoring.
                avg_conf = (conf_a + conf_b) / 2.0
                avg_area = (vehicles[i].get("area_ratio", 0) +
                            vehicles[j].get("area_ratio", 0)) / 2.0
                size_factor = min(1.0, avg_area / 0.02)
                conf_factor = min(1.0, avg_conf / 50.0)
                score = score * (0.5 + 0.25 * size_factor + 0.25 * conf_factor)

                # Apply cluster penalty — vehicles in large tight clusters
                # get their scores reduced (normal traffic, not crash)
                penalty_i = cluster_penalties.get(i, 1.0)
                penalty_j = cluster_penalties.get(j, 1.0)
                cluster_penalty = min(penalty_i, penalty_j)
                score = score * cluster_penalty

                # Store pair score for transitive closure
                pair_scores[(i, j)] = (score, signals)

                # Require at least one strong signal (actual overlap evidence)
                # Tightened gates to reduce false positives from perspective
                # overlap in busy traffic scenes
                has_strong_signal = (
                    signals.get("mask_overlap", 0) > 0.02 or
                    signals.get("bbox_iou", 0) > 0.15 or
                    signals.get("deformation", 0) > 0.35 or
                    signals.get("edge_proximity", 0) > 0.85
                )

                if score >= threshold and has_strong_signal:
                    involved_indices.add(i)
                    involved_indices.add(j)

                    # Store the highest score for each vehicle
                    for idx in [i, j]:
                        if idx not in crash_scores or score > crash_scores[idx]:
                            crash_scores[idx] = score

        # === TRANSITIVE CLOSURE ===
        # If A is involved with B, and B is involved with C (even at lower score),
        # then C should also be involved. This catches 3rd+ vehicles in pileups.
        involved_indices, crash_scores = self._apply_transitive_closure(
            involved_indices, crash_scores, pair_scores,
            threshold, len(vehicles)
        )

        # Also check for individually deformed vehicles (hit-and-run, single-vehicle)
        for i in range(len(vehicles)):
            if i not in involved_indices:
                deformation = self._compute_deformation_score(vehicles[i])
                if deformation > 0.7:
                    involved_indices.add(i)
                    crash_scores[i] = deformation * 0.6

        # Build result list
        involved = []
        for idx in sorted(involved_indices):
            v = vehicles[idx].copy()
            raw_score = crash_scores.get(idx, 0)
            v["crash_score"] = round(min(settings.MAX_CRASH_SCORE, raw_score), 3)
            involved.append(v)

        return involved

    def _compute_cluster_penalties(self, vehicles):
        """
        Compute penalty factors for vehicles in crowded clusters.

        In busy Indian traffic, many vehicles are naturally close together.
        A cluster of 4+ vehicles all within proximity threshold is likely
        normal congestion, not a crash.

        Returns:
            dict: {vehicle_index: penalty_factor} where penalty < 1.0
                  means the vehicle is in a large cluster and scores
                  should be reduced.
        """
        n = len(vehicles)
        if n < 4:
            # Too few vehicles for congestion
            return {i: 1.0 for i in range(n)}

        # Build adjacency: which vehicles are close to each other?
        proximity_threshold = settings.PROXIMITY_RATIO_THRESHOLD * 1.5
        adjacency = {i: set() for i in range(n)}

        for i in range(n):
            for j in range(i + 1, n):
                edge_dist = compute_edge_distance(
                    vehicles[i]["bbox"], vehicles[j]["bbox"]
                )
                avg_diag = (compute_diagonal(vehicles[i]["bbox"]) +
                            compute_diagonal(vehicles[j]["bbox"])) / 2
                if avg_diag > 0 and (edge_dist / avg_diag) < proximity_threshold:
                    adjacency[i].add(j)
                    adjacency[j].add(i)

        # For each vehicle, count its cluster size
        penalties = {}
        for i in range(n):
            cluster_size = len(adjacency[i]) + 1  # Include self

            if cluster_size >= 5:
                # Very crowded — heavy penalty
                penalties[i] = 0.50
            elif cluster_size >= 4:
                # Crowded — moderate penalty
                penalties[i] = 0.70
            elif cluster_size >= 3:
                # Somewhat crowded — mild penalty
                penalties[i] = 0.85
            else:
                # Normal — no penalty
                penalties[i] = 1.0

        return penalties

    def _apply_transitive_closure(self, involved_indices, crash_scores,
                                   pair_scores, threshold, n_vehicles):
        """
        Apply one-hop transitive closure to catch 3rd+ vehicles in pileups.

        If vehicle A is involved and vehicle B scored above a reduced threshold
        with A, then B is also involved.

        Args:
            involved_indices: set of already-involved vehicle indices
            crash_scores: dict of vehicle_index → best crash score
            pair_scores: dict of (i,j) → (score, signals)
            threshold: base crash threshold
            n_vehicles: total number of vehicles

        Returns:
            tuple: (updated involved_indices, updated crash_scores)
        """
        if not involved_indices:
            return involved_indices, crash_scores

        # Reduced threshold for transitive connections (60% of base)
        transitive_threshold = threshold * 0.60

        # One pass: check all non-involved vehicles against involved ones
        new_involved = set()
        for idx in range(n_vehicles):
            if idx in involved_indices:
                continue

            # Check if this vehicle has a pair score with any involved vehicle
            for inv_idx in involved_indices:
                pair_key = (min(idx, inv_idx), max(idx, inv_idx))
                if pair_key in pair_scores:
                    score, signals = pair_scores[pair_key]

                    # Require at least proximity signal for transitive
                    has_proximity = (
                        signals.get("edge_proximity", 0) > 0.3 or
                        signals.get("bbox_iou", 0) > 0.05 or
                        signals.get("mask_overlap", 0) > 0.005
                    )

                    if score >= transitive_threshold and has_proximity:
                        new_involved.add(idx)
                        if idx not in crash_scores or score > crash_scores[idx]:
                            crash_scores[idx] = score
                        break  # Found connection, no need to check more

        involved_indices = involved_indices | new_involved
        return involved_indices, crash_scores

    def _compute_adaptive_threshold(self, classifier_confidence):
        """
        Adjust crash score threshold based on classifier confidence.

        If the classifier is very confident this is an accident, we lower
        the geometric threshold (less proof needed). If it's skeptical,
        we raise it (need stronger geometric evidence).
        """
        if classifier_confidence is None:
            return self.base_threshold

        if classifier_confidence >= settings.CLASSIFIER_HIGH_CONFIDENCE:
            return self.base_threshold * settings.CRASH_SCORE_BOOST_FACTOR
        elif classifier_confidence <= settings.CLASSIFIER_LOW_CONFIDENCE:
            return self.base_threshold * settings.CRASH_SCORE_PENALTY_FACTOR
        else:
            # Linear interpolation between boost and penalty
            return self.base_threshold

    def _compute_crash_score(self, vehicle_a, vehicle_b, frame_shape):
        """
        Compute a weighted crash score for a pair of vehicles.

        Returns:
            tuple: (total_score, signal_dict)
        """
        signals = {}

        # 1. Mask overlap (pixel-level collision)
        has_collision, overlap_ratio = check_pixel_collision(
            vehicle_a.get("polygon", np.array([])),
            vehicle_b.get("polygon", np.array([])),
            frame_shape
        )
        mask_score = min(1.0, overlap_ratio / settings.MASK_OVERLAP_THRESHOLD) if has_collision else 0.0
        signals["mask_overlap"] = mask_score

        # 2. Bounding box IoU
        iou = compute_iou(vehicle_a["bbox"], vehicle_b["bbox"])
        iou_score = min(1.0, iou / (settings.BBOX_IOU_THRESHOLD * 2))
        signals["bbox_iou"] = iou_score

        # 3. Edge proximity
        edge_dist = compute_edge_distance(vehicle_a["bbox"], vehicle_b["bbox"])
        avg_diag = (compute_diagonal(vehicle_a["bbox"]) + compute_diagonal(vehicle_b["bbox"])) / 2
        if avg_diag > 0:
            prox_ratio = edge_dist / avg_diag
            # Closer = higher score (inverse relationship)
            if prox_ratio <= settings.PROXIMITY_RATIO_THRESHOLD:
                prox_score = 1.0 - (prox_ratio / settings.PROXIMITY_RATIO_THRESHOLD)
            else:
                prox_score = 0.0
        else:
            prox_score = 0.0
        signals["edge_proximity"] = prox_score

        # 4. Deformation (aspect ratio anomaly)
        deform_a = self._compute_deformation_score(vehicle_a)
        deform_b = self._compute_deformation_score(vehicle_b)
        deform_score = max(deform_a, deform_b)
        signals["deformation"] = deform_score

        # 5. Relative angle (vehicles at unusual angles suggest collision)
        angle = compute_relative_angle(vehicle_a["bbox"], vehicle_b["bbox"])
        # Angles near 0, 90, or 180 are normal; other angles suggest crash
        angle_deviation = min(
            abs(angle % 90),
            abs(90 - (angle % 90))
        )
        angle_score = angle_deviation / 45.0  # Max when at 45 degrees
        signals["relative_angle"] = min(1.0, angle_score)

        # 6. Scene position (both in center of road = more likely accident)
        frame_h, frame_w = frame_shape[:2]
        center_a = compute_box_center(vehicle_a["bbox"])
        center_b = compute_box_center(vehicle_b["bbox"])
        # Score higher if both are in the central area of frame
        cx_a = abs(center_a[0] / frame_w - 0.5) * 2  # 0 at center, 1 at edge
        cx_b = abs(center_b[0] / frame_w - 0.5) * 2
        position_score = 1.0 - (cx_a + cx_b) / 2  # Higher when both near center
        signals["scene_position"] = max(0.0, position_score)

        # 7. Size mismatch (e.g., car vs motorcycle = more damage likely)
        area_a = compute_box_area(vehicle_a["bbox"])
        area_b = compute_box_area(vehicle_b["bbox"])
        if min(area_a, area_b) > 0:
            size_ratio = min(area_a, area_b) / max(area_a, area_b)
            size_mismatch = 1.0 - size_ratio  # Higher when sizes differ more
        else:
            size_mismatch = 0.0
        signals["size_mismatch"] = size_mismatch

        # Compute weighted total
        total = 0.0
        for key, weight in self.weights.items():
            total += weight * signals.get(key, 0.0)

        return total, signals

    def _compute_deformation_score(self, vehicle):
        """
        Score how deformed a vehicle appears based on aspect ratio.

        Returns:
            float: 0.0 (normal) to 1.0 (severely deformed)
        """
        vehicle_type = vehicle.get("type", "car")
        ar = compute_aspect_ratio(vehicle["bbox"])
        area_ratio = vehicle.get("area_ratio", 0)

        # Skip tiny vehicles (can't judge deformation)
        if area_ratio < 0.02:
            return 0.0

        normal = NORMAL_ASPECT_RATIOS.get(vehicle_type, {"min": 0.5, "max": 3.0})
        min_ar = normal["min"]
        max_ar = normal["max"]

        if ar < min_ar:
            deviation = (min_ar - ar) / min_ar
        elif ar > max_ar:
            deviation = (ar - max_ar) / max_ar
        else:
            return 0.0

        return min(1.0, deviation)

    def _check_single_vehicle_accident(self, vehicles, frame_shape,
                                        classifier_confidence=None):
        """
        Check for single-vehicle accidents (rollover, hit-and-run).
        A single severely deformed vehicle may indicate an accident.

        When the classifier is very confident (>0.85), we lower the
        deformation requirement since the scene-level model is
        providing strong corroborating evidence.
        """
        involved = []
        cls_conf = classifier_confidence or 0.0

        for v in vehicles:
            deform = self._compute_deformation_score(v)

            # Two paths to single-vehicle detection:
            # 1. High deformation (traditional)
            if deform > 0.5:
                v_copy = v.copy()
                v_copy["crash_score"] = round(deform * 0.5, 3)
                involved.append(v_copy)
            # 2. Classifier very confident + any deformation or large vehicle
            elif (cls_conf >= settings.SINGLE_VEHICLE_CLASSIFIER_THRESHOLD
                  and deform >= settings.SINGLE_VEHICLE_DEFORMATION_THRESHOLD):
                v_copy = v.copy()
                v_copy["crash_score"] = round(
                    min(settings.MAX_CRASH_SCORE, cls_conf * 0.4 + deform * 0.2), 3
                )
                involved.append(v_copy)

        return involved

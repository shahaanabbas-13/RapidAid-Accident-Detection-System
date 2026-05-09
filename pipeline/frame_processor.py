"""
RapidAid — Frame Processor

Orchestrates all detection modules for a single frame.
This is the core pipeline: Frame → Classify → Detect → Score → Zone → Report

Supports classifier-geometric fusion: the frame classifier's confidence
modulates the crash score threshold for the geometric analysis.

Pipeline stages:
  0. Frame Classification (pre-filter, optional)
  1. Vehicle Detection
  2. Person Detection
  3. Accident Classification (vehicle-vehicle)
  3b. Vehicle-Pedestrian Collision Detection (conservative)
  3c. Damage Classification (M2 — per-vehicle damage scoring)
  4. Victim Classification
  5. Standalone victim / hit-and-run checks
  6. Quality filter (remove low-confidence involved vehicles)
  7. Zone Computation (collision-point centered)
  8. Report Generation
"""
# pyrefly: ignore [missing-import]
import cv2
from config import settings
from models.vehicle_detector import VehicleDetector
from models.person_detector import PersonDetector
from models.accident_classifier import AccidentClassifier
from models.victim_classifier import VictimClassifier
from models.frame_classifier import FrameClassifier
from models.damage_classifier import DamageClassifier
from models.accident_zone import AccidentZoneCalculator
from models.vehicle_pedestrian_detector import VehiclePedestrianDetector
from models.collision_detector import CollisionDetector
from pipeline.report_generator import ReportGenerator


class FrameProcessor:
    """
    Processes a single frame through the full RapidAid pipeline.

    Pipeline stages:
    0. Frame Classification (YOLOv8n-cls pre-filter)
    1. Vehicle Detection (YOLOv8s-seg)
    2. Person Detection (YOLOv8s-pose)
    3. Accident Classification (multi-signal scoring, classifier-fused)
    3b. Vehicle-Pedestrian Collision Detection (conservative)
    3c. Damage Classification (M2 — per-vehicle damage scoring)
    4. Victim Classification (keypoint + spatial analysis)
    5. Quality filter for involved vehicles
    6. Accident Zone Computation (collision-point centered)
    7. Report Generation (JSON + annotated frame)
    """

    def __init__(self, vehicle_model_path=None, pose_model_path=None):
        """
        Initialize all detection modules.

        Args:
            vehicle_model_path: custom path to vehicle detection model
            pose_model_path: custom path to pose estimation model
        """
        print("=" * 50)
        print("  RapidAid — Initializing Pipeline")
        print("=" * 50)

        self.vehicle_detector = VehicleDetector(vehicle_model_path)
        self.person_detector = PersonDetector(pose_model_path)
        self.accident_classifier = AccidentClassifier()
        self.victim_classifier = VictimClassifier()
        self.frame_classifier = FrameClassifier()
        self.damage_classifier = DamageClassifier()
        self.collision_detector = CollisionDetector()  # M4: collision zone detector
        self.zone_calculator = AccidentZoneCalculator()
        self.vp_detector = VehiclePedestrianDetector()
        self.report_generator = ReportGenerator()

        print("=" * 50)
        print("  RapidAid — Pipeline Ready")
        print("=" * 50)

    def process(self, frame, source_path=None, timestamp_sec=None,
                classifier_confidence=None):
        """
        Process a single frame through the full pipeline.

        Args:
            frame: numpy array (BGR image) or path to image file
            source_path: path to source file (for report)
            timestamp_sec: timestamp in seconds (for video mode)
            classifier_confidence: pre-computed classifier confidence (0-1).

        Returns:
            dict with pipeline results
        """
        # Load frame if path is given
        if isinstance(frame, str):
            source_path = source_path or frame
            frame = cv2.imread(frame)
            if frame is None:
                raise ValueError(f"Cannot load image: {source_path}")

        frame_h, frame_w = frame.shape[:2]

        # Stage 0: Pre-filter with classifier (if not already done)
        cls_conf = classifier_confidence
        if cls_conf is None:
            if self.frame_classifier.is_available():
                is_accident, cls_conf = self.frame_classifier.classify(frame)
                # IMPORTANT: Only use M1 as pre-filter when M4 is NOT available.
                # M4 detects collision zones with high confidence (0.7-0.95) on
                # frames where M1 gives near-zero confidence. If we reject here,
                # M4 never gets a chance to run.
                if (not self.collision_detector.is_available()
                        and not is_accident and cls_conf < 0.20):
                    result = self._empty_result(frame, source_path, timestamp_sec)
                    result["classifier_confidence"] = cls_conf
                    return result
            else:
                cls_conf = None

        # Stage 1: Detect all vehicles
        all_vehicles = self.vehicle_detector.detect(frame, filter_background=True)

        # Stage 2: Detect all persons
        all_persons = self.person_detector.detect(frame)

        # ================================================================
        # Stage 3-PRIMARY: M4 Collision Zone Detection (when available)
        # This is the paradigm shift: instead of guessing which vehicles
        # collided from geometric heuristics, we detect the collision
        # zone DIRECTLY, then find vehicles inside it.
        # ================================================================
        m4_used = False
        collision_zones = []
        involved_vehicles = []

        if self.collision_detector.is_available():
            collision_zones = self.collision_detector.detect(frame)
            if collision_zones:
                # M4 found collision zone(s) — use them to find involved vehicles
                involved_vehicles = self.collision_detector.get_involved_vehicles(
                    collision_zones, all_vehicles
                )

                # If no filtered vehicles overlap with the zone, retry with
                # UNFILTERED vehicles. This handles cases like Video 9 where
                # the background filter removes ALL vehicles (they appear
                # small/distant) but M4 confirms an accident at 0.95+ confidence.
                if not involved_vehicles:
                    all_vehicles_unfiltered = self.vehicle_detector.detect(
                        frame, filter_background=False
                    )
                    involved_vehicles = self.collision_detector.get_involved_vehicles(
                        collision_zones, all_vehicles_unfiltered
                    )
                    # Add the unfiltered vehicles to all_vehicles for downstream use
                    if involved_vehicles:
                        existing_keys = {tuple(v["bbox"]) for v in all_vehicles}
                        for v in all_vehicles_unfiltered:
                            if tuple(v["bbox"]) not in existing_keys:
                                all_vehicles.append(v)

                if involved_vehicles:
                    m4_used = True
                    print(f"  [M4] {len(collision_zones)} collision zone(s), "
                          f"{len(involved_vehicles)} vehicle(s) inside")

        # Stage 3-FALLBACK: Geometric analysis (when M4 unavailable or found nothing)
        if not m4_used or not involved_vehicles:
            involved_vehicles = self.accident_classifier.classify(
                all_vehicles, frame.shape, classifier_confidence=cls_conf
            )

        # Stage 3b: Vehicle-Pedestrian Collision Detection
        # CONSERVATIVE: only flags persons with actual contact evidence
        # + non-standing posture. Standing bystanders are filtered out.
        vp_collisions = self.vp_detector.detect(
            all_vehicles, all_persons, frame.shape
        )

        # Merge VP collision results into involved_vehicles and victims
        vp_victims = []
        involved_bbox_keys = {tuple(v["bbox"]) for v in involved_vehicles}
        for collision in vp_collisions:
            veh = collision["vehicle"]
            per = collision["person"]
            score = collision["score"]

            # Add vehicle to involved list (if not already)
            vbox_key = tuple(veh["bbox"])
            if vbox_key not in involved_bbox_keys:
                v_copy = veh.copy()
                v_copy["crash_score"] = round(score, 3)
                involved_vehicles.append(v_copy)
                involved_bbox_keys.add(vbox_key)

            # Determine victim status based on posture
            p_w = per.get("bbox_width", 0)
            p_h = per.get("bbox_height", 0)
            if p_h > 0 and p_w / p_h > 0.55:
                status = "fallen"
            elif score > 0.5:
                status = "hit"
            else:
                status = "nearby"

            vp_victims.append({
                "confidence": per["confidence"],
                "bbox": per["bbox"],
                "center": per["center"],
                "status": status,
            })

        # Stage 3c: Damage Classification (M2)
        # Classify EVERY detected vehicle as damaged or normal.
        # M2 scores are ALWAYS computed (for annotation display), but they
        # only influence crash scoring when M2_TRUST_ENABLED is True.
        damage_scores = {}
        if self.damage_classifier.is_available():
            damage_scores = self.damage_classifier.classify_all_vehicles(
                frame, all_vehicles
            )

            # Attach damage scores to all vehicles for annotation display
            for v in all_vehicles:
                bbox_key = tuple(v["bbox"])
                v["damage_score"] = round(damage_scores.get(bbox_key, 0.0), 3)

            # GATED: Only boost crash scores when M2 is trusted
            if settings.M2_TRUST_ENABLED:
                for v in involved_vehicles:
                    bbox_key = tuple(v["bbox"])
                    dmg_score = damage_scores.get(bbox_key, 0.0)
                    if dmg_score > 0.5:
                        old_score = v.get("crash_score", 0)
                        v["crash_score"] = round(
                            min(settings.MAX_CRASH_SCORE,
                                old_score + dmg_score * 0.2), 3
                        )
                        v["damage_score"] = round(dmg_score, 3)

        # Stage 3d: Damage-driven involved vehicle REFINEMENT
        # GATED: Only refine when M2 is trusted. When untrusted, M2 gives
        # false positives to normal Indian vehicles and causes wrong swaps.
        if (settings.M2_TRUST_ENABLED and damage_scores
                and involved_vehicles and len(all_vehicles) >= 2):
            involved_vehicles = self._refine_involved_by_damage(
                involved_vehicles, all_vehicles, damage_scores, frame.shape
            )

        # Stage 4: Compute initial accident zone (needed for victim classification)
        initial_zone = self.zone_calculator.compute(
            involved_vehicles, [], frame_w, frame_h
        )

        # Stage 5: Classify victims vs bystanders
        victims = self.victim_classifier.classify(
            all_persons, involved_vehicles, initial_zone
        )

        # Add VP-detected victims (avoid duplicates)
        victim_bbox_keys = {tuple(v["bbox"]) for v in victims}
        for vp_vic in vp_victims:
            if tuple(vp_vic["bbox"]) not in victim_bbox_keys:
                victims.append(vp_vic)
                victim_bbox_keys.add(tuple(vp_vic["bbox"]))

        # Stage 5b: Victim-only detection (hit-and-run / aftermath scenes)
        if not involved_vehicles and not victims:
            standalone_victims = self.victim_classifier.detect_standalone_victims(
                all_persons
            )
            if len(standalone_victims) >= settings.VICTIM_ONLY_MIN_COUNT:
                victims = standalone_victims

                # Stage 5c: Check if any vehicle is near the victims
                from utils.geometry import compute_edge_distance, compute_overlap_ratio
                involved_bboxes = set()
                for victim in victims:
                    for vehicle in all_vehicles:
                        vbox_key = tuple(vehicle["bbox"])
                        if vbox_key in involved_bboxes:
                            continue
                        overlap = compute_overlap_ratio(vehicle["bbox"], victim["bbox"])
                        edge_dist = compute_edge_distance(vehicle["bbox"], victim["bbox"])
                        if overlap > 0.05 or edge_dist < settings.VEHICLE_VICTIM_DISTANCE:
                            v_copy = vehicle.copy()
                            v_copy["crash_score"] = round(
                                min(settings.MAX_CRASH_SCORE, 0.4 + overlap), 3
                            )
                            involved_vehicles.append(v_copy)
                            involved_bboxes.add(vbox_key)

        # Stage 5d: Person-near-vehicle check (for non-lying persons near vehicles)
        # GATED: requires M1 scene confidence above threshold to prevent
        # flagging bystanders near parked cars in normal traffic.
        if (not involved_vehicles and not victims and len(all_persons) > 0
                and cls_conf is not None
                and cls_conf >= settings.VICTIM_VEHICLE_FALLBACK_MIN_SCENE_CONF):
            from utils.geometry import compute_edge_distance, compute_overlap_ratio
            for person in all_persons:
                for vehicle in all_vehicles:
                    overlap = compute_overlap_ratio(person["bbox"], vehicle["bbox"])
                    if overlap > 0.15:
                        victim = {
                            "confidence": person["confidence"],
                            "bbox": person["bbox"],
                            "center": person["center"],
                            "status": "trapped",
                        }
                        victims.append(victim)
                        v_copy = vehicle.copy()
                        v_copy["crash_score"] = round(
                            min(settings.MAX_CRASH_SCORE, 0.4 + overlap), 3
                        )
                        if tuple(vehicle["bbox"]) not in {tuple(v["bbox"]) for v in involved_vehicles}:
                            involved_vehicles.append(v_copy)
                        break

        # Stage 5e: Single-vehicle + classifier fusion
        if (not involved_vehicles and cls_conf is not None
                and cls_conf >= settings.SINGLE_VEHICLE_CLASSIFIER_THRESHOLD
                and len(all_vehicles) >= 1):
            single_involved = self.accident_classifier._check_single_vehicle_accident(
                all_vehicles, frame.shape, cls_conf
            )
            if single_involved:
                involved_vehicles = single_involved

        # Stage 5f: Vehicle-pedestrian collision (classifier-driven)
        # GATED: requires M1 scene confidence above threshold
        if (not involved_vehicles and not victims
                and cls_conf is not None
                and cls_conf >= settings.VICTIM_VEHICLE_FALLBACK_MIN_SCENE_CONF
                and len(all_vehicles) >= 1 and len(all_persons) >= 1):
            from utils.geometry import compute_edge_distance, compute_overlap_ratio
            veh_ped_found = False
            for person in all_persons:
                if veh_ped_found:
                    break
                for vehicle in all_vehicles:
                    edge_dist = compute_edge_distance(
                        person["bbox"], vehicle["bbox"]
                    )
                    overlap = compute_overlap_ratio(
                        person["bbox"], vehicle["bbox"]
                    )
                    if edge_dist < 150 or overlap > 0.03:
                        p_w = person.get("bbox_width", 0)
                        p_h = person.get("bbox_height", 0)
                        is_lying = (p_h > 0 and p_w / p_h > 0.55)
                        status = "fallen" if is_lying else "hit"
                        victim = {
                            "confidence": person["confidence"],
                            "bbox": person["bbox"],
                            "center": person["center"],
                            "status": status,
                        }
                        victims.append(victim)
                        v_copy = vehicle.copy()
                        v_copy["crash_score"] = round(
                            min(settings.MAX_CRASH_SCORE,
                                cls_conf * 0.5 + overlap * 0.2), 3
                        )
                        involved_vehicles.append(v_copy)
                        veh_ped_found = True
                        break

        # Stage 5g: Damage-only accident detection (M2-driven)
        # GATED: requires BOTH M2_TRUST_ENABLED AND strong M1 agreement.
        # Without M2 trust, this stage is skipped entirely — the current M2
        # gives 96% damage to normal Indian taxis, which triggers false
        # positives even when M1 says "NOT an accident".
        if (not involved_vehicles and not victims
                and settings.M2_TRUST_ENABLED
                and damage_scores and cls_conf is not None
                and cls_conf >= settings.DAMAGE_ONLY_MIN_SCENE_CONF):
            for vehicle in all_vehicles:
                bbox_key = tuple(vehicle["bbox"])
                dmg_score = damage_scores.get(bbox_key, 0.0)
                if dmg_score > settings.DAMAGE_ONLY_MIN_DAMAGE:
                    v_copy = vehicle.copy()
                    v_copy["crash_score"] = round(
                        min(settings.MAX_CRASH_SCORE,
                            dmg_score * 0.5 + cls_conf * 0.3), 3
                    )
                    v_copy["damage_score"] = round(dmg_score, 3)
                    involved_vehicles.append(v_copy)

        # NOTE: Stage 5g-ii (damage-dominant detection) has been REMOVED.
        # It was the primary source of false positives: M2 gave 90%+ damage
        # to normal vehicles, and the old threshold of M1>=0.20 was far too
        # low to gate it. This will be re-enabled after M2 is retrained
        # with Indian traffic data in Phase 2.

        # Stage 5h: Scene-only accident detection (M1-driven, no vehicles)
        # When YOLO can't detect ANY vehicles (smoke, debris, extreme
        # occlusion) but M1 scene classifier is very confident this is
        # an accident scene → flag with full-frame zone.
        # This specifically addresses Image 3 (cars crashing with smoke).
        if (not involved_vehicles and not victims
                and cls_conf is not None
                and cls_conf >= settings.SCENE_ONLY_CLASSIFIER_THRESHOLD
                and len(all_vehicles) == 0):
            # M1 is our only signal — create a scene-level detection
            # with a centered zone
            involved_vehicles = []  # No specific vehicles to highlight
            victims = []  # No specific victims
            # The zone will be computed as a centered region below

        # === Stage 6: Quality filter ===
        # Remove low-confidence or very small vehicles that were flagged
        # by transitive closure or VP detection but are likely false positives.
        # Only filter when we have multiple involved vehicles (keep single
        # vehicle detections even if low confidence).
        if len(involved_vehicles) > 1:
            involved_vehicles = self._filter_weak_involved(
                involved_vehicles, frame_h, frame_w
            )

        # Stage 7: Recompute final zone with victims included
        final_zone = self.zone_calculator.compute(
            involved_vehicles, victims, frame_w, frame_h
        )

        # Scene-only detection: if M1 is very confident but no entities
        # were detected, use a centered zone covering the middle of the frame
        scene_only_detection = False
        if (final_zone is None and cls_conf is not None
                and cls_conf >= settings.SCENE_ONLY_CLASSIFIER_THRESHOLD
                and len(all_vehicles) == 0):
            # Create a centered zone covering ~60% of the frame
            pad_x = int(frame_w * 0.20)
            pad_y = int(frame_h * 0.20)
            final_zone = [pad_x, pad_y, frame_w - pad_x, frame_h - pad_y]
            scene_only_detection = True

        # === Compute ensemble fused confidence ===
        max_crash = max((v.get("crash_score", 0) for v in involved_vehicles),
                        default=0.0)
        max_damage = max(damage_scores.values(), default=0.0) if damage_scores else 0.0
        scene_conf = cls_conf if cls_conf is not None else 0.0

        if max_crash > 0:
            # Geometric signal present — full ensemble
            fused_confidence = (
                settings.ENSEMBLE_WEIGHT_GEOMETRIC * max_crash +
                settings.ENSEMBLE_WEIGHT_SCENE * scene_conf +
                settings.ENSEMBLE_WEIGHT_DAMAGE * max_damage
            )
        elif scene_conf > 0 or max_damage > 0:
            # No geometric signal — M1+M2 only
            fused_confidence = (
                settings.ENSEMBLE_NO_GEO_WEIGHT_SCENE * scene_conf +
                settings.ENSEMBLE_NO_GEO_WEIGHT_DAMAGE * max_damage
            )
        else:
            fused_confidence = 0.0

        fused_confidence = round(min(1.0, fused_confidence), 3)

        # Stage 8: Generate report and annotated frame
        result = self.report_generator.generate(
            frame, involved_vehicles, victims, final_zone,
            source_path=source_path,
            timestamp_sec=timestamp_sec,
            classifier_confidence=scene_conf,
            max_damage_score=max_damage,
            fused_confidence=fused_confidence,
            scene_only=scene_only_detection,
        )

        # Add intermediate data for debugging/inspection
        result["involved_vehicles"] = involved_vehicles
        result["victims"] = victims
        result["accident_zone"] = final_zone
        result["all_vehicles"] = all_vehicles
        result["all_persons"] = all_persons
        result["classifier_confidence"] = cls_conf
        result["max_damage_score"] = max_damage
        result["fused_confidence"] = fused_confidence
        result["scene_only_detection"] = scene_only_detection

        return result

    def _refine_involved_by_damage(self, involved_vehicles, all_vehicles,
                                       damage_scores, frame_shape):
        """
        Refine which vehicles are 'involved' using per-vehicle M2 damage scores.

        CRITICAL: M2 gives false-positive HIGH damage to many vehicles in
        Indian traffic (old taxis, auto-rickshaws, worn trucks all look
        'damaged' to M2). To avoid this:

        1. Compute a SCENE-WIDE damage baseline. If most vehicles score
           high (>50%), M2 is unreliable for this scene — skip refinement.
        2. Only act when there's a clear DIFFERENTIAL: the outsider has
           significantly higher damage than both the involved vehicles
           AND the scene baseline.
        3. Limit additions to prevent zone explosion.
        """
        from utils.geometry import compute_edge_distance, compute_box_center

        # === Step 1: Compute scene-wide damage baseline ===
        all_damage_values = list(damage_scores.values())
        if not all_damage_values:
            return involved_vehicles

        scene_avg_damage = sum(all_damage_values) / len(all_damage_values)
        n_high_damage = sum(1 for d in all_damage_values if d > 0.50)
        pct_high_damage = n_high_damage / max(len(all_damage_values), 1)

        # If >50% of ALL vehicles in the scene score as "damaged",
        # M2 is unreliable (Indian traffic bias). Skip refinement.
        if pct_high_damage > 0.50:
            return involved_vehicles

        # === Step 2: Compute involved vs outsider damage ===
        involved_bbox_keys = {tuple(v["bbox"]) for v in involved_vehicles}
        involved_damage = []
        for v in involved_vehicles:
            bbox_key = tuple(v["bbox"])
            dmg = damage_scores.get(bbox_key, 0.0)
            involved_damage.append(dmg)

        avg_involved_damage = sum(involved_damage) / max(len(involved_damage), 1)

        # Find non-involved vehicles with damage ABOVE scene baseline
        damaged_outsiders = []
        for v in all_vehicles:
            bbox_key = tuple(v["bbox"])
            if bbox_key in involved_bbox_keys:
                continue
            dmg = damage_scores.get(bbox_key, 0.0)
            # Must be significantly above scene average to be meaningful
            if dmg > max(0.60, scene_avg_damage + 0.25):
                damaged_outsiders.append((v, dmg))

        if not damaged_outsiders:
            return involved_vehicles

        # Sort by damage score (highest first)
        damaged_outsiders.sort(key=lambda x: x[1], reverse=True)
        best_outsider_damage = damaged_outsiders[0][1]

        # === Step 3: Determine if outsider has significantly more damage ===
        damage_gap = best_outsider_damage - avg_involved_damage

        # Case 1: Involved vehicles have LOW damage, outsider has HIGH damage,
        # AND there's a clear gap → geometric picked the WRONG vehicles
        if (avg_involved_damage < 0.30
                and best_outsider_damage > 0.70
                and damage_gap > 0.35):
            new_involved = []
            # Only add the top 2 damaged outsiders (prevent zone explosion)
            for v, dmg in damaged_outsiders[:2]:
                v_copy = v.copy()
                v_copy["crash_score"] = round(dmg * 0.8, 3)
                v_copy["damage_score"] = round(dmg, 3)
                new_involved.append(v_copy)

            # Keep any originally-involved vehicle that's near a damaged one
            for v in involved_vehicles:
                v_center = compute_box_center(v["bbox"])
                for outsider, _ in damaged_outsiders[:2]:
                    o_center = compute_box_center(outsider["bbox"])
                    dist = ((v_center[0] - o_center[0])**2 +
                            (v_center[1] - o_center[1])**2) ** 0.5
                    frame_diag = (frame_shape[0]**2 + frame_shape[1]**2) ** 0.5
                    if dist < frame_diag * 0.20:
                        bbox_key = tuple(v["bbox"])
                        if bbox_key not in {tuple(nv["bbox"]) for nv in new_involved}:
                            new_involved.append(v)
                        break

            return new_involved

        # Case 2: Involved have moderate damage, outsider has much more
        # → Add the outsider but don't replace existing involved
        if damage_gap > 0.30:
            for v, dmg in damaged_outsiders[:1]:  # Add at most 1
                v_copy = v.copy()
                v_copy["crash_score"] = round(dmg * 0.7, 3)
                v_copy["damage_score"] = round(dmg, 3)
                bbox_key = tuple(v_copy["bbox"])
                if bbox_key not in involved_bbox_keys:
                    involved_vehicles.append(v_copy)

        return involved_vehicles

    def _filter_weak_involved(self, involved_vehicles, frame_h, frame_w):
        """
        Remove involved vehicles that are likely false positives:
        - Very low detection confidence (< 30%)
        - Very small area ratio AND low crash score
        - Very far from other involved vehicles (outlier)

        This prevents small, low-confidence detections (like a tiny
        motorcycle at 37%) from pulling the accident zone away from
        the actual crash area.
        """
        frame_area = frame_h * frame_w
        if len(involved_vehicles) <= 1:
            return involved_vehicles

        # Compute center of mass of involved vehicles
        centers = []
        for v in involved_vehicles:
            cx = (v["bbox"][0] + v["bbox"][2]) / 2
            cy = (v["bbox"][1] + v["bbox"][3]) / 2
            centers.append((cx, cy))

        avg_cx = sum(c[0] for c in centers) / len(centers)
        avg_cy = sum(c[1] for c in centers) / len(centers)

        filtered = []
        for i, v in enumerate(involved_vehicles):
            conf = v.get("confidence", 0)
            crash_score = v.get("crash_score", 0)
            area = (v["bbox"][2] - v["bbox"][0]) * (v["bbox"][3] - v["bbox"][1])
            area_ratio = area / frame_area if frame_area > 0 else 0

            # Distance from center of mass
            cx, cy = centers[i]
            dist_from_center = ((cx - avg_cx)**2 + (cy - avg_cy)**2) ** 0.5
            frame_diag = (frame_h**2 + frame_w**2) ** 0.5

            # Remove if: low confidence + small + far from center + low crash score
            is_weak = (
                conf < 30.0 and
                area_ratio < 0.01 and
                crash_score < 0.50
            )

            # Remove if: very far outlier (> 40% of frame diagonal from center)
            is_outlier = (
                dist_from_center > frame_diag * 0.40 and
                conf < 45.0 and
                crash_score < 0.50
            )

            if is_weak or is_outlier:
                continue

            filtered.append(v)

        # Safety: if filtering removed everything, keep the original
        return filtered if filtered else involved_vehicles

    def _empty_result(self, frame, source_path, timestamp_sec):
        """Return an empty no-accident result."""
        result = self.report_generator.generate(
            frame, [], [], None,
            source_path=source_path,
            timestamp_sec=timestamp_sec,
        )
        result["involved_vehicles"] = []
        result["victims"] = []
        result["accident_zone"] = None
        result["all_vehicles"] = []
        result["all_persons"] = []
        result["classifier_confidence"] = 0.0
        result["max_damage_score"] = 0.0
        result["fused_confidence"] = 0.0
        result["scene_only_detection"] = False
        return result

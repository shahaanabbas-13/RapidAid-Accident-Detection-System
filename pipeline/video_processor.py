"""
RapidAid — Video Processor

Processes video files to find and analyze accident frames.
Uses a hybrid scanning architecture:

  1. ALWAYS run the lightweight classifier on sampled frames
  2. Maintain smoothed classifier confidence (rolling average)
  3. Detect CHANGES in classifier confidence (rising = something happening)
  4. Enter ALERT MODE when classifier rise OR motion spike triggers
  5. In alert mode: run full geometric pipeline
  6. Confirm accident when detections accumulate over time
  7. After confirmation: scan forward for best aftermath frame

Improvements over previous version:
  - Extended baseline calibration (18 frames ~6 seconds)
  - Alert cooldown after non-confirmed alerts (3s suppression)
  - Higher confirmation count for single-vehicle detections
  - Vehicle count consistency check in detection window
"""
# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np
import json
from config import settings
from pipeline.frame_processor import FrameProcessor
from models.temporal_classifier import TemporalClassifier
from utils.helpers import display_frame


class VideoProcessor:
    """
    Processes video files to detect accidents.

    Hybrid Strategy:
    1. Sample frames at configured FPS rate
    2. Run classifier on EVERY sampled frame (fast: ~15ms)
    3. Compute motion score for motion spike detection
    4. Build a classifier baseline over the first few seconds
    5. Enter alert mode when EITHER:
       a. Classifier confidence rises significantly above baseline, OR
       b. Classifier is consistently high AND motion spike occurs, OR
       c. Classifier spikes (sudden jump from low to high)
    6. In alert mode: run full detection pipeline
    7. Require temporal confirmation (N detections in M frames)
    8. After confirmation: post-crash lookahead for best aftermath frame
    """

    def __init__(self, frame_processor=None):
        if frame_processor is None:
            self.processor = FrameProcessor()
        else:
            self.processor = frame_processor
        self.temporal_classifier = TemporalClassifier()

    def process_video(self, video_path, stop_on_first=True, show_result=True):
        """
        Process a video file for accident detection.

        Returns:
            dict with accident_detected, best_result, timestamp_sec, frames_analyzed
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {video_path}")
            return {"accident_detected": False, "best_result": None}

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, int(fps / settings.FRAMES_PER_SECOND_TO_ANALYZE))

        print(f"[VideoProcessor] Video: {video_path}")
        print(f"[VideoProcessor] {total_frames} frames @ {fps:.1f} FPS")
        print(f"[VideoProcessor] Analyzing 1 frame every {frame_interval} frames")
        print()

        frame_count = 0
        frames_analyzed = 0
        prev_gray = None
        best_result = None
        best_score = 0
        accident_timestamp = None

        # Motion tracking
        motion_history = []

        # Classifier tracking
        cls_history = []            # Full history for baseline
        cls_baseline = None         # Established baseline level
        cls_baseline_frames = 18    # Frames to build baseline (~6 seconds)

        # Alert mode state
        alert_mode = False
        alert_start_sec = None
        alert_cooldown_until = 0.0  # Suppress re-alerting until this time

        # Temporal consistency
        detection_window = []

        # Confirmed flag — once confirmed, switch to post-crash scan
        confirmed = False
        confirmation_time = None

        # M3 temporal classifier
        self.temporal_classifier.reset()
        m3_score = 0.0

        # Adaptive confirmation count: high-baseline scenes need more evidence
        confirm_count = settings.TEMPORAL_CONFIRM_COUNT  # Will be set after baseline

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Skip frames based on interval
            if frame_count % frame_interval != 0:
                continue

            t_sec = round(frame_count / fps, 2)
            frames_analyzed += 1

            # === STEP 1: Always compute motion score ===
            motion_score = self._compute_motion_score(frame, prev_gray)
            prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            motion_history.append(motion_score)
            if len(motion_history) > 30:
                motion_history.pop(0)

            # === STEP 2: Always run classifier (fast) ===
            cls_conf = 0.0
            if self.processor.frame_classifier.is_available():
                _, cls_conf = self.processor.frame_classifier.classify(frame)

            cls_history.append(cls_conf)

            # === STEP 2b: Feed frame to M3 temporal classifier ===
            if self.temporal_classifier.is_available():
                self.temporal_classifier.extract_and_add_frame(
                    frame, self.processor.frame_classifier
                )
                if self.temporal_classifier.has_enough_frames():
                    m3_score = self.temporal_classifier.classify_sequence()

            # === STEP 3: Build classifier baseline ===
            # The baseline represents the "normal" classifier level for this scene.
            # Some scenes (busy traffic) have a naturally high classifier response.
            if cls_baseline is None and len(cls_history) >= cls_baseline_frames:
                cls_baseline = float(np.median(cls_history[:cls_baseline_frames]))
                # High-baseline scenes need more temporal evidence to avoid
                # false positives from normal traffic perspective overlaps
                if cls_baseline > 0.70:
                    confirm_count = max(settings.TEMPORAL_CONFIRM_COUNT, 4)
                elif cls_baseline > 0.50:
                    confirm_count = max(settings.TEMPORAL_CONFIRM_COUNT, 3)
                print(f"  [baseline] Classifier baseline: {cls_baseline:.3f} "
                      f"(confirm_count={confirm_count})")

            # Smoothed classifier (recent window)
            recent_cls = cls_history[-settings.CLASSIFIER_SMOOTHING_WINDOW:]
            smoothed_cls = np.mean(recent_cls)

            # === STEP 4: Check alert triggers ===
            if not alert_mode and not confirmed:
                triggered = False
                trigger_reason = ""

                # Check cooldown: don't re-alert immediately after a failed alert
                if t_sec < alert_cooldown_until:
                    continue

                if cls_baseline is not None:
                    # Trigger A: Classifier RISE — smoothed confidence significantly
                    # above baseline (something changed in the scene)
                    cls_rise = smoothed_cls - cls_baseline
                    if cls_rise > 0.25 and smoothed_cls > 0.60:
                        triggered = True
                        trigger_reason = (f"classifier rise "
                                          f"(baseline={cls_baseline:.3f}, "
                                          f"now={smoothed_cls:.3f}, "
                                          f"rise={cls_rise:.3f})")

                    # Trigger B: Classifier SPIKE — single frame jumps very high
                    # from a previously low level (and sustained in recent window)
                    if not triggered and len(cls_history) >= 4:
                        prev_avg = np.mean(cls_history[-4:-1])
                        if (cls_conf > 0.75 and prev_avg < 0.30
                                and smoothed_cls > 0.40):
                            triggered = True
                            trigger_reason = (f"classifier spike "
                                              f"(prev={prev_avg:.3f}, "
                                              f"now={cls_conf:.3f}, "
                                              f"smoothed={smoothed_cls:.3f})")

                    # Trigger C: High classifier + motion spike
                    # Even if baseline is high, a motion spike during high
                    # classifier suggests something is happening NOW
                    if not triggered and smoothed_cls > 0.50:
                        if self._is_motion_spike(motion_score, motion_history):
                            triggered = True
                            trigger_reason = (f"classifier+motion "
                                              f"(cls={smoothed_cls:.3f}, "
                                              f"motion={motion_score:.1f})")

                    # Trigger D: Sustained near-perfect classifier
                    # For scenes with high baseline (busy traffic), the classifier
                    # can't rise much. But if it sustains near-perfect confidence
                    # (>0.95 smoothed), the scene has clearly changed.
                    if (not triggered and cls_baseline > 0.60
                            and smoothed_cls > 0.97):
                        triggered = True
                        trigger_reason = (f"sustained high classifier "
                                          f"(baseline={cls_baseline:.3f}, "
                                          f"smoothed={smoothed_cls:.3f})")

                    # Trigger E: M3 temporal classifier spike
                    # If M3 detects a temporal accident pattern even when
                    # M1 frame-level is ambiguous
                    if (not triggered and m3_score >= 0.80
                            and smoothed_cls > 0.30):
                        triggered = True
                        trigger_reason = (f"temporal classifier "
                                          f"(m3={m3_score:.3f}, "
                                          f"cls={smoothed_cls:.3f})")

                    # Trigger F: M4 collision-zone detection
                    # Run M4 periodically (every 5th sampled frame) to check
                    # for collision zones. M4 detects accidents at 0.63-0.95
                    # confidence even when M1 gives near-zero. This is the
                    # primary trigger for videos where M1 fails.
                    if (not triggered
                            and self.processor.collision_detector.is_available()
                            and frames_analyzed % 5 == 0):
                        m4_zones = self.processor.collision_detector.detect(frame)
                        if m4_zones and m4_zones[0]["confidence"] >= 0.50:
                            triggered = True
                            trigger_reason = (
                                f"M4 collision zone "
                                f"(conf={m4_zones[0]['confidence']:.3f}, "
                                f"zones={len(m4_zones)})")

                else:
                    # Baseline not yet established — wait for baseline
                    # Don't trigger alert until we understand what's normal
                    # for this scene (prevents false triggers from scene cuts)
                    pass

                if triggered:
                    alert_mode = True
                    alert_start_sec = t_sec
                    print(f"  [{t_sec}s] ALERT: {trigger_reason}")

                # Not in alert: skip expensive pipeline
                if not alert_mode:
                    continue

            # === STEP 5: In ALERT MODE — run full pipeline ===
            result = self.processor.process(
                frame, source_path=video_path, timestamp_sec=t_sec,
                classifier_confidence=cls_conf,
            )

            if result["accident_detected"]:
                n_vehicles = len(result["involved_vehicles"])
                n_victims = len(result["victims"])
                avg_crash = 0
                max_crash = 0
                if n_vehicles > 0:
                    crash_scores = [v.get("crash_score", 0)
                                    for v in result["involved_vehicles"]]
                    avg_crash = sum(crash_scores) / n_vehicles
                    max_crash = max(crash_scores)

                # Composite score: prioritize crash evidence quality over quantity.
                # max_crash (dominant) ensures the frame with the clearest collision
                # wins; vehicle count is capped at 2 to prevent large M4 zones
                # from biasing toward frames with many bystander vehicles.
                score = (max_crash * 0.7 + avg_crash * 0.3
                         + (n_victims * 0.15)
                         + (min(n_vehicles, 2) * 0.05))

                # M3 temporal bonus: boost score if temporal classifier agrees
                if (m3_score >= settings.TEMPORAL_SCORE_THRESHOLD
                        and self.temporal_classifier.is_available()):
                    score += m3_score * settings.TEMPORAL_SCORE_WEIGHT

                # Add to sliding window (only if classifier isn't strongly disagreeing)
                # This prevents normal traffic perspective overlaps from accumulating
                # when the classifier says "this is NOT an accident".
                # Gate raised from 0.15→0.30: at 0.15, frames where M1 says
                # "85% NOT accident" were accumulating temporal confirmations.
                # EXCEPTION: When M4 found collision zones, always allow
                # (M4 detects real accidents even when M1 gives near-zero).
                m4_override = (hasattr(result, '__contains__')
                               and len(result.get("involved_vehicles", [])) > 0
                               and self.processor.collision_detector.is_available())
                if cls_conf >= 0.30 or n_victims > 0 or m4_override:
                    detection_window.append((t_sec, score, result, n_vehicles))
                    if len(detection_window) > settings.TEMPORAL_WINDOW_SIZE:
                        detection_window.pop(0)

                status = (
                    f"  [{t_sec}s] Detection: "
                    f"{n_vehicles} vehicle(s), {n_victims} victim(s) "
                    f"(score={score:.3f}, cls={cls_conf:.3f}"
                    f"{f', m3={m3_score:.3f}' if m3_score > 0 else ''}) "
                    f"[{len(detection_window)}/{confirm_count}]"
                )
                print(status)

                # Check temporal consistency
                if (len(detection_window) >= confirm_count
                        and not confirmed):

                    # === Vehicle count consistency check ===
                    # Ensure detections in the window are consistent.
                    # If the vehicle count varies wildly, it's likely noise.
                    vehicle_counts = [w[3] for w in detection_window]
                    avg_veh_count = np.mean(vehicle_counts)

                    # For single-vehicle detections, require extra confirmation
                    effective_confirm = confirm_count
                    if avg_veh_count < 1.5:
                        effective_confirm = max(confirm_count, 3)

                    if len(detection_window) >= effective_confirm:
                        confirmed = True
                        confirmation_time = t_sec

                        # Pick best result from confirmed window
                        for w_time, w_score, w_result, _ in detection_window:
                            if w_score > best_score:
                                best_score = w_score
                                best_result = w_result
                                accident_timestamp = w_time

                        print(f"  [{t_sec}s] *** CONFIRMED *** "
                              f"(best at {accident_timestamp}s, "
                              f"score={best_score:.3f})")

                # Post-crash: keep looking for better aftermath frame
                if confirmed:
                    if score > best_score:
                        best_score = score
                        best_result = result
                        accident_timestamp = t_sec
                        print(f"  [{t_sec}s] Better frame "
                              f"(score={score:.3f})")

                    # Check if we've exceeded post-crash lookahead
                    time_since_confirm = t_sec - confirmation_time
                    if (time_since_confirm >= settings.POST_CRASH_LOOKAHEAD_SEC
                            and stop_on_first):
                        print(f"  [{t_sec}s] Post-crash scan complete")
                        break

            else:
                # No detection in this frame
                if detection_window:
                    detection_window.pop(0)

                # If confirmed, track how long since last detection
                if confirmed:
                    time_since_confirm = t_sec - confirmation_time
                    if (time_since_confirm >= settings.POST_CRASH_LOOKAHEAD_SEC
                            and stop_on_first):
                        print(f"  [{t_sec}s] Post-crash scan complete")
                        break

                # Alert mode timeout: reset if too long without detection
                if alert_mode and not confirmed:
                    time_in_alert = t_sec - alert_start_sec
                    if time_in_alert > 6.0 and not detection_window:
                        alert_mode = False
                        alert_start_sec = None
                        # Set cooldown: don't re-alert for 3 seconds
                        alert_cooldown_until = t_sec + 3.0
                        print(f"  [{t_sec}s] Alert expired, "
                              f"cooldown until {alert_cooldown_until:.1f}s")

        cap.release()

        # Final output
        if best_result is not None:
            print()
            print("=" * 50)
            print(f"  ACCIDENT CONFIRMED at {accident_timestamp}s")
            print(f"  Vehicles: {len(best_result['involved_vehicles'])}")
            print(f"  Victims: {len(best_result['victims'])}")
            print("=" * 50)

            saved = self.processor.report_generator.save(best_result)
            print(f"  Annotated frame: {saved['frame_path']}")
            print(f"  JSON report: {saved['report_path']}")

            if show_result:
                display_frame(best_result["annotated_frame"],
                              f"Accident at {accident_timestamp}s")

            print()
            print(json.dumps(best_result["report"], indent=2, default=str))

        else:
            print()
            print("[VideoProcessor] Video complete. No accidents detected.")

        return {
            "accident_detected": best_result is not None,
            "best_result": best_result,
            "timestamp_sec": accident_timestamp,
            "frames_analyzed": frames_analyzed,
            "temporal_score_m3": round(m3_score, 3),
        }

    def _compute_motion_score(self, frame, prev_gray):
        """
        Compute motion intensity between current and previous frame.
        Returns mean absolute pixel difference.
        """
        if prev_gray is None:
            return 0.0

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, curr_gray)
        return float(np.mean(diff))

    def _is_motion_spike(self, current_score, history):
        """
        Detect if current motion score is a significant spike
        relative to the recent baseline.

        A spike means sudden scene change (crash, vehicle collision).
        Normal traffic has gradual, consistent motion.

        Requires BOTH absolute AND relative threshold to reduce
        false triggers in high-motion scenes.
        """
        if len(history) < 5:
            return False  # Need baseline

        # Compute baseline stats
        baseline = history[:-1]  # Exclude current
        mean_baseline = np.mean(baseline)
        std_baseline = max(np.std(baseline), 1.0)  # Avoid div by zero

        # Spike if current >> baseline (both conditions must be true)
        absolute_spike = current_score > settings.MOTION_SPIKE_THRESHOLD
        relative_spike = current_score > (mean_baseline + 2.5 * std_baseline)

        return absolute_spike and relative_spike

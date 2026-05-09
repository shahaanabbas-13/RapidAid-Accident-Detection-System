"""
RapidAid — Report Generator

Builds structured JSON reports and annotated frames from analysis results.
This module is the final stage of the pipeline.
"""
import json
from utils.helpers import get_timestamp, save_annotated_frame, save_report
from utils.visualization import annotate_frame


class ReportGenerator:
    """Generates JSON reports and annotated frames from pipeline results."""

    def generate(self, frame, involved_vehicles, victims, zone,
                 source_path=None, timestamp_sec=None,
                 classifier_confidence=None, max_damage_score=None,
                 fused_confidence=None, scene_only=False):
        """
        Generate a complete report from analysis results.

        Args:
            frame: original BGR frame (numpy array)
            involved_vehicles: list of involved vehicle dicts
            victims: list of victim dicts
            zone: accident zone [x1, y1, x2, y2] or None
            source_path: path to source image/video
            timestamp_sec: timestamp in seconds (for video)
            classifier_confidence: M1 scene classifier confidence (0-1)
            max_damage_score: M2 max damage score across vehicles (0-1)
            fused_confidence: ensemble fused confidence (0-1)
            scene_only: True if detection is based on M1 alone (no vehicles)

        Returns:
            dict with keys:
                - 'report': JSON-serializable report dict
                - 'annotated_frame': annotated BGR frame (numpy array)
                - 'accident_detected': bool
        """
        accident_detected = (
            len(involved_vehicles) > 0 or len(victims) > 0 or scene_only
        )

        # Build the JSON report
        report = self._build_report(
            involved_vehicles, victims, zone,
            source_path, timestamp_sec, accident_detected,
            classifier_confidence=classifier_confidence,
            max_damage_score=max_damage_score,
            fused_confidence=fused_confidence,
            scene_only=scene_only,
        )

        # Build annotated frame
        ts_label = f"{timestamp_sec}" if timestamp_sec is not None else None
        annotated = annotate_frame(
            frame, involved_vehicles, victims, zone,
            timestamp=ts_label,
            classifier_confidence=classifier_confidence,
            scene_only=scene_only,
        )

        return {
            "report": report,
            "annotated_frame": annotated,
            "accident_detected": accident_detected,
        }

    def save(self, result, frame_filename=None, report_filename=None):
        """
        Save annotated frame and JSON report to disk.

        Args:
            result: dict from generate()
            frame_filename: optional filename for annotated frame
            report_filename: optional filename for JSON report

        Returns:
            dict with 'frame_path' and 'report_path'
        """
        frame_path = save_annotated_frame(
            result["annotated_frame"], frame_filename
        )
        report_path = save_report(result["report"], report_filename)

        # Update report with saved paths
        result["report"]["annotated_frame"] = frame_path
        save_report(result["report"], report_filename)

        return {
            "frame_path": frame_path,
            "report_path": report_path,
        }

    def _build_report(self, involved_vehicles, victims, zone,
                       source_path, timestamp_sec, accident_detected,
                       classifier_confidence=None, max_damage_score=None,
                       fused_confidence=None, scene_only=False):
        """Build the structured JSON report dict."""
        report = {
            "accident_detected": accident_detected,
            "timestamp": get_timestamp(),
            "source_frame": source_path or "unknown",
        }

        if timestamp_sec is not None:
            report["timestamp_sec"] = timestamp_sec

        # Classifier scores
        if classifier_confidence is not None:
            report["classifier_confidence_m1"] = round(classifier_confidence, 3)
        if max_damage_score is not None:
            report["damage_score_m2"] = round(max_damage_score, 3)
        if fused_confidence is not None:
            report["fused_confidence"] = round(fused_confidence, 3)
        if scene_only:
            report["detection_mode"] = "scene_only"
        else:
            report["detection_mode"] = "ensemble"

        # Vehicle summary
        vehicle_list = []
        for v in involved_vehicles:
            veh_entry = {
                "type": v.get("display_name", v.get("type", "Unknown")),
                "confidence": v.get("confidence", 0),
                "crash_score": v.get("crash_score", 0),
                "bbox": v["bbox"],
            }
            if "damage_score" in v:
                veh_entry["damage_score"] = v["damage_score"]
            vehicle_list.append(veh_entry)

        report["total_vehicles_in_scene"] = len(involved_vehicles)
        report["involved_vehicles"] = vehicle_list

        # Accident zone
        report["accident_zone"] = zone

        # Victim summary
        victim_list = []
        for i, victim in enumerate(victims):
            victim_list.append({
                "id": i + 1,
                "confidence": victim.get("confidence", 0),
                "status": victim.get("status", "unknown"),
                "bbox": victim["bbox"],
            })

        report["victims_in_zone"] = len(victims)
        report["victims"] = victim_list

        # Annotated frame path (will be filled after save)
        report["annotated_frame"] = None

        return report

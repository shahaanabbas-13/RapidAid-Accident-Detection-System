"""
RapidAid — Visualization Utilities

Functions for drawing annotations on frames:
- Involved vehicle bounding boxes and labels
- Victim bounding boxes and labels
- Accident zone rectangle
- Status banner
"""
import cv2
import numpy as np
from config import settings


def draw_banner(frame, text, is_accident=True):
    """
    Draw a status banner at the top of the frame.

    Args:
        frame: numpy array (BGR image)
        text: banner text
        is_accident: True for red alert, False for green safe
    """
    h, w = frame.shape[:2]
    color = settings.COLORS["banner_alert"] if is_accident else settings.COLORS["banner_safe"]

    # Draw filled rectangle for banner
    cv2.rectangle(frame, (0, 0), (w, settings.BANNER_HEIGHT), color, -1)

    # Draw text with shadow for readability
    text_color = settings.COLORS["text"]
    cv2.putText(frame, text, (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, settings.FONT_SCALE_LARGE,
                settings.COLORS["text_shadow"], settings.FONT_THICKNESS + 1)
    cv2.putText(frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, settings.FONT_SCALE_LARGE,
                text_color, settings.FONT_THICKNESS)


def draw_vehicle(frame, vehicle, index=None):
    """
    Draw involved vehicle annotation on the frame.

    Args:
        frame: numpy array (BGR image)
        vehicle: dict with keys 'bbox', 'type', 'confidence', 'polygon'
        index: optional vehicle index number
    """
    color = settings.COLORS["involved_vehicle"]
    bbox = vehicle["bbox"]
    x1, y1, x2, y2 = bbox

    # Draw segmentation polygon if available
    polygon = vehicle.get("polygon", None)
    if polygon is not None and len(polygon) > 0:
        poly = np.int32([polygon])
        cv2.polylines(frame, poly, True, color, 3)

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Draw label background
    label = vehicle.get("display_name", vehicle.get("type", "Vehicle"))
    conf = vehicle.get("confidence", 0)
    dmg = vehicle.get("damage_score", 0)
    if index is not None:
        label_text = f"{label} #{index+1} ({conf:.0f}%)"
    else:
        label_text = f"{label} ({conf:.0f}%)"

    # Append damage score if M2 detected damage
    if dmg > 0.5:
        label_text += f" DMG:{dmg*100:.0f}%"

    text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX,
                                 settings.FONT_SCALE_SMALL, settings.FONT_THICKNESS)[0]
    label_w = text_size[0] + 10
    label_h = 25

    cv2.rectangle(frame, (x1, y1 - label_h), (x1 + label_w, y1), color, -1)
    cv2.putText(frame, label_text, (x1 + 5, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, settings.FONT_SCALE_SMALL,
                settings.COLORS["text"], settings.FONT_THICKNESS)


def draw_victim(frame, victim, index):
    """
    Draw victim annotation on the frame.

    Args:
        frame: numpy array (BGR image)
        victim: dict with keys 'bbox', 'confidence', 'status'
        index: victim number (1-based for display)
    """
    color = settings.COLORS["victim"]
    bbox = victim["bbox"]
    x1, y1, x2, y2 = bbox

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    # Draw label
    status = victim.get("status", "victim")
    label_text = f"Victim {index} ({status})"
    text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX,
                                 settings.FONT_SCALE_SMALL, settings.FONT_THICKNESS)[0]
    label_w = text_size[0] + 10
    label_h = 25

    cv2.rectangle(frame, (x1, y1 - label_h), (x1 + label_w, y1), color, -1)
    cv2.putText(frame, label_text, (x1 + 5, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, settings.FONT_SCALE_SMALL,
                settings.COLORS["text"], settings.FONT_THICKNESS)


def draw_accident_zone(frame, zone):
    """
    Draw accident zone rectangle on the frame.

    Args:
        frame: numpy array (BGR image)
        zone: [x1, y1, x2, y2]
    """
    if zone is None:
        return

    color = settings.COLORS["accident_zone"]
    x1, y1, x2, y2 = zone

    # Draw dashed-effect rectangle (thick, partially transparent)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    # Draw zone label
    label = "ACCIDENT ZONE"
    cv2.putText(frame, label, (x1 + 5, y1 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, settings.FONT_SCALE_MEDIUM,
                color, settings.FONT_THICKNESS)


def annotate_frame(frame, involved_vehicles, victims, zone, timestamp=None,
                   classifier_confidence=None, scene_only=False):
    """
    Full annotation pipeline for a single frame.

    Args:
        frame: numpy array (BGR image) — will be modified in place
        involved_vehicles: list of vehicle dicts
        victims: list of victim dicts
        zone: [x1, y1, x2, y2] or None
        timestamp: optional timestamp string
        classifier_confidence: M1 scene classifier confidence (0-1)
        scene_only: True if detection is from M1 alone (no vehicles)

    Returns:
        numpy array: annotated frame
    """
    output = frame.copy()

    # 1. Draw accident zone (bottom layer)
    if zone is not None:
        draw_accident_zone(output, zone)

    # 2. Draw involved vehicles
    for i, v in enumerate(involved_vehicles):
        draw_vehicle(output, v, index=i)

    # 3. Draw victims
    for i, victim in enumerate(victims):
        draw_victim(output, victim, index=i + 1)

    # 4. Draw banner
    n_vehicles = len(involved_vehicles)
    n_victims = len(victims)
    cls_str = ""
    if classifier_confidence is not None and classifier_confidence > 0:
        cls_str = f" | M1:{classifier_confidence*100:.0f}%"

    if scene_only:
        ts_str = f" @ {timestamp}s" if timestamp else ""
        banner_text = f"ACCIDENT DETECTED (Scene) | M1:{classifier_confidence*100:.0f}%{ts_str}"
        draw_banner(output, banner_text, is_accident=True)
    elif n_vehicles > 0 or n_victims > 0:
        ts_str = f" @ {timestamp}s" if timestamp else ""
        banner_text = f"ACCIDENT DETECTED{cls_str} | {n_vehicles} Vehicle(s) | {n_victims} Victim(s){ts_str}"
        draw_banner(output, banner_text, is_accident=True)
    else:
        draw_banner(output, "MONITORING - No Accident Detected", is_accident=False)

    return output

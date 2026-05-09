"""
RapidAid — Geometry Utilities

Core geometric functions for collision detection, distance computation,
bounding box operations, and polygon analysis.
"""
import math
import numpy as np
import cv2


def compute_iou(box_a, box_b):
    """
    Compute Intersection over Union (IoU) for two bounding boxes.

    Args:
        box_a: [x1, y1, x2, y2]
        box_b: [x1, y1, x2, y2]

    Returns:
        float: IoU value between 0 and 1
    """
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
    if inter_area == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def compute_overlap_ratio(box_a, box_b):
    """
    Compute what fraction of the smaller box is overlapped by the larger.
    More useful than IoU for detecting trapped persons inside vehicles.

    Args:
        box_a: [x1, y1, x2, y2]
        box_b: [x1, y1, x2, y2]

    Returns:
        float: Overlap ratio (intersection / min_area)
    """
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
    if inter_area == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    min_area = min(area_a, area_b)

    if min_area <= 0:
        return 0.0

    return inter_area / min_area


def compute_edge_distance(box_a, box_b):
    """
    Compute minimum edge-to-edge distance between two bounding boxes.
    Returns 0 if boxes overlap.

    Args:
        box_a: [x1, y1, x2, y2]
        box_b: [x1, y1, x2, y2]

    Returns:
        float: Minimum edge distance (0 if overlapping)
    """
    x1_a, y1_a, x2_a, y2_a = box_a
    x1_b, y1_b, x2_b, y2_b = box_b

    left = x2_b < x1_a
    right = x1_b > x2_a
    bottom = y2_b < y1_a
    top = y1_b > y2_a

    if top and left:
        return math.hypot(x1_a - x2_b, y2_a - y1_b)
    elif left and bottom:
        return math.hypot(x1_a - x2_b, y1_a - y2_b)
    elif bottom and right:
        return math.hypot(x2_a - x1_b, y1_a - y2_b)
    elif right and top:
        return math.hypot(x2_a - x1_b, y2_a - y1_b)
    elif left:
        return x1_a - x2_b
    elif right:
        return x1_b - x2_a
    elif bottom:
        return y1_a - y2_b
    elif top:
        return y1_b - y2_a

    return 0.0  # Overlapping


def compute_diagonal(box):
    """Compute diagonal length of a bounding box."""
    return math.hypot(box[2] - box[0], box[3] - box[1])


def compute_box_area(box):
    """Compute area of a bounding box."""
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def compute_box_center(box):
    """Compute center point of a bounding box."""
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    return (cx, cy)


def compute_aspect_ratio(box):
    """Compute width/height aspect ratio of a bounding box."""
    w = box[2] - box[0]
    h = box[3] - box[1]
    if h <= 0:
        return 0.0
    return w / h


def check_pixel_collision(poly1, poly2, img_shape):
    """
    Check if two segmentation polygons overlap at the pixel level.

    Args:
        poly1: numpy array of polygon points for object 1
        poly2: numpy array of polygon points for object 2
        img_shape: (height, width, channels) of the frame

    Returns:
        tuple: (bool: collision detected, float: overlap ratio)
    """
    if len(poly1) == 0 or len(poly2) == 0:
        return False, 0.0

    mask1 = np.zeros(img_shape[:2], dtype=np.uint8)
    mask2 = np.zeros(img_shape[:2], dtype=np.uint8)

    cv2.fillPoly(mask1, [np.int32(poly1)], 1)
    cv2.fillPoly(mask2, [np.int32(poly2)], 1)

    inter_area = cv2.countNonZero(cv2.bitwise_and(mask1, mask2))
    if inter_area == 0:
        return False, 0.0

    area1 = cv2.countNonZero(mask1)
    area2 = cv2.countNonZero(mask2)
    min_area = min(area1, area2)

    if min_area <= 0:
        return False, 0.0

    overlap_ratio = inter_area / (min_area + 1e-6)
    return True, overlap_ratio


def compute_relative_angle(box_a, box_b):
    """
    Compute the angle (in degrees) between the centers of two bounding boxes.
    Used to detect if vehicles are at unusual angles to each other.

    Returns:
        float: Angle in degrees (0-180)
    """
    center_a = compute_box_center(box_a)
    center_b = compute_box_center(box_b)

    dx = center_b[0] - center_a[0]
    dy = center_b[1] - center_a[1]

    angle = abs(math.degrees(math.atan2(dy, dx)))
    return angle


def point_in_box(point, box):
    """
    Check if a point is inside a bounding box.

    Args:
        point: (x, y)
        box: [x1, y1, x2, y2]

    Returns:
        bool
    """
    return (box[0] <= point[0] <= box[2]) and (box[1] <= point[1] <= box[3])


def expand_box(box, padding_x, padding_y, max_w, max_h):
    """
    Expand a bounding box by padding, clamped to frame bounds.

    Args:
        box: [x1, y1, x2, y2]
        padding_x: horizontal padding in pixels
        padding_y: vertical padding in pixels
        max_w: frame width
        max_h: frame height

    Returns:
        list: expanded [x1, y1, x2, y2]
    """
    return [
        max(0, int(box[0] - padding_x)),
        max(0, int(box[1] - padding_y)),
        min(max_w, int(box[2] + padding_x)),
        min(max_h, int(box[3] + padding_y)),
    ]


def merge_boxes(boxes):
    """
    Compute the bounding box that encloses all given boxes.

    Args:
        boxes: list of [x1, y1, x2, y2]

    Returns:
        list: merged [x1, y1, x2, y2] or None if empty
    """
    if not boxes:
        return None

    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)

    return [x1, y1, x2, y2]

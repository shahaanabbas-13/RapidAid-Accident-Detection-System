"""
RapidAid — General Helper Utilities

File I/O, timestamp generation, frame saving, and other common operations.
"""
import os
import cv2
import json
from datetime import datetime
from config import settings


def get_timestamp():
    """Get current timestamp string for reports."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def save_annotated_frame(frame, filename=None):
    """
    Save an annotated frame to the outputs directory.

    Args:
        frame: numpy array (BGR image)
        filename: optional filename (auto-generated if None)

    Returns:
        str: path to saved file
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"accident_{timestamp}.jpg"

    filepath = os.path.join(settings.ANNOTATED_DIR, filename)
    cv2.imwrite(filepath, frame)
    return filepath


def save_report(report, filename=None):
    """
    Save a JSON report to the outputs directory.

    Args:
        report: dict containing the report data
        filename: optional filename (auto-generated if None)

    Returns:
        str: path to saved file
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"report_{timestamp}.json"

    filepath = os.path.join(settings.REPORTS_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(report, f, indent=2, default=str)
    return filepath


def load_frame(path):
    """
    Load a frame from file path.

    Args:
        path: path to image file

    Returns:
        numpy array (BGR image) or None if failed
    """
    frame = cv2.imread(path)
    if frame is None:
        print(f"[ERROR] Cannot load frame: {path}")
    return frame


def get_frame_dimensions(frame):
    """Get (height, width) of a frame."""
    return frame.shape[:2]


def compute_frame_area(frame):
    """Get total pixel area of a frame."""
    h, w = frame.shape[:2]
    return h * w


def display_frame(frame, title="RapidAid", wait=0):
    """
    Display a frame using matplotlib (works in both local and Colab).

    Args:
        frame: numpy array (BGR image)
        title: window title
        wait: milliseconds to wait (0 = wait for key)
    """
    try:
        import matplotlib.pyplot as plt
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(14, 8))
        plt.imshow(rgb)
        plt.axis("off")
        plt.title(title)
        plt.tight_layout()
        plt.show()
    except ImportError:
        cv2.imshow(title, frame)
        cv2.waitKey(wait)
        cv2.destroyAllWindows()

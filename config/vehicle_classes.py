"""
RapidAid — Vehicle Class Mappings

Maps COCO class IDs to RapidAid vehicle categories.
Each vehicle type has specific deformation thresholds and properties.
"""

# COCO class IDs that are vehicles
VEHICLE_CLASS_IDS = {2, 3, 5, 6, 7}

# COCO ID → RapidAid label mapping
COCO_TO_RAPIDAID = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    6: "train",
    7: "truck",
}

# RapidAid label → Display name
DISPLAY_NAMES = {
    "car": "Car",
    "motorcycle": "Motorcycle",
    "bus": "Heavy Vehicle",
    "truck": "Heavy Vehicle",
    "train": "Industrial Vehicle",
}

# Normal aspect ratio ranges (width/height) for each vehicle type
# Vehicles outside these ranges may be deformed/crashed
NORMAL_ASPECT_RATIOS = {
    "car":        {"min": 0.6,  "max": 2.8},
    "motorcycle":  {"min": 0.3,  "max": 2.5},
    "bus":         {"min": 0.4,  "max": 4.5},
    "truck":       {"min": 0.4,  "max": 4.0},
    "train":       {"min": 0.3,  "max": 5.0},
}

# Typical size ranges (relative to frame area) for foreground vehicles
# Used to filter out very distant or very close (partial) vehicles
VEHICLE_SIZE_RANGE = {
    "car":        {"min": 0.008, "max": 0.45},
    "motorcycle":  {"min": 0.005, "max": 0.35},
    "bus":         {"min": 0.025, "max": 0.55},
    "truck":       {"min": 0.020, "max": 0.50},
    "train":       {"min": 0.020, "max": 0.60},
}


def get_rapidaid_label(coco_class_id: int) -> str:
    """Convert COCO class ID to RapidAid vehicle label."""
    return COCO_TO_RAPIDAID.get(coco_class_id, "unknown")


def get_display_name(rapidaid_label: str) -> str:
    """Get human-readable display name for a vehicle type."""
    return DISPLAY_NAMES.get(rapidaid_label, rapidaid_label.title())


def is_vehicle_class(coco_class_id: int) -> bool:
    """Check if a COCO class ID corresponds to a vehicle."""
    return coco_class_id in VEHICLE_CLASS_IDS

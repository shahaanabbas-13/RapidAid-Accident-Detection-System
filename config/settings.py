"""
RapidAid — Centralized Configuration

All tunable thresholds, paths, and constants in one place.
Adjust these values to fine-tune detection accuracy.
"""
import os

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
ANNOTATED_DIR = os.path.join(OUTPUTS_DIR, "annotated")
REPORTS_DIR = os.path.join(OUTPUTS_DIR, "reports")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Model weight files
VEHICLE_MODEL = os.path.join(WEIGHTS_DIR, "yolov8s-seg.pt")
POSE_MODEL = os.path.join(WEIGHTS_DIR, "yolov8s-pose.pt")
ACCIDENT_CLASSIFIER_MODEL = os.path.join(WEIGHTS_DIR, "accident_classifier.pt")
DAMAGE_CLASSIFIER_MODEL = os.path.join(WEIGHTS_DIR, "damage_classifier.pt")
TEMPORAL_CLASSIFIER_MODEL = os.path.join(WEIGHTS_DIR, "temporal_classifier.pt")

# ============================================================
# VEHICLE DETECTION THRESHOLDS
# ============================================================
# Minimum confidence for vehicle detection
VEHICLE_CONFIDENCE_THRESHOLD = 0.15

# Minimum vehicle bounding box area as fraction of frame area
# Vehicles smaller than this are considered background/distant
MIN_VEHICLE_AREA_RATIO = 0.005

# Maximum vehicle bounding box area as fraction of frame area
# Vehicles larger than this are likely partial/too close
MAX_VEHICLE_AREA_RATIO = 0.55

# Vehicles in the top X% of frame are likely distant background
BACKGROUND_Y_THRESHOLD_RATIO = 0.15

# ============================================================
# ACCIDENT CLASSIFICATION THRESHOLDS
# ============================================================
# Pixel-level mask overlap threshold for collision detection
MASK_OVERLAP_THRESHOLD = 0.12

# Bounding box IoU threshold for overlap detection
BBOX_IOU_THRESHOLD = 0.10

# Edge proximity: distance / avg_diagonal ratio
# If vehicles are closer than this ratio of their average size, suspicious
PROXIMITY_RATIO_THRESHOLD = 0.20

# Minimum crash score to classify a vehicle pair as involved
CRASH_SCORE_THRESHOLD = 0.35

# Classifier-geometric fusion: modulate crash threshold based on classifier confidence
CLASSIFIER_HIGH_CONFIDENCE = 0.80
CLASSIFIER_LOW_CONFIDENCE = 0.30
CRASH_SCORE_BOOST_FACTOR = 0.80    # Multiply threshold when classifier is high confidence
CRASH_SCORE_PENALTY_FACTOR = 1.20  # Multiply threshold when classifier is low confidence

# Maximum crash score (clamp all crash_scores to this value)
MAX_CRASH_SCORE = 1.0

# Single-vehicle crash: allow crash detection with only 1 vehicle if classifier is very confident
SINGLE_VEHICLE_CLASSIFIER_THRESHOLD = 0.85
SINGLE_VEHICLE_DEFORMATION_THRESHOLD = 0.35

# Scene-only detection: M1 confidence to flag accident when NO vehicles are detected
# (handles smoke/debris scenes where YOLO can't see anything)
SCENE_ONLY_CLASSIFIER_THRESHOLD = 0.85

# Minimum confidence for BOTH vehicles in a pair to be considered
MIN_PAIR_CONFIDENCE = 25.0

# ============================================================
# M2 DAMAGE CLASSIFIER TRUST GATE
# ============================================================
# CRITICAL: Set to False until M2 is retrained with Indian traffic data.
# The current M2 gives 96% "damaged" to normal Indian taxis/trucks.
# When False: M2 scores are computed but NOT used to boost crash scores
# or trigger damage-dominant fallback detection.
M2_TRUST_ENABLED = True

# Damage-only accident detection (stage 5g): require strong scene evidence
DAMAGE_ONLY_MIN_SCENE_CONF = 0.60     # M1 must agree at 60%+
DAMAGE_ONLY_MIN_DAMAGE = 0.80          # M2 must be very confident

# Damage-dominant detection (stage 5g-ii): DISABLED when M2 untrusted
# Even when enabled, require strong scene evidence
DAMAGE_DOMINANT_MIN_SCENE_CONF = 0.65  # Was effectively 0.20
DAMAGE_DOMINANT_MIN_DAMAGE = 0.95       # Was 0.90

# Minimum M1 confidence for victim-vehicle fallback detections (stages 5d, 5f)
VICTIM_VEHICLE_FALLBACK_MIN_SCENE_CONF = 0.40

# Weights for crash score components
CRASH_SCORE_WEIGHTS = {
    "mask_overlap":     0.25,
    "bbox_iou":         0.30,
    "edge_proximity":   0.15,
    "deformation":      0.10,
    "relative_angle":   0.05,
    "scene_position":   0.05,
    "size_mismatch":    0.10,
}

# ============================================================
# VICTIM CLASSIFICATION THRESHOLDS
# ============================================================
# Minimum confidence for person detection
PERSON_CONFIDENCE_THRESHOLD = 0.10

# Person standing ratio: if height > STANDING_RATIO * width → standing
STANDING_RATIO = 1.1

# Minimum keypoint confidence to use a keypoint
KEYPOINT_CONFIDENCE_THRESHOLD = 0.25

# Minimum number of victims lying on ground to detect victim-only accident
VICTIM_ONLY_MIN_COUNT = 1

# Lying ratio: if width > LYING_RATIO * height → person is lying down
LYING_RATIO = 0.75

# Person-vehicle overlap threshold for "trapped" classification
TRAPPED_OVERLAP_THRESHOLD = 0.25

# Horizontal body angle threshold (degrees from horizontal)
# If body angle is within this many degrees of horizontal → lying
LYING_ANGLE_THRESHOLD = 35.0

# Maximum distance from accident zone center for victim consideration
# (as multiple of zone diagonal)
VICTIM_ZONE_PROXIMITY = 1.2

# Maximum pixel distance between vehicle and victim to flag vehicle as involved
VEHICLE_VICTIM_DISTANCE = 80

# ============================================================
# ACCIDENT ZONE COMPUTATION
# ============================================================
# Base padding around accident zone (fraction of frame dimension)
ZONE_PADDING_RATIO = 0.05

# Minimum zone size (fraction of frame dimension)
MIN_ZONE_SIZE_RATIO = 0.10

# Maximum zone size (fraction of frame dimension)
MAX_ZONE_SIZE_RATIO = 0.85

# ============================================================
# VIDEO PROCESSING
# ============================================================
# Frames to analyze per second of video
FRAMES_PER_SECOND_TO_ANALYZE = 3

# Number of frames to look ahead after initial detection
# to find the best (highest confidence) accident frame
LOOKAHEAD_FRAMES = 10

# Frame difference threshold for motion spike detection
MOTION_SPIKE_THRESHOLD = 14.0

# Temporal consistency: require accident detection in at least
# this many frames within a sliding window before confirming
TEMPORAL_CONFIRM_COUNT = 2
TEMPORAL_WINDOW_SIZE = 8

# Classifier-based alert for video processing
CLASSIFIER_ALERT_THRESHOLD = 0.55      # Smoothed confidence to trigger alert mode
CLASSIFIER_SMOOTHING_WINDOW = 5        # Frames for rolling average

# Post-crash analysis: continue scanning after first confirmation
# to find the best aftermath frame with most visible evidence
POST_CRASH_LOOKAHEAD_SEC = 3.0

# ============================================================
# ENSEMBLE FUSION WEIGHTS
# ============================================================
# Weights for combining geometric, M1 (scene), and M2 (damage) scores
# into a unified fused confidence. Must sum to 1.0.
ENSEMBLE_WEIGHT_GEOMETRIC = 0.40
ENSEMBLE_WEIGHT_SCENE = 0.35      # M1 scene classifier
ENSEMBLE_WEIGHT_DAMAGE = 0.25     # M2 damage classifier

# When geometric signal is absent (M1+M2 only)
ENSEMBLE_NO_GEO_WEIGHT_SCENE = 0.55
ENSEMBLE_NO_GEO_WEIGHT_DAMAGE = 0.45

# M3 temporal classifier weight for video processing
# Applied as a bonus multiplier on top of frame-level fused confidence
TEMPORAL_SCORE_WEIGHT = 0.15          # Bonus weight for M3 temporal signal
TEMPORAL_SCORE_THRESHOLD = 0.60       # Min M3 score to count as temporal confirmation

# ============================================================
# VISUALIZATION / ANNOTATION
# ============================================================
# Colors in BGR format for OpenCV
COLORS = {
    "involved_vehicle": (0, 0, 220),      # Red
    "victim":           (0, 140, 255),     # Orange
    "accident_zone":    (255, 100, 0),     # Blue
    "banner_alert":     (0, 0, 200),       # Red banner
    "banner_safe":      (0, 160, 0),       # Green banner
    "text":             (255, 255, 255),   # White
    "text_shadow":      (0, 0, 0),         # Black
}

# Banner height in pixels
BANNER_HEIGHT = 45

# Font settings
FONT_SCALE_LARGE = 0.8
FONT_SCALE_MEDIUM = 0.6
FONT_SCALE_SMALL = 0.5
FONT_THICKNESS = 2

# ============================================================
# ENSURE DIRECTORIES EXIST
# ============================================================
for _dir in [WEIGHTS_DIR, OUTPUTS_DIR, ANNOTATED_DIR, REPORTS_DIR, DATA_DIR]:
    os.makedirs(_dir, exist_ok=True)

"""Diagnose Acc Video 9 failure."""
# pyrefly: ignore [missing-import]
import cv2
from models.frame_classifier import FrameClassifier
from models.collision_detector import CollisionDetector
from models.vehicle_detector import VehicleDetector

fc = FrameClassifier()
cd = CollisionDetector()
vd = VehicleDetector()

cap = cv2.VideoCapture(r"data\test_videos\Acc Video 9.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Acc Video 9: {total} frames, {fps}fps, {total/fps:.1f}s")

for i in range(0, total, 10):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    if not ret:
        break
    t = i / fps
    _, cls = fc.classify(frame)
    zones = cd.detect(frame)
    vehs = vd.detect(frame, filter_background=True)
    all_v = vd.detect(frame, filter_background=False)
    if zones:
        zc = zones[0]["confidence"]
        zone_info = f"conf={zc:.3f}"
    else:
        zone_info = "none"
    print(f"  @{t:5.1f}s: M1={cls:.3f} M4={zone_info} "
          f"vehs_filtered={len(vehs)} vehs_raw={len(all_v)}")

cap.release()

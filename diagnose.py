"""Diagnose why test frames and videos are failing."""
import cv2
import os
from models.frame_classifier import FrameClassifier
from models.collision_detector import CollisionDetector
from models.damage_classifier import DamageClassifier
from models.vehicle_detector import VehicleDetector

fc = FrameClassifier()
cd = CollisionDetector()
dc = DamageClassifier()
vd = VehicleDetector()

print("=" * 60)
print("  FRAME DIAGNOSIS")
print("=" * 60)

frames_dir = r"data\test_frames"
for fname in sorted(os.listdir(frames_dir)):
    fpath = os.path.join(frames_dir, fname)
    frame = cv2.imread(fpath)
    if frame is None:
        print(f"{fname}: FAILED TO READ")
        continue
    h, w = frame.shape[:2]

    # M1
    is_acc, cls_conf = fc.classify(frame)

    # M4
    zones = cd.detect(frame)
    zone_confs = [z["confidence"] for z in zones]

    # Vehicles
    vehicles = vd.detect(frame, filter_background=True)
    veh_types = [v["type"] for v in vehicles]

    # Also check without background filter
    all_vehicles = vd.detect(frame, filter_background=False)

    # M2 damage
    dmg_scores = []
    for v in vehicles:
        dmg = dc.classify_vehicle(frame, v["bbox"])
        dmg_scores.append(round(dmg, 3))

    print(f"\n{fname} ({w}x{h}):")
    print(f"  M1: is_accident={is_acc}, conf={cls_conf:.3f}")
    print(f"  M4: {len(zones)} zone(s), confs={zone_confs}")
    print(f"  Vehicles (filtered): {len(vehicles)} {veh_types}")
    print(f"  Vehicles (unfiltered): {len(all_vehicles)}")
    print(f"  M2 damage: {dmg_scores}")

    if not vehicles and all_vehicles:
        print(f"  ⚠️  ALL VEHICLES FILTERED OUT! Unfiltered types:")
        for v in all_vehicles:
            print(f"     {v['type']} conf={v['confidence']} "
                  f"area_ratio={v['area_ratio']:.4f} "
                  f"bbox={v['bbox']}")


print("\n" + "=" * 60)
print("  VIDEO FAILURE DIAGNOSIS (3 failing videos)")
print("=" * 60)

# Check first frames of failing videos
failing_videos = ["Acc Video 5.mp4", "Acc Video 9.mp4", "Acc video 3.mp4"]
videos_dir = r"data\test_videos"

for vname in failing_videos:
    vpath = os.path.join(videos_dir, vname)
    if not os.path.exists(vpath):
        print(f"\n{vname}: FILE NOT FOUND")
        continue

    cap = cv2.VideoCapture(vpath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps if fps > 0 else 0

    print(f"\n{vname} ({total} frames, {fps:.0f}fps, {duration:.1f}s):")

    # Sample frames at different points in the video
    sample_points = [0.3, 0.5, 0.7, 0.9]  # 30%, 50%, 70%, 90% through
    for pct in sample_points:
        frame_num = int(total * pct)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue

        t = frame_num / fps
        is_acc, cls_conf = fc.classify(frame)
        zones = cd.detect(frame)
        vehicles = vd.detect(frame, filter_background=True)
        all_vehs = vd.detect(frame, filter_background=False)

        print(f"  @{t:.1f}s ({pct*100:.0f}%): M1={cls_conf:.3f} "
              f"M4_zones={len(zones)} "
              f"vehs={len(vehicles)}/{len(all_vehs)} "
              f"is_acc={is_acc}")

        if zones:
            for z in zones:
                print(f"    Zone: conf={z['confidence']} bbox={z['bbox']}")

    cap.release()

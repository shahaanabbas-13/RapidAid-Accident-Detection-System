"""Debug which vehicles get involved via geometric scoring"""
from pipeline.frame_processor import FrameProcessor
import cv2

fp = FrameProcessor()
frame = cv2.imread('data/test_frames/Car accident(8).png')

# Run full pipeline
result = fp.process(frame, source_path='Car accident(8).png')

print(f"Accident: {result['accident_detected']}")
print(f"Involved vehicles: {len(result['involved_vehicles'])}")
for i, v in enumerate(result['involved_vehicles']):
    vtype = v["type"]
    conf = v["confidence"]
    cs = v.get("crash_score", 0)
    dmg = v.get("damage_score", 0)
    print(f"  [{i+1}] {vtype:15s} conf={conf:5.1f}% crash={cs:.3f} dmg={dmg:.2f} bbox={v['bbox']}")

print(f"\nAll vehicles: {len(result['all_vehicles'])}")
for i, v in enumerate(result['all_vehicles']):
    vtype = v["type"]
    conf = v["confidence"]
    dmg = v.get("damage_score", 0)
    print(f"  [{i+1}] {vtype:15s} conf={conf:5.1f}% dmg={dmg:.2f} bbox={v['bbox']}")

print(f"\nVictims: {len(result['victims'])}")
print(f"Zone: {result['accident_zone']}")

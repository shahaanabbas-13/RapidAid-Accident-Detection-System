"""
RapidAid — Comprehensive Test Suite
Tests all 6 frames + 9 videos and reports accuracy.
"""
import os
import sys
import time
import json
import traceback

# Test configuration
FRAMES_DIR = r"data\test_frames"
VIDEOS_DIR = r"data\test_videos"
OUTPUT_DIR = r"outputs\test_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("  RapidAid — Comprehensive Test Suite")
print("=" * 60)

# Initialize pipeline ONCE
print("\n[1/3] Initializing pipeline...")
start = time.time()
from pipeline.frame_processor import FrameProcessor
from pipeline.video_processor import VideoProcessor

fp = FrameProcessor()
vp = VideoProcessor(frame_processor=fp)
init_time = time.time() - start
print(f"  Pipeline ready in {init_time:.1f}s\n")

# ============================================================
# TEST PART 1: Frame Analysis (6 test frames)
# ============================================================
print("=" * 60)
print("  PART 1: Frame Analysis (6 test frames)")
print("=" * 60)

frame_files = sorted([
    f for f in os.listdir(FRAMES_DIR)
    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
])

frame_results = []
for i, fname in enumerate(frame_files, 1):
    fpath = os.path.join(FRAMES_DIR, fname)
    print(f"\n--- Frame {i}/{len(frame_files)}: {fname} ---")
    
    try:
        start = time.time()
        result = fp.process(fpath)
        elapsed = time.time() - start
        
        detected = result.get("accident_detected", False)
        n_vehicles = len(result.get("involved_vehicles", []))
        n_victims = len(result.get("victims", []))
        
        # Get crash scores
        crash_scores = []
        damage_scores = []
        for v in result.get("involved_vehicles", []):
            crash_scores.append(v.get("crash_score", 0))
            damage_scores.append(v.get("damage_score", 0))
        
        # Get confidence info
        fused_conf = result.get("report", {}).get("confidence", {})
        
        status = "ACCIDENT DETECTED" if detected else "NO ACCIDENT"
        print(f"  Result: {status}")
        print(f"  Vehicles: {n_vehicles}, Victims: {n_victims}")
        if crash_scores:
            print(f"  Crash scores: {crash_scores}")
        if damage_scores:
            print(f"  Damage scores: {damage_scores}")
        print(f"  Time: {elapsed:.2f}s")
        
        frame_results.append({
            "file": fname,
            "detected": detected,
            "vehicles": n_vehicles,
            "victims": n_victims,
            "crash_scores": crash_scores,
            "damage_scores": damage_scores,
            "time_sec": round(elapsed, 2),
        })
        
        # Save annotated frame if accident detected
        if detected and "annotated_frame" in result:
            import cv2
            out_path = os.path.join(OUTPUT_DIR, f"frame_{i}_{fname}")
            cv2.imwrite(out_path, result["annotated_frame"])
            print(f"  Saved: {out_path}")
            
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        frame_results.append({
            "file": fname,
            "detected": False,
            "error": str(e),
        })

# Frame summary
print("\n" + "=" * 60)
print("  FRAME RESULTS SUMMARY")
print("=" * 60)
frame_detected = sum(1 for r in frame_results if r.get("detected", False))
print(f"  Detected accidents: {frame_detected}/{len(frame_results)}")
for r in frame_results:
    status = "✓ DETECTED" if r.get("detected") else "✗ MISSED"
    print(f"  {status} | {r['file']} | "
          f"V:{r.get('vehicles',0)} P:{r.get('victims',0)} | "
          f"CS:{r.get('crash_scores',[])} | "
          f"{r.get('time_sec',0):.1f}s")


# ============================================================
# TEST PART 2: Video Analysis (9 test videos)
# ============================================================
print("\n" + "=" * 60)
print("  PART 2: Video Analysis (9 test videos)")
print("=" * 60)

video_files = sorted([
    f for f in os.listdir(VIDEOS_DIR)
    if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
])

video_results = []
for i, fname in enumerate(video_files, 1):
    vpath = os.path.join(VIDEOS_DIR, fname)
    print(f"\n{'='*50}")
    print(f"--- Video {i}/{len(video_files)}: {fname} ---")
    print(f"{'='*50}")
    
    try:
        start = time.time()
        result = vp.process_video(vpath, stop_on_first=True, show_result=False)
        elapsed = time.time() - start
        
        detected = result.get("accident_detected", False)
        timestamp = result.get("timestamp_sec", None)
        frames_analyzed = result.get("frames_analyzed", 0)
        m3_score = result.get("temporal_score_m3", 0)
        
        best = result.get("best_result", {}) or {}
        n_vehicles = len(best.get("involved_vehicles", []))
        n_victims = len(best.get("victims", []))
        
        crash_scores = [v.get("crash_score", 0) for v in best.get("involved_vehicles", [])]
        
        status = "ACCIDENT DETECTED" if detected else "NO ACCIDENT"
        print(f"\n  Result: {status}")
        if detected:
            print(f"  Timestamp: {timestamp}s")
            print(f"  Vehicles: {n_vehicles}, Victims: {n_victims}")
            print(f"  Crash scores: {crash_scores}")
        print(f"  Frames analyzed: {frames_analyzed}")
        print(f"  M3 temporal: {m3_score}")
        print(f"  Time: {elapsed:.1f}s")
        
        video_results.append({
            "file": fname,
            "detected": detected,
            "timestamp": timestamp,
            "vehicles": n_vehicles,
            "victims": n_victims,
            "crash_scores": crash_scores,
            "frames_analyzed": frames_analyzed,
            "m3_score": m3_score,
            "time_sec": round(elapsed, 1),
        })
        
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        video_results.append({
            "file": fname,
            "detected": False,
            "error": str(e),
        })

# Video summary
print("\n" + "=" * 60)
print("  VIDEO RESULTS SUMMARY")
print("=" * 60)
video_detected = sum(1 for r in video_results if r.get("detected", False))
print(f"  Detected accidents: {video_detected}/{len(video_results)}")
for r in video_results:
    status = "✓ DETECTED" if r.get("detected") else "✗ MISSED"
    ts = f"@{r.get('timestamp','-')}s" if r.get("detected") else ""
    print(f"  {status} | {r['file']} | "
          f"V:{r.get('vehicles',0)} P:{r.get('victims',0)} | "
          f"CS:{r.get('crash_scores',[])} {ts} | "
          f"{r.get('time_sec',0):.0f}s")


# ============================================================
# OVERALL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("  OVERALL ACCURACY")
print("=" * 60)

total_tests = len(frame_results) + len(video_results)
total_detected = frame_detected + video_detected
accuracy = (total_detected / total_tests * 100) if total_tests > 0 else 0

print(f"  Frames: {frame_detected}/{len(frame_results)} detected")
print(f"  Videos: {video_detected}/{len(video_results)} detected")
print(f"  TOTAL:  {total_detected}/{total_tests} ({accuracy:.1f}%)")
print("=" * 60)

# Save results JSON
results_json = {
    "frame_results": frame_results,
    "video_results": video_results,
    "summary": {
        "frames_detected": frame_detected,
        "frames_total": len(frame_results),
        "videos_detected": video_detected,
        "videos_total": len(video_results),
        "total_detected": total_detected,
        "total_tests": total_tests,
        "accuracy_pct": round(accuracy, 1),
    }
}

json_path = os.path.join(OUTPUT_DIR, "test_results.json")
with open(json_path, "w") as f:
    json.dump(results_json, f, indent=2)
print(f"\n  Results saved to: {json_path}")

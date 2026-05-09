"""
Run RapidAid pipeline on all test frames and videos.
Produces annotated images and JSON reports for each.

Now includes M1 (scene classifier) and M2 (damage detector) status
and scores in the summary output.
"""
import sys, os, glob, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.frame_processor import FrameProcessor
from pipeline.video_processor import VideoProcessor

def main():
    processor = FrameProcessor()

    # ===== MODEL AVAILABILITY =====
    print("\n" + "=" * 60)
    print("  MODEL STATUS")
    print("=" * 60)
    m1_status = "LOADED" if processor.frame_classifier.is_available() else "NOT FOUND"
    m2_status = "LOADED" if processor.damage_classifier.is_available() else "NOT FOUND"
    print(f"  M1 Scene Classifier:  {m1_status}")
    print(f"  M2 Damage Detector:   {m2_status}")
    # M3 status checked later when VideoProcessor is created
    print("=" * 60)

    # ===== PROCESS ALL TEST FRAMES =====
    frames = sorted(glob.glob("data/test_frames/*"))
    print("\n" + "=" * 60)
    print("  PROCESSING {} TEST FRAMES".format(len(frames)))
    print("=" * 60)

    frame_results = []
    for f in frames:
        name = os.path.basename(f)
        print("\n>>> {}".format(name))
        t0 = time.time()
        result = processor.process(f, source_path=f)
        dt = time.time() - t0

        detected = result["accident_detected"]
        n_v = len(result["involved_vehicles"])
        n_vic = len(result["victims"])
        cls_conf = result.get("classifier_confidence", 0) or 0
        max_dmg = result.get("max_damage_score", 0) or 0
        fused = result.get("fused_confidence", 0) or 0
        scene_only = result.get("scene_only_detection", False)
        status = "ACCIDENT" if detected else "No Accident"

        # Save outputs
        saved = processor.report_generator.save(result)

        mode = "scene-only" if scene_only else "ensemble"
        print("  Result: {} | Vehicles: {} | Victims: {} | Time: {:.1f}s".format(
            status, n_v, n_vic, dt
        ))
        print("  M1: {:.1%} | M2: {:.1%} | Fused: {:.1%} | Mode: {}".format(
            cls_conf, max_dmg, fused, mode
        ))
        print("  Saved: {}".format(saved["frame_path"]))

        frame_results.append({
            "file": name,
            "detected": detected,
            "vehicles": n_v,
            "victims": n_vic,
            "cls_conf": round(cls_conf, 3),
            "max_damage": round(max_dmg, 3),
            "fused": round(fused, 3),
            "scene_only": scene_only,
            "time_sec": round(dt, 1),
        })

    # ===== SUMMARY TABLE =====
    print("\n" + "=" * 80)
    print("  FRAME RESULTS SUMMARY")
    print("=" * 80)
    header = "  {:<28s} {:>8s} {:>5s} {:>5s} {:>6s} {:>6s} {:>6s}".format(
        "Frame", "Accident", "Vehs", "Vics", "M1%", "M2%", "Fused"
    )
    print(header)
    print("  " + "-" * 76)
    for r in frame_results:
        label = "YES" if r["detected"] else "No"
        suffix = " *" if r["scene_only"] else ""
        print("  {:<28s} {:>8s} {:>5d} {:>5d} {:>5.0f}% {:>5.0f}% {:>5.0f}%{}".format(
            r["file"][:28], label, r["vehicles"], r["victims"],
            r["cls_conf"] * 100, r["max_damage"] * 100, r["fused"] * 100,
            suffix
        ))
    print("  (* = scene-only detection, no vehicles detected by YOLO)")

    # ===== PROCESS ALL TEST VIDEOS =====
    videos = sorted(glob.glob("data/test_videos/*"))
    print("\n" + "=" * 80)
    print("  PROCESSING {} TEST VIDEOS".format(len(videos)))
    print("=" * 80)

    video_proc = VideoProcessor(frame_processor=processor)
    m3_status = "LOADED" if video_proc.temporal_classifier.is_available() else "NOT FOUND"
    print(f"  M3 Temporal Classifier: {m3_status}")
    video_results = []
    for v in videos:
        name = os.path.basename(v)
        print("\n>>> {}".format(name))
        t0 = time.time()
        result = video_proc.process_video(v, stop_on_first=True, show_result=False)
        dt = time.time() - t0

        detected = result["accident_detected"]
        ts = result.get("timestamp_sec")
        n_analyzed = result.get("frames_analyzed", 0)

        # Extract classifier info from best result if available
        best = result.get("best_result", {}) or {}
        cls_conf = best.get("classifier_confidence", 0) or 0
        max_dmg = best.get("max_damage_score", 0) or 0
        fused = best.get("fused_confidence", 0) or 0

        # Extract M3 temporal score from video result
        m3_score = result.get("temporal_score_m3", 0) or 0

        print("  Result: {} | Timestamp: {}s | Frames analyzed: {} | Time: {:.1f}s".format(
            "ACCIDENT" if detected else "No Accident", ts, n_analyzed, dt
        ))
        if detected:
            print("  M1: {:.1%} | M2: {:.1%} | Fused: {:.1%} | M3: {:.1%}".format(
                cls_conf, max_dmg, fused, m3_score
            ))

        video_results.append({
            "file": name,
            "detected": detected,
            "timestamp_sec": ts,
            "frames_analyzed": n_analyzed,
            "cls_conf": round(cls_conf, 3),
            "max_damage": round(max_dmg, 3),
            "fused": round(fused, 3),
            "m3_temporal": round(m3_score, 3),
            "time_sec": round(dt, 1),
        })

    # ===== VIDEO SUMMARY =====
    print("\n" + "=" * 80)
    print("  VIDEO RESULTS SUMMARY")
    print("=" * 80)
    print("  {:<22s} {:>8s} {:>8s} {:>7s} {:>6s} {:>6s} {:>6s} {:>5s} {:>7s}".format(
        "Video", "Accident", "Time(s)", "Frames", "M1%", "M2%", "Fused", "M3%", "RunSec"
    ))
    print("  " + "-" * 82)
    for r in video_results:
        label = "YES" if r["detected"] else "No"
        ts = str(r["timestamp_sec"]) if r["timestamp_sec"] else "-"
        print("  {:<22s} {:>8s} {:>8s} {:>7d} {:>5.0f}% {:>5.0f}% {:>5.0f}% {:>4.0f}% {:>7.1f}".format(
            r["file"][:22], label, ts, r["frames_analyzed"],
            r["cls_conf"] * 100, r["max_damage"] * 100, r["fused"] * 100,
            r["m3_temporal"] * 100,
            r["time_sec"]
        ))

    # ===== SAVE CONSOLIDATED SUMMARY =====
    summary = {
        "m1_available": processor.frame_classifier.is_available(),
        "m2_available": processor.damage_classifier.is_available(),
        "m3_available": video_proc.temporal_classifier.is_available(),
        "frame_results": frame_results,
        "video_results": video_results,
    }
    summary_path = os.path.join("outputs", "test_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n  Summary saved: {}".format(summary_path))
    print("\nDone! Check outputs/annotated/ and outputs/reports/ for results.\n")


if __name__ == "__main__":
    main()

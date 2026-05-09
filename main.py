"""
RapidAid — CLI Entry Point

Usage:
    # Process a single image:
    python main.py --image path/to/frame.jpg

    # Process a video:
    python main.py --video path/to/video.mp4

    # Process a video without stopping at first accident:
    python main.py --video path/to/video.mp4 --no-stop

    # Save results without displaying:
    python main.py --image path/to/frame.jpg --no-display
"""
import sys
import os
import argparse
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.frame_processor import FrameProcessor
from pipeline.video_processor import VideoProcessor
from utils.helpers import display_frame


def process_image(image_path, show=True):
    """Process a single image through the RapidAid pipeline."""
    print(f"\n[RapidAid] Processing image: {image_path}\n")

    processor = FrameProcessor()
    result = processor.process(image_path)

    # Save outputs
    saved = processor.report_generator.save(result)

    # Print results
    print("\n" + "=" * 50)
    if result["accident_detected"]:
        n_v = len(result["involved_vehicles"])
        n_vic = len(result["victims"])
        print(f"  ACCIDENT DETECTED")
        print(f"  Involved Vehicles: {n_v}")
        print(f"  Victims: {n_vic}")
    else:
        print(f"  No Accident Detected")
    print("=" * 50)

    print(f"\n  Annotated frame saved: {saved['frame_path']}")
    print(f"  JSON report saved: {saved['report_path']}")

    # Print JSON report
    print("\n--- JSON Report ---")
    print(json.dumps(result["report"], indent=2, default=str))

    # Display
    if show:
        display_frame(result["annotated_frame"], "RapidAid Analysis")

    return result


def process_video(video_path, stop_on_first=True, show=True):
    """Process a video through the RapidAid pipeline."""
    print(f"\n[RapidAid] Processing video: {video_path}\n")

    processor = FrameProcessor()
    video_proc = VideoProcessor(frame_processor=processor)
    result = video_proc.process_video(
        video_path,
        stop_on_first=stop_on_first,
        show_result=show,
    )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="RapidAid — Automated Accident Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --image data/test_frames/accident1.jpg
  python main.py --video data/test_videos/crash.mp4
  python main.py --video data/test_videos/crash.mp4 --no-stop
        """
    )

    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--no-stop", action="store_true",
                        help="Don't stop at first accident (video mode)")
    parser.add_argument("--no-display", action="store_true",
                        help="Don't display results (save only)")

    args = parser.parse_args()

    if not args.image and not args.video:
        parser.print_help()
        print("\n[ERROR] Please provide --image or --video argument.")
        sys.exit(1)

    show = not args.no_display

    if args.image:
        if not os.path.exists(args.image):
            print(f"[ERROR] Image file not found: {args.image}")
            sys.exit(1)
        process_image(args.image, show=show)

    elif args.video:
        if not os.path.exists(args.video):
            print(f"[ERROR] Video file not found: {args.video}")
            sys.exit(1)
        process_video(args.video, stop_on_first=not args.no_stop, show=show)


if __name__ == "__main__":
    main()

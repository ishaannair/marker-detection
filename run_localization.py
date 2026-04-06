"""
run_localization.py
-------------------
End-to-end demo: detect custom markers and estimate their pose.

Usage
-----
# On a single image file:
    python run_localization.py --source input/page_2.png

# On a live camera (index 0):
    python run_localization.py --source 0

# On a video file:
    python run_localization.py --source path/to/video.mp4

# Save output frames to folder:
    python run_localization.py --source input/page_2.png --output output/

Camera calibration
------------------
Replace the placeholder K and dist below with YOUR calibrated values, or
pass --calib path/to/calib.npz  (npz file with keys 'K' and 'dist').
"""

import argparse
import sys
import os
import cv2
import numpy as np

from marker_detector import detect_markers
from pose_estimator  import estimate_pose, draw_pose_axes


# ── placeholder calibration (replace with your values!) ───────────────────────
# These are typical values for a 1080p webcam.  Use your own from calibration.
DEFAULT_K = np.array([
    [1200,    0,  960],
    [   0, 1200,  540],
    [   0,    0,    1],
], dtype=np.float64)

DEFAULT_DIST = np.zeros((5,), dtype=np.float64)   # assume no distortion as placeholder


def load_calib(path: str):
    data = np.load(path)
    return data["K"].astype(np.float64), data["dist"].astype(np.float64)


def process_frame(frame: np.ndarray,
                  K:     np.ndarray,
                  dist:  np.ndarray,
                  debug: bool = True) -> np.ndarray:
    """
    Detect markers, estimate pose, draw results.
    Returns annotated frame.
    """
    markers = detect_markers(frame, debug=debug)

    if not markers:
        cv2.putText(frame, "No markers detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return frame

    for i, marker in enumerate(markers):
        pose = estimate_pose(marker, K, dist)

        # ── draw pose axes ─────────────────────────────────────────────────────
        draw_pose_axes(frame, pose, K, dist, axis_length=0.10)

        # ── draw keypoints ────────────────────────────────────────────────────
        if marker.debug_img is not None:
            # blend debug overlay
            frame = cv2.addWeighted(frame, 0.7, marker.debug_img, 0.3, 0)

        # ── print pose info on frame ──────────────────────────────────────────
        y0 = 40 + i * 90
        tag  = marker.marker_type.upper()
        info = str(pose)
        cv2.putText(frame, f"[{tag}] {info}", (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

        # also print to console
        print(f"Marker {i+1} ({marker.marker_type}): {pose}")

    return frame


def main():
    parser = argparse.ArgumentParser(description="Custom marker localisation")
    parser.add_argument("--source",  default="0",
                        help="Image/video path or camera index (default: 0)")
    parser.add_argument("--calib",   default=None,
                        help="Path to .npz calibration file with keys K, dist")
    parser.add_argument("--output",  default=None,
                        help="Directory to save annotated frames")
    parser.add_argument("--no-debug", action="store_true",
                        help="Disable keypoint debug overlay")
    args = parser.parse_args()

    # ── load calibration ───────────────────────────────────────────────────────
    if args.calib:
        K, dist = load_calib(args.calib)
        print(f"Loaded calibration from {args.calib}")
    else:
        K, dist = DEFAULT_K, DEFAULT_DIST
        print("WARNING: Using placeholder calibration. Pose values will be wrong!")
        print("         Provide --calib path/to/calib.npz for real results.\n")

    debug = not args.no_debug

    if args.output:
        os.makedirs(args.output, exist_ok=True)

    # ── open source ────────────────────────────────────────────────────────────
    source = args.source
    is_image = False
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    elif os.path.isfile(source):
        ext = os.path.splitext(source)[1].lower()
        if ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff"):
            is_image = True
        else:
            cap = cv2.VideoCapture(source)
    else:
        print(f"Source not found: {source}")
        sys.exit(1)

    frame_idx = 0

    if is_image:
        frame = cv2.imread(source)
        if frame is None:
            print(f"Could not read image: {source}")
            sys.exit(1)
        result = process_frame(frame, K, dist, debug)
        cv2.imshow("Marker Localisation", result)
        if args.output:
            out_path = os.path.join(args.output, f"result_{frame_idx:04d}.png")
            cv2.imwrite(out_path, result)
            print(f"Saved: {out_path}")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = process_frame(frame, K, dist, debug)
            cv2.imshow("Marker Localisation  (q to quit)", result)

            if args.output:
                out_path = os.path.join(args.output, f"result_{frame_idx:04d}.png")
                cv2.imwrite(out_path, result)

            frame_idx += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

"""
run_pose.py
-----------
End-to-end marker detection + pose estimation.

Uses the full detect.py pipeline (geometry validation, ID classification)
and solves PnP to get 6-DOF pose for each detected marker.

Usage
-----
  # single image
  python run_pose.py --source input/page_2.png

  # folder of images
  python run_pose.py --source input/

  # live camera (index 0 or 1 etc.)
  python run_pose.py --source 1

  # with your own calibration file
  python run_pose.py --source 1 --calib calib.npz

  # save annotated output
  python run_pose.py --source input/ --output output/

Camera calibration file
-----------------------
Pass a .npz file containing keys 'K' (3x3) and 'dist' (1D coefficients).
Without --calib, placeholder values are used and pose numbers will be wrong.

Marker model units
------------------
Edit MARKER_WIDTH / MARKER_HEIGHT in marker_model.py to match your
physical print size.  Default is 0.40 m x 0.50 m.
"""

import argparse
import os
import sys
import cv2
import numpy as np

from detect        import detect, annotate, MarkerClassifier, Tracker
from pose_estimator import estimate_pose, draw_pose_axes


# ── placeholder calibration  (replace with your own!) ─────────────────────────
DEFAULT_K = np.array([
    [1200,    0,  960],
    [   0, 1200,  540],
    [   0,    0,    1],
], dtype=np.float64)

DEFAULT_DIST = np.zeros((5,), dtype=np.float64)

DISPLAY_MAX_DIM = 900   # max window dimension for display


# ── helpers ────────────────────────────────────────────────────────────────────

def _load_calib(path: str):
    data = np.load(path)
    return data["K"].astype(np.float64), data["dist"].astype(np.float64)


def _fit_display(img, max_dim=DISPLAY_MAX_DIM):
    h, w = img.shape[:2]
    scale = min(max_dim / max(h, w, 1), 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return img


def _open_camera(index: int):
    """Try MSMF -> DSHOW -> ANY backends."""
    for backend in (cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY):
        try:
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                return cap
            cap.release()
        except Exception:
            pass
    return None


# ── per-frame processing ───────────────────────────────────────────────────────

def process_frame(frame: np.ndarray,
                  K:     np.ndarray,
                  dist:  np.ndarray,
                  classifier=None) -> np.ndarray:
    """
    Detect markers, estimate pose for each, draw results.
    Returns annotated BGR frame.
    """
    detections = detect(frame, classifier=classifier)

    # draw standard detection boxes / labels first
    out = annotate(frame, detections)

    for i, det in enumerate(detections):
        # skip if we don't have enough keypoints for PnP
        if det.shape_corners is None or len(det.circle_centres) != 2:
            continue

        pose = estimate_pose(det, K, dist)

        # draw 3-D axes at marker origin
        draw_pose_axes(out, pose, K, dist, axis_length=0.10)

        # pose text overlay
        y0  = 40 + i * 80
        tag = det.marker_type.upper()
        id_str = f"  ID:{det.marker_id}" if det.marker_id else ""
        line1 = f"[{tag}{id_str}]  conf={det.confidence:.2f}"
        line2 = f"  {pose}"

        cv2.putText(out, line1, (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(out, line2, (10, y0 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 255), 2, cv2.LINE_AA)

        # console
        print(f"  Marker {i+1} ({det.marker_type}{id_str}): {pose}")

    return out


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Marker detection + pose estimation")
    parser.add_argument("--source",    default="0",
                        help="Image path, folder, video, or camera index (default: 0)")
    parser.add_argument("--calib",     default=None,
                        help="Path to .npz calibration file with keys K, dist")
    parser.add_argument("--templates", default=None,
                        help="Folder with page_N.png reference images for ID matching")
    parser.add_argument("--output",    default=None,
                        help="Directory to save annotated frames")
    parser.add_argument("--no-show",   action="store_true",
                        help="Suppress display window")
    args = parser.parse_args()

    # ── calibration ───────────────────────────────────────────────────────────
    if args.calib:
        K, dist = _load_calib(args.calib)
        print(f"Loaded calibration from {args.calib}")
    else:
        K, dist = DEFAULT_K, DEFAULT_DIST
        print("WARNING: Using placeholder calibration - pose values will be wrong!")
        print("         Use --calib path/to/calib.npz for real results.\n")

    # ── classifier (optional) ─────────────────────────────────────────────────
    classifier = None
    if args.templates:
        classifier = MarkerClassifier()
        n = classifier.build_from_dir(args.templates)
        print(f"Loaded {n} ID templates from {args.templates}")

    if args.output:
        os.makedirs(args.output, exist_ok=True)

    source = args.source
    show   = not args.no_show

    # ── image / folder ────────────────────────────────────────────────────────
    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

    if os.path.isdir(source):
        files = sorted(f for f in os.listdir(source)
                       if os.path.splitext(f)[1].lower() in IMAGE_EXTS)
        for fname in files:
            fpath = os.path.join(source, fname)
            frame = cv2.imread(fpath)
            if frame is None:
                continue
            print(f"\n--- {fname} ---")
            result = process_frame(frame, K, dist, classifier)
            if show:
                cv2.imshow("Pose Estimation", _fit_display(result))
                key = cv2.waitKey(0)
                if key == ord("q"):
                    break
            if args.output:
                out_path = os.path.join(args.output, fname)
                cv2.imwrite(out_path, result)
                print(f"  Saved -> {out_path}")
        cv2.destroyAllWindows()
        return

    if os.path.isfile(source):
        ext = os.path.splitext(source)[1].lower()
        if ext in IMAGE_EXTS:
            frame = cv2.imread(source)
            if frame is None:
                print(f"Could not read: {source}")
                sys.exit(1)
            print(f"\n--- {os.path.basename(source)} ---")
            result = process_frame(frame, K, dist, classifier)
            if show:
                cv2.imshow("Pose Estimation", _fit_display(result))
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            if args.output:
                out_path = os.path.join(args.output,
                                        "result_" + os.path.basename(source))
                cv2.imwrite(out_path, result)
                print(f"Saved -> {out_path}")
            return

        # video file fallthrough to cap below
        cap = cv2.VideoCapture(source)
    elif source.isdigit():
        cap = _open_camera(int(source))
        if cap is None:
            print(f"Could not open camera {source}")
            sys.exit(1)
    else:
        print(f"Source not found: {source}")
        sys.exit(1)

    # ── camera / video loop ───────────────────────────────────────────────────
    tracker   = Tracker()
    frame_idx = 0
    print("Running... press Q to quit.")

    while True:
        try:
            ret, frame = cap.read()
        except Exception:
            ret, frame = False, None

        if not ret or frame is None or frame.size == 0:
            if os.path.isfile(source):
                break       # end of video
            continue        # camera glitch - keep going

        # use tracker for camera smoothing; raw detections for pose
        raw_dets = detect(frame, classifier=classifier)
        smooth_dets, stale_flags = tracker.update(raw_dets)

        # pose on fresh detections only (stale = no new corners available)
        out = annotate(frame, smooth_dets, stale_flags)

        for i, (det, stale) in enumerate(zip(smooth_dets, stale_flags)):
            if stale:
                continue
            if det.shape_corners is None or len(det.circle_centres) != 2:
                continue
            pose = estimate_pose(det, K, dist)
            draw_pose_axes(out, pose, K, dist, axis_length=0.10)
            y0  = 40 + i * 80
            tag = det.marker_type.upper()
            id_str = f"  ID:{det.marker_id}" if det.marker_id else ""
            cv2.putText(out, f"[{tag}{id_str}] {pose}",
                        (10, y0), cv2.FONT_HERSHEY_SIMPLEX,
                        0.50, (255, 255, 0), 2, cv2.LINE_AA)

        disp = _fit_display(out)
        if show:
            cv2.imshow("Pose Estimation  (Q to quit)", disp)

        if args.output:
            out_path = os.path.join(args.output, f"frame_{frame_idx:05d}.png")
            cv2.imwrite(out_path, out)

        frame_idx += 1
        if show and (cv2.waitKey(1) & 0xFF == ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

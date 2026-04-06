"""
detect.py  –  Custom marker detection with bounding-box output
--------------------------------------------------------------
Detects the custom geometric markers (inverted-triangle or diamond shape
with two circles) in images or a live camera feed.

Usage
-----
  # single image
  python detect.py --source input/page_2.png

  # folder of images
  python detect.py --source input/

  # live camera (index 0 = default webcam)
  python detect.py --source 0

  # save output alongside source images
  python detect.py --source input/ --save

  # show result without a pop-up window (headless)
  python detect.py --source input/page_2.png --no-show

Output
------
  Prints to console:  filename | marker type | bounding box (x, y, w, h)
  Annotated image:    green bounding box + label drawn on the frame
"""

import argparse
import os
import sys
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
#  Detection parameters  (tune if needed)
# ──────────────────────────────────────────────────────────────────────────────

MIN_SHAPE_AREA        = 2_000   # px²  – absolute minimum area for outer shape
MIN_SHAPE_FRAC        = 0.0008  # shape must be ≥ 0.08 % of total image area
MIN_CIRCLE_AREA_FRAC  = 0.03    # circle must be ≥ 3 % of outer shape area
POLY_EPS              = 0.04    # polygon approximation tolerance
CIRCULARITY_THRESH    = 0.50    # how round a blob must be to count as a circle
DISPLAY_MAX_DIM       = 900     # resize for display (keeps window manageable)


TEMPLATE_SIZE    = (128, 128)   # fixed size for normalised shape crops
ID_MATCH_THRESH  = 0.60         # minimum NCC score to accept an ID


# ──────────────────────────────────────────────────────────────────────────────
#  Marker classifier  (template matching on the inner symbol)
# ──────────────────────────────────────────────────────────────────────────────

class MarkerClassifier:
    """
    Rotation-invariant marker ID classifier.

    Strategy
    --------
    1. Crop the detected outer-shape region and binarise.
    2. Resize to TEMPLATE_SIZE.
    3. Sweep the crop over SWEEP_ANGLES rotation angles (covering the full
       expected rotation range).  For each rotated query, compute NCC against
       every stored template.
    4. Return the (template_id, max_score) across all angles and templates.

    This is simpler and more robust than trying to estimate orientation from
    noisy/rotated polygon vertices, which tends to flip to the wrong vertex and
    actively harm accuracy.
    """

    # Sweep the full expected rotation range in equal steps.
    # 36 angles × 10 templates × 128×128 = ~6 ms on modern hardware.
    SWEEP_ANGLES = np.linspace(-180, 175, 36)   # every 10°, full circle

    def __init__(self):
        self._templates: dict[int, np.ndarray] = {}

    # ── helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _rotate_img(img: np.ndarray, angle: float) -> np.ndarray:
        if abs(angle) < 0.5:
            return img
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=0)

    def _crop_and_binarise(self, frame: np.ndarray,
                            shape_bbox: tuple) -> np.ndarray:
        """Crop to shape_bbox, binarise, resize to TEMPLATE_SIZE."""
        sx, sy, sw, sh = shape_bbox
        fh, fw = frame.shape[:2]
        sx, sy = max(sx, 0), max(sy, 0)
        ex, ey = min(sx + sw, fw), min(sy + sh, fh)
        crop   = frame[sy:ey, sx:ex]
        if crop.size == 0:
            return np.zeros(TEMPLATE_SIZE, dtype=np.uint8)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        _, binary = cv2.threshold(gray, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.resize(binary, TEMPLATE_SIZE, interpolation=cv2.INTER_AREA)

    # ── template building ──────────────────────────────────────────────────────

    def build_from_dir(self, directory: str) -> int:
        """
        Scan `directory` for page_<N>.png reference images, detect the marker,
        and store one binary template per page number.
        """
        self._templates.clear()
        built = 0
        for fname in sorted(os.listdir(directory)):
            if not (fname.startswith("page_") and fname.endswith(".png")):
                continue
            try:
                page_num = int(fname[5:-4])
            except ValueError:
                continue

            img = cv2.imread(os.path.join(directory, fname))
            if img is None:
                continue

            dets = detect(img)
            if not dets:
                print(f"  [classifier] WARNING: no marker in {fname}, skipping")
                continue

            tmpl = self._crop_and_binarise(img, dets[0].shape_bbox)
            self._templates[page_num] = tmpl
            built += 1

        return built

    # ── classification ─────────────────────────────────────────────────────────

    def classify(self, frame: np.ndarray,
                 shape_bbox: tuple,
                 marker_type: Optional[str] = None) -> tuple:
        """
        Match the shape crop against all templates by sweeping over a full
        360° rotation range (every 10°).

        Returns (marker_id, best_score).  marker_id is None if
        best_score < ID_MATCH_THRESH.
        """
        if not self._templates:
            return None, 0.0

        base = self._crop_and_binarise(frame, shape_bbox)

        best_id    = None
        best_score = -1.0

        for angle in self.SWEEP_ANGLES:
            query = self._rotate_img(base, float(angle)).astype(np.float32)
            for page_num, tmpl in self._templates.items():
                score = float(cv2.matchTemplate(
                    query, tmpl.astype(np.float32),
                    cv2.TM_CCOEFF_NORMED)[0, 0])
                if score > best_score:
                    best_score = score
                    best_id    = page_num

        if best_score < ID_MATCH_THRESH:
            return None, best_score

        return best_id, best_score

    def __len__(self):
        return len(self._templates)


# ──────────────────────────────────────────────────────────────────────────────
#  Result dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MarkerDetection:
    marker_type   : str           # 'triangle' | 'diamond'
    bbox          : tuple         # (x, y, w, h) — full marker incl. circles
    shape_bbox    : tuple         # (x, y, w, h) — outer shape only
    circle_centres: list          # [(cx, cy), (cx, cy)]  — may be empty
    confidence    : float         # 0–1  (1 = both circles found)
    marker_id     : Optional[int] = None   # page number (2–11), None if unknown
    id_score      : float         = 0.0   # template match score 0–1
    shape_corners : Optional[np.ndarray] = None  # ordered polygon corners (N,2) float32
                                                  # triangle: [TL, TR, AP]
                                                  # diamond:  [T, L, R, B]

    def __str__(self):
        x, y, w, h = self.bbox
        id_str = f"ID={self.marker_id}({self.id_score:.2f})" if self.marker_id else "ID=?"
        return (f"[{self.marker_type:<8s}]  "
                f"bbox=({x:4d},{y:4d},{w:4d},{h:4d})  "
                f"circles={'yes' if self.circle_centres else 'no '}  "
                f"conf={self.confidence:.2f}  {id_str}")


# ──────────────────────────────────────────────────────────────────────────────
#  Core detection logic
# ──────────────────────────────────────────────────────────────────────────────

def _order_triangle_corners(pts: np.ndarray) -> np.ndarray:
    """Return triangle corners as [TL, TR, AP] matching MODEL_POINTS_TRIANGLE order."""
    pts = pts[np.argsort(pts[:, 1])]   # sort by Y ascending
    top_two = pts[:2]
    apex    = pts[2:3]
    top_two = top_two[np.argsort(top_two[:, 0])]   # sort top pair left→right
    return np.vstack([top_two, apex])              # TL, TR, AP


def _order_diamond_corners(pts: np.ndarray) -> np.ndarray:
    """Return diamond corners as [T, L, R, B] matching MODEL_POINTS_DIAMOND order."""
    order  = np.argsort(pts[:, 1])
    top    = pts[order[0:1]]           # min Y
    bottom = pts[order[-1:]]           # max Y
    mid    = pts[order[1:3]]
    mid    = mid[np.argsort(mid[:, 0])]   # left, right
    return np.vstack([top, mid, bottom])  # T, L, R, B


def _circularity(cnt) -> float:
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    return 4 * np.pi * area / (peri ** 2) if peri > 0 else 0.0


def _contour_centre(cnt):
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return None
    return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))


def _union_bbox(*bboxes):
    """Return bounding rect that covers all supplied (x,y,w,h) rects."""
    x1 = min(b[0] for b in bboxes)
    y1 = min(b[1] for b in bboxes)
    x2 = max(b[0] + b[2] for b in bboxes)
    y2 = max(b[1] + b[3] for b in bboxes)
    return (x1, y1, x2 - x1, y2 - y1)


def _circles_match_shape(shape_bbox: tuple,
                          c1: tuple, c1_area: float,
                          c2: tuple, c2_area: float) -> bool:
    """
    Validate that two circle candidates are geometrically consistent with
    the marker layout:

      1. Same height  – circles sit at roughly the same Y level
      2. Below shape  – both circle centres are below the shape's centre
      3. Symmetric    – horizontal midpoint of circles aligns with shape centre
      4. Similar size – neither circle is more than 3x the area of the other
      5. Sensible gap – horizontal separation is between 20 % and 120 % of shape width

    Returns True only if all five checks pass.
    """
    sx, sy, sw, sh = shape_bbox
    shape_cx = sx + sw / 2
    shape_cy = sy + sh / 2

    lx, ly = c1   # left  circle centre  (already sorted left→right)
    rx, ry = c2   # right circle centre

    # 1. same height: Y difference < 25 % of shape height
    if abs(ly - ry) > 0.25 * sh:
        return False

    # 2. both circles below the shape's vertical centre
    avg_cy = (ly + ry) / 2
    if avg_cy < shape_cy:
        return False

    # 3. horizontal midpoint within 25 % of shape width from shape centre
    mid_x = (lx + rx) / 2
    if abs(mid_x - shape_cx) > 0.25 * sw:
        return False

    # 4. similar size: area ratio between 1/3 and 3
    if c1_area <= 0 or c2_area <= 0:
        return False
    ratio = max(c1_area, c2_area) / min(c1_area, c2_area)
    if ratio > 3.0:
        return False

    # 5. horizontal separation between 20 % and 120 % of shape width
    sep = rx - lx
    if not (0.20 * sw < sep < 1.20 * sw):
        return False

    return True


def detect(frame: np.ndarray,
           classifier: Optional[MarkerClassifier] = None) -> list[MarkerDetection]:
    """
    Run marker detection on a single BGR (or grayscale) frame.
    If `classifier` is provided, each detection is also assigned a marker_id.
    Returns a list of MarkerDetection objects, one per marker found.
    """
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours     = sorted(contours, key=cv2.contourArea, reverse=True)

    results     : list[MarkerDetection] = []
    used_indices: set[int] = set()
    img_area     = frame.shape[0] * frame.shape[1]
    min_area     = max(MIN_SHAPE_AREA, img_area * MIN_SHAPE_FRAC)

    for idx, cnt in enumerate(contours):
        if idx in used_indices:
            continue
        area = cv2.contourArea(cnt)
        if area < min_area:
            break

        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, POLY_EPS * peri, True)
        n      = len(approx)

        # ── classify outer shape ───────────────────────────────────────────────
        marker_type   = None
        shape_corners = None
        pts = approx.reshape(-1, 2).astype(np.float32)

        if n == 3:
            marker_type   = "triangle"
            shape_corners = _order_triangle_corners(pts)

        elif n == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / h if h > 0 else 0
            if 0.65 < aspect < 1.5:          # square-ish → diamond or rectangle
                marker_type   = "diamond"
                shape_corners = _order_diamond_corners(pts)

        if marker_type is None:
            continue

        used_indices.add(idx)
        shape_bbox = cv2.boundingRect(cnt)
        sx, sy, sw, sh = shape_bbox

        # ── find circles: round blobs that sit below the outer shape ──────────
        # "below" = centre-Y > upper third of image
        img_h     = frame.shape[0]
        lower_y   = img_h * 0.30          # circles expected in lower 70 %
        min_c_area = area * MIN_CIRCLE_AREA_FRAC

        circle_candidates = []
        for j, c in enumerate(contours):
            if j in used_indices:
                continue
            c_area = cv2.contourArea(c)
            if c_area < min_c_area:
                continue
            if _circularity(c) < CIRCULARITY_THRESH:
                continue
            centre = _contour_centre(c)
            if centre is None:
                continue
            cx, cy = centre
            if cy < lower_y:
                continue
            circle_candidates.append((j, cx, cy, c_area, cv2.boundingRect(c)))

        # pick the best pair of circles that passes geometric validation
        circle_candidates.sort(key=lambda x: -x[3])   # largest first

        circle_centres = []
        circle_bboxes  = []
        picked_indices = []

        # try every pair (largest first) until one passes the layout check
        for i in range(len(circle_candidates)):
            for k in range(i + 1, len(circle_candidates)):
                ji, cxi, cyi, ai, bi = circle_candidates[i]
                jk, cxk, cyk, ak, bk = circle_candidates[k]
                # sort left→right before checking
                if cxi <= cxk:
                    lc, rc = (cxi, cyi), (cxk, cyk)
                    la, ra = ai, ak
                    li, ri = ji, jk
                    lb, rb = bi, bk
                else:
                    lc, rc = (cxk, cyk), (cxi, cyi)
                    la, ra = ak, ai
                    li, ri = jk, ji
                    lb, rb = bk, bi

                if _circles_match_shape(shape_bbox, lc, la, rc, ra):
                    circle_centres = [lc, rc]
                    circle_bboxes  = [lb, rb]
                    picked_indices = [li, ri]
                    break
            if circle_centres:
                break

        for j in picked_indices:
            used_indices.add(j)

        # order circles left → right
        circle_centres.sort(key=lambda p: p[0])

        # ── compute full marker bounding box ──────────────────────────────────
        all_bboxes = [shape_bbox] + circle_bboxes
        full_bbox  = _union_bbox(*all_bboxes) if circle_bboxes else shape_bbox

        # confidence: 1.0 if both circles found, 0.6 if one, 0.3 if none
        confidence = {2: 1.0, 1: 0.6, 0: 0.3}[len(circle_centres)]

        # ── classify marker ID ─────────────────────────────────────────────────
        marker_id, id_score = (None, 0.0)
        if classifier is not None:
            marker_id, id_score = classifier.classify(frame, shape_bbox,
                                                        marker_type=marker_type)

        results.append(MarkerDetection(
            marker_type    = marker_type,
            bbox           = full_bbox,
            shape_bbox     = shape_bbox,
            circle_centres = circle_centres,
            confidence     = confidence,
            marker_id      = marker_id,
            id_score       = id_score,
            shape_corners  = shape_corners,
        ))

    return results


# ──────────────────────────────────────────────────────────────────────────────
#  Annotation / drawing
# ──────────────────────────────────────────────────────────────────────────────

# colours  BGR
COL_BOX_GOOD  = (0, 220, 0)      # green  – conf 1.0 (both circles found)
COL_BOX_BAD   = (0, 0, 220)      # red    – conf < 1.0
COL_BOX_STALE = (0, 180, 255)    # orange – held from previous frame (smoothed)
COL_SHAPE     = (0, 165, 255)    # orange – outer shape bbox (optional)
COL_CIRCLE    = (255, 80,  80)   # blue   – circle centres
COL_TEXT      = (255, 255, 255)  # white  – label text

THICKNESS = 2


def annotate(frame: np.ndarray,
             detections: list[MarkerDetection],
             stale_flags: Optional[list] = None,
             show_shape_box: bool = False) -> np.ndarray:
    """
    Draw bounding boxes and labels.

    Box colour:
      GREEN  – conf 1.0  (both circles found, fresh detection)
      RED    – conf < 1.0 (circles missing, fresh detection)
      ORANGE – held from a previous frame (temporal smoothing)
    """
    out = frame.copy()
    if stale_flags is None:
        stale_flags = [False] * len(detections)

    for det, stale in zip(detections, stale_flags):
        x, y, w, h = det.bbox

        # ── choose box colour ──────────────────────────────────────────────────
        if stale:
            col_box = COL_BOX_STALE
        elif det.confidence >= 1.0:
            col_box = COL_BOX_GOOD
        else:
            col_box = COL_BOX_BAD

        col_bg = tuple(max(0, c - 80) for c in col_box)   # darker shade for label bg

        # ── main bounding box ──────────────────────────────────────────────────
        cv2.rectangle(out, (x, y), (x + w, y + h), col_box, THICKNESS)

        # ── optional inner shape box ───────────────────────────────────────────
        if show_shape_box:
            sx, sy, sw, sh = det.shape_bbox
            cv2.rectangle(out, (sx, sy), (sx + sw, sy + sh),
                          COL_SHAPE, THICKNESS)

        # ── circle centres ─────────────────────────────────────────────────────
        for cx, cy in det.circle_centres:
            cv2.drawMarker(out, (cx, cy), COL_CIRCLE,
                           cv2.MARKER_CROSS, 20, THICKNESS)

        # ── label ──────────────────────────────────────────────────────────────
        status   = "LOCKED" if det.confidence >= 1.0 else ("STALE" if stale else "PARTIAL")
        id_str   = f"  ID:{det.marker_id}" if det.marker_id is not None else "  ID:?"
        label    = f"{det.marker_type}  {status}{id_str}"
        font     = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, min(1.2, w / 400))
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, THICKNESS)
        lx, ly = x, max(y - 8, th + 8)
        cv2.rectangle(out, (lx, ly - th - baseline), (lx + tw + 4, ly + baseline),
                      col_bg, -1)
        cv2.putText(out, label, (lx + 2, ly), font,
                    font_scale, COL_TEXT, THICKNESS, cv2.LINE_AA)

    # ── "nothing found" message ────────────────────────────────────────────────
    if not detections:
        cv2.putText(out, "No markers detected", (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 200), 2, cv2.LINE_AA)

    return out


# ──────────────────────────────────────────────────────────────────────────────
#  Temporal smoother
# ──────────────────────────────────────────────────────────────────────────────

MAX_STALE_FRAMES = 8    # hold a detection for this many frames after it disappears
EMA_ALPHA        = 0.4  # bbox smoothing: 0=frozen, 1=no smoothing


class _Track:
    """Holds state for one tracked marker type."""
    def __init__(self, det: MarkerDetection):
        self.det        = det
        self.bbox_ema   = list(det.bbox)   # smoothed bbox as floats
        self.stale_age  = 0                # frames since last real detection

    def update(self, det: MarkerDetection):
        """Called when a fresh detection of this marker type arrives."""
        self.det       = det
        self.stale_age = 0
        bx, by, bw, bh = det.bbox
        self.bbox_ema[0] += EMA_ALPHA * (bx - self.bbox_ema[0])
        self.bbox_ema[1] += EMA_ALPHA * (by - self.bbox_ema[1])
        self.bbox_ema[2] += EMA_ALPHA * (bw - self.bbox_ema[2])
        self.bbox_ema[3] += EMA_ALPHA * (bh - self.bbox_ema[3])

    def smoothed_detection(self, stale: bool) -> tuple[MarkerDetection, bool]:
        """Return a copy of the detection with the smoothed bbox applied."""
        from dataclasses import replace
        smoothed = replace(self.det,
                           bbox=tuple(int(v) for v in self.bbox_ema))
        return smoothed, stale


class Tracker:
    """
    Frame-to-frame smoother for camera detections.

    - Keeps the last confident detection for up to MAX_STALE_FRAMES frames
      so the box doesn't flicker when the detector misses for a frame or two.
    - Applies exponential moving average (EMA) to bbox coordinates so the
      box glides smoothly instead of jumping.
    """

    def __init__(self):
        self._tracks: dict[str, _Track] = {}   # keyed by marker_type

    def update(self, detections: list[MarkerDetection]
               ) -> tuple[list[MarkerDetection], list[bool]]:
        """
        Feed current-frame detections in, get back smoothed detections + stale flags.

        Returns
        -------
        out_dets   : list of MarkerDetection (smoothed bbox)
        stale_flags: parallel list of bool  (True = held from previous frame)
        """
        seen_types = set()

        # update / create tracks for this frame's detections
        for det in detections:
            t = det.marker_type
            seen_types.add(t)
            if t in self._tracks:
                self._tracks[t].update(det)
            else:
                self._tracks[t] = _Track(det)

        # age out tracks that weren't seen this frame
        to_remove = []
        for t, track in self._tracks.items():
            if t not in seen_types:
                track.stale_age += 1
                if track.stale_age > MAX_STALE_FRAMES:
                    to_remove.append(t)
        for t in to_remove:
            del self._tracks[t]

        # build output: fresh first, then stale
        out_dets, stale_flags = [], []
        for t, track in self._tracks.items():
            is_stale = t not in seen_types
            det, flag = track.smoothed_detection(stale=is_stale)
            out_dets.append(det)
            stale_flags.append(flag)

        return out_dets, stale_flags


def _fit_for_display(img: np.ndarray, max_dim: int = DISPLAY_MAX_DIM) -> np.ndarray:
    """Downscale image so the largest dimension ≤ max_dim (for readable display)."""
    h, w = img.shape[:2]
    scale = min(max_dim / max(h, w, 1), 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return img


# ──────────────────────────────────────────────────────────────────────────────
#  Save helper
# ──────────────────────────────────────────────────────────────────────────────

def _save_path(source_path: str) -> str:
    """Return output path: same folder, filename prefixed with 'detected_'."""
    folder = os.path.dirname(source_path) or "."
    name   = "detected_" + os.path.basename(source_path)
    return os.path.join(folder, name)


# ──────────────────────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────────────────────

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


def run_on_image(path: str, show: bool = True, save: bool = False,
                 show_shape_box: bool = False,
                 classifier: Optional[MarkerClassifier] = None):
    frame = cv2.imread(path)
    if frame is None:
        print(f"  ERROR: cannot read {path}")
        return

    detections = detect(frame, classifier=classifier)
    annotated  = annotate(frame, detections, show_shape_box=show_shape_box)

    # console output
    print(f"\n{path}  ->  {len(detections)} marker(s)")
    for i, d in enumerate(detections):
        print(f"  [{i+1}] {d}")

    if save:
        out_path = _save_path(path)
        cv2.imwrite(out_path, annotated)
        print(f"  Saved: {out_path}")

    if show:
        display = _fit_for_display(annotated)
        cv2.imshow(f"Marker Detection - {os.path.basename(path)}", display)
        print("  Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def _open_camera(index: int) -> cv2.VideoCapture:
    """
    Try several backends in order until the camera opens.
    On Windows, MSMF (Media Foundation) is more reliable than DSHOW.
    """
    backends = [
        (cv2.CAP_MSMF,  "MSMF"),
        (cv2.CAP_DSHOW, "DSHOW"),
        (cv2.CAP_ANY,   "ANY"),
    ]
    for api, name in backends:
        cap = cv2.VideoCapture(index, api)
        if cap.isOpened():
            print(f"Camera {index} opened via {name} backend")
            return cap
        cap.release()
    return cv2.VideoCapture()     # return unopened cap so caller can error-check


def run_on_camera(camera_index: int = 0, save_frames: bool = False,
                  output_dir: str = "output", show_shape_box: bool = False,
                  classifier: Optional[MarkerClassifier] = None):
    cap = _open_camera(camera_index)
    if not cap.isOpened():
        # list available indices as a hint
        available = [i for i in range(4)
                     if cv2.VideoCapture(i, cv2.CAP_MSMF).isOpened()]
        print(f"ERROR: Cannot open camera index {camera_index}")
        if available:
            print(f"  Available camera indices: {available}")
            print(f"  Try:  python detect.py --source {available[0]}")
        else:
            print("  No cameras detected. Check device manager / camera permissions.")
        sys.exit(1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Resolution: {actual_w}x{actual_h}  |  press  q  to quit,  s  to save snapshot")

    if save_frames:
        os.makedirs(output_dir, exist_ok=True)

    tracker   = Tracker()
    frame_idx = 0
    prev_summary = ""

    while True:
        try:
            ret, frame = cap.read()
        except cv2.error:
            ret = False

        if not ret or frame is None or frame.size == 0:
            continue   # skip bad frames, don't crash

        raw_dets            = detect(frame, classifier=classifier)
        detections, stalens = tracker.update(raw_dets)
        annotated           = annotate(frame, detections,
                                       stale_flags=stalens,
                                       show_shape_box=show_shape_box)

        # console: print only when something meaningful changes
        summary_parts = []
        for d, stale in zip(detections, stalens):
            status = "STALE" if stale else ("LOCKED" if d.confidence >= 1.0 else "PARTIAL")
            id_str = f"ID:{d.marker_id}" if d.marker_id is not None else "ID:?"
            summary_parts.append(f"{d.marker_type}:{status}:{id_str}")
        summary = "  ".join(summary_parts) if summary_parts else "none"

        if summary != prev_summary:
            print(f"  [{frame_idx:05d}] {summary}")
            for i, (d, stale) in enumerate(zip(detections, stalens)):
                status = "STALE" if stale else ("LOCKED" if d.confidence >= 1.0 else "PARTIAL")
                x, y, w, h = d.bbox
                id_str = f"ID:{d.marker_id}(score={d.id_score:.2f})" if d.marker_id else "ID:?"
                print(f"    [{i+1}] {d.marker_type:<8s}  bbox=({x:4d},{y:4d},{w:4d},{h:4d})"
                      f"  circles={'yes' if d.circle_centres else 'no '}"
                      f"  {status}  {id_str}")
            prev_summary = summary

        # overlay frame counter
        cv2.putText(annotated, f"frame {frame_idx}", (10, annotated.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        display = _fit_for_display(annotated)
        cv2.imshow("Marker Detection  (q=quit, s=snapshot)", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            snap_path = os.path.join(output_dir if save_frames else ".",
                                     f"snapshot_{frame_idx:05d}.png")
            cv2.imwrite(snap_path, annotated)
            print(f"  Snapshot saved: {snap_path}")

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Custom marker detector – bounding-box output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--source", default="0",
                        help="Image file, folder of images, or camera index "
                             "(default: 0 = webcam)")
    parser.add_argument("--templates", default="input",
                        help="Folder containing page_<N>.png reference images "
                             "for ID classification (default: input/)")
    parser.add_argument("--save", action="store_true",
                        help="Save annotated output next to source image(s)")
    parser.add_argument("--no-show", action="store_true",
                        help="Do not open a display window")
    parser.add_argument("--shape-box", action="store_true",
                        help="Also draw the inner shape bounding rect in orange")
    args = parser.parse_args()

    show = not args.no_show
    src  = args.source

    # ── build classifier from reference images ─────────────────────────────────
    classifier = None
    if os.path.isdir(args.templates):
        classifier = MarkerClassifier()
        n = classifier.build_from_dir(args.templates)
        print(f"Classifier ready: {n} templates loaded from {args.templates}/")
        if n == 0:
            print("  WARNING: no templates built — ID detection disabled.")
            classifier = None
    else:
        print(f"WARNING: templates folder '{args.templates}' not found — ID detection disabled.")

    # ── camera ─────────────────────────────────────────────────────────────────
    if src.isdigit():
        run_on_camera(int(src), save_frames=args.save,
                      show_shape_box=args.shape_box,
                      classifier=classifier)
        return

    # ── folder ─────────────────────────────────────────────────────────────────
    if os.path.isdir(src):
        files = sorted(
            f for f in os.listdir(src)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS
        )
        if not files:
            print(f"No images found in {src}")
            sys.exit(1)
        print(f"Processing {len(files)} image(s) in {src}/")
        for fname in files:
            run_on_image(os.path.join(src, fname),
                         show=show, save=args.save,
                         show_shape_box=args.shape_box,
                         classifier=classifier)
        return

    # ── single image ───────────────────────────────────────────────────────────
    if os.path.isfile(src):
        run_on_image(src, show=show, save=args.save,
                     show_shape_box=args.shape_box,
                     classifier=classifier)
        return

    print(f"ERROR: source not found: {src}")
    sys.exit(1)


if __name__ == "__main__":
    main()

"""
marker_detector.py
------------------
Detects custom geometric markers in an image and extracts the 2-D keypoints
needed for pose estimation.

Detection strategy (no ML required for keypoint extraction):
  1. Threshold → find contours
  2. Classify outer shape:  inverted-triangle  vs  diamond/rotated-square
  3. Locate the two circles (below the main shape)
  4. Return ordered 2-D image points that match MODEL_POINTS_TRIANGLE /
     MODEL_POINTS_DIAMOND in marker_model.py

The detector returns a list of DetectedMarker dataclass instances,
one per marker found in the frame.
"""

from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ── tuneable parameters ────────────────────────────────────────────────────────
MIN_MARKER_AREA   = 5_000    # px²  – ignore tiny blobs
APPROX_POLY_EPS   = 0.04     # fraction of arc-length for polygon approx
CIRCLE_CIRCULARITY_THRESH = 0.55   # how round a blob must be to count as circle


# ── data class ─────────────────────────────────────────────────────────────────
@dataclass
class DetectedMarker:
    marker_type: str                     # 'triangle' or 'diamond'
    image_points: np.ndarray             # shape (N, 2) float32
    bbox: tuple                          # (x, y, w, h) of bounding rect
    sign_id: Optional[int] = None        # filled in by classifier (Sign 1 … N)
    debug_img: Optional[np.ndarray] = None


# ── helpers ────────────────────────────────────────────────────────────────────

def _circularity(contour) -> float:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0.0
    return 4 * np.pi * area / (perimeter ** 2)


def _order_triangle_corners(pts: np.ndarray) -> np.ndarray:
    """
    Given 3 triangle vertices, return them in order:
    [top-left, top-right, apex(bottom)]
    matching MODEL_POINTS_TRIANGLE[0:3].
    """
    # sort by Y ascending (top = small Y in image coords)
    pts = pts[np.argsort(pts[:, 1])]
    top_two = pts[:2]
    apex    = pts[2:3]
    # sort top two by X
    top_two = top_two[np.argsort(top_two[:, 0])]
    return np.vstack([top_two, apex])  # TL, TR, AP


def _order_diamond_corners(pts: np.ndarray) -> np.ndarray:
    """
    Given 4 diamond vertices return [top, left, right, bottom]
    matching MODEL_POINTS_DIAMOND[0:4].
    """
    cx, cy = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    # top=-π/2, right=0, bottom=+π/2, left=±π
    # sort: top (smallest y), left, right, bottom
    order = np.argsort(pts[:, 1])          # by Y
    top    = pts[order[0:1]]              # smallest Y
    bottom = pts[order[-1:]]              # largest Y
    mid    = pts[order[1:3]]
    mid    = mid[np.argsort(mid[:, 0])]   # left, right
    return np.vstack([top, mid, bottom])  # T, L, R, B


def _find_circles_below(binary: np.ndarray,
                         main_bbox: tuple,
                         contours: list) -> list[np.ndarray]:
    """
    Among `contours`, find the two most circular blobs that sit
    below (or at the bottom of) the main marker shape.
    Returns a list of up to 2 centre points [(cx,cy), ...].
    """
    _, _, _, bh = main_bbox
    img_h = binary.shape[0]
    threshold_y = img_h * 0.40   # circles expected in lower 60 %

    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_MARKER_AREA * 0.05:
            continue
        circ = _circularity(c)
        if circ < CIRCLE_CIRCULARITY_THRESH:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        if cy > threshold_y:
            candidates.append((cx, cy, area, circ))

    # pick the two largest
    candidates.sort(key=lambda x: -x[2])
    centres = [np.array([c[0], c[1]], dtype=np.float32) for c in candidates[:2]]
    # order left→right
    if len(centres) == 2:
        centres.sort(key=lambda p: p[0])
    return centres


# ── main detection function ────────────────────────────────────────────────────

def detect_markers(frame: np.ndarray,
                   debug: bool = False) -> list[DetectedMarker]:
    """
    Detect all custom markers in `frame` (BGR or grayscale).

    Returns a list of DetectedMarker objects.
    Each object's `image_points` are ordered to match the corresponding
    MODEL_POINTS_* array in marker_model.py.
    """
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()

    # ── 1. Binarise (markers are high-contrast B&W) ────────────────────────────
    # Blur first to smooth jagged contours in high-res scans
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # ── 2. Find all external contours ─────────────────────────────────────────
    contours, _ = cv2.findContours(binary,
                                    cv2.RETR_LIST,
                                    cv2.CHAIN_APPROX_SIMPLE)

    markers_found: list[DetectedMarker] = []
    used_contours: set[int] = set()

    # sort by area (largest first → process big shapes before circles)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for idx, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < MIN_MARKER_AREA:
            break   # all remaining are too small

        if idx in used_contours:
            continue

        # ── 3. Approximate polygon ──────────────────────────────────────────────
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, APPROX_POLY_EPS * peri, True)
        n_verts = len(approx)
        pts = approx.reshape(-1, 2).astype(np.float32)

        marker_type: Optional[str] = None
        shape_corners: Optional[np.ndarray] = None

        if n_verts == 3:
            marker_type   = "triangle"
            shape_corners = _order_triangle_corners(pts)

        elif n_verts == 4:
            # check if it is diamond-like (roughly equal sides, rotated ~45°)
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / h if h > 0 else 0
            if 0.7 < aspect < 1.3:   # roughly square bounding box → diamond
                marker_type   = "diamond"
                shape_corners = _order_diamond_corners(pts)

        if marker_type is None:
            continue

        used_contours.add(idx)
        bbox = cv2.boundingRect(cnt)

        # ── 4. Find the two circles ─────────────────────────────────────────────
        remaining = [c for i, c in enumerate(contours)
                     if i not in used_contours and cv2.contourArea(c) > 100]
        circle_centres = _find_circles_below(binary, bbox, remaining)

        if len(circle_centres) != 2:
            # circles not found – skip (can't do full PnP)
            continue

        # mark circle contours as used
        for c_pt in circle_centres:
            for j, c in enumerate(contours):
                if j in used_contours:
                    continue
                M = cv2.moments(c)
                if M["m00"] == 0:
                    continue
                ccx = M["m10"] / M["m00"]
                ccy = M["m01"] / M["m00"]
                if abs(ccx - c_pt[0]) < 20 and abs(ccy - c_pt[1]) < 20:
                    used_contours.add(j)
                    break

        # ── 5. Assemble ordered image_points ───────────────────────────────────
        # Order must match MODEL_POINTS_TRIANGLE / MODEL_POINTS_DIAMOND
        lc = circle_centres[0].reshape(1, 2)
        rc = circle_centres[1].reshape(1, 2)
        image_points = np.vstack([shape_corners, lc, rc]).astype(np.float32)

        # ── 6. (Optional) draw debug overlay ──────────────────────────────────
        dbg = None
        if debug:
            dbg = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            colours = [(0, 255, 0), (0, 200, 0), (0, 150, 0),
                       (255, 0, 0), (200, 0, 0), (150, 0, 0)]
            labels  = (["TL", "TR", "AP"] if marker_type == "triangle"
                       else ["T", "L", "R", "B"])
            labels += ["LC", "RC"]
            for i, (pt, col, lbl) in enumerate(zip(image_points, colours, labels)):
                x_i, y_i = int(pt[0]), int(pt[1])
                cv2.circle(dbg, (x_i, y_i), 8, col, -1)
                cv2.putText(dbg, lbl, (x_i + 10, y_i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

        markers_found.append(DetectedMarker(
            marker_type  = marker_type,
            image_points = image_points,
            bbox         = bbox,
            debug_img    = dbg,
        ))

    return markers_found

"""
marker_model.py
---------------
Defines the 3D model points for each marker in the marker's own coordinate frame.

Coordinate convention:
  - Origin at the geometric center of the marker's bounding rectangle
  - X → right, Y → up, Z → out of the marker (toward camera)
  - All points lie on the Z=0 plane (flat marker)

Units: metres.  Set MARKER_WIDTH / MARKER_HEIGHT to the physical print size.
"""

import numpy as np

# ---------- Physical size of the printed marker (metres) ----------
MARKER_WIDTH  = 0.40   # total bounding-box width  (e.g. 40 cm)
MARKER_HEIGHT = 0.50   # total bounding-box height (e.g. 50 cm)

W, H = MARKER_WIDTH / 2, MARKER_HEIGHT / 2   # half-extents

# ---- Relative positions of circles (measured from images, as fractions of bbox) ----
# Both circles sit near the bottom of the bbox.
# Approximate positions (tweak after measuring your printed markers):
CIRCLE_Y_FRAC   = -0.35   # ~35 % below center  (negative = below)
CIRCLE_X_FRAC   =  0.35   # ~35 % left/right of center

CX = CIRCLE_X_FRAC * MARKER_WIDTH
CY = CIRCLE_Y_FRAC * MARKER_HEIGHT

# ---- Triangle-type markers (Signs 1-8): inverted equilateral-ish triangle ----
# Detected keypoints: 3 triangle corners + 2 circle centers  (5 pts total)
#
#   TL --------- TR        (top-left, top-right of triangle base)
#     \         /
#      \       /
#       \     /
#        \ _ /
#         AP               (apex pointing down)
#
#  LC           RC         (left circle, right circle)

TRIANGLE_TOP_Y_FRAC =  0.40    # fraction of H above centre
TRIANGLE_APX_Y_FRAC = -0.15    # fraction of H below centre (apex)

MODEL_POINTS_TRIANGLE = np.array([
    [-W,  TRIANGLE_TOP_Y_FRAC * MARKER_HEIGHT, 0],   # top-left  (TL)
    [ W,  TRIANGLE_TOP_Y_FRAC * MARKER_HEIGHT, 0],   # top-right (TR)
    [ 0,  TRIANGLE_APX_Y_FRAC * MARKER_HEIGHT, 0],   # apex      (AP)
    [-CX, CY, 0],                                     # left circle centre (LC)
    [ CX, CY, 0],                                     # right circle centre (RC)
], dtype=np.float64)

# ---- Diamond-type markers (Signs 9+): rotated square ----
# Detected keypoints: 4 diamond corners + 2 circle centers  (6 pts)
#
#        T                 (top corner)
#       / \
#      /   \
#     L     R              (left, right corners)
#      \   /
#       \ /
#        B                 (bottom corner of diamond)
#
#  LC           RC

DIAMOND_HALF = W * 0.90   # diamond "radius" (half-diagonal ≈ 90 % of half-width)

MODEL_POINTS_DIAMOND = np.array([
    [0,              DIAMOND_HALF, 0],   # top    (T)
    [-DIAMOND_HALF,  0,            0],   # left   (L)
    [ DIAMOND_HALF,  0,            0],   # right  (R)
    [0,             -DIAMOND_HALF * 0.55, 0],  # bottom (B) — partially hidden by circles
    [-CX, CY, 0],                        # left circle centre (LC)
    [ CX, CY, 0],                        # right circle centre (RC)
], dtype=np.float64)


def get_model_points(marker_type: str) -> np.ndarray:
    """
    Returns (N, 3) float64 array of 3-D model points for the given marker type.

    Parameters
    ----------
    marker_type : 'triangle' or 'diamond'
    """
    if marker_type == "triangle":
        return MODEL_POINTS_TRIANGLE
    elif marker_type == "diamond":
        return MODEL_POINTS_DIAMOND
    else:
        raise ValueError(f"Unknown marker type: {marker_type!r}")

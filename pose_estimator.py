"""
pose_estimator.py
-----------------
Wraps cv2.solvePnP to estimate the 6-DOF pose of each detected marker
relative to the camera.

Inputs:
  - MarkerDetection  (from detect.py)
  - camera_matrix   (3x3 intrinsic matrix K)
  - dist_coeffs     (distortion coefficients from calibration)

Outputs:
  - PoseResult dataclass: rvec, tvec, rotation matrix R, translation t,
    and convenience helpers (distance, euler angles).
"""

from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional

from marker_model import get_model_points
from detect import MarkerDetection


@dataclass
class PoseResult:
    success:   bool
    rvec:      Optional[np.ndarray] = None   # (3,1) rotation vector
    tvec:      Optional[np.ndarray] = None   # (3,1) translation vector (metres)
    R:         Optional[np.ndarray] = None   # (3,3) rotation matrix
    rms_error: float = 0.0                   # reprojection RMS (pixels)

    @property
    def distance(self) -> float:
        """Euclidean distance from camera to marker centre (metres)."""
        if self.tvec is None:
            return float("nan")
        return float(np.linalg.norm(self.tvec))

    @property
    def euler_degrees(self) -> tuple[float, float, float]:
        """Roll, pitch, yaw in degrees (Rodrigues -> rotation matrix -> Euler)."""
        if self.R is None:
            return (float("nan"),) * 3
        sy = np.sqrt(self.R[0, 0] ** 2 + self.R[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            roll  = np.degrees(np.arctan2( self.R[2, 1],  self.R[2, 2]))
            pitch = np.degrees(np.arctan2(-self.R[2, 0],  sy))
            yaw   = np.degrees(np.arctan2( self.R[1, 0],  self.R[0, 0]))
        else:
            roll  = np.degrees(np.arctan2(-self.R[1, 2],  self.R[1, 1]))
            pitch = np.degrees(np.arctan2(-self.R[2, 0],  sy))
            yaw   = 0.0
        return roll, pitch, yaw

    def __str__(self) -> str:
        if not self.success:
            return "PoseResult(failed)"
        r, p, y = self.euler_degrees
        return (f"dist={self.distance:.3f}m  "
                f"roll={r:.1f}deg  pitch={p:.1f}deg  yaw={y:.1f}deg  "
                f"rms={self.rms_error:.2f}px")


# ── main estimator ─────────────────────────────────────────────────────────────

def estimate_pose(marker:        MarkerDetection,
                  camera_matrix: np.ndarray,
                  dist_coeffs:   np.ndarray,
                  use_extrinsic_guess: bool = False,
                  rvec_init: Optional[np.ndarray] = None,
                  tvec_init: Optional[np.ndarray] = None,
                  ) -> PoseResult:
    """
    Estimate the 6-DOF pose of a single detected marker.

    Requires marker.shape_corners and marker.circle_centres to be set
    (both are populated by detect.detect() when confidence == 1.0).

    Parameters
    ----------
    marker        : MarkerDetection from detect.detect()
    camera_matrix : (3,3) float64 - camera intrinsics K
    dist_coeffs   : (4|5|8,) float64 - distortion coefficients
    use_extrinsic_guess : pass True + rvec/tvec_init for tracking mode
    """
    if marker.shape_corners is None:
        return PoseResult(success=False)
    if len(marker.circle_centres) != 2:
        return PoseResult(success=False)

    # Build ordered 2-D image_points matching model point order:
    #   triangle: [TL, TR, AP, LC, RC]
    #   diamond:  [T,  L,  R,  B,  LC, RC]
    lc = np.array(marker.circle_centres[0], dtype=np.float64).reshape(1, 2)
    rc = np.array(marker.circle_centres[1], dtype=np.float64).reshape(1, 2)
    image_pts = np.vstack([
        marker.shape_corners.astype(np.float64),
        lc,
        rc,
    ])  # (N, 2)

    model_pts = get_model_points(marker.marker_type)   # (N, 3)

    if model_pts.shape[0] != image_pts.shape[0]:
        return PoseResult(success=False)

    ok, rvec, tvec = cv2.solvePnP(
        objectPoints      = model_pts,
        imagePoints       = image_pts,
        cameraMatrix      = camera_matrix,
        distCoeffs        = dist_coeffs,
        rvec              = rvec_init,
        tvec              = tvec_init,
        useExtrinsicGuess = use_extrinsic_guess and rvec_init is not None,
        flags             = cv2.SOLVEPNP_ITERATIVE,
    )

    if not ok:
        return PoseResult(success=False)

    R, _ = cv2.Rodrigues(rvec)

    # reprojection error
    proj, _ = cv2.projectPoints(model_pts, rvec, tvec, camera_matrix, dist_coeffs)
    proj = proj.reshape(-1, 2)
    rms  = float(np.sqrt(np.mean(np.sum((proj - image_pts) ** 2, axis=1))))

    return PoseResult(success=True, rvec=rvec, tvec=tvec, R=R, rms_error=rms)


def draw_pose_axes(frame:         np.ndarray,
                   pose:          PoseResult,
                   camera_matrix: np.ndarray,
                   dist_coeffs:   np.ndarray,
                   axis_length:   float = 0.10) -> np.ndarray:
    """
    Draw XYZ axes on the frame at the marker's pose origin.
    Red=X, Green=Y, Blue=Z.  axis_length in metres.
    """
    if not pose.success:
        return frame

    origin = np.zeros((1, 3), dtype=np.float64)
    axes   = np.float64([[axis_length, 0, 0],
                          [0, axis_length, 0],
                          [0, 0, axis_length]])

    o_pt,  _ = cv2.projectPoints(origin, pose.rvec, pose.tvec, camera_matrix, dist_coeffs)
    ax_pts, _ = cv2.projectPoints(axes,  pose.rvec, pose.tvec, camera_matrix, dist_coeffs)

    o  = tuple(o_pt.reshape(2).astype(int))
    xp = tuple(ax_pts[0].reshape(2).astype(int))
    yp = tuple(ax_pts[1].reshape(2).astype(int))
    zp = tuple(ax_pts[2].reshape(2).astype(int))

    cv2.arrowedLine(frame, o, xp, (0, 0, 255), 2, tipLength=0.2)   # X red
    cv2.arrowedLine(frame, o, yp, (0, 255, 0), 2, tipLength=0.2)   # Y green
    cv2.arrowedLine(frame, o, zp, (255, 0, 0), 2, tipLength=0.2)   # Z blue

    return frame

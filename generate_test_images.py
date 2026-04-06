"""
generate_test_images.py
-----------------------
Generates synthetic test images from the reference markers in input/.

Each reference marker is rendered into a realistic scene by:
  - Cropping the marker from its white background
  - Scaling  (simulating different distances)
  - Rotating (simulating camera tilt)
  - Adding a real-looking background (solid colour, gradient, or texture)
  - Applying noise, blur, and brightness jitter (simulating camera quality)
  - Optionally placing 2 markers in one frame (multi-marker test)

Output goes to  test/  folder.
Filename convention:
  test_p<page>_<variant>.png   — single marker
  test_multi_<a>_<b>_<v>.png  — two markers in one frame

Usage
-----
  python generate_test_images.py               # generate everything
  python generate_test_images.py --count 5     # 5 variants per marker
  python generate_test_images.py --out custom/ # different output folder
  python generate_test_images.py --no-multi    # skip multi-marker images
  python generate_test_images.py --preview     # show each image as it's made
"""

import argparse
import os
import random
import sys
import cv2
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
#  Config
# ──────────────────────────────────────────────────────────────────────────────

INPUT_DIR       = "input"
OUTPUT_DIR      = "test"
CANVAS_SIZE     = (720, 1280)   # (H, W) of every output image
VARIANTS        = 8             # number of augmented variants per marker

SCALE_RANGE     = (0.15, 0.55)  # marker occupies 15–55 % of canvas height
ROTATE_RANGE    = (-18, 18)     # degrees
BRIGHTNESS_RANGE= (0.55, 1.35)  # multiplicative brightness jitter
BLUR_PROB       = 0.5           # probability of applying gaussian blur
NOISE_PROB      = 0.6           # probability of adding noise
NOISE_STD       = 6             # gaussian noise std-dev (0–255 scale)
JPEG_PROB       = 0.4           # probability of JPEG compression artefacts
JPEG_QUALITY    = (40, 80)      # JPEG quality range when applied

SEED            = 42

# Perspective tilt (simulating side-view camera angle)
TILT_ANGLE_RANGE = (25, 75)   # degrees from frontal — 25=gentle, 75=extreme side
TILT_VARIANTS    = 6          # dedicated tilt images per marker

# ──────────────────────────────────────────────────────────────────────────────
#  Background generators
# ──────────────────────────────────────────────────────────────────────────────

def _bg_solid(h, w, rng):
    """Plain coloured background — light colours only (simulates floor/table)."""
    base = rng.integers(160, 240, 3).tolist()
    return np.full((h, w, 3), base, dtype=np.uint8)


def _bg_gradient(h, w, rng):
    """Two-colour gradient."""
    c1 = rng.integers(120, 230, 3).astype(np.float32).reshape(1, 1, 3)
    c2 = rng.integers(120, 230, 3).astype(np.float32).reshape(1, 1, 3)
    direction = rng.choice(["h", "v", "d"])
    if direction == "h":
        t = np.tile(np.linspace(0, 1, w, dtype=np.float32).reshape(1, w, 1), (h, 1, 1))
    elif direction == "v":
        t = np.tile(np.linspace(0, 1, h, dtype=np.float32).reshape(h, 1, 1), (1, w, 1))
    else:
        tx = np.linspace(0, 1, w, dtype=np.float32).reshape(1, w, 1)
        ty = np.linspace(0, 1, h, dtype=np.float32).reshape(h, 1, 1)
        t  = np.clip((tx + ty) / 2, 0, 1)
    img = (1 - t) * c1 + t * c2
    return img.clip(0, 255).astype(np.uint8)


def _bg_texture(h, w, rng):
    """Low-frequency smooth texture (simulates concrete / tarmac)."""
    # use large patches so the upsampled result is smooth, not pixel-noise
    small_h = max(h // 6, 8)
    small_w = max(w // 6, 8)
    base_col = int(rng.integers(140, 210))
    noise = rng.integers(max(base_col - 30, 0), min(base_col + 30, 255),
                         (small_h, small_w, 3), dtype=np.uint8)
    # upsample with cubic to get smooth blobs, then blur to soften edges
    big = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(big, (15, 15), 0)


def _bg_grid(h, w, rng):
    """Light grid pattern (concrete / tile floor)."""
    img  = np.full((h, w, 3), int(rng.integers(180, 220)), dtype=np.uint8)
    step = int(rng.integers(30, 80))
    col  = int(rng.integers(130, 170))
    for x in range(0, w, step):
        cv2.line(img, (x, 0), (x, h - 1), (col, col, col), 1)
    for y in range(0, h, step):
        cv2.line(img, (0, y), (w - 1, y), (col, col, col), 1)
    return img


BG_GENERATORS = [_bg_solid, _bg_gradient, _bg_texture, _bg_grid]


# ──────────────────────────────────────────────────────────────────────────────
#  Marker extraction  (white-background crop → RGBA sprite)
# ──────────────────────────────────────────────────────────────────────────────

def extract_marker_sprite(img_bgr: np.ndarray) -> np.ndarray:
    """
    Given a reference image (white background, black marker),
    return an RGBA sprite where the white background is transparent.
    The sprite is cropped tightly around the marker content.
    """
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # threshold: white BG → 0, marker content → 255
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # find tight bounding box of non-white content
    coords  = cv2.findNonZero(mask)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    # small margin
    pad = 10
    x  = max(x - pad, 0);  y = max(y - pad, 0)
    x2 = min(x + w + 2*pad, img_bgr.shape[1])
    y2 = min(y + h + 2*pad, img_bgr.shape[0])

    crop  = img_bgr[y:y2, x:x2]
    alpha = mask[y:y2, x:x2]
    rgba  = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha
    return rgba


# ──────────────────────────────────────────────────────────────────────────────
#  Augmentation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _rotate_rgba(sprite: np.ndarray, angle: float) -> np.ndarray:
    """Rotate an RGBA image around its centre, expanding canvas to fit."""
    h, w = sprite.shape[:2]
    cx, cy = w / 2, h / 2
    M      = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    # compute new bounding size
    cos = abs(M[0, 0]);  sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += (new_w / 2) - cx
    M[1, 2] += (new_h / 2) - cy

    return cv2.warpAffine(sprite, M, (new_w, new_h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(255, 255, 255, 0))


def _apply_perspective_tilt(sprite: np.ndarray,
                             tilt_deg: float,
                             axis: str,
                             near_side: str) -> np.ndarray:
    """
    Apply a perspective (projective) warp to simulate the marker being viewed
    from the side — i.e. the camera is no longer facing the marker head-on.

    Parameters
    ----------
    sprite   : RGBA image of the marker
    tilt_deg : angle from frontal view in degrees
                 0  = perfectly frontal (no distortion)
                 45 = camera at 45° to the marker plane
                 75 = extreme side view (marker almost edge-on)
    axis     : 'h' = tilt left↔right   (horizontal perspective foreshortening)
               'v' = tilt top↔bottom   (vertical   perspective foreshortening)
    near_side: which side of the marker is closer to the camera
               'h' axis → 'left' or 'right'
               'v' axis → 'top'  or 'bottom'

    How it works
    ------------
    A marker tilted by θ around a vertical axis looks like a trapezoid:
    the far side is compressed by cos(θ), the near side stays full size.
    We realise this with cv2.getPerspectiveTransform on the 4 corners.
    """
    h, w = sprite.shape[:2]
    compress = float(np.cos(np.radians(tilt_deg)))   # 0 (edge-on) … 1 (frontal)

    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    if axis == 'h':
        # horizontal tilt: left/right edges shrink on the far side
        margin = h * (1.0 - compress) / 2.0      # pixels lost on each end of far edge
        if near_side == 'left':
            # left side = near (full height), right side = far (compressed)
            dst = np.float32([
                [0,   0],           # TL stays
                [w,   margin],      # TR moves in
                [w,   h - margin],  # BR moves in
                [0,   h],           # BL stays
            ])
        else:                                     # near_side == 'right'
            dst = np.float32([
                [0,   margin],      # TL moves in
                [w,   0],           # TR stays
                [w,   h],           # BR stays
                [0,   h - margin],  # BL moves in
            ])
    else:
        # vertical tilt: top/bottom edges shrink on the far side
        margin = w * (1.0 - compress) / 2.0
        if near_side == 'bottom':
            # bottom = near (full width), top = far (compressed)
            dst = np.float32([
                [margin,     0],    # TL moves in
                [w - margin, 0],    # TR moves in
                [w,          h],    # BR stays
                [0,          h],    # BL stays
            ])
        else:                                     # near_side == 'top'
            dst = np.float32([
                [0,          0],    # TL stays
                [w,          0],    # TR stays
                [w - margin, h],    # BR moves in
                [margin,     h],    # BL moves in
            ])

    M      = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(sprite, M, (w, h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(255, 255, 255, 0))
    return warped


def _paste_rgba_on_bgr(canvas: np.ndarray,
                        sprite: np.ndarray,
                        cx: int, cy: int) -> np.ndarray:
    """
    Alpha-blend RGBA `sprite` onto `canvas` (BGR) centred at (cx, cy).
    Returns modified canvas.
    """
    sh, sw = sprite.shape[:2]
    ch, cw = canvas.shape[:2]

    # compute paste region
    x0 = cx - sw // 2;  y0 = cy - sh // 2
    x1 = x0 + sw;        y1 = y0 + sh

    # clip to canvas bounds
    sx0 = max(-x0, 0);  sy0 = max(-y0, 0)
    sx1 = sw - max(x1 - cw, 0)
    sy1 = sh - max(y1 - ch, 0)
    cx0 = max(x0, 0);   cy0 = max(y0, 0)
    cx1 = cx0 + (sx1 - sx0)
    cy1 = cy0 + (sy1 - sy0)

    if cx0 >= cw or cy0 >= ch or sx0 >= sx1 or sy0 >= sy1:
        return canvas   # completely off-canvas

    roi  = canvas[cy0:cy1, cx0:cx1].astype(np.float32)
    src  = sprite[sy0:sy1, sx0:sx1]
    alpha = src[:, :, 3:4].astype(np.float32) / 255.0
    bgr   = src[:, :, :3].astype(np.float32)

    blended = alpha * bgr + (1 - alpha) * roi
    canvas[cy0:cy1, cx0:cx1] = blended.astype(np.uint8)
    return canvas


def _apply_camera_effects(img: np.ndarray, rng,
                           blur: bool, noise: bool,
                           brightness: float, jpeg: bool) -> np.ndarray:
    """Apply brightness, blur, noise, JPEG artefacts."""
    out = img.astype(np.float32)

    # brightness
    out = (out * brightness).clip(0, 255)

    if noise:
        n   = rng.normal(0, NOISE_STD, out.shape).astype(np.float32)
        out = (out + n).clip(0, 255)

    out = out.astype(np.uint8)

    if blur:
        k = int(rng.choice([3, 5, 7]))
        out = cv2.GaussianBlur(out, (k, k), 0)

    if jpeg:
        q = int(rng.integers(*JPEG_QUALITY))
        _, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])
        out    = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    return out


# ──────────────────────────────────────────────────────────────────────────────
#  Single-marker image builder
# ──────────────────────────────────────────────────────────────────────────────

def make_single(sprite: np.ndarray,
                rng,
                canvas_h: int = CANVAS_SIZE[0],
                canvas_w: int = CANVAS_SIZE[1]) -> np.ndarray:
    """Render one marker sprite onto a random background with augmentation."""

    # ── random augmentation params ─────────────────────────────────────────────
    scale      = float(rng.uniform(*SCALE_RANGE))
    angle      = float(rng.uniform(*ROTATE_RANGE))
    brightness = float(rng.uniform(*BRIGHTNESS_RANGE))
    do_blur    = rng.random() < BLUR_PROB
    do_noise   = rng.random() < NOISE_PROB
    do_jpeg    = rng.random() < JPEG_PROB
    bg_fn      = BG_GENERATORS[int(rng.integers(len(BG_GENERATORS)))]

    # ── scale sprite ──────────────────────────────────────────────────────────
    sh, sw = sprite.shape[:2]
    target_h = int(canvas_h * scale)
    factor   = target_h / max(sh, 1)
    new_h    = int(sh * factor)
    new_w    = int(sw * factor)
    scaled   = cv2.resize(sprite, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # ── rotate ────────────────────────────────────────────────────────────────
    rotated  = _rotate_rgba(scaled, angle)

    # ── background ────────────────────────────────────────────────────────────
    canvas   = bg_fn(canvas_h, canvas_w, rng)

    # ── position: random but mostly centred ──────────────────────────────────
    rh, rw = rotated.shape[:2]
    margin  = 0.15
    cx = int(rng.uniform(canvas_w * margin + rw // 2,
                          canvas_w * (1 - margin) - rw // 2))
    cy = int(rng.uniform(canvas_h * margin + rh // 2,
                          canvas_h * (1 - margin) - rh // 2))
    cx = int(np.clip(cx, rw // 2, canvas_w - rw // 2))
    cy = int(np.clip(cy, rh // 2, canvas_h - rh // 2))

    canvas = _paste_rgba_on_bgr(canvas, rotated, cx, cy)
    canvas = _apply_camera_effects(canvas, rng, do_blur, do_noise,
                                    brightness, do_jpeg)
    return canvas


# ──────────────────────────────────────────────────────────────────────────────
#  Multi-marker image builder
# ──────────────────────────────────────────────────────────────────────────────

def make_multi(sprites: list,
               rng,
               canvas_h: int = CANVAS_SIZE[0],
               canvas_w: int = CANVAS_SIZE[1]) -> np.ndarray:
    """Render 2 different marker sprites onto the same background."""

    bg_fn  = BG_GENERATORS[int(rng.integers(len(BG_GENERATORS)))]
    canvas = bg_fn(canvas_h, canvas_w, rng)

    # place each sprite in a different half of the canvas
    halves = [(0, canvas_w // 2), (canvas_w // 2, canvas_w)]

    half_w = canvas_w // 2
    for sprite, (x_lo, x_hi) in zip(sprites, halves):
        # scale so sprite height <= 45% canvas and width fits in its half
        max_scale_h = 0.45
        max_scale_w = (x_hi - x_lo) / max(sprite.shape[1], 1) * 0.80
        scale   = float(rng.uniform(0.12, min(max_scale_h, max_scale_w)))
        angle   = float(rng.uniform(*ROTATE_RANGE))
        sh, sw  = sprite.shape[:2]
        new_h   = int(canvas_h * scale)
        factor  = new_h / max(sh, 1)
        new_w   = int(sw * factor)
        scaled  = cv2.resize(sprite, (max(new_w, 1), max(new_h, 1)),
                             interpolation=cv2.INTER_AREA)
        rotated = _rotate_rgba(scaled, angle)

        rh, rw  = rotated.shape[:2]
        cx_min  = x_lo + rw // 2 + 5
        cx_max  = x_hi - rw // 2 - 5
        cy_min  = rh // 2 + 5
        cy_max  = canvas_h - rh // 2 - 5
        # fall back to centre of the half if sprite somehow still too large
        cx = int(rng.uniform(cx_min, cx_max)) if cx_min < cx_max else (x_lo + x_hi) // 2
        cy = int(rng.uniform(cy_min, cy_max)) if cy_min < cy_max else canvas_h // 2
        canvas = _paste_rgba_on_bgr(canvas, rotated, cx, cy)

    brightness = float(rng.uniform(*BRIGHTNESS_RANGE))
    do_blur    = rng.random() < BLUR_PROB
    do_noise   = rng.random() < NOISE_PROB
    do_jpeg    = rng.random() < JPEG_PROB
    canvas = _apply_camera_effects(canvas, rng, do_blur, do_noise,
                                    brightness, do_jpeg)
    return canvas


# ──────────────────────────────────────────────────────────────────────────────
#  Tilted-view image builder
# ──────────────────────────────────────────────────────────────────────────────

# All 4 tilt directions cycled evenly across variants
_TILT_CONFIGS = [
    ('h', 'left'),    # camera to the right  → left side of marker is closer
    ('h', 'right'),   # camera to the left   → right side is closer
    ('v', 'top'),     # camera below          → top of marker is closer
    ('v', 'bottom'),  # camera above          → bottom is closer
]


def make_tilted(sprite: np.ndarray,
                rng,
                tilt_deg: float,
                tilt_axis: str,
                near_side: str,
                canvas_h: int = CANVAS_SIZE[0],
                canvas_w: int = CANVAS_SIZE[1]) -> np.ndarray:
    """
    Render one marker with a perspective side-view tilt onto a random background.

    The pipeline is:
      scale → perspective tilt → small in-plane rotation → paste → camera fx
    """
    # ── scale (use a slightly larger range so tilted markers are visible) ──────
    scale      = float(rng.uniform(0.25, 0.60))
    in_plane   = float(rng.uniform(-8, 8))        # small in-plane rotation on top
    brightness = float(rng.uniform(*BRIGHTNESS_RANGE))
    do_blur    = rng.random() < BLUR_PROB
    do_noise   = rng.random() < NOISE_PROB
    do_jpeg    = rng.random() < JPEG_PROB
    bg_fn      = BG_GENERATORS[int(rng.integers(len(BG_GENERATORS)))]

    sh, sw   = sprite.shape[:2]
    target_h = int(canvas_h * scale)
    factor   = target_h / max(sh, 1)
    scaled   = cv2.resize(sprite,
                           (max(int(sw * factor), 1), max(int(sh * factor), 1)),
                           interpolation=cv2.INTER_AREA)

    # ── perspective tilt ───────────────────────────────────────────────────────
    tilted = _apply_perspective_tilt(scaled, tilt_deg, tilt_axis, near_side)

    # ── small in-plane rotation on top ────────────────────────────────────────
    if abs(in_plane) > 0.5:
        tilted = _rotate_rgba(tilted, in_plane)

    # ── background & paste ────────────────────────────────────────────────────
    canvas = bg_fn(canvas_h, canvas_w, rng)
    rh, rw = tilted.shape[:2]
    margin = 0.12
    cx = int(rng.uniform(canvas_w * margin + rw // 2,
                          canvas_w * (1 - margin) - rw // 2))
    cy = int(rng.uniform(canvas_h * margin + rh // 2,
                          canvas_h * (1 - margin) - rh // 2))
    cx = int(np.clip(cx, rw // 2, canvas_w - rw // 2))
    cy = int(np.clip(cy, rh // 2, canvas_h - rh // 2))
    canvas = _paste_rgba_on_bgr(canvas, tilted, cx, cy)
    canvas = _apply_camera_effects(canvas, rng, do_blur, do_noise,
                                    brightness, do_jpeg)
    return canvas


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic test images for marker detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input",    default=INPUT_DIR,
                        help=f"Reference image folder (default: {INPUT_DIR}/)")
    parser.add_argument("--out",      default=OUTPUT_DIR,
                        help=f"Output folder (default: {OUTPUT_DIR}/)")
    parser.add_argument("--count",    type=int, default=VARIANTS,
                        help=f"Variants per marker (default: {VARIANTS})")
    parser.add_argument("--no-multi",   action="store_true",
                        help="Skip multi-marker composite images")
    parser.add_argument("--tilt-count", type=int, default=TILT_VARIANTS,
                        help=f"Tilted-view variants per marker (default: {TILT_VARIANTS}). "
                              "Set 0 to skip.")
    parser.add_argument("--seed",       type=int, default=SEED,
                        help=f"Random seed (default: {SEED})")
    parser.add_argument("--preview",    action="store_true",
                        help="Show each generated image (press any key)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)

    # ── load reference images ──────────────────────────────────────────────────
    refs = {}   # {page_num: sprite_rgba}
    for fname in sorted(os.listdir(args.input)):
        if not (fname.startswith("page_") and fname.endswith(".png")):
            continue
        try:
            page_num = int(fname[5:-4])
        except ValueError:
            continue
        img = cv2.imread(os.path.join(args.input, fname))
        if img is None:
            print(f"  WARNING: cannot read {fname}, skipping")
            continue
        sprite = extract_marker_sprite(img)
        if sprite is None:
            print(f"  WARNING: no marker content found in {fname}, skipping")
            continue
        refs[page_num] = sprite
        print(f"  Loaded page_{page_num}: sprite {sprite.shape[1]}x{sprite.shape[0]} px")

    if not refs:
        print("ERROR: no reference images found.")
        sys.exit(1)

    total = 0

    # ── single-marker variants ────────────────────────────────────────────────
    print(f"\nGenerating {args.count} variants x {len(refs)} markers "
          f"= {args.count * len(refs)} single-marker images...")

    for page_num, sprite in refs.items():
        for v in range(args.count):
            img  = make_single(sprite, rng)
            name = f"test_p{page_num}_v{v:02d}.png"
            path = os.path.join(args.out, name)
            cv2.imwrite(path, img)
            total += 1
            if args.preview:
                cv2.imshow(name, cv2.resize(img, (900, 506)))
                if cv2.waitKey(0) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    print("Preview aborted.")
                    sys.exit(0)

    # ── multi-marker composites ───────────────────────────────────────────────
    if not args.no_multi:
        page_nums = sorted(refs.keys())
        combos = [(page_nums[i], page_nums[j])
                  for i in range(len(page_nums))
                  for j in range(i + 1, len(page_nums))]

        # generate one composite per combo (or cap at 20)
        combos = combos[:20]
        print(f"Generating {len(combos)} multi-marker composite images...")

        for a, b in combos:
            img  = make_multi([refs[a], refs[b]], rng)
            name = f"test_multi_p{a}_p{b}.png"
            path = os.path.join(args.out, name)
            cv2.imwrite(path, img)
            total += 1
            if args.preview:
                cv2.imshow(name, cv2.resize(img, (900, 506)))
                if cv2.waitKey(0) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    sys.exit(0)

    # ── tilted-view variants ──────────────────────────────────────────────────
    n_tilt = args.tilt_count
    if n_tilt > 0:
        # spread variants evenly across the 4 tilt directions, sampling random
        # angle within TILT_ANGLE_RANGE for each
        print(f"Generating {n_tilt} tilted-view variants x {len(refs)} markers "
              f"= {n_tilt * len(refs)} tilt images...")

        for page_num, sprite in refs.items():
            for v in range(n_tilt):
                cfg_idx  = v % len(_TILT_CONFIGS)
                axis, near_side = _TILT_CONFIGS[cfg_idx]
                tilt_deg = float(rng.uniform(*TILT_ANGLE_RANGE))

                img  = make_tilted(sprite, rng, tilt_deg, axis, near_side)
                name = f"test_p{page_num}_tilt_v{v:02d}.png"
                path = os.path.join(args.out, name)
                cv2.imwrite(path, img)
                total += 1
                if args.preview:
                    label = f"{name}  ({axis}-axis {near_side}  {tilt_deg:.0f}deg)"
                    cv2.imshow(label, cv2.resize(img, (900, 506)))
                    if cv2.waitKey(0) & 0xFF == ord("q"):
                        cv2.destroyAllWindows()
                        sys.exit(0)

    cv2.destroyAllWindows()
    print(f"\nDone. {total} images saved to {args.out}/")
    print(f"  Standard:  {args.count * len(refs)}")
    if not args.no_multi:
        print(f"  Multi:     {len(combos)}")
    if n_tilt > 0:
        print(f"  Tilted:    {n_tilt * len(refs)}")
    print(f"\nTest with:  python detect.py --source {args.out}/")


if __name__ == "__main__":
    main()

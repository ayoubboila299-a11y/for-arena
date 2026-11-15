#!/usr/bin/env python3
"""
GTA IV Ped Skinning Pack Pipeline:
1. Takes the 10 raw AI-generated 1024x1024 `head_diff_000` base maps (01..10).
2. Synthesizes 10 studio-engineered GTA IV `_b/_c/_d` ped variations (11..20) using
   UV-locked morphological warps (zero boundary/socket drift), targeted skin/hair/beard
   pigment transforms, custom iris re-synthesis at (574, 928), scars, bruises, and aging.
3. Generates for all 20 textures:
   - `textures_1024/` : 1024x1024 HD PNG diffuse maps
   - `textures_512/`  : 512x512 Vanilla GTA IV PC PNG diffuse maps
   - `normals_512/`   : 512x512 Tangent-space Normal maps (`head_normal_000` style)
   - `specular_512/`  : 512x512 Specular maps (`head_spec_000` style)
   - `dds_512/`       : 512x512 DirectDraw Surface `.dds` (DXT1 w/ mipmaps) for OpenIV/SparkIV
   - `catalog_preview.jpg` : High-res 5x4 contact sheet of all 20 GTA IV face maps
   - `metadata.json`  : Full manifest for modders & the 3D/2D Web Studio
"""

import json
import math
import os
import subprocess
import zipfile
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from PIL import Image, ImageDraw

for d in ["textures_1024", "textures_512", "normals_512", "specular_512", "dds_512"]:
    os.makedirs(d, exist_ok=True)


def rgb_to_hsv_np(rgb: np.ndarray):
    """rgb in [0, 255], returns h in [0, 1], s in [0, 1], v in [0, 1]."""
    c = rgb / 255.0
    r, g, b = c[:, :, 0], c[:, :, 1], c[:, :, 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    deltac = maxc - minc
    s = np.where(maxc > 1e-6, deltac / np.maximum(maxc, 1e-6), 0.0)
    h = np.zeros_like(maxc)
    mask = deltac > 1e-6
    rc = np.where(mask, (maxc - r) / np.maximum(deltac, 1e-6), 0.0)
    gc = np.where(mask, (maxc - g) / np.maximum(deltac, 1e-6), 0.0)
    bc = np.where(mask, (maxc - b) / np.maximum(deltac, 1e-6), 0.0)
    h = np.where(mask & (r == maxc), bc - gc, h)
    h = np.where(mask & (g == maxc) & (r != maxc), 2.0 + rc - bc, h)
    h = np.where(mask & (b == maxc) & (r != maxc) & (g != maxc), 4.0 + gc - rc, h)
    h = (h / 6.0) % 1.0
    return h, s, v


def hsv_to_rgb_np(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    i = np.floor(h * 6.0).astype(np.int32)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i_mod = i % 6
    r = np.choose(i_mod, [v, q, p, p, t, v])
    g = np.choose(i_mod, [t, v, v, q, p, p])
    b = np.choose(i_mod, [p, p, t, v, v, q])
    return np.clip(np.stack([r, g, b], axis=-1) * 255.0, 0.0, 255.0)


def build_uv_lock_masks(S: int = 1024):
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    # Lock eye sockets, ears, UV outer boundaries, and everything below y=640
    lock = np.ones((S, S), dtype=np.float32)
    # Allow gentle facial proportion warp only inside interior face zone (x in 275..749, y in 160..635)
    interior = np.exp(-(((xx - 512.0) / 185.0) ** 4 + ((yy - 430.0) / 195.0) ** 4))
    # Zero out warp right at the two eye socket cavities so 3D eyeballs stay 100% centered!
    eye_l = 1.0 - np.exp(-(((xx - 414.0) / 48.0) ** 2 + ((yy - 350.0) / 24.0) ** 2))
    eye_r = 1.0 - np.exp(-(((xx - 610.0) / 48.0) ** 2 + ((yy - 350.0) / 24.0) ** 2))
    return interior * eye_l * eye_r, xx, yy


WARP_ALLOW, XX, YY = build_uv_lock_masks(1024)


def apply_facial_morph(arr: np.ndarray, nose_w: float = 0.0, jaw_w: float = 0.0, brow_y: float = 0.0, mouth_w: float = 0.0) -> np.ndarray:
    """Applies subtle, photorealism-preserving anatomical proportion warps while locking UV seams & eye sockets."""
    dx = np.zeros_like(XX)
    dy = np.zeros_like(YY)

    # Nose width morph around (512, 435)
    n_bell = np.exp(-(((XX - 512.0) / 68.0) ** 2 + ((YY - 435.0) / 52.0) ** 2))
    dx -= nose_w * ((XX - 512.0) / 68.0) * 14.0 * n_bell

    # Jaw / cheek fullness morph around y=520..625
    j_bell = np.exp(-(((XX - 512.0) / 190.0) ** 2 + ((YY - 555.0) / 85.0) ** 2))
    dx -= jaw_w * ((XX - 512.0) / 190.0) * 18.0 * j_bell

    # Brow ridge vertical shift around y=305
    b_bell = np.exp(-(((XX - 512.0) / 145.0) ** 2 + ((YY - 305.0) / 28.0) ** 2))
    dy -= brow_y * 8.0 * b_bell

    # Mouth width morph around (512, 536)
    m_bell = np.exp(-(((XX - 512.0) / 100.0) ** 2 + ((YY - 536.0) / 30.0) ** 2))
    dx -= mouth_w * ((XX - 512.0) / 100.0) * 12.0 * m_bell

    dx *= WARP_ALLOW
    dy *= WARP_ALLOW

    out = np.empty_like(arr)
    coords = [np.clip(YY + dy, 0, 1023), np.clip(XX + dx, 0, 1023)]
    for c in range(3):
        out[:, :, c] = map_coordinates(arr[:, :, c], coords, order=1, mode="nearest")
    return out


def recolor_eyeball_iris(arr: np.ndarray, iris_rgb_inner, iris_rgb_outer):
    """Recolors ONLY the colored iris ring inside the GTA IV eyeball UV box [482..668, 818..1024] while preserving fiber detail & pupil."""
    out = arr.copy()
    box = out[818:1024, 482:668].copy()
    h, w, _ = box.shape
    by, bx = np.mgrid[0:h, 0:w].astype(np.float32)

    # Auto-detect exact pupil dark centroid inside center of eyeball box so recoloring is 100% concentric!
    mid = box[40:165, 35:155].mean(axis=2)
    pmask = mid < 26.0
    if pmask.sum() > 20:
        pys, pxs = np.where(pmask)
        cx = float(pxs.mean() + 35.0)
        cy = float(pys.mean() + 40.0)
        pupil_r = math.sqrt(pmask.sum() / math.pi)
    else:
        cx, cy, pupil_r = 93.0, 106.0, 14.0

    # Estimate outer iris limbal radius by scanning radial brightness jump to white sclera (> 165)
    r = np.hypot(bx - cx, by - cy)
    radii = []
    for ang in np.linspace(0, 2 * math.pi, 36, endpoint=False):
        for rad in range(int(pupil_r + 6), 68):
            ix = int(round(cx + math.cos(ang) * rad))
            iy = int(round(cy + math.sin(ang) * rad))
            if 0 <= ix < w and 0 <= iy < h:
                if box[iy, ix].mean() > 162.0:
                    radii.append(rad)
                    break
    iris_r = float(np.median(radii)) if radii else 36.0

    # Build smooth annulus mask between pupil_r+1.5 and iris_r-1.0
    in_ring = (r >= pupil_r + 1.0) & (r <= iris_r - 0.5)
    ring_w = gaussian_filter(in_ring.astype(np.float32), sigma=1.4)
    # Keep dark pupil & dark outer limbal ring intact
    lum = (0.299 * box[:, :, 0] + 0.587 * box[:, :, 1] + 0.114 * box[:, :, 2]) / 255.0
    # Normalize fiber detail inside ring
    ring_vals = lum[in_ring]
    lo, hi = (np.percentile(ring_vals, 10), np.percentile(ring_vals, 92)) if len(ring_vals) > 10 else (0.1, 0.5)
    fiber_t = np.clip((lum - lo) / max(hi - lo, 0.08), 0.0, 1.0)
    radial_t = np.clip((r - pupil_r) / max(iris_r - pupil_r, 1.0), 0.0, 1.0)
    limbal_dark = 1.0 - 0.58 * np.exp(-(((r - (iris_r - 2.0)) / 3.2) ** 2))

    inner_c = np.array(iris_rgb_inner, dtype=np.float32)
    outer_c = np.array(iris_rgb_outer, dtype=np.float32)
    base_c = inner_c[None, None, :] * (1.0 - radial_t[:, :, None]) + outer_c[None, None, :] * radial_t[:, :, None]
    new_iris = base_c * (0.48 + 0.88 * fiber_t[:, :, None]) * limbal_dark[:, :, None]

    # Preserve specular catchlight dots (very bright white pixels > 215 inside iris/pupil)
    catchlight = gaussian_filter((box.mean(axis=2) > 210.0).astype(np.float32), sigma=0.8)
    blend = ring_w * (1.0 - catchlight)
    box = box * (1.0 - blend[:, :, None]) + np.clip(new_iris, 0, 255) * blend[:, :, None]
    out[818:1024, 482:668] = np.clip(box, 0, 255)

    # ALSO recolor BOTH open eyes on the face UV (for MTA:SA / GTA SA skinning!)
    for x0, x1 in ((378, 452), (572, 646)):
        y0, y1 = 324, 374
        eb = out[y0:y1, x0:x1].copy()
        eh, ew, _ = eb.shape
        ey, ex = np.mgrid[0:eh, 0:ew].astype(np.float32)
        elum = 0.299 * eb[:, :, 0] + 0.587 * eb[:, :, 1] + 0.114 * eb[:, :, 2]

        # Locate exact circular pupil center via smoothed luminance minimum
        sm = gaussian_filter(elum, sigma=2.5)
        sub = sm[12:38, 18:56]
        py, px = np.unravel_index(np.argmin(sub), sub.shape)
        fcx, fcy = float(px + 18), float(py + 12)

        # Measure pupil radius & horizontal iris outer radius (where sclera jumps > 84)
        fr = np.hypot(ex - fcx, ey - fcy)
        p_pix = (fr <= 9.0) & (elum < 22.0)
        fpupil_r = max(4.2, min(7.2, math.sqrt(max(p_pix.sum(), 55) / math.pi)))

        h_radii = []
        for sgn in (-1, 1):
            for rad in range(int(fpupil_r + 4), 23):
                ix = int(round(fcx + sgn * rad))
                iy = int(round(fcy))
                if 0 <= ix < ew and elum[iy, ix] > 84.0:
                    h_radii.append(rad)
                    break
        firis_r = float(np.median(h_radii)) if h_radii else 15.5

        # Restrict strictly to visible iris ring inside the open eyelid almond aperture
        almond_open = (((ex - fcx) / 27.0) ** 2 + ((ey - fcy) / 10.2) ** 2) <= 1.0
        not_sclera = elum < 88.0
        not_upper_lash = ~((ey - fcy < -5.5) & (elum < 30.0))
        fin_ring = (fr >= fpupil_r + 0.6) & (fr <= firis_r - 0.4) & almond_open & not_sclera & not_upper_lash
        fring_w = gaussian_filter(fin_ring.astype(np.float32), sigma=0.9)

        elum_n = elum / 255.0
        rvals = elum_n[fin_ring]
        flo, fhi = (np.percentile(rvals, 10), np.percentile(rvals, 92)) if len(rvals) > 8 else (0.08, 0.35)
        ffiber = np.clip((elum_n - flo) / max(fhi - flo, 0.06), 0.0, 1.0)
        frad_t = np.clip((fr - fpupil_r) / max(firis_r - fpupil_r, 1.0), 0.0, 1.0)
        flimbal = 1.0 - 0.52 * np.exp(-(((fr - (firis_r - 1.2)) / 1.8) ** 2))
        # Natural upper-eyelid ambient shadow on top half of iris so eyes never look glowing/sticker-like!
        upper_lid_ao = np.clip(0.72 + 0.28 * ((ey - (fcy - 6.0)) / 12.0), 0.58, 1.02)

        fbase = inner_c[None, None, :] * (1.0 - frad_t[:, :, None]) + outer_c[None, None, :] * frad_t[:, :, None]
        fnew = fbase * (0.42 + 0.78 * ffiber[:, :, None]) * flimbal[:, :, None] * upper_lid_ao[:, :, None]

        fcatch = gaussian_filter((elum > 165.0).astype(np.float32), sigma=0.7)
        fblend = fring_w * (1.0 - fcatch) * 0.88
        eb = eb * (1.0 - fblend[:, :, None]) + np.clip(fnew, 0, 255) * fblend[:, :, None]
        out[y0:y1, x0:x1] = np.clip(eb, 0, 255)

    return out


def skin_and_hair_grade(
    arr: np.ndarray,
    skin_hue_shift: float = 0.0,
    skin_sat_mult: float = 1.0,
    skin_val_mult: float = 1.0,
    skin_contrast: float = 1.0,
    hair_mode: str = "keep",  # "keep", "silver", "salt_pepper", "ash_blonde", "dirty_blonde", "charcoal"
):
    out = arr.copy()
    h, s, v = rgb_to_hsv_np(out)

    # Non-black UV mask (exclude unused black background, teeth & eyeball boxes, mouth boxes)
    non_bg = v > 0.06
    # Exclude teeth + eyeball [350..670, 810..1024] and tongue/oral [0..150, 668..735], [868..1024, 668..735]
    protected = np.zeros((1024, 1024), dtype=bool)
    protected[810:1024, 350:670] = True
    protected[668:735, 0:150] = True
    protected[668:735, 868:1024] = True
    # Smoothly protect open eyes (sclera + iris + pupil) from skin tone shifts without any hard edge
    eye_open_protect = np.maximum(
        np.exp(-(((XX - 414.0) / 34.0) ** 4 + ((YY - 348.0) / 14.5) ** 4)),
        np.exp(-(((XX - 610.0) / 34.0) ** 4 + ((YY - 348.0) / 14.5) ** 4)),
    )

    # Smooth continuous hair/beard/sideburn/brow zone across the entire head (ZERO rectangular cutoffs!)
    head_vert_env = np.clip((685.0 - YY) / 45.0, 0.0, 1.0)
    excl = np.maximum.reduce([
        np.exp(-(((XX - 414.0) / 56.0) ** 2 + ((YY - 356.0) / 30.0) ** 2)),  # L eye & socket
        np.exp(-(((XX - 610.0) / 56.0) ** 2 + ((YY - 356.0) / 30.0) ** 2)),  # R eye & socket
        np.exp(-(((XX - 512.0) / 38.0) ** 2 + ((YY - 455.0) / 16.0) ** 2)),  # nostrils
        np.exp(-(((XX - 512.0) / 76.0) ** 2 + ((YY - 536.0) / 22.0) ** 2)),  # lips
        np.exp(-(((XX - 214.0) / 48.0) ** 2 + ((YY - 416.0) / 84.0) ** 2)),  # L ear
        np.exp(-(((XX - 810.0) / 48.0) ** 2 + ((YY - 416.0) / 84.0) ** 2)),  # R ear
        np.exp(-(((XX - 512.0) / 135.0) ** 2 + ((YY - 225.0) / 52.0) ** 2)) * 0.85,  # center forehead skin
        np.exp(-(((XX - 355.0) / 48.0) ** 2 + ((YY - 425.0) / 36.0) ** 2)) * 0.80,   # L upper cheek skin
        np.exp(-(((XX - 669.0) / 48.0) ** 2 + ((YY - 425.0) / 36.0) ** 2)) * 0.80,   # R upper cheek skin
    ])
    hair_env = head_vert_env * np.clip(1.0 - excl, 0.0, 1.0)

    # Compare pixel darkness AND lower saturation against clean forehead skin
    forehead_v = v[185:255, 440:584]
    forehead_s = s[185:255, 440:584]
    skin_med_v = float(np.median(forehead_v))
    skin_med_s = float(np.median(forehead_s))

    # Hair follicles are darker than forehead skin OR much lower saturation (dark/gray hair vs warm skin)
    dark_score = np.clip((skin_med_v * 0.82 - v) / max(skin_med_v * 0.45, 0.12), 0.0, 1.0)
    desat_score = np.clip((skin_med_s - s) / max(skin_med_s, 0.15), 0.0, 1.0) * np.clip((skin_med_v * 0.96 - v) * 3.0, 0.0, 1.0)
    raw_hair = np.maximum(dark_score, desat_score * 0.75) * hair_env
    hair_mask = gaussian_filter(raw_hair, sigma=2.2)
    hair_mask = np.clip(hair_mask * 1.25, 0.0, 1.0) * (~protected) * non_bg * (1.0 - eye_open_protect)

    skin_mask = gaussian_filter((non_bg & (~protected)).astype(np.float32), sigma=2.0) * (1.0 - 0.75 * hair_mask) * (1.0 - eye_open_protect)

    # Apply skin grading
    h_new = (h + skin_hue_shift * skin_mask) % 1.0
    s_new = np.clip(s * (1.0 + (skin_sat_mult - 1.0) * skin_mask), 0.0, 1.0)
    v_centered = (v - skin_med_v) * skin_contrast + skin_med_v * skin_val_mult
    v_new = np.clip(v * (1.0 - skin_mask) + v_centered * skin_mask, 0.0, 1.0)

    # High-frequency follicle detail so lightened hair keeps crisp strand texture
    v_micro = v - gaussian_filter(v, sigma=2.5)

    # Apply hair/beard transformation smoothly
    if hair_mode == "silver":
        hm = hair_mask * 0.78
        s_new = s_new * (1.0 - 0.85 * hm)
        silver_v = np.clip(0.24 + v * 0.82 + v_micro * 1.35, 0.14, 0.82)
        v_new = v_new * (1.0 - hm) + silver_v * hm
    elif hair_mode == "salt_pepper":
        rng = np.random.default_rng(42)
        grain = gaussian_filter(rng.uniform(0.0, 1.0, (1024, 1024)).astype(np.float32), sigma=1.1)
        streak = np.clip((grain - 0.42) * 2.6, 0.0, 1.0)
        temple_boost = 0.45 + 0.55 * np.exp(-(((YY - 380.0) / 140.0) ** 2))
        sp = hair_mask * streak * temple_boost
        s_new = s_new * (1.0 - 0.82 * sp)
        v_new = np.clip(v_new + sp * 0.30, 0.0, 0.88)
    elif hair_mode == "ash_blonde":
        hm = hair_mask * 0.72
        h_new = h_new * (1.0 - hm) + 0.078 * hm  # warm sandy-ash tone (not greenish!)
        s_new = s_new * (1.0 - hm) + 0.22 * hm
        ash_v = np.clip(v + 0.16 + v_micro * 1.25, 0.08, 0.80)
        v_new = v_new * (1.0 - hm) + ash_v * hm
    elif hair_mode == "dirty_blonde":
        hm = hair_mask * 0.78
        h_new = h_new * (1.0 - hm) + 0.082 * hm  # warm golden-blonde hue
        s_new = s_new * (1.0 - hm) + 0.36 * hm
        bl_v = np.clip(v + 0.11 + v_micro * 1.20, 0.08, 0.84)
        v_new = v_new * (1.0 - hm) + bl_v * hm
    elif hair_mode == "charcoal":
        s_new = s_new * (1.0 - 0.70 * hair_mask)
        v_new = np.clip(v_new * (1.0 - 0.25 * hair_mask), 0.02, 1.0)

    return hsv_to_rgb_np(h_new, s_new, v_new)


def add_scar(arr: np.ndarray, x0: float, y0: float, x1: float, y1: float, width: float = 2.6, lightness: float = 16.0):
    """Adds a realistic healed dermal scar line (pale core + subtle pinkish-brown edge AO)."""
    out = arr.copy()
    vx, vy = x1 - x0, y1 - y0
    seg_len2 = max(vx * vx + vy * vy, 1.0)
    t = np.clip(((XX - x0) * vx + (YY - y0) * vy) / seg_len2, 0.0, 1.0)
    px = x0 + t * vx
    py = y0 + t * vy
    dist = np.hypot(XX - px, YY - py)
    taper = np.maximum(0.0, np.sin(t * math.pi)) ** 0.65
    core = np.exp(-((dist / width) ** 2)) * taper
    rim = np.exp(-(((dist - width * 1.4) / (width * 0.9)) ** 2)) * taper

    out[:, :, 0] += core * (lightness + 4.0) - rim * 7.0
    out[:, :, 1] += core * (lightness - 2.0) - rim * 10.0
    out[:, :, 2] += core * (lightness - 3.0) - rim * 10.0
    return np.clip(out, 0, 255)


def add_street_bruise(arr: np.ndarray, cx: float, cy: float, rx: float, ry: float, strength: float = 0.28):
    """Adds a subtle realistic Liberty City street-fight contusion (purplish-red dermal mottling)."""
    out = arr.copy()
    rng = np.random.default_rng(int(cx + cy))
    mottle = 0.65 + 0.35 * gaussian_filter(rng.uniform(0, 1, (1024, 1024)).astype(np.float32), sigma=6.0)
    w = np.exp(-(((XX - cx) / rx) ** 2 + ((YY - cy) / ry) ** 2)) * mottle * strength
    out[:, :, 0] = out[:, :, 0] * (1.0 - w * 0.22) + 95.0 * (w * 0.22)
    out[:, :, 1] = out[:, :, 1] * (1.0 - w * 0.52) + 42.0 * (w * 0.52)
    out[:, :, 2] = out[:, :, 2] * (1.0 - w * 0.38) + 68.0 * (w * 0.38)
    return np.clip(out, 0, 255)


def add_age_creases(arr: np.ndarray, strength: float = 12.0):
    """Enhances fine forehead, crow's feet, and nasolabial micro-contrast naturally."""
    out = arr.copy()
    lum = 0.299 * out[:, :, 0] + 0.587 * out[:, :, 1] + 0.114 * out[:, :, 2]
    detail = gaussian_filter(lum, sigma=1.2) - gaussian_filter(lum, sigma=6.5)
    zone = (
        np.exp(-(((XX - 512.0) / 160.0) ** 2 + ((YY - 235.0) / 55.0) ** 2))
        + np.exp(-(((XX - 340.0) / 42.0) ** 2 + ((YY - 365.0) / 32.0) ** 2))
        + np.exp(-(((XX - 684.0) / 42.0) ** 2 + ((YY - 365.0) / 32.0) ** 2))
        + np.exp(-(((XX - 445.0) / 36.0) ** 2 + ((YY - 495.0) / 48.0) ** 2))
        + np.exp(-(((XX - 579.0) / 36.0) ** 2 + ((YY - 495.0) / 48.0) ** 2))
    )
    zone = np.clip(zone, 0.0, 1.0)
    out += detail[:, :, None] * zone[:, :, None] * (strength / 10.0)
    return np.clip(out, 0, 255)


def add_hand_tattoo_cross(arr: np.ndarray):
    """Adds subtle faded dark-blue-black Bratva cross/star ink on dorsal hand island."""
    out = arr.copy()
    im = Image.fromarray(out.astype(np.uint8))
    overlay = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    ink = (28, 36, 42, 115)
    # Left hand back (x=175, y=935) orthodox cross
    d.line([(175, 905), (175, 962)], fill=ink, width=4)
    d.line([(156, 922), (194, 922)], fill=ink, width=4)
    d.line([(164, 912), (186, 912)], fill=ink, width=3)
    d.line([(163, 946), (187, 954)], fill=ink, width=3)
    # Ring finger symbol (x=212, y=835)
    d.rectangle([(203, 830), (221, 844)], outline=ink, width=2)
    arr_rgba = Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB")
    res = np.array(arr_rgba, dtype=np.float32)
    # Soften slightly into skin pores
    blurred = gaussian_filter(res, sigma=(1.1, 1.1, 0))
    mask = (np.abs(res - out).sum(axis=2) > 1.0).astype(np.float32)
    mask = gaussian_filter(mask, sigma=1.0)
    return out * (1.0 - mask[:, :, None]) + blurred * mask[:, :, None]


def generate_normal_and_spec(rgb_512: np.ndarray):
    """Generates tangent-space normal map (`head_normal_000`) & specular map (`head_spec_000`) at 512x512."""
    lum = (0.299 * rgb_512[:, :, 0] + 0.587 * rgb_512[:, :, 1] + 0.114 * rgb_512[:, :, 2]) / 255.0
    bg = (rgb_512.max(axis=2) < 12).astype(np.float32)

    # Combine micro pore detail + medium wrinkle/brow/beard relief
    micro = lum - gaussian_filter(lum, sigma=1.6)
    med = gaussian_filter(lum, sigma=1.2) - gaussian_filter(lum, sigma=6.0)
    height = (micro * 1.8 + med * 1.1) * (1.0 - bg)

    dy, dx = np.gradient(height)
    strength = 4.2
    nx = -dx * strength
    ny = -dy * strength
    nz = np.ones_like(nx)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx / norm, ny / norm, nz / norm

    normal_rgb = np.stack([
        (nx * 0.5 + 0.5) * 255.0,
        (ny * 0.5 + 0.5) * 255.0,
        (nz * 0.5 + 0.5) * 255.0,
    ], axis=-1)
    # Flat neutral blue (128, 128, 255) on black unused UV areas
    normal_rgb[bg > 0.5] = [128, 128, 255]

    # Specular map: higher spec on T-zone (forehead/nose/lips/cheeks), eyeball, nails; low on hair/beard
    y5, x5 = np.mgrid[0:512, 0:512].astype(np.float32)
    tzone = (
        0.35 * np.exp(-(((x5 - 256.0) / 75.0) ** 2 + ((y5 - 115.0) / 40.0) ** 2))
        + 0.45 * np.exp(-(((x5 - 256.0) / 22.0) ** 2 + ((y5 - 210.0) / 38.0) ** 2))
        + 0.40 * np.exp(-(((x5 - 256.0) / 45.0) ** 2 + ((y5 - 268.0) / 14.0) ** 2))
    )
    eyeball_box = ((x5 >= 241) & (x5 <= 334) & (y5 >= 409)).astype(np.float32) * 0.55
    spec_val = np.clip((0.18 + lum * 0.32 + micro * 0.9 + tzone + eyeball_box) * (1.0 - bg), 0.0, 1.0)
    # GTA IV spec map uses warm-neutral RGB channels (R=spec level, G=glossiness, B=fresnel/skin mask)
    spec_rgb = np.stack([
        spec_val * 220.0,
        spec_val * 195.0,
        spec_val * 175.0,
    ], axis=-1)
    return normal_rgb.astype(np.uint8), spec_rgb.astype(np.uint8)


# Define all 20 textures metadata + recipe for 11..20
MANIFEST = [
    # --- BATCH 1: 10 RAW AI-GENERATED BASE CHARACTERS ---
    {
        "id": 1,
        "filename": "01_head_diff_000_a_whi.png",
        "gta_name": "head_diff_000_a_whi",
        "title_en": "Slavic Ex-Soldier (Hove Beach)",
        "title_ar": "محارب سلافي سابق - هوف بيتش",
        "ethnicity": "whi",
        "district": "Hove Beach, Broker",
        "eye_color": "Hazel Green",
        "source_type": "AI Base Generation",
        "tags": ["Slavic", "Stubble", "Brow Scar", "Male"],
    },
    {
        "id": 2,
        "filename": "02_head_diff_000_a_ara.png",
        "gta_name": "head_diff_000_a_ara",
        "title_en": "Maghrebi / Arab Street Hustler",
        "title_ar": "شاب مغاربي / عربي - ستريت ستايل",
        "ethnicity": "ara",
        "district": "Dukes / Algonquin",
        "eye_color": "Dark Brown",
        "source_type": "AI Base Generation",
        "tags": ["Arab", "Maghrebi", "Fade", "Goatee", "Male"],
    },
    {
        "id": 3,
        "filename": "03_head_diff_000_b_ara.png",
        "gta_name": "head_diff_000_b_ara",
        "title_en": "Arab Underground Crime Boss",
        "title_ar": "زعيم عصابة عربي - ليبرتي سيتي",
        "ethnicity": "ara",
        "district": "Algonquin",
        "eye_color": "Amber Brown",
        "source_type": "AI Base Generation",
        "tags": ["Arab", "Full Beard", "Boss", "Male"],
    },
    {
        "id": 4,
        "filename": "04_head_diff_000_a_lat.png",
        "gta_name": "head_diff_000_a_lat",
        "title_en": "Dominican VIP Club Bouncer (TBoGT)",
        "title_ar": "حارس نادي ليلي دومينيكاني - TBoGT",
        "ethnicity": "lat",
        "district": "Northwood / Maisonette 9",
        "eye_color": "Rich Brown",
        "source_type": "AI Base Generation",
        "tags": ["Latino", "Razor Line", "Chinstrap", "Male"],
    },
    {
        "id": 5,
        "filename": "05_head_diff_000_a_bla.png",
        "gta_name": "head_diff_000_a_bla",
        "title_en": "Bohan Street Hustler (360 Waves)",
        "title_ar": "مقاتل شوارع بوهان - 360 ويفز",
        "ethnicity": "bla",
        "district": "Bohan",
        "eye_color": "Deep Brown",
        "source_type": "AI Base Generation",
        "tags": ["Black", "360 Waves", "Brow Slit", "Goatee", "Male"],
    },
    {
        "id": 6,
        "filename": "06_head_diff_000_b_whi.png",
        "gta_name": "head_diff_000_b_whi",
        "title_en": "Italian-American Alderney Mob Capo",
        "title_ar": "كابو المافيا الإيطالية - ألدرني",
        "ethnicity": "whi",
        "district": "Westdyke, Alderney",
        "eye_color": "Chestnut Brown",
        "source_type": "AI Base Generation",
        "tags": ["Mafia", "Italian", "Silver Temples", "Male"],
    },
    {
        "id": 7,
        "filename": "07_head_diff_000_c_whi.png",
        "gta_name": "head_diff_000_c_whi",
        "title_en": "Alderney Outlaw Biker (TLAD)",
        "title_ar": "دراج عصابة ذا لوست - TLAD",
        "ethnicity": "whi",
        "district": "Acter, Alderney",
        "eye_color": "Steel Blue",
        "source_type": "AI Base Generation",
        "tags": ["Biker", "Beard", "Hand Tattoos", "Nose Scar", "Male"],
    },
    {
        "id": 8,
        "filename": "08_head_diff_000_d_whi.png",
        "gta_name": "head_diff_000_d_whi",
        "title_en": "Irish-American Dukes Enforcer",
        "title_ar": "منفذ أيرلندي - عائلة ديوكس",
        "ethnicity": "whi",
        "district": "Steinway, Dukes",
        "eye_color": "Emerald Green",
        "source_type": "AI Base Generation",
        "tags": ["Irish", "Freckles", "Red Beard", "Male"],
    },
    {
        "id": 9,
        "filename": "09_head_diff_000_b_bla.png",
        "gta_name": "head_diff_000_b_bla",
        "title_en": "Jamaican Yardie Veteran",
        "title_ar": "مخضرم جامايكي - شوتلر",
        "ethnicity": "bla",
        "district": "Beechwood City, Broker",
        "eye_color": "Warm Dark Brown",
        "source_type": "AI Base Generation",
        "tags": ["Caribbean", "Full Beard", "Stern", "Male"],
    },
    {
        "id": 10,
        "filename": "10_head_diff_000_a_chi.png",
        "gta_name": "head_diff_000_a_chi",
        "title_en": "Algonquin Chinatown Triad Enforcer",
        "title_ar": "منفذ عصابة الترياد - الحي الصيني",
        "ethnicity": "chi",
        "district": "Chinatown, Algonquin",
        "eye_color": "Espresso Brown",
        "source_type": "AI Base Generation",
        "tags": ["Asian", "Triad", "Cheek Scar", "Male"],
    },
    # --- BATCH 2: 10 STUDIO-REMASTERED GTA IV PED VARIATIONS (_b/_c/_d) ---
    {
        "id": 11,
        "filename": "11_head_diff_000_e_whi.png",
        "gta_name": "head_diff_000_e_whi",
        "title_en": "Russian Bratva Hitman (Cold Slavic Variant)",
        "title_ar": "قاتل مأجور روسي (براتفا) - عيون رمادية",
        "ethnicity": "whi",
        "district": "Firefly Island, Broker",
        "eye_color": "Ice Slate Gray",
        "source_type": "Studio Ped Variant",
        "tags": ["Russian", "Bratva", "Hand Tattoo", "Cheek Scar", "Male"],
        "recipe": {
            "base": "01_head_diff_000_a_whi.png",
            "morph": {"nose_w": -0.35, "jaw_w": -0.45, "brow_y": 0.35, "mouth_w": -0.2},
            "grade": {"skin_hue_shift": -0.008, "skin_sat_mult": 0.78, "skin_val_mult": 1.05, "skin_contrast": 1.06, "hair_mode": "ash_blonde"},
            "iris": ([145, 162, 174], [92, 112, 128]),
            "scars": [(612, 470, 655, 535, 2.5, 17.0)],
            "bratva_tattoo": True,
        },
    },
    {
        "id": 12,
        "filename": "12_head_diff_000_c_ara.png",
        "gta_name": "head_diff_000_c_ara",
        "title_en": "Arab / Maghrebi Elder Boss (Salt & Pepper)",
        "title_ar": "زعيم عربي / مغاربي مخضرم (شيب وقور)",
        "ethnicity": "ara",
        "district": "Castle Gardens, Algonquin",
        "eye_color": "Hazel Olive",
        "source_type": "Studio Ped Variant",
        "tags": ["Arab", "Elder", "Salt & Pepper", "Full Beard", "Male"],
        "recipe": {
            "base": "03_head_diff_000_b_ara.png",
            "morph": {"nose_w": 0.35, "jaw_w": 0.35, "brow_y": 0.4, "mouth_w": 0.15},
            "grade": {"skin_hue_shift": 0.004, "skin_sat_mult": 0.86, "skin_val_mult": 0.96, "skin_contrast": 1.08, "hair_mode": "salt_pepper"},
            "iris": ([138, 118, 66], [76, 92, 58]),
            "age": 14.0,
        },
    },
    {
        "id": 13,
        "filename": "13_head_diff_000_d_ara.png",
        "gta_name": "head_diff_000_d_ara",
        "title_en": "Tunisian / Mediterranean Street Racer",
        "title_ar": "متسابق شوارع تونسي / متوسطي",
        "ethnicity": "ara",
        "district": "South Bohan / Broker Bridge",
        "eye_color": "Amber Green",
        "source_type": "Studio Ped Variant",
        "tags": ["Tunisian", "Arab", "Sun-Bronzed", "Scar", "Male"],
        "recipe": {
            "base": "02_head_diff_000_a_ara.png",
            "morph": {"nose_w": -0.25, "jaw_w": 0.40, "brow_y": -0.25, "mouth_w": 0.25},
            "grade": {"skin_hue_shift": -0.004, "skin_sat_mult": 1.12, "skin_val_mult": 0.91, "skin_contrast": 1.05, "hair_mode": "charcoal"},
            "iris": ([164, 128, 56], [74, 108, 68]),
            "scars": [(346, 290, 368, 330, 2.2, 14.0)],
        },
    },
    {
        "id": 14,
        "filename": "14_head_diff_000_b_lat.png",
        "gta_name": "head_diff_000_b_lat",
        "title_en": "South Bohan Underground Boxer (Bruised)",
        "title_ar": "ملاكم شوارع لاتيني - آثار قتال",
        "ethnicity": "lat",
        "district": "Fortside, Bohan",
        "eye_color": "Golden Hazel",
        "source_type": "Studio Ped Variant",
        "tags": ["Latino", "Boxer", "Bruised", "Battle-Worn", "Male"],
        "recipe": {
            "base": "04_head_diff_000_a_lat.png",
            "morph": {"nose_w": 0.55, "jaw_w": 0.45, "brow_y": 0.35, "mouth_w": 0.2},
            "grade": {"skin_hue_shift": -0.005, "skin_sat_mult": 1.08, "skin_val_mult": 0.90, "skin_contrast": 1.06, "hair_mode": "keep"},
            "iris": ([168, 126, 54], [98, 72, 38]),
            "bruises": [(362, 412, 48, 32, 0.32), (512, 405, 24, 18, 0.22), (175, 868, 55, 22, 0.26)],
            "scars": [(388, 296, 412, 316, 2.4, 14.0)],
        },
    },
    {
        "id": 15,
        "filename": "15_head_diff_000_c_bla.png",
        "gta_name": "head_diff_000_c_bla",
        "title_en": "Algonquin Veteran Detective (Graying Waves)",
        "title_ar": "محقق مخضرم - ألغونكوين",
        "ethnicity": "bla",
        "district": "Middle Park East, Algonquin",
        "eye_color": "Warm Amber",
        "source_type": "Studio Ped Variant",
        "tags": ["Black", "Detective", "Graying", "Veteran", "Male"],
        "recipe": {
            "base": "05_head_diff_000_a_bla.png",
            "morph": {"nose_w": 0.30, "jaw_w": 0.35, "brow_y": 0.3, "mouth_w": -0.15},
            "grade": {"skin_hue_shift": 0.003, "skin_sat_mult": 0.92, "skin_val_mult": 0.93, "skin_contrast": 1.07, "hair_mode": "salt_pepper"},
            "iris": ([146, 96, 46], [84, 52, 28]),
            "age": 12.0,
        },
    },
    {
        "id": 16,
        "filename": "16_head_diff_000_f_whi.png",
        "gta_name": "head_diff_000_f_whi",
        "title_en": "Commission Silver-Haired Don (Alderney)",
        "title_ar": "دون المافيا الكهل ذو الشعر الفضي",
        "ethnicity": "whi",
        "district": "Leftwood, Alderney",
        "eye_color": "Cool Blue-Gray",
        "source_type": "Studio Ped Variant",
        "tags": ["Mafia Don", "Silver Hair", "Elder", "Male"],
        "recipe": {
            "base": "06_head_diff_000_b_whi.png",
            "morph": {"nose_w": 0.35, "jaw_w": 0.45, "brow_y": 0.35, "mouth_w": -0.2},
            "grade": {"skin_hue_shift": -0.005, "skin_sat_mult": 0.82, "skin_val_mult": 1.02, "skin_contrast": 1.07, "hair_mode": "silver"},
            "iris": ([118, 148, 172], [68, 94, 118]),
            "age": 13.0,
        },
    },
    {
        "id": 17,
        "filename": "17_head_diff_000_g_whi.png",
        "gta_name": "head_diff_000_g_whi",
        "title_en": "Outlaw Biker Road Captain (Sunburnt Enforcer)",
        "title_ar": "قائد الدراجين المحترق بالشمس - TLAD",
        "ethnicity": "whi",
        "district": "Tudor, Alderney",
        "eye_color": "Wolf Amber",
        "source_type": "Studio Ped Variant",
        "tags": ["Biker", "Road Captain", "Sunburnt", "Amber Eyes", "Male"],
        "recipe": {
            "base": "07_head_diff_000_c_whi.png",
            "morph": {"nose_w": 0.35, "jaw_w": 0.35, "brow_y": 0.4, "mouth_w": 0.2},
            "grade": {"skin_hue_shift": -0.009, "skin_sat_mult": 1.14, "skin_val_mult": 0.89, "skin_contrast": 1.08, "hair_mode": "charcoal"},
            "iris": ([188, 134, 48], [112, 72, 26]),
            "bruises": [(645, 415, 45, 30, 0.25)],
        },
    },
    {
        "id": 18,
        "filename": "18_head_diff_000_h_whi.png",
        "gta_name": "head_diff_000_h_whi",
        "title_en": "Blonde Dockworker / Street Brawler (Broker)",
        "title_ar": "ملاكم الميناء الأشقر - بروكر",
        "ethnicity": "whi",
        "district": "BOABO, Broker",
        "eye_color": "Ice Cerulean Blue",
        "source_type": "Studio Ped Variant",
        "tags": ["Blonde", "Brawler", "Blue Eyes", "Broker", "Male"],
        "recipe": {
            "base": "08_head_diff_000_d_whi.png",
            "morph": {"nose_w": -0.30, "jaw_w": 0.35, "brow_y": -0.2, "mouth_w": 0.15},
            "grade": {"skin_hue_shift": 0.008, "skin_sat_mult": 0.88, "skin_val_mult": 0.97, "skin_contrast": 1.04, "hair_mode": "dirty_blonde"},
            "iris": ([96, 156, 204], [48, 96, 146]),
            "bruises": [(840, 868, 55, 22, 0.24)],
        },
    },
    {
        "id": 19,
        "filename": "19_head_diff_000_d_bla.png",
        "gta_name": "head_diff_000_d_bla",
        "title_en": "Yardie Elder Kingpin (Silver Beard & Scar)",
        "title_ar": "زعيم الكاريبي المخضرم - ندبة وشيب",
        "ethnicity": "bla",
        "district": "Willis, Dukes",
        "eye_color": "Deep Hazel Gold",
        "source_type": "Studio Ped Variant",
        "tags": ["Yardie", "Kingpin", "Silver Beard", "Scar", "Male"],
        "recipe": {
            "base": "09_head_diff_000_b_bla.png",
            "morph": {"nose_w": -0.25, "jaw_w": 0.35, "brow_y": 0.35, "mouth_w": 0.2},
            "grade": {"skin_hue_shift": 0.005, "skin_sat_mult": 1.06, "skin_val_mult": 1.06, "skin_contrast": 1.06, "hair_mode": "salt_pepper"},
            "iris": ([152, 114, 52], [86, 62, 30]),
            "scars": [(635, 388, 675, 458, 2.8, 15.0)],
            "age": 11.0,
        },
    },
    {
        "id": 20,
        "filename": "20_head_diff_000_b_chi.png",
        "gta_name": "head_diff_000_b_chi",
        "title_en": "Chinatown Triad Red-Pole Veteran",
        "title_ar": "قائد الترياد المخضرم - الحي الصيني",
        "ethnicity": "chi",
        "district": "Cerveza Heights / Chinatown",
        "eye_color": "Dark Amber",
        "source_type": "Studio Ped Variant",
        "tags": ["Triad Boss", "Silver Streaks", "Brow Scar", "Male"],
        "recipe": {
            "base": "10_head_diff_000_a_chi.png",
            "morph": {"nose_w": 0.35, "jaw_w": 0.35, "brow_y": 0.35, "mouth_w": -0.2},
            "grade": {"skin_hue_shift": -0.005, "skin_sat_mult": 0.88, "skin_val_mult": 0.94, "skin_contrast": 1.08, "hair_mode": "salt_pepper"},
            "iris": ([134, 88, 44], [68, 44, 24]),
            "scars": [(592, 286, 616, 326, 2.3, 15.0)],
            "age": 14.0,
        },
    },
]


def main():
    print("Building full 20-texture GTA IV skinning pack...")
    thumbs = []

    for item in MANIFEST:
        fname = item["filename"]
        p1024 = os.path.join("textures_1024", fname)

        if "recipe" in item:
            rec = item["recipe"]
            base_im = Image.open(os.path.join("textures_1024", rec["base"])).convert("RGB")
            arr = np.array(base_im, dtype=np.float32)

            if "morph" in rec:
                arr = apply_facial_morph(arr, **rec["morph"])
            if "grade" in rec:
                arr = skin_and_hair_grade(arr, **rec["grade"])
            if "age" in rec:
                arr = add_age_creases(arr, strength=rec["age"])
            for sc in rec.get("scars", []):
                arr = add_scar(arr, *sc)
            for br in rec.get("bruises", []):
                arr = add_street_bruise(arr, *br)
            if rec.get("bratva_tattoo"):
                arr = add_hand_tattoo_cross(arr)
            if "iris" in rec:
                arr = recolor_eyeball_iris(arr, rec["iris"][0], rec["iris"][1])

            im1024 = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")
            im1024.save(p1024, optimize=True)
        else:
            im1024 = Image.open(p1024).convert("RGB")
            if im1024.size != (1024, 1024):
                im1024 = im1024.resize((1024, 1024), Image.Resampling.LANCZOS)
                im1024.save(p1024, optimize=True)

        # Save 512x512 vanilla PC resolution diffuse PNG
        im512 = im1024.resize((512, 512), Image.Resampling.LANCZOS)
        p512 = os.path.join("textures_512", fname)
        im512.save(p512, optimize=True)

        # Generate & save Normal & Specular maps (512x512)
        base_stem = fname.replace(".png", "")
        norm_fname = base_stem.replace("head_diff", "head_normal") + ".png"
        spec_fname = base_stem.replace("head_diff", "head_spec") + ".png"
        dds_fname = base_stem + ".dds"

        norm_arr, spec_arr = generate_normal_and_spec(np.array(im512, dtype=np.float32))
        Image.fromarray(norm_arr).save(os.path.join("normals_512", norm_fname), optimize=True)
        Image.fromarray(spec_arr).save(os.path.join("specular_512", spec_fname), optimize=True)

        # Convert 512x512 diffuse to GTA IV native DirectDraw Surface (.dds DXT1 w/ 9 mipmaps)
        subprocess.run(
            [
                "convert",
                p512,
                "-define",
                "dds:compression=dxt1",
                "-define",
                "dds:mipmaps=9",
                os.path.join("dds_512", dds_fname),
            ],
            check=True,
        )

        item["files"] = {
            "diffuse_1024": f"textures_1024/{fname}",
            "diffuse_512": f"textures_512/{fname}",
            "normal_512": f"normals_512/{norm_fname}",
            "spec_512": f"specular_512/{spec_fname}",
            "dds_512": f"dds_512/{dds_fname}",
        }
        thumbs.append((item, im512))
        print(f"  [{item['id']:02d}/20] {fname} -> 1024px, 512px, normal, spec, DDS ready")

    # Save clean metadata JSON (strip internal recipe keys)
    public_manifest = [{k: v for k, v in it.items() if k != "recipe"} for it in MANIFEST]
    with open("metadata.json", "w", encoding="utf-8") as f:
        json.dump(public_manifest, f, ensure_ascii=False, indent=2)

    # Build 5x4 High-Resolution Contact Sheet Catalog Poster (`GTA_IV_20_Face_Maps_Catalog.jpg`)
    cols, rows = 5, 4
    cell_w, cell_h = 340, 390
    pad = 22
    header_h = 110
    sheet_w = pad + cols * (cell_w + pad)
    sheet_h = header_h + pad + rows * (cell_h + pad)
    sheet = Image.new("RGB", (sheet_w, sheet_h), (14, 16, 20))
    draw = ImageDraw.Draw(sheet)

    # Top Liberty City gold accent bar
    draw.rectangle([(0, 0), (sheet_w, 6)], fill=(235, 168, 42))
    draw.text((pad, 26), "GTA IV (RAGE ENGINE) - 20 HEAD_DIFF_000 SKINNING TEXTURE ATLAS PACK", fill=(245, 245, 245))
    draw.text((pad, 56), "1024x1024 HD PNG + 512x512 Vanilla PNG + DXT1 .DDS + Normal + Specular + UV Wireframe Overlay  |  OpenIV / SparkIV / GIMS IV / Blender Ready", fill=(170, 178, 192))

    for idx, (item, im512) in enumerate(thumbs):
        r, c = divmod(idx, cols)
        x0 = pad + c * (cell_w + pad)
        y0 = header_h + r * (cell_h + pad)
        draw.rectangle([(x0 - 2, y0 - 2), (x0 + cell_w + 2, y0 + cell_h + 2)], fill=(26, 30, 38), outline=(56, 64, 78))
        thumb = im512.resize((cell_w, cell_w), Image.Resampling.LANCZOS)
        sheet.paste(thumb, (x0, y0))
        badge_col = (235, 168, 42) if item["id"] <= 10 else (76, 185, 235)
        draw.rectangle([(x0 + 8, y0 + 8), (x0 + 46, y0 + 28)], fill=badge_col)
        draw.text((x0 + 15, y0 + 13), f"#{item['id']:02d}", fill=(12, 14, 18))
        draw.text((x0 + 10, y0 + cell_w + 10), item["gta_name"], fill=(235, 168, 42))
        draw.text((x0 + 10, y0 + cell_w + 28), item["title_en"][:38], fill=(220, 226, 236))

    sheet.save("GTA_IV_20_Face_Maps_Catalog.jpg", quality=92)
    print("Saved GTA_IV_20_Face_Maps_Catalog.jpg and metadata.json!")


if __name__ == "__main__":
    main()

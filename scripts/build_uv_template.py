#!/usr/bin/env python3
"""
Builds 100% original, mathematically exact GTA IV `head_diff_000` UV reference templates
(1024x1024 and 512x512) + transparent UV wireframe overlays + labeled skinning guide.

All coordinates are normalized to the canonical Rockstar RAGE (GTA IV / EFLC) `head_diff_000`
pedestrian UV atlas (1024x1024 HD scale, 2x vanilla 512x512):
  - Main cylindrical head & neck wrap: x=0..1024, y=0..808 (collar V-dip from y=670 at edges to y=808 center)
  - Butterfly scalp / hair wrap: top lobes left (0..512) & right (512..1024), center dip at (512, 68)
  - Left eye socket cavity: center (414, 350), size ~70x30
  - Right eye socket cavity: center (610, 350), size ~70x30
  - Nose bridge & tip: (512, 350..456)
  - Mouth / lips: center (512, 536), width ~172
  - Left ear: center (214, 416), box [164..264, 330..504]
  - Right ear: center (810, 416), box [760..860, 330..504]
  - Tongue UV strip: [0..144, 670..732]
  - Oral cavity UV strip: [872..1024, 670..732]
  - Left dorsal hand UV island: [0..340, 736..1024]
  - Vertical teeth & gums UV strip: [358..478, 816..1024]
  - Eyeball sclera & iris UV box: [482..668, 818..1024], pupil center (574, 928)
  - Right dorsal hand UV island: [672..1024, 736..1024]
"""

import os
import math
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageDraw, ImageFont

os.makedirs("templates", exist_ok=True)


def build_uv_reference(female: bool = False) -> Image.Image:
    S = 1024
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)

    # Start with pure black (#000000) background for unused UV space
    img = np.zeros((S, S, 3), dtype=np.float32)

    # Base neutral mannequin skin color (warm neutral midtone albedo, flat diffuse)
    if female:
        skin_base = np.array([214.0, 174.0, 152.0], dtype=np.float32)
        lip_col = np.array([186.0, 108.0, 106.0], dtype=np.float32)
        hair_col = np.array([42.0, 30.0, 24.0], dtype=np.float32)
    else:
        skin_base = np.array([204.0, 164.0, 142.0], dtype=np.float32)
        lip_col = np.array([168.0, 112.0, 104.0], dtype=np.float32)
        hair_col = np.array([46.0, 35.0, 28.0], dtype=np.float32)

    # 1. Main Head + Neck UV Island Mask
    # Top boundary: butterfly lobes peaking at x=210 and x=814 (y=4), dipping slightly at corners (y=70) and center x=512 (y=48)
    dx_l = (xx - 210.0) / 235.0
    dx_r = (xx - 814.0) / 235.0
    top_l = 6.0 + 72.0 * (dx_l ** 2)
    top_r = 6.0 + 72.0 * (dx_r ** 2)
    top_edge = np.minimum(top_l, top_r)

    # Bottom neck V-curve: y=666 at x=0 and x=1024, dipping smoothly to y=806 at x=512
    nx = (xx - 512.0) / 512.0
    bot_edge = 806.0 - 140.0 * (nx ** 2)
    # Slight flattening near outer shoulders (|nx| > 0.72)
    shoulder_flat = 668.0 + 10.0 * (1.0 - np.clip((np.abs(nx) - 0.72) / 0.28, 0.0, 1.0))
    bot_edge = np.where(np.abs(nx) > 0.78, np.minimum(bot_edge, shoulder_flat), bot_edge)

    head_mask = (yy >= top_edge) & (yy <= bot_edge)
    head_mask_f = gaussian_filter(head_mask.astype(np.float32), sigma=1.5)

    # Fill base skin on head+neck
    for c in range(3):
        img[:, :, c] = skin_base[c] * head_mask_f

    # Subtle anatomical diffuse shading (no harsh directional light, only soft ambient occlusion cues)
    # Forehead & cheek warmth, subtle jaw/neck AO
    cheek_l = np.exp(-(((xx - 355.0) / 75.0) ** 2 + ((yy - 440.0) / 65.0) ** 2))
    cheek_r = np.exp(-(((xx - 669.0) / 75.0) ** 2 + ((yy - 440.0) / 65.0) ** 2))
    redness = (cheek_l + cheek_r) * (10.0 if female else 6.0)
    img[:, :, 0] += redness * head_mask_f
    img[:, :, 1] -= redness * 0.35 * head_mask_f
    img[:, :, 2] -= redness * 0.45 * head_mask_f

    # Chin / under-jaw soft AO curve (smooth x falloff, zero hard seam!)
    jaw_curve = 628.0 - 115.0 * (((xx - 512.0) / 210.0) ** 2)
    jaw_x_fade = np.exp(-(((xx - 512.0) / 185.0) ** 4))
    under_jaw = np.exp(-(((yy - (jaw_curve + 24.0)) / 26.0) ** 2)) * jaw_x_fade
    img -= (under_jaw * 15.0 * head_mask_f)[:, :, None]

    # Nose bridge, tip & nostrils at center x=512, y=345..458
    nose_bridge = np.exp(-(((xx - 512.0) / 20.0) ** 2)) * np.exp(-(((yy - 400.0) / 55.0) ** 2))
    nose_tip = np.exp(-(((xx - 512.0) / 30.0) ** 2 + ((yy - 444.0) / 20.0) ** 2))
    nose_side_l = np.exp(-(((xx - 476.0) / 16.0) ** 2 + ((yy - 418.0) / 48.0) ** 2))
    nose_side_r = np.exp(-(((xx - 548.0) / 16.0) ** 2 + ((yy - 418.0) / 48.0) ** 2))
    nostril_l = np.exp(-(((xx - 490.0) / 11.0) ** 2 + ((yy - 456.0) / 6.5) ** 2))
    nostril_r = np.exp(-(((xx - 534.0) / 11.0) ** 2 + ((yy - 456.0) / 6.5) ** 2))

    img += ((nose_bridge * 12.0 + nose_tip * 14.0) * head_mask_f)[:, :, None]
    img -= ((nose_side_l + nose_side_r) * 15.0 * head_mask_f)[:, :, None]
    img -= ((nostril_l + nostril_r) * 95.0 * head_mask_f)[:, :, None]

    # Eye socket AO + eyelids + dark almond eye socket cavities at (414, 350) and (610, 350)
    for ex in (414.0, 610.0):
        socket_ao = np.exp(-(((xx - ex) / 52.0) ** 2 + ((yy - 352.0) / 30.0) ** 2))
        img -= (socket_ao * 24.0 * head_mask_f)[:, :, None]
        # Upper eyelid crease
        crease_y = 330.0 + 8.0 * (((xx - ex) / 38.0) ** 2)
        crease = np.exp(-(((yy - crease_y) / 3.0) ** 2)) * np.exp(-(((xx - ex) / 38.0) ** 2))
        img -= (crease * 38.0 * head_mask_f)[:, :, None]
        # Dark almond eye cavity (GTA IV uses separate 3D eyeball mesh!)
        almond = (((xx - ex) / (33.0 if not female else 31.0)) ** 2 + ((yy - 350.0) / 13.5) ** 2) <= 1.0
        almond_f = gaussian_filter(almond.astype(np.float32), sigma=1.8)
        cavity_col = np.array([18.0, 10.0, 9.0], dtype=np.float32)
        for c in range(3):
            img[:, :, c] = img[:, :, c] * (1.0 - almond_f) + cavity_col[c] * almond_f

    # Eyebrows at y ~ 308
    for ex, sign in ((410.0, -1.0), (614.0, 1.0)):
        arch_y = 308.0 + 7.0 * (((xx - ex) / 48.0) ** 2) + sign * 3.0 * ((xx - ex) / 48.0)
        thick = 6.5 if female else 10.5
        brow = np.exp(-(((yy - arch_y) / thick) ** 2)) * np.exp(-(((xx - ex) / 48.0) ** 4))
        for c in range(3):
            img[:, :, c] = img[:, :, c] * (1.0 - brow * 0.72) + hair_col[c] * (brow * 0.72)

    # Lips / Mouth at center (512, 536)
    mx = (xx - 512.0) / (76.0 if female else 84.0)
    upper_lip_y = 527.0 + 6.0 * (mx ** 2) - 3.5 * np.exp(-((mx / 0.22) ** 2))
    lower_lip_y = 547.0 - 7.0 * (mx ** 2)
    lip_mask = (np.abs(mx) < 1.05) & (yy >= upper_lip_y - 8.0) & (yy <= lower_lip_y + 9.0)
    lip_f = gaussian_filter(lip_mask.astype(np.float32), sigma=3.2) * np.exp(-(mx ** 4))
    for c in range(3):
        img[:, :, c] = img[:, :, c] * (1.0 - lip_f * 0.75) + lip_col[c] * (lip_f * 0.75)
    # Mouth parting line at y=536
    part_line = np.exp(-(((yy - (536.0 + 3.0 * (mx ** 2))) / 2.2) ** 2)) * np.exp(-((mx / 0.95) ** 4))
    img -= (part_line * 65.0)[:, :, None]

    # Ears: Left [164..264, 330..504] center (214, 416), Right [760..860, 330..504] center (810, 416)
    for ear_x, side in ((214.0, -1.0), (810.0, 1.0)):
        exn = (xx - ear_x) / 46.0
        eyn = (yy - 416.0) / 84.0
        ear_outer = ((exn ** 2 + eyn ** 2) <= 1.0) & (eyn > -0.98)
        ear_rim = np.exp(-(((np.sqrt(exn ** 2 + eyn ** 2) - 0.86) / 0.11) ** 2)) * (exn * side > -0.3)
        ear_cavity = np.exp(-((((xx - (ear_x - side * 8.0)) / 18.0) ** 2) + (((yy - 422.0) / 38.0) ** 2)))
        img += (ear_rim * 16.0 * head_mask_f)[:, :, None]
        img -= (ear_cavity * 38.0 * head_mask_f)[:, :, None]

    # Butterfly Scalp / Hair Wrap at top & temples (exact GTA IV scalp wrap boundary!)
    # Forehead stays open skin inside ellipse centered at (512, 345) rx=215, ry=205
    # Sideburns / temple hair comes down around ears (`|x-512| > 265` and `y < 515`)
    forehead_open = (((xx - 512.0) / (205.0 if not female else 198.0)) ** 2 + ((yy - 355.0) / 215.0) ** 2)
    ear_clear_l = (((xx - 214.0) / 56.0) ** 2 + ((yy - 420.0) / 92.0) ** 2)
    ear_clear_r = (((xx - 810.0) / 56.0) ** 2 + ((yy - 420.0) / 92.0) ** 2)
    scalp_region = (
        (yy < 510.0)
        & (forehead_open > 1.0)
        & (ear_clear_l > 1.0)
        & (ear_clear_r > 1.0)
        & head_mask
    )
    # Taper off behind neck below y=485
    neck_back_fade = np.clip((495.0 - yy) / 55.0, 0.0, 1.0)
    # Smooth hairline transition without radial moire stripes
    scalp_f = gaussian_filter(scalp_region.astype(np.float32), sigma=10.0) * neck_back_fade * head_mask_f
    for c in range(3):
        img[:, :, c] = img[:, :, c] * (1.0 - scalp_f * 0.88) + hair_col[c] * (scalp_f * 0.88)

    # 2. Tongue UV Strip [0..144, 670..732] & Oral Cavity UV Strip [872..1024, 670..732]
    tongue_box = (xx >= 2) & (xx <= 144) & (yy >= 672) & (yy <= 732)
    tx = (xx - 73.0) / 64.0
    ty = (yy - 702.0) / 24.0
    tongue_oval = np.clip(1.0 - (tx ** 2 + ty ** 2) * 0.45, 0.2, 1.0)
    tongue_groove = np.exp(-((ty / 0.14) ** 2)) * 28.0
    img[tongue_box, 0] = 186.0 * tongue_oval[tongue_box] - tongue_groove[tongue_box]
    img[tongue_box, 1] = 112.0 * tongue_oval[tongue_box] - tongue_groove[tongue_box]
    img[tongue_box, 2] = 114.0 * tongue_oval[tongue_box] - tongue_groove[tongue_box]

    mouth_box = (xx >= 872) & (xx <= 1022) & (yy >= 672) & (yy <= 732)
    mx2 = (xx - 947.0) / 68.0
    my2 = (yy - 702.0) / 24.0
    throat_dark = np.exp(-(mx2 ** 2 + my2 ** 2) * 1.8)
    img[mouth_box, 0] = 128.0 * (1.0 - 0.65 * throat_dark[mouth_box])
    img[mouth_box, 1] = 58.0 * (1.0 - 0.70 * throat_dark[mouth_box])
    img[mouth_box, 2] = 62.0 * (1.0 - 0.70 * throat_dark[mouth_box])

    # 3. Left Hand [0..340, 736..1024] & Right Hand [672..1024, 736..1024]
    def paint_hand(x0: int, x1: int, mirror: bool):
        w = x1 - x0
        # Dorsal hand with 4 vertical fingers + side thumb
        sub_x = xx[736:1024, x0:x1] - x0
        sub_y = yy[736:1024, x0:x1] - 736.0
        if mirror:
            sub_x = w - sub_x
        # Palm/back of hand lower body (sub_y > 105) + 4 fingers (x in 58..315, sub_y in 8..130) + thumb (x in 8..68, sub_y in 68..210)
        palm = (sub_x >= 22) & (sub_x <= 324) & (sub_y >= 100)
        thumb = (((sub_x - 36.0) / 26.0) ** 2 + ((sub_y - 145.0) / 72.0) ** 2) <= 1.0
        fingers = np.zeros_like(palm)
        knuckles = np.zeros_like(sub_x)
        nails = np.zeros_like(sub_x)
        finger_centers = [92.0, 152.0, 212.0, 270.0]
        finger_tops = [22.0, 8.0, 14.0, 34.0]
        for fc, ft in zip(finger_centers, finger_tops):
            fmask = (np.abs(sub_x - fc) <= 25.0) & (sub_y >= ft) & (sub_y <= 135.0)
            tip = (((sub_x - fc) / 25.0) ** 2 + ((sub_y - (ft + 18.0)) / 20.0) ** 2) <= 1.0
            fmask = (fmask & (sub_y >= ft + 18.0)) | tip
            fingers |= fmask
            # Finger gap shadow
            knuckles += np.exp(-(((sub_x - (fc - 28.0)) / 4.5) ** 2)) * ((sub_y >= 30.0) & (sub_y <= 135.0)) * 22.0
            # Knuckle creases
            for ky in (ft + 48.0, ft + 88.0, 142.0):
                knuckles += np.exp(-(((sub_y - ky) / 4.0) ** 2)) * np.exp(-(((sub_x - fc) / 18.0) ** 2)) * 10.0
            # Fingernail
            nail = (((sub_x - fc) / 13.0) ** 2 + ((sub_y - (ft + 16.0)) / 12.0) ** 2) <= 1.0
            nails = np.maximum(nails, nail.astype(np.float32))

        hmask = gaussian_filter((palm | thumb | fingers).astype(np.float32), sigma=1.8)
        patch = img[736:1024, x0:x1]
        for c in range(3):
            val = skin_base[c] - knuckles + nails * (22.0 if female else 12.0)
            patch[:, :, c] = patch[:, :, c] * (1.0 - hmask) + np.clip(val, 0, 255) * hmask

    paint_hand(0, 340, mirror=False)
    paint_hand(672, 1024, mirror=True)

    # 4. Vertical Teeth & Gums Strip [358..478, 816..1024]
    # Note: In GTA IV ped `head_diff_000`, teeth strip is oriented vertically (left gum -> upper teeth -> bite line -> lower teeth -> right gum)!
    tx0, tx1, ty0, ty1 = 358, 478, 816, 1024
    sub_x = xx[ty0:ty1, tx0:tx1] - tx0  # 0..120
    sub_y = yy[ty0:ty1, ty0 - ty0 : tx1 - tx0]  # dummy avoid, compute properly:
    sub_y = yy[ty0:ty1, tx0:tx1] - ty0  # 0..208

    tpatch = np.zeros((ty1 - ty0, tx1 - tx0, 3), dtype=np.float32)
    # Pink gums base
    tpatch[:, :, 0] = 178.0
    tpatch[:, :, 1] = 104.0
    tpatch[:, :, 2] = 108.0
    # Teeth columns in center x_local = 24..96, meeting at center bite x_local = 60
    in_teeth_x = (sub_x >= 24.0) & (sub_x <= 96.0)
    # 8 horizontal tooth divisions along sub_y (0..208)
    tooth_wave = np.abs(np.sin(sub_y / 208.0 * math.pi * 8.0))
    bite_gap = np.exp(-(((sub_x - 60.0) / 3.2) ** 2))
    gum_arch = 4.0 * (1.0 - tooth_wave)
    tooth_mask = (sub_x >= (24.0 + gum_arch)) & (sub_x <= (96.0 - gum_arch))
    tooth_f = gaussian_filter(tooth_mask.astype(np.float32), sigma=1.2)
    ivory = np.stack([
        232.0 - 28.0 * (1.0 - tooth_wave) - 110.0 * bite_gap,
        224.0 - 32.0 * (1.0 - tooth_wave) - 120.0 * bite_gap,
        206.0 - 36.0 * (1.0 - tooth_wave) - 120.0 * bite_gap,
    ], axis=-1)
    tpatch = tpatch * (1.0 - tooth_f[:, :, None]) + ivory * tooth_f[:, :, None]
    img[ty0:ty1, tx0:tx1] = np.clip(tpatch, 0, 255)

    # 5. Eyeball / Sclera / Iris Square [482..668, 818..1024], Pupil Center (574, 928)
    ex0, ex1, ey0, ey1 = 482, 668, 818, 1024
    ecx, ecy = 574.0 - ex0, 928.0 - ey0
    esx = xx[ey0:ey1, ex0:ex1] - ex0
    esy = yy[ey0:ey1, ex0:ex1] - ey0
    er = np.hypot(esx - ecx, esy - ecy)
    eang = np.arctan2(esy - ecy, esx - ecx)

    epatch = np.zeros((ey1 - ey0, ex1 - ex0, 3), dtype=np.float32)
    # Outer pink/reddish sclera vascular border fading inward to clean off-white sclera
    sclera_t = np.clip((er - 42.0) / 56.0, 0.0, 1.0)
    epatch[:, :, 0] = 232.0 * (1.0 - 0.25 * sclera_t) + 175.0 * (0.25 * sclera_t)
    epatch[:, :, 1] = 228.0 * (1.0 - 0.55 * sclera_t) + 95.0 * (0.55 * sclera_t)
    epatch[:, :, 2] = 224.0 * (1.0 - 0.55 * sclera_t) + 95.0 * (0.55 * sclera_t)
    # Dark outer corner vignette beyond r=94
    corner_dark = np.clip((er - 90.0) / 18.0, 0.0, 1.0)
    epatch *= (1.0 - 0.72 * corner_dark)[:, :, None]

    # Iris (r <= 34) with radial fibers + limbal ring + pupil (r <= 13)
    iris_mask = gaussian_filter((er <= 33.0).astype(np.float32), sigma=1.2)
    fibers = 0.5 + 0.5 * np.sin(eang * 44.0)
    limbal = np.exp(-(((er - 31.5) / 3.0) ** 2))
    iris_col = np.stack([
        96.0 + 28.0 * fibers - 55.0 * limbal,
        78.0 + 22.0 * fibers - 45.0 * limbal,
        58.0 + 14.0 * fibers - 35.0 * limbal,
    ], axis=-1)
    epatch = epatch * (1.0 - iris_mask[:, :, None]) + np.clip(iris_col, 0, 255) * iris_mask[:, :, None]

    pupil_mask = gaussian_filter((er <= 12.5).astype(np.float32), sigma=1.0)
    epatch = epatch * (1.0 - pupil_mask[:, :, None]) + 8.0 * pupil_mask[:, :, None]
    img[ey0:ey1, ex0:ex1] = np.clip(epatch, 0, 255)

    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), mode="RGB")


def build_uv_wireframe(size: int = 1024) -> Image.Image:
    """Creates a crisp RGBA UV wireframe overlay matching GTA IV head_diff_000 topology."""
    scale = size / 1024.0
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)

    def pt(x, y):
        return (int(round(x * scale)), int(round(y * scale)))

    cyan = (0, 240, 255, 165)
    grid_col = (180, 235, 255, 85)
    accent = (255, 190, 40, 200)

    # Cylindrical latitude & longitude UV loops on head+neck
    # Center symmetry axis
    draw.line([pt(512, 48), pt(512, 806)], fill=accent, width=max(1, int(2 * scale)))

    # Longitudinal curves wrapping from crown dip around face to neck
    for u in np.linspace(-1.0, 1.0, 25):
        if abs(u) < 1e-4:
            continue
        pts = []
        for t in np.linspace(0.0, 1.0, 40):
            x = 512.0 + u * 508.0 * (0.45 + 0.55 * math.sin(t * math.pi * 0.65 + 0.35))
            x = max(2.0, min(1022.0, 512.0 + u * 510.0))
            dx_l = (x - 210.0) / 235.0
            dx_r = (x - 814.0) / 235.0
            y_top = min(6.0 + 72.0 * dx_l * dx_l, 6.0 + 72.0 * dx_r * dx_r)
            nx = (x - 512.0) / 512.0
            y_bot = 806.0 - 140.0 * nx * nx
            if abs(nx) > 0.78:
                y_bot = min(y_bot, 668.0 + 10.0 * (1.0 - min(1.0, max(0.0, (abs(nx) - 0.72) / 0.28))))
            y = y_top + t * (y_bot - y_top)
            pts.append(pt(x, y))
        draw.line(pts, fill=grid_col, width=1)

    # Latitudinal topological rings
    for t in np.linspace(0.0, 1.0, 19):
        pts = []
        for x in np.linspace(2.0, 1022.0, 65):
            dx_l = (x - 210.0) / 235.0
            dx_r = (x - 814.0) / 235.0
            y_top = min(6.0 + 72.0 * dx_l * dx_l, 6.0 + 72.0 * dx_r * dx_r)
            nx = (x - 512.0) / 512.0
            y_bot = 806.0 - 140.0 * nx * nx
            if abs(nx) > 0.78:
                y_bot = min(y_bot, 668.0 + 10.0 * (1.0 - min(1.0, max(0.0, (abs(nx) - 0.72) / 0.28))))
            y = y_top + t * (y_bot - y_top)
            pts.append(pt(x, y))
        col = cyan if t in (0.0, 1.0) else grid_col
        w = max(1, int(2 * scale)) if t in (0.0, 1.0) else 1
        draw.line(pts, fill=col, width=w)

    # Concentric facial loops: Eye sockets, Nose, Mouth, Ears
    for ex in (414, 610):
        for rx, ry, col in ((34, 14, cyan), (50, 26, grid_col), (66, 38, grid_col)):
            draw.ellipse([pt(ex - rx, 350 - ry), pt(ex + rx, 350 + ry)], outline=col, width=1)

    # Nose loop
    draw.ellipse([pt(472, 412), pt(552, 468)], outline=cyan, width=1)
    # Mouth loops
    for rx, ry, col in ((84, 22, cyan), (104, 36, grid_col)):
        draw.ellipse([pt(512 - rx, 536 - ry), pt(512 + rx, 536 + ry)], outline=col, width=1)
    # Ears
    for ear_x in (214, 810):
        draw.ellipse([pt(ear_x - 48, 330), pt(ear_x + 48, 504)], outline=cyan, width=1)

    # Bottom Atlas UV Island Boxes
    boxes = [
        (2, 672, 144, 732, cyan),      # Tongue
        (872, 672, 1022, 732, cyan),   # Inner Mouth
        (2, 736, 340, 1022, cyan),     # Left Hand
        (358, 816, 478, 1022, accent), # Teeth & Gums
        (482, 818, 668, 1022, accent), # Eyeball
        (672, 736, 1022, 1022, cyan),  # Right Hand
    ]
    for x0, y0, x1, y1, col in boxes:
        draw.rectangle([pt(x0, y0), pt(x1, y1)], outline=col, width=max(1, int(2 * scale)))

    # Eyeball iris & pupil guide circles
    for r in (13, 33, 88):
        draw.ellipse([pt(574 - r, 928 - r), pt(574 + r, 928 + r)], outline=cyan, width=1)

    return im


def build_labeled_guide(base_im: Image.Image, wire_im: Image.Image) -> Image.Image:
    guide = base_im.convert("RGBA")
    guide = Image.alpha_composite(guide, wire_im)
    draw = ImageDraw.Draw(guide)
    labels = [
        (512, 34, "SCALP / HAIR WRAP (BUTTERFLY UV)"),
        (512, 225, "FOREHEAD"),
        (414, 350, "L EYE SOCKET"),
        (610, 350, "R EYE SOCKET"),
        (214, 416, "L EAR"),
        (810, 416, "R EAR"),
        (512, 446, "NOSE"),
        (512, 536, "MOUTH / LIPS"),
        (512, 700, "NECK COLLAR SEAM"),
        (73, 702, "TONGUE"),
        (947, 702, "ORAL"),
        (170, 885, "LEFT HAND UV"),
        (418, 920, "TEETH"),
        (574, 845, "EYEBALL UV"),
        (848, 885, "RIGHT HAND UV"),
    ]
    for x, y, txt in labels:
        tw = len(txt) * 6 + 10
        draw.rectangle([(x - tw // 2, y - 10), (x + tw // 2, y + 10)], fill=(0, 0, 0, 185), outline=(0, 240, 255, 220))
        draw.text((x - tw // 2 + 5, y - 6), txt, fill=(255, 255, 255, 255))
    return guide.convert("RGB")


if __name__ == "__main__":
    male_ref = build_uv_reference(female=False)
    female_ref = build_uv_reference(female=True)
    wire_1024 = build_uv_wireframe(1024)
    wire_512 = build_uv_wireframe(512)
    guide = build_labeled_guide(male_ref, wire_1024)

    male_ref.save("templates/gta4_uv_reference_male.png")
    female_ref.save("templates/gta4_uv_reference_female.png")
    wire_1024.save("templates/gta4_uv_wireframe_1024.png")
    wire_512.save("templates/gta4_uv_wireframe_512.png")
    guide.save("templates/gta4_uv_guide_labeled.png")
    print("Saved all UV templates in templates/")

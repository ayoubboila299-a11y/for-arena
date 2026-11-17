#!/usr/bin/env python3
"""Draw DFF UVs over the experimental patches and flag rectangle boundaries.

Requires Pillow, numpy and a checkout of DragonFF (gtaLib). Pass paths to the
PF_Skins and sa-skins checkouts at the revisions listed in uv_review/README.md.
This is a 2D raster inspection aid, not a RenderWare renderer or an in-game test.
"""
import argparse
import csv
import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser()
parser.add_argument('--dragonff', type=Path, required=True)
parser.add_argument('--skins', type=Path, required=True, help='PF_Skins checkout')
parser.add_argument('--originals', type=Path, required=True, help='sa-skins checkout')
parser.add_argument('--review', type=Path, default=Path('uv_review'))
args = parser.parse_args()
sys.path.insert(0, str(args.dragonff))
from gtaLib import dff, txd

size = 512
rows = []
pages = []
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
for batch in range(1, 7):
    with zipfile.ZipFile(args.review / f'UV-patch-batch-{batch:02d}.zip') as z:
        manifest = json.loads(z.read('manifest.json'))
        for item in manifest:
            model = Path(item['filename']).stem
            original_path = args.originals / 'public/skins' / f'{model}.dff'
            if not original_path.is_file():
                raise FileNotFoundError(original_path)
            d = dff.dff()
            d.load_file(str(original_path))
            t = txd.txd()
            t.load_memory(z.read('txd/' + item['filename']))
            candidates = [x for x in t.native_textures if (x.width, x.height) == (1024, 1024)]
            if len(candidates) != 1:
                raise ValueError(f'{model}: could not identify patched native texture')
            texture = candidates[0]
            texname = texture.name.lower()
            result = Image.frombytes('RGBA', (1024, 1024), bytes(texture.to_rgba())).convert('RGB')
            original = Image.open(args.skins / f"SkinID{item['id']}.png").convert('RGB').resize((1024, 1024), Image.Resampling.LANCZOS)
            enhanced = Image.open(args.skins / 'Authentic_Peds_Remaster' / f"SkinID{item['id']}.png").convert('RGB')
            patchmask = Image.new('1', (size, size), 0)
            pd = ImageDraw.Draw(patchmask)
            for x1, y1, x2, y2 in item['original_restored_rectangles_px']:
                # Conservative approximation at this resolution.
                pd.rectangle((math.floor(x1*size/1024), math.floor(y1*size/1024), math.ceil(x2*size/1024)-1, math.ceil(y2*size/1024)-1), fill=1)
            patch = np.asarray(patchmask, dtype=bool)
            coverage = Image.new('1', (size, size), 0)
            lines = Image.new('RGBA', (size, size), (0, 0, 0, 0))
            covdraw = ImageDraw.Draw(coverage)
            linedraw = ImageDraw.Draw(lines)
            tri_count = crossings = 0
            for clump in d.clumps:
                for g in clump.geometry_list:
                    if not g.uv_layers:
                        continue
                    uv = g.uv_layers[0]
                    for tri in g.triangles:
                        mat = g.materials[tri.material]
                        refs = [x for x in mat.textures if x.name.lower() == texname]
                        if not refs:
                            continue
                        if any(x.uv_addressing != 0x11 for x in refs):
                            raise ValueError(f'{model}: unexpected texture addressing')
                        coords = [uv[tri.a], uv[tri.b], uv[tri.c]]
                        us, vs = [p.u for p in coords], [p.v for p in coords]
                        if any(not math.isfinite(x) or abs(x) > 10 for x in us + vs):
                            raise ValueError(f'{model}: invalid or unsupported UV range')
                        one = Image.new('1', (size, size), 0)
                        onedraw = ImageDraw.Draw(one)
                        # Wrap by drawing translated copies clipped by the tile.
                        for uoff in range(math.floor(min(us)), math.floor(max(us))+1):
                            for voff in range(math.floor(min(vs)), math.floor(max(vs))+1):
                                pts = [((p.u-uoff)*size, (p.v-voff)*size) for p in coords]
                                onedraw.polygon(pts, fill=1)
                                covdraw.polygon(pts, fill=1)
                                linedraw.line(pts+[pts[0]], fill=(255, 0, 0, 135), width=1)
                        mask = np.asarray(one, dtype=bool)
                        if mask.any():
                            tri_count += 1
                            if (mask & patch).any() and (mask & ~patch).any():
                                crossings += 1
            uv_mask = np.asarray(coverage, dtype=bool)
            if not tri_count or not uv_mask.any():
                raise ValueError(f'{model}: no triangles using {texname}')
            covered = round(100 * (uv_mask & patch).sum() / uv_mask.sum(), 1)
            rows.append({'id': item['id'], 'model': model, 'batch': batch, 'uv_triangles': tri_count, 'triangles_crossing_restore_boundary_approx': crossings, 'uv_coverage_in_restored_rectangles_percent_approx': covered})
            page = Image.new('RGB', (1560, 630), 'white')
            draw = ImageDraw.Draw(page)
            draw.text((12, 10), f"ID {item['id']} / {model}.txd  |  batch {batch}", font=font, fill='black')
            draw.text((12, 38), f'{tri_count} mapped triangles; approx {crossings} cross a restoration boundary. UV overlap with restored area: {covered}%.', font=small, fill='black')
            for n, (label, image) in enumerate([('Original', original), ('Enhanced', enhanced), ('Experimental patch', result)]):
                pos = (n*520+4, 98)
                im = image.resize((size, size), Image.Resampling.LANCZOS).convert('RGBA')
                im = Image.alpha_composite(im, lines)
                page.paste(im.convert('RGB'), pos)
                draw.text((pos[0], 72), label + ' + DFF UV wireframe', fill='black', font=small)
            pages.append(page)
            print(model, tri_count, crossings, covered, flush=True)
with (args.review / 'patch-uv-triangle-analysis.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator='\n')
    w.writeheader()
    w.writerows(rows)
pages[0].save(args.review / 'patch-uv-wireframes.pdf', save_all=True, append_images=pages[1:], resolution=130.0, quality=85)

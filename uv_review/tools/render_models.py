#!/usr/bin/env python3
"""Make simplified, static front/back DFF texture previews (not game renders).

Needs numpy, pillow, numba and a DragonFF checkout. See uv_review/README.md.
The renderer omits animation, lighting, mipmaps, bilinear filtering, blending,
material colours and game-specific frame transforms. Axes are estimated from
mesh bounds, so some previews may be rotated or otherwise misleading. It uses
orthographic projection, a z-buffer and nearest-neighbour texture wrapping.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from numba import njit
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument('--dragonff', type=Path, required=True)
ap.add_argument('--skins', type=Path, required=True)
ap.add_argument('--originals', type=Path, required=True)
ap.add_argument('--review', type=Path, default=Path('uv_review'))
ap.add_argument('--images', type=Path, required=True, help='temporary output directory for per-model JPEGs')
args = ap.parse_args()
sys.path.insert(0, str(args.dragonff))
from gtaLib import dff, txd

@njit
def raster(vertices, uv, triangles, texture, image, depth, direction):
    h, w = image.shape[:2]
    th, tw = texture.shape[:2]
    for tri in triangles:
        a, b, c = tri
        x0, y0 = vertices[a, 0], vertices[a, 1]
        x1, y1 = vertices[b, 0], vertices[b, 1]
        x2, y2 = vertices[c, 0], vertices[c, 1]
        den = (y1-y2)*(x0-x2) + (x2-x1)*(y0-y2)
        if abs(den) < 1e-8:
            continue
        left, right = max(0, int(math.floor(min(x0, x1, x2)))), min(w-1, int(math.ceil(max(x0, x1, x2))))
        top, bottom = max(0, int(math.floor(min(y0, y1, y2)))), min(h-1, int(math.ceil(max(y0, y1, y2))))
        for y in range(top, bottom+1):
            for x in range(left, right+1):
                px, py = x+0.5, y+0.5
                u0 = ((y1-y2)*(px-x2)+(x2-x1)*(py-y2))/den
                u1 = ((y2-y0)*(px-x2)+(x0-x2)*(py-y2))/den
                u2 = 1-u0-u1
                if u0 >= -1e-6 and u1 >= -1e-6 and u2 >= -1e-6:
                    z = (u0*vertices[a, 2]+u1*vertices[b, 2]+u2*vertices[c, 2])*direction
                    if z > depth[y, x]:
                        u = u0*uv[a, 0]+u1*uv[b, 0]+u2*uv[c, 0]
                        v = u0*uv[a, 1]+u1*uv[b, 1]+u2*uv[c, 1]
                        u -= math.floor(u)
                        v -= math.floor(v)
                        px_tex = texture[min(th-1, int(v*th)), min(tw-1, int(u*tw))]
                        if px_tex[3] >= 128:
                            image[y, x, :] = px_tex[:3]
                            depth[y, x] = z

font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 17)
small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
args.images.mkdir(parents=True, exist_ok=True)
models = json.loads((args.originals/'data/mta.json').read_text())
rows = []
for item in models:
    ident, model = item['id'], item['model']
    imagefile = args.skins/'Authentic_Peds_Remaster'/f'SkinID{ident}.png'
    if ident == 0 or not imagefile.exists():
        continue
    row = {'id': ident, 'model': model, 'texture': '', 'status': 'error', 'notes': ''}
    try:
        original = Image.open(args.skins/f'SkinID{ident}.png').convert('RGBA')
        enhanced = Image.open(imagefile).convert('RGBA')
        if enhanced.size != (1024, 1024):
            raise ValueError('enhanced texture is not 1024x1024')
        t = txd.txd()
        t.load_file(str(args.originals/'public/skins'/f'{model}.txd'))
        candidates = []
        for tex in t.native_textures:
            if (tex.width, tex.height) == original.size:
                actual = np.frombuffer(bytes(tex.to_rgba()), np.uint8).reshape(tex.height, tex.width, 4)
                error = np.abs(actual[:,:,:3].astype(np.int16)-np.asarray(original)[:,:,:3].astype(np.int16)).mean()
                if error < 1:
                    candidates.append(tex)
        if len(candidates) != 1:
            raise ValueError(f'expected one matching TXD image, found {len(candidates)}')
        target = candidates[0].name.lower()
        row['texture'] = target
        textures = {tex.name.lower(): np.frombuffer(bytes(tex.to_rgba()), np.uint8).reshape(tex.height, tex.width, 4).copy() for tex in t.native_textures}
        d = dff.dff()
        d.load_file(str(args.originals/'public/skins'/f'{model}.dff'))
        geometries = [g for c in d.clumps for g in c.geometry_list if g.vertices and g.triangles and g.uv_layers]
        if len(geometries) != 1:
            raise ValueError(f'expected one textured geometry, found {len(geometries)}')
        g = geometries[0]
        vertices, uv = np.array(g.vertices, dtype=np.float64), np.array(g.uv_layers[0], dtype=np.float64)
        if not np.isfinite(vertices).all() or not np.isfinite(uv).all():
            raise ValueError('invalid vertex or UV')
        mat_groups = []
        missing_textures = []
        target_triangles = 0
        for index, mat in enumerate(g.materials):
            tris = np.array([[tri.a, tri.b, tri.c] for tri in g.triangles if tri.material == index], dtype=np.int64).reshape(-1, 3)
            if not len(tris):
                continue
            refs = [ref.name.lower() for ref in mat.textures]
            matching = [r for r in refs if r in textures]
            if not matching and len(refs) == 1:
                # Some models refer to a shared texture not present in their TXD.
                # Retain geometry with an explicit neutral placeholder, never
                # substitute an unrelated image or claim a complete preview.
                name = refs[0]
                missing_textures.append(name)
                textures[name] = np.array([[[180, 180, 180, 255]]], dtype=np.uint8)
                matching = [name]
            if len(matching) != 1:
                raise ValueError(f'material {index}: expected one texture, found {matching}')
            for ref in mat.textures:
                if ref.name.lower() == matching[0] and ref.uv_addressing != 0x11:
                    raise ValueError(f'material {index}: unsupported address mode {ref.uv_addressing}')
            if matching[0] == target:
                target_triangles += len(tris)
            mat_groups.append((matching[0], tris))
        if not target_triangles:
            raise ValueError('no material references the matched main texture')
        if sum(len(tris) for _, tris in mat_groups) != len(g.triangles):
            raise ValueError('some triangles have invalid material indices')
        lo, hi = vertices.min(axis=0), vertices.max(axis=0)
        extent = hi-lo
        depth_axis, horiz_axis, vertical_axis = np.argsort(extent)
        if extent[depth_axis] <= 0:
            raise ValueError('degenerate mesh bounds')
        width, height = 320, 480
        scale = min((width-20)/extent[horiz_axis], (height-20)/extent[vertical_axis])
        v = np.empty((len(vertices), 3), dtype=np.float64)
        v[:,0] = width/2+(vertices[:,horiz_axis]-(hi[horiz_axis]+lo[horiz_axis])/2)*scale
        v[:,1] = height/2-(vertices[:,vertical_axis]-(hi[vertical_axis]+lo[vertical_axis])/2)*scale
        v[:,2] = vertices[:,depth_axis]
        page = Image.new('RGB', (width*4, height+55), 'white')
        draw = ImageDraw.Draw(page)
        warning = f" - PLACEHOLDER: {','.join(missing_textures)}" if missing_textures else ''
        draw.text((5, 3), f'ID {ident} - {model}.txd - internal texture {target}{warning}', font=font, fill='black')
        for mode in (0, 1):
            texs = dict(textures)
            texs[target] = np.asarray(original if mode == 0 else enhanced)
            for side in (0, 1):
                img = np.full((height, width, 3), 230, np.uint8)
                z = np.full((height, width), -1e10, np.float64)
                for name, tris in mat_groups:
                    raster(v, uv, tris, texs[name], img, z, 1 if side == 0 else -1)
                page.paste(Image.fromarray(img), (width*(mode*2+side), 55))
                draw.text((width*(mode*2+side)+5, 30), f"{'Original' if mode == 0 else 'Enhanced'} - view {'A' if side == 0 else 'B'}", font=small, fill='black')
        page.save(args.images/f'{ident:03d}-{model}.jpg', quality=88)
        row['status'] = 'partial' if missing_textures else 'ok'
        row['notes'] = f'{target_triangles} target triangles; estimated vertical axis {int(vertical_axis)}'
        if missing_textures:
            row['notes'] += '; neutral placeholder for missing texture(s): ' + ','.join(missing_textures)
    except Exception as ex:
        row['notes'] = f'{type(ex).__name__}: {ex}'
    rows.append(row)
    print(ident, model, row['status'], row['notes'], flush=True)

with (args.review/'model-render-status.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['id', 'model', 'texture', 'status', 'notes'], lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)

ok = [r for r in rows if r['status'] in ('ok', 'partial')]
for i in range(0, len(ok), 75):
    pages = [Image.open(args.images/f"{r['id']:03d}-{r['model']}.jpg").convert('RGB') for r in ok[i:i+75]]
    pages[0].save(args.review/f'model-comparison-{i//75+1:02d}.pdf', save_all=True, append_images=pages[1:], resolution=120, quality=83)
print('TOTAL', len(rows), 'OK', sum(r['status'] == 'ok' for r in rows), 'PARTIAL', sum(r['status'] == 'partial' for r in rows), 'ERRORS', sum(r['status'] == 'error' for r in rows))

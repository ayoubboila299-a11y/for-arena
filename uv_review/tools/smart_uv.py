#!/usr/bin/env python3
"""Experimental UV-aware texture change triage and opt-in TXD proposal export.

Not semantic recognition: scores are only a heuristic. See SMART-UV.md.
"""
import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from numba import njit

SIZE = 256
GRID = 16


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chunk_iter(data):
    pos = 0
    while pos < len(data):
        if pos + 12 > len(data):
            raise ValueError('truncated chunk header')
        typ, length, version = struct.unpack_from('<III', data, pos)
        end = pos + 12 + length
        if end > len(data):
            raise ValueError('truncated chunk')
        yield typ, version, data[pos+12:end]
        pos = end


def pack(typ, version, data):
    return struct.pack('<III', typ, len(data), version) + data


def replace_texture(source, name, image):
    if len(source) < 12:
        raise ValueError('truncated TXD')
    typ, length, version = struct.unpack_from('<III', source)
    if typ != 22 or length + 12 > len(source) or any(source[length+12:]):
        raise ValueError('unsupported TXD container or trailing data')
    out, count = [], 0
    for ct, cv, body in chunk_iter(source[12:length+12]):
        if ct == 21:
            sub = []
            for st, sv, data in chunk_iter(body):
                if st == 1 and len(data) >= 88 and data[8:40].split(b'\0', 1)[0].decode('ascii').lower() == name.lower():
                    header = bytearray(data[:88])
                    if struct.unpack_from('<I', header)[0] != 9:
                        raise ValueError('only D3D9 native textures supported')
                    flags = struct.unpack_from('<I', header, 72)[0]
                    struct.pack_into('<I', header, 72, (flags & 255) | 0x600)
                    struct.pack_into('<I', header, 76, 22)
                    struct.pack_into('<HHBBB', header, 80, image.width, image.height, 32, 1, header[86])
                    header[87], header[4] = 0, 2
                    pixels = image.tobytes('raw', 'BGRA')
                    data = bytes(header) + struct.pack('<I', len(pixels)) + pixels
                    count += 1
                sub.append(pack(st, sv, data))
            body = b''.join(sub)
        out.append(pack(ct, cv, body))
    if count != 1:
        raise ValueError(f'expected one texture named {name}, found {count}')
    return pack(typ, version, b''.join(out)) + source[length+12:]


def coverage(d, target):
    mask = Image.new('1', (SIZE, SIZE))
    draw = ImageDraw.Draw(mask)
    triangles = []
    for clump in d.clumps:
        for g in clump.geometry_list:
            if not g.uv_layers:
                continue
            uv = g.uv_layers[0]
            for tri in g.triangles:
                mat = g.materials[tri.material]
                refs = [x for x in mat.textures if x.name.lower() == target.lower()]
                if not refs:
                    continue
                if any(x.uv_addressing != 0x11 for x in refs):
                    raise ValueError('only wrapped UV addressing is supported')
                points = [uv[tri.a], uv[tri.b], uv[tri.c]]
                us, vs = [p.u for p in points], [p.v for p in points]
                if any(not math.isfinite(x) or abs(x) > 10 for x in us + vs):
                    raise ValueError('invalid or unsupported UV range')
                polygons = []
                for uoff in range(math.floor(min(us)), math.floor(max(us)) + 1):
                    for voff in range(math.floor(min(vs)), math.floor(max(vs)) + 1):
                        poly = [((p.u-uoff)*SIZE, (p.v-voff)*SIZE) for p in points]
                        draw.polygon(poly, fill=1)
                        polygons.append(poly)
                triangles.append(polygons)
    if not triangles or not np.asarray(mask).any():
        raise ValueError(f'no mapped triangles for texture {target}')
    return np.asarray(mask, dtype=bool), triangles


def triangle_crossings(triangles, selected):
    crossings = 0
    for polys in triangles:
        img = Image.new('1', (SIZE, SIZE))
        draw = ImageDraw.Draw(img)
        for poly in polys:
            draw.polygon(poly, fill=1)
        triangle = np.asarray(img, dtype=bool)
        if (triangle & selected).any() and (triangle & ~selected).any():
            crossings += 1
    return crossings


def score_tiles(original, enhanced, uv, threshold, max_fraction):
    """Find UV-covered tiles with colour/edge changes. Scores are not probabilities."""
    a = np.asarray(original.convert('RGB').resize((SIZE, SIZE), Image.Resampling.LANCZOS), dtype=np.float32) / 255
    b = np.asarray(enhanced.convert('RGB').resize((SIZE, SIZE), Image.Resampling.LANCZOS), dtype=np.float32) / 255
    # Spatially averaged colour and edge changes, not just pixel-level HD detail.
    a = np.asarray(Image.fromarray((a * 255).astype('uint8')).filter(ImageFilter.GaussianBlur(2)), dtype=np.float32) / 255
    b = np.asarray(Image.fromarray((b * 255).astype('uint8')).filter(ImageFilter.GaussianBlur(2)), dtype=np.float32) / 255
    colour = np.abs(a-b).mean(axis=2)
    gray_a, gray_b = a.mean(axis=2), b.mean(axis=2)
    gy_a, gx_a = np.gradient(gray_a)
    gy_b, gx_b = np.gradient(gray_b)
    edges = (np.abs(gx_a-gx_b) + np.abs(gy_a-gy_b)) / 2
    scores = np.zeros((GRID, GRID), dtype=np.float32)
    eligible = []
    tile = SIZE // GRID
    for y in range(GRID):
        for x in range(GRID):
            region = np.s_[y*tile:(y+1)*tile, x*tile:(x+1)*tile]
            m = uv[region]
            if m.mean() < 0.10:
                continue
            score = float(colour[region][m].mean() + 2 * edges[region][m].mean())
            scores[y, x] = score
            if score >= threshold:
                eligible.append((score, y, x))
    # Hard cap avoids silently restoring a whole atlas; still requires review.
    limit = int(GRID * GRID * max_fraction)
    eligible.sort(reverse=True)
    selected = np.zeros((SIZE, SIZE), dtype=bool)
    chosen = []
    for score, y, x in eligible[:limit]:
        selected[y*tile:(y+1)*tile, x*tile:(x+1)*tile] = True
        chosen.append({'x': x, 'y': y, 'score': round(score, 4)})
    return scores, selected, chosen, len(eligible)


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


def model_preview(mesh, old, target, original, enhanced, fixed, out):
    geometries = [g for c in mesh.clumps for g in c.geometry_list if g.vertices and g.triangles and g.uv_layers]
    if len(geometries) != 1:
        raise ValueError('model preview supports exactly one textured geometry')
    g = geometries[0]
    vertices, uv = np.array(g.vertices, dtype=np.float64), np.array(g.uv_layers[0], dtype=np.float64)
    if not np.isfinite(vertices).all() or not np.isfinite(uv).all():
        raise ValueError('invalid model coordinates')
    textures = {t.name.lower(): np.frombuffer(bytes(t.to_rgba()), np.uint8).reshape(t.height, t.width, 4).copy() for t in old.native_textures}
    groups, missing = [], []
    for index, mat in enumerate(g.materials):
        tris = np.array([[t.a, t.b, t.c] for t in g.triangles if t.material == index], dtype=np.int64).reshape(-1, 3)
        if not len(tris):
            continue
        refs = [r.name.lower() for r in mat.textures]
        found = [r for r in refs if r in textures]
        if not found and len(refs) == 1:
            name = refs[0]
            missing.append(name)
            textures[name] = np.array([[[180, 180, 180, 255]]], dtype=np.uint8)
            found = [name]
        if len(found) != 1:
            raise ValueError(f'material {index}: cannot identify its texture')
        if any(r.uv_addressing != 0x11 for r in mat.textures if r.name.lower() == found[0]):
            raise ValueError(f'material {index}: unsupported UV addressing')
        groups.append((found[0], tris))
    if sum(len(t) for _, t in groups) != len(g.triangles):
        raise ValueError('invalid triangle material indices')
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    extent = hi - lo
    depth, horiz, vertical = np.argsort(extent)
    if min(extent) <= 0:
        raise ValueError('degenerate mesh bounds')
    width, height = 240, 360
    scale = min((width-20)/extent[horiz], (height-20)/extent[vertical])
    v = np.empty((len(vertices), 3), dtype=np.float64)
    v[:, 0] = width/2 + (vertices[:, horiz]-(hi[horiz]+lo[horiz])/2)*scale
    v[:, 1] = height/2 - (vertices[:, vertical]-(hi[vertical]+lo[vertical])/2)*scale
    v[:, 2] = vertices[:, depth]
    page = Image.new('RGB', (width*6, height+40), 'white')
    draw = ImageDraw.Draw(page)
    for mode, texture in enumerate([original, enhanced, fixed]):
        texs = dict(textures)
        texs[target.lower()] = np.asarray(texture.convert('RGBA'))
        for side in range(2):
            img = np.full((height, width, 3), 230, np.uint8)
            z = np.full((height, width), -1e10, np.float64)
            for name, tris in groups:
                raster(v, uv, tris, texs[name], img, z, 1 if side == 0 else -1)
            x = width*(mode*2+side)
            draw.text((x+5, 8), f"{['Original', 'Enhanced', 'Proposal'][mode]} view {side+1}", fill='black')
            page.paste(Image.fromarray(img), (x, 40))
    page.save(out / 'model-preview.jpg', quality=90)
    return missing

def prepare(args, dff, txd):
    models = json.loads((args.originals / 'data/mta.json').read_text())
    items = [m for m in models if m['id'] == args.id and m['id'] != 0]
    if len(items) != 1:
        raise ValueError('ID not found or CJ excluded')
    model = items[0]['model']
    if Path(model).name != model:
        raise ValueError('unsafe model name')
    original_path = args.skins / f'SkinID{args.id}.png'
    enhanced_path = args.skins / 'Authentic_Peds_Remaster' / f'SkinID{args.id}.png'
    txd_path = args.originals / 'public/skins' / f'{model}.txd'
    dff_path = args.originals / 'public/skins' / f'{model}.dff'
    original = Image.open(original_path).convert('RGBA')
    enhanced = Image.open(enhanced_path).convert('RGBA')
    if enhanced.size != (1024, 1024) or original.getchannel('A').getextrema() != (255, 255) or enhanced.getchannel('A').getextrema() != (255, 255):
        raise ValueError('requires opaque original and 1024x1024 opaque enhanced image')
    old = txd.txd()
    old.load_file(str(txd_path))
    candidates = []
    for t in old.native_textures:
        if (t.width, t.height) == original.size:
            actual = np.frombuffer(bytes(t.to_rgba()), dtype=np.uint8).reshape(t.height, t.width, 4)
            if np.abs(actual[:, :, :3].astype(np.int16) - np.asarray(original)[:, :, :3].astype(np.int16)).mean() < 1:
                candidates.append(t)
    if len(candidates) != 1:
        raise ValueError(f'expected one matching TXD texture, found {len(candidates)}')
    target = candidates[0].name
    mesh = dff.dff()
    mesh.load_file(str(dff_path))
    uv, triangles = coverage(mesh, target)
    scores, selected, chosen, eligible = score_tiles(original, enhanced, uv, args.threshold, args.max_fraction)
    mask = Image.fromarray((selected * 255).astype('uint8')).resize(enhanced.size, Image.Resampling.NEAREST)
    fixed = Image.composite(original.resize(enhanced.size, Image.Resampling.LANCZOS), enhanced, mask)
    args.out.mkdir(parents=True, exist_ok=False)
    original.resize(enhanced.size, Image.Resampling.LANCZOS).save(args.out / 'original.png')
    enhanced.save(args.out / 'enhanced.png')
    mask.save(args.out / 'mask.png')
    fixed.save(args.out / 'proposal.png')
    heat = Image.new('RGB', (SIZE, SIZE), (30, 30, 30))
    hd = ImageDraw.Draw(heat)
    for y in range(GRID):
        for x in range(GRID):
            intensity = min(255, int(scores[y, x] / 0.5 * 255))
            hd.rectangle((x*16, y*16, x*16+15, y*16+15), fill=(intensity, max(0, 100-intensity//2), 0))
            if selected[y*16, x*16]:
                hd.rectangle((x*16, y*16, x*16+15, y*16+15), outline='white', width=1)
    heat = heat.resize((512, 512), Image.Resampling.NEAREST)
    page = Image.new('RGB', (2048, 550), 'white')
    draw = ImageDraw.Draw(page)
    for i, (label, image) in enumerate([('Original (upscaled)', original), ('Enhanced', enhanced), ('Change score (not diagnosis)', heat), ('Proposed restoration', fixed)]):
        draw.text((i*512+8, 8), label, fill='black')
        page.paste(image.convert('RGB').resize((512, 512), Image.Resampling.LANCZOS), (i*512, 30))
    page.save(args.out / 'preview.jpg', quality=90)
    missing = model_preview(mesh, old, target, original, enhanced, fixed, args.out)
    report = {'id': args.id, 'model': model, 'texture': target, 'threshold': args.threshold, 'max_atlas_fraction': args.max_fraction,
              'model_preview_missing_textures': missing, 'uv_triangles': len(triangles), 'approx_triangles_crossing_boundary': triangle_crossings(triangles, selected),
              'uv_coverage_fraction': round(float(uv.mean()), 4), 'restored_atlas_fraction': round(float(selected.mean()), 4),
              'restored_uv_fraction': round(float((uv & selected).sum() / uv.sum()), 4),
              'eligible_tiles': eligible, 'selected_tiles': chosen,
              'note': 'Heuristic change triage only; no semantic body-part or wrong-UV classification. False positives and negatives likely. Review model and in game.',
              'inputs': {str(p.resolve()): sha(p) for p in [original_path, enhanced_path, txd_path, dff_path]},
              'txd_path': str(txd_path.resolve()), 'artifacts': {name: sha(args.out / name) for name in ['original.png', 'enhanced.png', 'mask.png', 'proposal.png']}}
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['id', 'model', 'texture', 'eligible_tiles', 'restored_atlas_fraction', 'approx_triangles_crossing_boundary']}, indent=2))


def approve(args, txd):
    folder = args.proposal
    report = json.loads((folder / 'report.json').read_text())
    for name, expected in report['artifacts'].items():
        if Path(name).name != name or sha(folder / name) != expected:
            raise ValueError(f'proposal artifact changed: {name}; regenerate instead')
    for path, expected in report['inputs'].items():
        if sha(Path(path)) != expected:
            raise ValueError(f'input changed: {path}; regenerate instead')
    if not report['selected_tiles']:
        raise ValueError('no candidate restoration; nothing to export')
    model = report['model']
    if Path(model).name != model:
        raise ValueError('unsafe model name')
    dest = args.output / (model + '.txd')
    args.output.mkdir(parents=True, exist_ok=True)
    source = Path(report['txd_path']).read_bytes()
    before = txd.txd()
    before.load_memory(source)
    proposal = Image.open(folder / 'proposal.png').convert('RGBA')
    result = replace_texture(source, report['texture'], proposal)
    after = txd.txd()
    after.load_memory(result)
    if len(before.native_textures) != len(after.native_textures):
        raise ValueError('native texture count changed')
    matched = [t for t in after.native_textures if t.name.lower() == report['texture'].lower()]
    if len(matched) != 1 or bytes(matched[0].to_rgba()) != proposal.tobytes():
        raise ValueError('output texture verification failed')
    for tex in before.native_textures:
        if tex.name.lower() != report['texture'].lower():
            others = [x for x in after.native_textures if x.name == tex.name]
            if len(others) != 1 or tex.to_mem() != others[0].to_mem():
                raise ValueError('additional texture changed')
    with dest.open('xb') as f:
        f.write(result)
    print(f'Experimental TXD written to {dest}; test in game after backup.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare', help='generate a reviewable proposal, never modifies input')
    p.add_argument('--dragonff', type=Path, required=True)
    p.add_argument('--skins', type=Path, required=True)
    p.add_argument('--originals', type=Path, required=True)
    p.add_argument('--id', type=int, required=True)
    p.add_argument('--out', type=Path, required=True, help='new output directory')
    p.add_argument('--threshold', type=float, default=0.18)
    p.add_argument('--max-fraction', type=float, default=0.15)
    p = sub.add_parser('approve', help='explicitly export verified proposed TXD')
    p.add_argument('--dragonff', type=Path, required=True)
    p.add_argument('--proposal', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.command == 'prepare' and (not 0 < args.threshold <= 1 or not 0 < args.max_fraction <= 1):
        ap.error('threshold and max-fraction must be between 0 and 1')
    sys.path.insert(0, str(args.dragonff))
    from gtaLib import dff, txd
    try:
        if args.command == 'prepare':
            prepare(args, dff, txd)
        else:
            approve(args, txd)
    except (ValueError, OSError, KeyError) as e:
        ap.exit(1, f'Error: {e}\n')


if __name__ == '__main__':
    main()

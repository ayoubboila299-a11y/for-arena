#!/usr/bin/env python3
"""Build optional, experimental UV-guided patch batch 05.

Requires Pillow, numpy, and the gtaLib package from a DragonFF checkout.
Input checkouts must use the revisions in uv_review/README.md. See README for
important limitations; visual inspection on a DFF renderer/in game is required.
"""
import argparse
import json
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser()
parser.add_argument('--dragonff', type=Path, required=True)
parser.add_argument('--skins', type=Path, required=True)
parser.add_argument('--originals', type=Path, required=True)
parser.add_argument('--review', type=Path, default=Path('uv_review'))
args = parser.parse_args()
sys.path.insert(0, str(args.dragonff))
from gtaLib import txd

# Pixel bounds: [left, top, right, bottom) in the 1024x1024 texture.
# The selected areas address visible mapped-atlas changes in simplified DFF previews.
spec = [
    (14, 'bmori', [(410, 250, 1024, 532)], 'restore skin of arms that was replaced by white cloth'),
    (15, 'bmost', [(0, 360, 800, 615)], 'restore trousers where architectural imagery was added'),
    (78, 'wmotr1', [(220, 290, 375, 600)], 'restore shirt area where a nose and face texture were added'),
    (93, 'wfyst', [(600, 0, 1024, 330)], 'restore black top overwritten by skin texture'),
    (115, 'vla2', [(0, 0, 525, 535)], 'restore jacket area overwritten by skin texture'),
    (117, 'triada', [(0, 0, 560, 340)], 'restore back of suit affected by an added shirt collar'),
    (122, 'dnb2', [(0, 500, 515, 705)], 'restore front of black shirt overwritten by skin and tattoos'),
    (162, 'cwmyhb1', [(480, 0, 1024, 90)], 'restore hand region overwritten by a black area'),
    (282, 'lvpd1', [(0, 0, 545, 250)], 'restore upper back of shirt overwritten by black'),
    (295, 'torino', [(815, 0, 1024, 370)], 'restore jacket sleeves overwritten by hand imagery'),
]

def chunks(data):
    pos = 0
    while pos < len(data):
        if pos + 12 > len(data):
            raise ValueError('incomplete chunk header')
        typ, length, ver = struct.unpack_from('<III', data, pos)
        end = pos + 12 + length
        if end > len(data):
            raise ValueError('incomplete chunk body')
        yield typ, ver, data[pos+12:end]
        pos = end

def pack(typ, ver, body):
    return struct.pack('<III', typ, len(body), ver) + body

def replace_texture(source, name, image):
    typ, length, ver = struct.unpack_from('<III', source)
    if typ != 22 or length + 12 > len(source) or any(source[length+12:]):
        raise ValueError('unexpected TXD container')
    out = []
    matches = 0
    for ct, cv, body in chunks(source[12:length+12]):
        if ct == 21:
            sub = []
            for st, sv, data in chunks(body):
                if st == 1 and data[8:40].split(b'\0', 1)[0].decode('ascii').lower() == name.lower():
                    header = bytearray(data[:88])
                    if struct.unpack_from('<I', header)[0] != 9:
                        raise ValueError('only D3D9 native textures are supported')
                    flags = struct.unpack_from('<I', header, 72)[0]
                    struct.pack_into('<I', header, 72, (flags & 255) | 0x600)
                    struct.pack_into('<I', header, 76, 22)
                    struct.pack_into('<HHBBB', header, 80, 1024, 1024, 32, 1, header[86])
                    header[87] = 0
                    header[4] = 2
                    pixels = image.tobytes('raw', 'BGRA')
                    data = bytes(header) + struct.pack('<I', len(pixels)) + pixels
                    matches += 1
                sub.append(pack(st, sv, data))
            body = b''.join(sub)
        out.append(pack(ct, cv, body))
    if matches != 1:
        raise ValueError(f'{name}: expected exactly one matching texture, found {matches}')
    return pack(typ, ver, b''.join(out)) + source[length+12:]

rows = []
pages = []
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 20)
with zipfile.ZipFile(args.review/'UV-patch-batch-05.zip', 'w', zipfile.ZIP_DEFLATED, compresslevel=7) as archive:
    for ident, model, rectangles, reason in spec:
        raw = Image.open(args.skins/f'SkinID{ident}.png').convert('RGBA')
        enhanced = Image.open(args.skins/'Authentic_Peds_Remaster'/f'SkinID{ident}.png').convert('RGBA')
        if enhanced.size != (1024, 1024) or raw.getchannel('A').getextrema() != (255, 255):
            raise ValueError('unexpected input image format')
        original = raw.resize(enhanced.size, Image.Resampling.LANCZOS)
        mask = Image.new('L', enhanced.size)
        draw = ImageDraw.Draw(mask)
        for x1, y1, x2, y2 in rectangles:
            draw.rectangle((x1, y1, x2-1, y2-1), fill=255)
        fixed = Image.composite(original, enhanced, mask)
        m = np.asarray(mask) > 0
        assert np.array_equal(np.asarray(fixed)[m], np.asarray(original)[m])
        assert np.array_equal(np.asarray(fixed)[~m], np.asarray(enhanced)[~m])
        source = (args.originals/'public/skins'/f'{model}.txd').read_bytes()
        old = txd.txd()
        old.load_memory(source)
        matches = [x for x in old.native_textures if (x.width, x.height) == raw.size]
        if len(matches) != 1:
            raise ValueError(f'{model}: cannot uniquely identify source texture')
        target = matches[0]
        old_image = Image.frombytes('RGBA', raw.size, bytes(target.to_rgba())).convert('RGB')
        assert np.abs(np.asarray(old_image, dtype=np.int16) - np.asarray(raw.convert('RGB'), dtype=np.int16)).mean() < 1
        result = replace_texture(source, target.name, fixed)
        new = txd.txd()
        new.load_memory(result)
        assert bytes(next(x for x in new.native_textures if x.name == target.name).to_rgba()) == fixed.tobytes()
        assert len(old.native_textures) == len(new.native_textures)
        for tex in old.native_textures:
            if tex.name != target.name:
                assert tex.to_mem() == next(x for x in new.native_textures if x.name == tex.name).to_mem()
        archive.writestr('txd/'+model+'.txd', result)
        rows.append({'id': ident, 'filename': model+'.txd', 'reason': reason, 'original_restored_rectangles_px': rectangles, 'original_area_percent': round(float(m.mean())*100, 1)})
        page = Image.new('RGB', (1500, 560), 'white')
        ImageDraw.Draw(page).text((15, 8), f'ID {ident} - {model}.txd: Original / Enhanced input / Experimental patch', font=font, fill='black')
        for i, image in enumerate((original, enhanced, fixed)):
            page.paste(image.convert('RGB').resize((480, 480), Image.Resampling.LANCZOS), (i*500+10, 45))
        pages.append(page)
    archive.writestr('manifest.json', json.dumps(rows, indent=2))
    archive.writestr('README.txt', 'Experimental UV-guided batch. Restore portions of the original, upscaled to 1024, in places where source changes visibly affect the simplified model preview. Read manifest.json for rectangles. Other image areas and other native textures are preserved. This does not guarantee a fix; restored areas lose enhanced detail and boundaries may cause seams. Not tested in-game; simplified model previews are not equivalent to the game. Back up gta3.img before testing.\n')
with zipfile.ZipFile(args.review/'UV-patch-batch-05.zip') as z:
    assert z.testzip() is None
pages[0].save(args.review/'patch-batch-05-comparison.pdf', save_all=True, append_images=pages[1:], resolution=130.0, quality=85)
print(json.dumps(rows, indent=2))

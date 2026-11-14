#!/usr/bin/env python3
"""Generate 512px derivatives and a ZIP for a batch of face PNGs.

Run from the repository root: python scripts/build_new_faces_batch.py 51 60
Requires Pillow, NumPy, SciPy and ImageMagick's convert with DDS support.
The original 1024px maps must exist in textures_1024/ first.
"""
import io
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from PIL import Image, ImageDraw
from build_pack_pipeline import generate_normal_and_spec


def main(first, last):
    root = Path(__file__).resolve().parent.parent
    rows = (last - first + 5) // 5
    sheet = Image.new('RGB', (1280, rows * 280 + 45), (17, 20, 25))
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 17), f'FACE TEXTURES {first:02}-{last:02} | OPEN EYES', fill='white')
    files = []
    manifest = []
    for index in range(first, last + 1):
        name = f'{index:02}_head_diff_000.png'
        source = root / 'textures_1024' / name
        with Image.open(source) as im:
            image = im.convert('RGB')
        if image.size != (1024, 1024):
            raise ValueError(f'Unexpected size for {source}: {image.size}')
        small = image.resize((512, 512), Image.Resampling.LANCZOS)
        diffuse = root / 'textures_512' / name
        small.save(diffuse, optimize=True)
        normal, spec = generate_normal_and_spec(np.asarray(small, dtype=np.float32))
        normal_path = root / 'normals_512' / name.replace('head_diff', 'head_normal')
        spec_path = root / 'specular_512' / name.replace('head_diff', 'head_spec')
        Image.fromarray(normal).save(normal_path, optimize=True)
        Image.fromarray(spec).save(spec_path, optimize=True)
        dds = root / 'dds_512' / name.replace('.png', '.dds')
        subprocess.run(['convert', str(diffuse), '-define', 'dds:compression=dxt1',
                        '-define', 'dds:mipmaps=9', str(dds)], check=True)
        paths = [source, diffuse, normal_path, spec_path, dds]
        files.extend(paths)
        manifest.append({'id': index, 'files': [p.relative_to(root).as_posix() for p in paths]})
        x = (index - first) % 5 * 256
        y = 45 + (index - first) // 5 * 280
        sheet.paste(image.resize((256, 256), Image.Resampling.LANCZOS), (x, y))
        draw.text((x + 8, y + 258), f'#{index:02}', fill='white')
    out = root / f'Faces_{first:02}-{last:02}.zip'
    with ZipFile(out, 'w', ZIP_DEFLATED, compresslevel=4) as z:
        for path in files:
            z.write(path, path.relative_to(root).as_posix())
        for path in sorted((root / 'templates').glob('*')):
            if path.is_file():
                z.write(path, path.relative_to(root).as_posix())
        buffer = io.BytesIO()
        sheet.save(buffer, format='JPEG', quality=90)
        z.writestr('catalog.jpg', buffer.getvalue())
        z.writestr('manifest.json', json.dumps(manifest, indent=2))
        z.writestr('READ_ME.txt', f'''Face texture batch {first:02}-{last:02}. PNG diffuse maps at 1024 and 512, 512 DXT1 DDS, approximate diffuse-derived normal and specular PNGs, and UV templates.\n\nThese are GTA IV-style UV atlases, not finished DFF/TXD skins. To use in MTA:SA, the model's UV layout must match or be remapped and the TXD texture name must match its material. Test on the target model.\n''')
    with ZipFile(out) as z:
        if z.testzip():
            raise RuntimeError('ZIP integrity check failed')
        for folder in ('textures_1024', 'textures_512', 'normals_512', 'specular_512', 'dds_512'):
            count = sum(n.startswith(folder + '/') for n in z.namelist())
            if count != last - first + 1:
                raise RuntimeError(f'{folder}: {count} entries')
    print(f'Verified {out.name}: {len(manifest)} sets, {out.stat().st_size / 1048576:.1f} MiB')

if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit('Usage: build_new_faces_batch.py FIRST LAST')
    start, end = map(int, sys.argv[1:])
    if start < 1 or end < start:
        raise SystemExit('Invalid range')
    main(start, end)

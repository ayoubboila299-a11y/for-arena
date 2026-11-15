#!/usr/bin/env python3
"""Build a GitHub-sized ZIP of the five batches 51–100. Requires Pillow.

Place Faces_51-60.zip through Faces_91-100.zip in the repo root and run
python scripts/build_extra_50_zip.py. Original full-resolution PNGs remain in
those batch archives and textures_1024/; 1024px images here are JPEG quality 95.
"""
from collections import Counter
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / 'Faces_51-100_GitHub.zip'
FOLDERS = ('textures_512', 'dds_512', 'normals_512', 'specular_512')


def main():
    sheet = Image.new('RGB', (1280, 2850), (17, 20, 25))
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 17), '50 ADDITIONAL FACE TEXTURES 51-100 | OPEN EYES', fill='white')
    manifest = []
    with ZipFile(OUTPUT, 'w', ZIP_DEFLATED, compresslevel=4) as output:
        for start in range(51, 100, 10):
            end = start + 9
            archive = ROOT / f'Faces_{start}-{end}.zip'
            with ZipFile(archive) as batch:
                if batch.testzip():
                    raise RuntimeError(f'Corrupt batch: {archive}')
                if start == 51:
                    for member in batch.namelist():
                        if member.startswith('templates/') and not member.endswith('/'):
                            output.writestr(member, batch.read(member))
                for number in range(start, end + 1):
                    stem = f'{number}_head_diff_000'
                    with Image.open(BytesIO(batch.read(f'textures_1024/{stem}.png'))) as original:
                        image = original.convert('RGB')
                    if image.size != (1024, 1024):
                        raise ValueError(f'{stem}: unexpected resolution {image.size}')
                    buffer = BytesIO()
                    image.save(buffer, 'JPEG', quality=95, subsampling=0, optimize=True)
                    jpg = f'textures_1024_jpg/{stem}.jpg'
                    output.writestr(jpg, buffer.getvalue())
                    names = {
                        'textures_512': f'{stem}.png',
                        'dds_512': f'{stem}.dds',
                        'normals_512': f'{number}_head_normal_000.png',
                        'specular_512': f'{number}_head_spec_000.png',
                    }
                    paths = {'textures_1024_jpg': jpg}
                    for folder, filename in names.items():
                        name = f'{folder}/{filename}'
                        output.writestr(name, batch.read(name))
                        paths[folder] = name
                    manifest.append({'id': number, 'files': paths})
                    x = (number - 51) % 5 * 256
                    y = 45 + (number - 51) // 5 * 280
                    sheet.paste(image.resize((256, 256), Image.Resampling.LANCZOS), (x, y))
                    draw.text((x + 6, y + 258), f'#{number}', fill='white')
        buffer = BytesIO()
        sheet.save(buffer, format='JPEG', quality=90)
        output.writestr('catalog_51-100.jpg', buffer.getvalue())
        output.writestr('manifest_51-100.json', json.dumps(manifest, ensure_ascii=False, indent=2))
        output.writestr('READ_ME_FIRST.txt', '''50 ADDITIONAL FACE TEXTURES 51-100\n\nContains 50 PNG diffuse textures at 512px, 50 high-quality JPEG diffuse textures at 1024px, 50 DXT1 DDS files, 50 approximate normal maps, 50 approximate specular maps, templates and a catalog. 1024px images are JPEG quality 95 to fit GitHub's 100 MB single-file limit. Full-quality 1024px PNGs are in Faces_51-60.zip, Faces_61-70.zip, Faces_71-80.zip, Faces_81-90.zip and Faces_91-100.zip on the same branch.\n\nFor MTA:SA import into your TXD and match texture names to the DFF material. These are GTA IV-style UV atlases, so the DFF head must use matching UVs or be remapped; test on your target model. These are textures, not finished DFF/TXD skins. Normal and specular maps are approximations derived from diffuse.\n''')
    with ZipFile(OUTPUT) as z:
        if z.testzip():
            raise RuntimeError('ZIP integrity check failed')
        counts = Counter(n.split('/')[0] for n in z.namelist())
        for folder in ('textures_1024_jpg', *FOLDERS):
            if counts[folder] != 50:
                raise RuntimeError(f'Expected 50 files in {folder}, got {counts[folder]}')
        if OUTPUT.stat().st_size >= 100_000_000:
            raise RuntimeError('ZIP exceeds GitHub 100 MB limit')
        print('Verified', dict(counts), 'size MiB:', round(OUTPUT.stat().st_size / 1048576, 1))


if __name__ == '__main__':
    main()

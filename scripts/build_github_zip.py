#!/usr/bin/env python3
"""Build the GitHub-friendly 50-face archive from the repository's source files.

Requires Pillow. The original 1024px PNGs stay in textures_1024/ in the repo;
the archive stores high-quality JPEG copies to remain below GitHub's file limit.
Run from any working directory: python scripts/build_github_zip.py
"""
import io
import json
from collections import Counter
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / '50_Faces_MTA_SA_GitHub.zip'


def main():
    sheet = Image.new('RGB', (1280, 2850), (17, 20, 25))
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 17), '50 FACE TEXTURES | OPEN EYES', fill='white')
    manifest = []
    with ZipFile(OUTPUT, 'w', ZIP_DEFLATED, compresslevel=4) as z:
        for i in range(1, 51):
            files = list((ROOT / 'textures_1024').glob(f'{i:02}_*.png'))
            if len(files) != 1:
                raise RuntimeError(f'Expected exactly one 1024px texture for #{i}: {files}')
            source = files[0]
            name = source.name
            jpeg_name = Path(name).with_suffix('.jpg').name
            with Image.open(source) as original:
                im = original.convert('RGB')
                if im.size != (1024, 1024):
                    raise ValueError(f'{source} is {im.size}, expected 1024x1024')
                buffer = io.BytesIO()
                im.save(buffer, 'JPEG', quality=95, subsampling=0, optimize=True)
                z.writestr(f'textures_1024_jpg/{jpeg_name}', buffer.getvalue())
                x, y = ((i - 1) % 5) * 256, 45 + ((i - 1) // 5) * 280
                sheet.paste(im.resize((256, 256), Image.Resampling.LANCZOS), (x, y))
                draw.text((x + 6, y + 258), f'#{i:02}', fill='white')
            names = {
                'textures_512': name,
                'dds_512': name.replace('.png', '.dds'),
                'normals_512': name.replace('head_diff', 'head_normal'),
                'specular_512': name.replace('head_diff', 'head_spec'),
            }
            paths = {'textures_1024_jpg': f'textures_1024_jpg/{jpeg_name}'}
            for folder, filename in names.items():
                p = ROOT / folder / filename
                if not p.is_file():
                    raise FileNotFoundError(p)
                arcname = f'{folder}/{filename}'
                z.write(p, arcname)
                paths[folder] = arcname
            manifest.append({'id': i, 'files': paths, 'source_type': 'edited variant of an earlier face' if 11 <= i <= 20 else 'AI generated'})
        for p in sorted((ROOT / 'templates').glob('*')):
            if p.is_file():
                z.write(p, p.relative_to(ROOT).as_posix())
        for name in ('README.md', 'metadata.json', 'server.py', 'web/index.html',
                     'scripts/build_uv_template.py', 'scripts/build_pack_pipeline.py',
                     'scripts/build_50_pack.py', 'scripts/build_github_zip.py'):
            z.write(ROOT / name, name)
        buffer = io.BytesIO()
        sheet.save(buffer, 'JPEG', quality=90)
        z.writestr('catalog_50.jpg', buffer.getvalue())
        z.writestr('manifest_50.json', json.dumps(manifest, ensure_ascii=False, indent=2))
        z.writestr('READ_ME_FIRST.txt', '''50 FACE TEXTURE PACK - MTA:SA / GTA IV\n\n50 sets, numbered 01-50. 512px diffuse textures are original PNGs, with DDS, derived normal and specular PNGs, templates and catalog. To fit GitHub's 100 MB file limit, the 1024px versions in this ZIP are JPEG quality 95, not lossless PNG. The original full-quality 1024px PNG files are available in the repository under textures_1024/.\n\nFaces 11-20 are edited variants of 01-10, not fully independent characters. Faces 21-30 were regenerated because the previous batch was unavailable.\n\nFor MTA:SA, import into your TXD and match texture names to the DFF material. These use a GTA IV-style UV layout; the DFF head needs matching UVs or must be remapped. Test on your model. These are textures, not a ready-made DFF or TXD. Normal/specular maps are approximations derived from the diffuse.\n\nThe original web studio and metadata.json cover only faces 01-20. manifest_50.json lists all 50.\n''')
    with ZipFile(OUTPUT) as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError(f'Corrupt ZIP entry: {bad}')
        counts = Counter(n.split('/')[0] for n in z.namelist())
        for folder in ('textures_1024_jpg', 'textures_512', 'dds_512', 'normals_512', 'specular_512'):
            if counts[folder] != 50:
                raise RuntimeError(f'Expected 50 files in {folder}: {counts[folder]}')
        if OUTPUT.stat().st_size >= 100_000_000:
            raise RuntimeError('Archive exceeds GitHub 100 MB file limit')
        print('Verified', dict(counts), 'size MiB:', round(OUTPUT.stat().st_size / 1048576, 1))

if __name__ == '__main__':
    main()

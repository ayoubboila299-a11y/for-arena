#!/usr/bin/env python3
"""Rebuild the 50-face catalog/archive. Run from repo root (Pillow required).

The original studio manifest covers 01-20 only. This pack's separate manifest
covers all 50. The generated ZIP is a distribution artifact, not source control.
"""
from collections import Counter
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / '50_Faces_MTA_SA_Complete.zip'
FOLDERS = ('textures_1024', 'textures_512', 'normals_512', 'specular_512', 'dds_512')

def main():
    originals = json.loads((ROOT / 'metadata.json').read_text(encoding='utf-8'))
    by_id = {entry['id']: entry for entry in originals}
    sheet = Image.new('RGB', (1280, 2850), (17, 20, 25))
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 17), '50 FACE TEXTURE MAPS  |  OPEN EYES', fill='white')
    manifest = []
    with ZipFile(OUT, 'w', ZIP_DEFLATED, compresslevel=4) as z:
        for number in range(1, 51):
            files = list((ROOT / 'textures_1024').glob(f'{number:02}_*.png'))
            if len(files) != 1:
                raise RuntimeError(f'Expected exactly one texture for #{number}, found {files}')
            path = files[0]
            name = path.name
            names = (name, name, name.replace('head_diff', 'head_normal'),
                     name.replace('head_diff', 'head_spec'), name.replace('.png', '.dds'))
            for folder, filename in zip(FOLDERS, names):
                src = ROOT / folder / filename
                if not src.is_file():
                    raise FileNotFoundError(src)
                z.write(src, f'{folder}/{filename}')
            with Image.open(path) as im:
                if im.size != (1024, 1024):
                    raise ValueError(f'Unexpected image size for {path}: {im.size}')
                thumb = im.convert('RGB').resize((256, 256), Image.Resampling.LANCZOS)
            x, y = ((number - 1) % 5) * 256, 45 + ((number - 1) // 5) * 280
            sheet.paste(thumb, (x, y))
            draw.text((x + 6, y + 259), f'#{number:02}', fill='white')
            manifest.append({'id': number, 'filename': name,
                             'source_type': by_id[number].get('source_type', 'original') if number in by_id else 'AI generated',
                             'files': dict(zip(FOLDERS, [f'{folder}/{name}' for folder, name in zip(FOLDERS, names)]))})
        for p in sorted((ROOT / 'templates').glob('*')):
            if p.is_file(): z.write(p, p.relative_to(ROOT).as_posix())
        for name in ('README.md', 'metadata.json', 'server.py', 'web/index.html',
                     'scripts/build_uv_template.py', 'scripts/build_pack_pipeline.py', 'scripts/build_50_pack.py'):
            z.write(ROOT / name, name)
        b = BytesIO()
        sheet.save(b, format='JPEG', quality=90)
        z.writestr('catalog_50.jpg', b.getvalue())
        z.writestr('manifest_50.json', json.dumps(manifest, ensure_ascii=False, indent=2))
        z.writestr('READ_ME_FIRST.txt', '''50 FACE TEXTURE PACK - MTA:SA / GTA IV\n\n50 texture sets, numbered 01-50. Includes 1024 and 512 PNG diffuse textures, 512 DDS, derived normal and specular maps, templates and a catalog. Some original faces (11-20) are edited variants of 01-10, not fully independent characters. Faces 21-30 were regenerated because the earlier batch was unavailable.\n\nFor MTA:SA import into your TXD and match texture names to the DFF material. These use a GTA IV-style UV layout; the target DFF head must have matching UVs, or be remapped in Blender/DragonFF. Test on your model. This is a texture pack, not a ready-made DFF or TXD skin. Normal/specular maps are approximations derived from the diffuse.\n\nThe included original web studio and metadata.json describe only the original 20 faces. manifest_50.json lists all 50.\n''')
    with ZipFile(OUT) as z:
        bad = z.testzip()
        if bad: raise RuntimeError(f'Corrupt archive entry: {bad}')
        counts = Counter(n.split('/')[0] for n in z.namelist())
        assert all(counts[folder] == 50 for folder in FOLDERS), counts
        print('Archive verified:', counts, 'size MiB:', round(OUT.stat().st_size / 1024**2, 1))

if __name__ == '__main__':
    main()

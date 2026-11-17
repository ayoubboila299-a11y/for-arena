#!/usr/bin/env python3
"""Combine selected experimental TXD overrides into two convenient archives.

Uses the most recent experimental version of duplicate names. Selection is
based on approximate static previews, not game testing. No source ZIP is edited.
"""
import hashlib
import json
import zipfile
from pathlib import Path

review = Path('uv_review')
# Omitted after model review: 26, 28, 32, 52, 55, 99, 123, 147.
# These have no clearly demonstrated benefit in the simplified previews, or
# discard too much enhanced detail. Do not infer that other IDs are verified.
selected = [14, 15, 18, 35, 38, 41, 47, 78, 93, 114, 115, 117, 122, 154, 162, 174, 175, 196, 198, 241, 249, 261, 278, 282, 295]
latest = {}
for batch in range(1, 7):
    path = review/f'UV-patch-batch-{batch:02d}.zip'
    with zipfile.ZipFile(path) as archive:
        if archive.testzip():
            raise ValueError(f'corrupt source archive: {path}')
        for row in json.loads(archive.read('manifest.json')):
            data = archive.read('txd/'+row['filename'])
            latest[row['id']] = (batch, row, data)
if len(set(selected)) != len(selected) or any(i not in latest for i in selected):
    raise ValueError('duplicate or missing selection')
for part, ids in enumerate((selected[:13], selected[13:]), 1):
    path = review/f'Reviewed-overrides-{part:02d}.zip'
    manifest = []
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED, compresslevel=7) as z:
        for ident in ids:
            batch, row, data = latest[ident]
            z.writestr(row['filename'], data)
            manifest.append({**row, 'source_batch': batch, 'sha256': hashlib.sha256(data).hexdigest()})
        z.writestr('manifest.json', json.dumps(manifest, indent=2))
        z.writestr('README.txt', '''GTA SA skin experimental reviewed overrides.\n\nInstall the nine OriginalNames archives first, then optionally replace these TXD files by their original filenames in gta3.img. The two Reviewed-overrides archives together contain 25 selected TXDs, not all skins. Some restored regions are the original texture resized to 1024x1024, not high-definition improvements. See manifest.json for exactly what was replaced.\n\nSelection uses a simplified static preview of the original DFF models, not in-game tests. Changes can have seams or colour differences and some may be undesirable. Back up gta3.img before importing, and test individually. Model comparisons and full documentation: https://github.com/ayoubboila299-a11y/for-arena/tree/arena/2e1209bd-for-arena/uv_review\n''')
    with zipfile.ZipFile(path) as z:
        if z.testzip():
            raise ValueError(f'corrupt output archive: {path}')
        for item in manifest:
            if hashlib.sha256(z.read(item['filename'])).hexdigest() != item['sha256']:
                raise ValueError(f'mismatch in {item["filename"]}')
    print(path, len(manifest), hashlib.sha256(path.read_bytes()).hexdigest())

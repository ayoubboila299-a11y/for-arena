import importlib.util
import struct
import unittest
from pathlib import Path
from types import SimpleNamespace as Obj

import numpy as np
from PIL import Image

path = Path(__file__).resolve().parents[1] / 'tools' / 'smart_uv.py'
spec = importlib.util.spec_from_file_location('smart_uv', path)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


class SmartUVTests(unittest.TestCase):
    def test_unchanged_texture_has_no_candidates(self):
        im = Image.new('RGB', (256, 256), 'gray')
        _, mask, chosen, eligible = s.score_tiles(im, im, np.ones((256, 256), bool), .18, .15)
        self.assertFalse(mask.any())
        self.assertEqual((chosen, eligible), ([], 0))

    def test_covered_changes_are_capped_and_unmapped_ignored(self):
        original = Image.new('RGB', (256, 256), 'black')
        changed = Image.new('RGB', (256, 256), 'white')
        uv = np.zeros((256, 256), bool)
        uv[:128, :128] = True
        _, mask, chosen, eligible = s.score_tiles(original, changed, uv, .18, .02)
        self.assertEqual(eligible, 64)
        self.assertEqual(len(chosen), 5)
        self.assertFalse(mask[128:, :].any())
        self.assertFalse(mask[:, 128:].any())

    def test_wrap_and_boundary_crossing(self):
        uv = [Obj(u=.95, v=.2), Obj(u=1.05, v=.2), Obj(u=1, v=.4)]
        mat = Obj(textures=[Obj(name='foo', uv_addressing=0x11)])
        g = Obj(uv_layers=[uv], triangles=[Obj(a=0, b=1, c=2, material=0)], materials=[mat])
        d = Obj(clumps=[Obj(geometry_list=[g])])
        covered, tris = s.coverage(d, 'FOO')
        self.assertTrue(covered[:, :15].any())
        self.assertTrue(covered[:, -15:].any())
        selected = np.zeros((256, 256), bool)
        selected[:, :20] = True
        self.assertEqual(s.triangle_crossings(tris, selected), 1)
        mat.textures[0].uv_addressing = 0x22
        with self.assertRaises(ValueError):
            s.coverage(d, 'foo')

    def test_txd_replace_preserves_other_chunks_and_rejects_missing(self):
        header = bytearray(88)
        struct.pack_into('<I', header, 0, 9)
        header[8:11] = b'foo'
        native = s.pack(21, 1, s.pack(1, 1, bytes(header) + b'old'))
        extra = s.pack(42, 1, b'untouched')
        source = s.pack(22, 1, native + extra)
        image = Image.new('RGBA', (2, 2), (1, 2, 3, 255))
        output = s.replace_texture(source, 'FOO', image)
        self.assertEqual(list(s.chunk_iter(output[12:]))[1], (42, 1, b'untouched'))
        body = list(s.chunk_iter(output[12:]))[0][2]
        data = list(s.chunk_iter(body))[0][2]
        self.assertEqual(struct.unpack_from('<HH', data, 80), (2, 2))
        self.assertEqual(data[92:], image.tobytes('raw', 'BGRA'))
        with self.assertRaises(ValueError):
            s.replace_texture(source, 'nope', image)
        with self.assertRaises(ValueError):
            s.replace_texture(source[:-1], 'foo', image)


if __name__ == '__main__':
    unittest.main()

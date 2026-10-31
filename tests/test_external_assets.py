import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/download_external_assets.py'
SPEC = importlib.util.spec_from_file_location('download_external_assets', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ExternalAssetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.payload = b'verified asset'
        self.entry = {'path': 'assets/grid.usd', 'url': 'https://example.org/grid.usd',
                      'bytes': len(self.payload), 'sha256': hashlib.sha256(self.payload).hexdigest()}

    def test_download_and_verified_cache(self):
        with patch.object(MODULE.urllib.request, 'urlopen', return_value=io.BytesIO(self.payload)) as request:
            target = MODULE.download(self.entry, self.root)
            self.assertEqual(target.read_bytes(), self.payload)
            MODULE.download(self.entry, self.root)
            request.assert_called_once()

    def test_invalid_download_preserves_existing_file(self):
        target = self.root / self.entry['path']
        target.parent.mkdir()
        target.write_bytes(b'previous asset')
        for payload in (b'bad', b'x' * self.entry['bytes'], self.payload + b'extra'):
            with self.subTest(payload=payload), patch.object(MODULE.urllib.request, 'urlopen', return_value=io.BytesIO(payload)):
                with self.assertRaises(ValueError):
                    MODULE.download(self.entry, self.root)
                self.assertEqual(target.read_bytes(), b'previous asset')
                self.assertFalse(list(target.parent.glob('*.part')))

    def test_destination_cannot_escape_root(self):
        with self.assertRaises(ValueError):
            MODULE.download(dict(self.entry, path='../outside.usd'), self.root)

    def test_non_https_download_rejected(self):
        with self.assertRaises(ValueError):
            MODULE.download(dict(self.entry, url='http://example.org/grid.usd'), self.root)


if __name__ == '__main__':
    unittest.main()

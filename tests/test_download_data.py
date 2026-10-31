"""Integrity and extraction boundaries for published dataset archives."""

import hashlib
import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('download_data', Path(__file__).resolve().parents[1] / 'scripts/download_data.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def archive(self, names, link=False):
        archive = self.root / 'dataset.tar'
        with tarfile.open(archive, 'w') as stream:
            for name in names:
                member = tarfile.TarInfo(name)
                if link:
                    member.type = tarfile.SYMTYPE
                    member.linkname = '/tmp/outside'
                    stream.addfile(member)
                else:
                    member.size = 4
                    stream.addfile(member, io.BytesIO(b'data'))
        return archive

    def test_extract_and_no_overwrite(self):
        archive = self.archive(['task1/data/file', 'task1/videos/file', 'task1/meta/file'])
        target = module.extract(archive, self.root / 'out', 'task1')
        self.assertEqual((target / 'data/file').read_bytes(), b'data')
        with self.assertRaises(ValueError):
            module.extract(archive, self.root / 'out', 'task1')

    def test_unsafe_paths_and_links_rejected(self):
        for name in ['../escape', '/absolute', 'task1/../../escape', 'task2/data/file']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                module.extract(self.archive([name]), self.root / 'out', 'task1')
        with self.assertRaises(ValueError):
            module.extract(self.archive(['task1/link'], link=True), self.root / 'out', 'task1')
        self.assertFalse((self.root / 'out/task1').exists())

    def test_download_verified_before_extraction(self):
        source = self.root / 'source.tar'
        source.write_bytes(b'archive')
        entry = {'archive': 'cached.tar', 'url': source.as_uri(), 'archive_bytes': 7,
                 'sha256': hashlib.sha256(b'archive').hexdigest()}
        result = module.download(entry, self.root / 'cache')
        self.assertEqual(result.read_bytes(), b'archive')
        result.write_bytes(b'changed')
        with self.assertRaises(ValueError):
            module.download(entry, self.root / 'cache')

    def test_invalid_download_not_promoted(self):
        source = self.root / 'source.tar'
        source.write_bytes(b'wrong')
        entry = {'archive': 'cached.tar', 'url': source.as_uri(), 'archive_bytes': 5, 'sha256': '0' * 64}
        with self.assertRaises(ValueError):
            module.download(entry, self.root / 'cache')
        self.assertFalse((self.root / 'cache/cached.tar').exists())

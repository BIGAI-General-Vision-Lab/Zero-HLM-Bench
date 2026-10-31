"""Checkpoint download integrity, reuse, and filesystem boundaries."""

import copy
import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    'download_models', Path(__file__).resolve().parents[1] / 'scripts/download_models.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ModelDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.destination = self.root / 'checkpoints'
        self.payload = b'checkpoint content'
        self.entry = {'path': 'processor/config.json', 'size': len(self.payload),
                      'sha256': hashlib.sha256(self.payload).hexdigest()}
        self.manifest = {'repo_id': 'Example/Models', 'revision': 'a' * 40,
                         'tasks': {task: {'files': [copy.deepcopy(self.entry)]}
                                   for task in module.TASKS}}

    def download(self):
        return module.download_file(self.manifest, 'task3', self.entry, self.destination)

    def assert_no_partial_files(self):
        self.assertEqual(list(self.destination.rglob('*.part')), [])

    def test_download_uses_pinned_url_and_preserves_processor_directory(self):
        module.validate_manifest(self.manifest)
        with patch.object(module.urllib.request, 'urlopen', return_value=io.BytesIO(self.payload)) as request:
            target = self.download()
        self.assertEqual(target, self.destination / 'task3/processor/config.json')
        self.assertEqual(target.read_bytes(), self.payload)
        request.assert_called_once_with(
            'https://huggingface.co/Example/Models/resolve/' + 'a' * 40
            + '/task3/processor/config.json', timeout=120)
        self.assert_no_partial_files()

    def test_valid_existing_checkpoint_requires_no_network_or_rewrite(self):
        target = self.destination / 'task3/processor/config.json'
        target.parent.mkdir(parents=True)
        target.write_bytes(self.payload)
        before = target.stat().st_mtime_ns
        with patch.object(module.urllib.request, 'urlopen') as request:
            self.assertEqual(self.download(), target)
        request.assert_not_called()
        self.assertEqual(target.stat().st_mtime_ns, before)

    def test_corrupt_existing_file_is_rejected_without_overwrite(self):
        target = self.destination / 'task3/processor/config.json'
        target.parent.mkdir(parents=True)
        damaged = b'x' * len(self.payload)
        target.write_bytes(damaged)
        with patch.object(module.urllib.request, 'urlopen') as request:
            with self.assertRaisesRegex(ValueError, 'Existing checkpoint failed'):
                self.download()
        request.assert_not_called()
        self.assertEqual(target.read_bytes(), damaged)

    def test_corrupt_short_and_oversized_downloads_are_not_promoted(self):
        for payload in (b'x' * len(self.payload), self.payload[:-1], self.payload + b'x'):
            with self.subTest(payload=payload):
                with patch.object(module.urllib.request, 'urlopen', return_value=io.BytesIO(payload)):
                    with self.assertRaises(ValueError):
                        self.download()
                self.assertFalse((self.destination / 'task3/processor/config.json').exists())
                self.assert_no_partial_files()

    def test_interrupted_stream_removes_temporary_file_and_can_retry(self):
        class InterruptedStream(io.BytesIO):
            def read(self, size=-1):
                if self.tell():
                    raise ConnectionError('connection interrupted')
                return super().read(4)

        with patch.object(module.urllib.request, 'urlopen', return_value=InterruptedStream(self.payload)):
            with self.assertRaises(ConnectionError):
                self.download()
        self.assertFalse((self.destination / 'task3/processor/config.json').exists())
        self.assert_no_partial_files()
        with patch.object(module.urllib.request, 'urlopen', return_value=io.BytesIO(self.payload)):
            self.assertEqual(self.download().read_bytes(), self.payload)

    def test_network_failure_does_not_leave_partial_file(self):
        with patch.object(module.urllib.request, 'urlopen', side_effect=TimeoutError):
            with self.assertRaises(TimeoutError):
                self.download()
        self.assert_no_partial_files()

    def test_unsafe_paths_rejected_before_network(self):
        for path in ('../outside', '/absolute', 'a/../../outside', 'a//b', './config',
                     'a/./b', 'a\\b', 'a\x00b', '', '.'):
            with self.subTest(path=path):
                self.entry['path'] = path
                with patch.object(module.urllib.request, 'urlopen') as request:
                    with self.assertRaises(ValueError):
                        self.download()
                    request.assert_not_called()

    def test_symlink_directory_and_dangling_target_are_rejected(self):
        self.destination.mkdir()
        outside = self.root / 'outside'
        outside.mkdir()
        (self.destination / 'task3').symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            self.download()
        self.assertEqual(list(outside.iterdir()), [])
        (self.destination / 'task3').unlink()
        target = self.destination / 'task3/processor/config.json'
        target.parent.mkdir(parents=True)
        target.symlink_to(outside / 'missing')
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            self.download()
        self.assertFalse((outside / 'missing').exists())

    def test_manifest_rejects_unpinned_and_conflicting_entries(self):
        invalid = copy.deepcopy(self.manifest)
        invalid['revision'] = 'main'
        with self.assertRaisesRegex(ValueError, 'pinned'):
            module.validate_manifest(invalid)
        for entry in (copy.deepcopy(self.entry), dict(self.entry, path='processor')):
            invalid = copy.deepcopy(self.manifest)
            invalid['tasks']['task3']['files'].append(entry)
            with self.assertRaises(ValueError):
                module.validate_manifest(invalid)
        for key, value in (('size', -1), ('size', True), ('sha256', 'unknown')):
            invalid = copy.deepcopy(self.manifest)
            invalid['tasks']['task3']['files'][0][key] = value
            with self.assertRaises(ValueError):
                module.validate_manifest(invalid)


if __name__ == '__main__':
    unittest.main()

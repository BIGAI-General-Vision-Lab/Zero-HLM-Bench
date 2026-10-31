#!/usr/bin/env python3
"""Download and verify the published GR00T checkpoints."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tempfile
import urllib.parse
import urllib.request


TASKS = ('task1', 'task2', 'task3')
CHUNK_BYTES = 4 * 1024 * 1024


def validate_path(value):
    if not isinstance(value, str) or not value or '\\' in value or '\x00' in value:
        raise ValueError(f'Invalid checkpoint path: {value!r}')
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ('', '.', '..') for part in value.split('/')):
        raise ValueError(f'Unsafe checkpoint path: {value!r}')
    return path


def validate_manifest(manifest):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', manifest['repo_id']):
        raise ValueError('Invalid Hugging Face model repository')
    if not re.fullmatch(r'[0-9a-f]{40}', manifest['revision']):
        raise ValueError('Checkpoint revision must be a pinned commit SHA')
    if set(manifest['tasks']) != set(TASKS):
        raise ValueError('Checkpoint manifest must contain all three tasks')
    for task, definition in manifest['tasks'].items():
        entries = definition['files']
        if not entries:
            raise ValueError(f'No checkpoint files specified for {task}')
        paths = set()
        for entry in entries:
            path = validate_path(entry['path'])
            if path in paths:
                raise ValueError(f'Duplicate checkpoint path: {task}/{path}')
            paths.add(path)
            if type(entry['size']) is not int or entry['size'] < 0:
                raise ValueError(f'Invalid checkpoint size: {task}/{path}')
            if not re.fullmatch(r'[0-9a-f]{64}', entry['sha256']):
                raise ValueError(f'Invalid checkpoint checksum: {task}/{path}')
        if any(parent in paths for path in paths for parent in path.parents):
            raise ValueError(f'Conflicting checkpoint paths in {task}')
    return manifest


def verified(path, entry):
    if not path.is_file() or path.stat().st_size != entry['size']:
        return False
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b''):
            checksum.update(chunk)
    return checksum.hexdigest() == entry['sha256']


def download_file(manifest, task, entry, destination):
    if task not in TASKS:
        raise ValueError(f'Invalid task: {task}')
    relative = validate_path(entry['path'])
    destination = Path(destination).resolve()
    target = destination / task / relative
    current = destination
    for component in (task, *relative.parts):
        current = current / component
        if current.is_symlink():
            raise ValueError(f'Symlink in checkpoint destination: {current}')
    if target.exists():
        if verified(target, entry):
            return target
        raise ValueError(f'Existing checkpoint failed integrity verification: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    remote_path = urllib.parse.quote(f'{task}/{relative}', safe='/')
    url = f"https://huggingface.co/{manifest['repo_id']}/resolve/{manifest['revision']}/{remote_path}"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f'.{target.name}.', suffix='.part',
                                         dir=target.parent, delete=False) as output:
            temporary = Path(output.name)
            checksum = hashlib.sha256()
            size = 0
            with urllib.request.urlopen(url, timeout=120) as response:
                for chunk in iter(lambda: response.read(CHUNK_BYTES), b''):
                    size += len(chunk)
                    if size > entry['size']:
                        raise ValueError(f'Download exceeds expected size: {task}/{relative}')
                    checksum.update(chunk)
                    output.write(chunk)
        if size != entry['size'] or checksum.hexdigest() != entry['sha256']:
            raise ValueError(f'Download failed integrity verification: {task}/{relative}')
        temporary.chmod(0o644)
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=(*TASKS, 'all'), default='all')
    parser.add_argument('--destination', type=Path, default=Path('checkpoints'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = validate_manifest(json.loads((root / 'configs/pretrained_models.json').read_text()))
    for task in TASKS if args.task == 'all' else (args.task,):
        for entry in manifest['tasks'][task]['files']:
            print(f"Verifying {task}/{entry['path']}", flush=True)
            download_file(manifest, task, entry, args.destination)
        print(f'Checkpoint ready: {args.destination / task}', flush=True)


if __name__ == '__main__':
    main()

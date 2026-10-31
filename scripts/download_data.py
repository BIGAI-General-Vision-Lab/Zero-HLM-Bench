#!/usr/bin/env python3
"""Download, verify, and extract the benchmark training datasets."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import urllib.request


def sha256(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def download(entry, cache):
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / entry['archive']
    if target.exists():
        if target.stat().st_size == entry['archive_bytes'] and sha256(target) == entry['sha256']:
            return target
        raise ValueError(f'Cached archive failed integrity verification: {target}')
    partial = target.with_suffix(target.suffix + '.part')
    with urllib.request.urlopen(entry['url'], timeout=120) as response, partial.open('wb') as output:
        shutil.copyfileobj(response, output, 4 * 1024 * 1024)
    if partial.stat().st_size != entry['archive_bytes'] or sha256(partial) != entry['sha256']:
        raise ValueError(f'Download failed integrity verification: {partial}')
    partial.replace(target)
    return target


def extract(archive, destination, task):
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / task
    if target.exists():
        raise ValueError(f'Destination already exists: {target}')
    with tempfile.TemporaryDirectory(prefix='.extract-', dir=destination) as temporary:
        staging = Path(temporary)
        with tarfile.open(archive, 'r:') as stream:
            members = stream.getmembers()
            for member in members:
                path = PurePosixPath(member.name)
                if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0] != task:
                    raise ValueError(f'Unsafe archive path: {member.name}')
                if not (member.isfile() or member.isdir()):
                    raise ValueError(f'Unsupported archive member: {member.name}')
            for member in members:
                output = staging / member.name
                if member.isdir():
                    output.mkdir(parents=True, exist_ok=True)
                else:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    with stream.extractfile(member) as source, output.open('wb') as sink:
                        shutil.copyfileobj(source, sink, 4 * 1024 * 1024)
                    os.chmod(output, 0o644)
        for name in ('data', 'videos', 'meta'):
            if not (staging / task / name).is_dir():
                raise ValueError(f'Missing dataset directory: {task}/{name}')
        (staging / task).rename(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=('task1', 'task2', 'task3', 'all'), default='all')
    parser.add_argument('--destination', type=Path, default=Path('data'))
    parser.add_argument('--cache', type=Path, default=Path('.cache/downloads'))
    parser.add_argument('--download-only', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    entries = json.loads((root / 'configs/training_data.json').read_text())
    for entry in entries:
        if args.task not in ('all', entry['task']):
            continue
        print(f"Downloading {entry['task']}: {entry['episodes']} episodes", flush=True)
        archive = download(entry, args.cache)
        result = archive if args.download_only else extract(archive, args.destination, entry['task'])
        print(f'Verified: {result}', flush=True)


if __name__ == '__main__':
    main()

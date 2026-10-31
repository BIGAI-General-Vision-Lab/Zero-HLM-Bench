#!/usr/bin/env python3
"""Fetch separately licensed assets directly from their official distributor."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request


def matches(path, entry):
    if not path.is_file() or path.stat().st_size != entry['bytes']:
        return False
    return hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']


def download(entry, root):
    root = root.resolve()
    destination = (root / entry['path']).resolve()
    if not destination.is_relative_to(root) or destination == root:
        raise ValueError('External asset destination must be inside the project directory')
    if not entry['url'].startswith('https://'):
        raise ValueError('External assets require HTTPS')
    if matches(destination, entry):
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix='.part', delete=False) as output:
            temporary = Path(output.name)
            with urllib.request.urlopen(entry['url'], timeout=120) as response:
                remaining = entry['bytes']
                while remaining:
                    chunk = response.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    output.write(chunk)
                    remaining -= len(chunk)
                if response.read(1):
                    raise ValueError(f"External asset exceeds expected size: {entry['path']}")
        if not matches(temporary, entry):
            raise ValueError(f"External asset failed integrity verification: {entry['path']}")
        temporary.replace(destination)
        return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    manifest = args.manifest or args.root / 'configs/external_assets.json'
    entries = json.loads(manifest.read_text())['files']
    for entry in entries:
        if args.verify_only:
            if not matches(args.root / entry['path'], entry):
                raise SystemExit(f"Missing or invalid external asset: {entry['path']}")
        else:
            download(entry, args.root)
        print(f"Verified {entry['path']}")


if __name__ == '__main__':
    main()

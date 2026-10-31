#!/usr/bin/env python3
"""Install or verify the bundled GR00T N1.7 inference integration."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "integrations" / "groot"


def verify_source():
    manifest = json.loads((SOURCE / "source_manifest.json").read_text())
    for entry in manifest["files"]:
        path = SOURCE / entry["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Bundled GR00T source checksum mismatch: {entry['path']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--check", action="store_true", help="Check source, imports, and server options without installation")
    parser.add_argument("--with-dependencies", action="store_true", help="Install the pinned Python dependencies; install PyTorch separately first")
    args = parser.parse_args()
    verify_source()
    if not args.check:
        if args.with_dependencies:
            subprocess.run([args.python, "-m", "pip", "install", "-r", str(SOURCE / "requirements.txt")], check=True)
        subprocess.run([args.python, "-m", "pip", "install", "--no-deps", "-e", str(SOURCE)], check=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(SOURCE), env.get("PYTHONPATH")]))
    env["NO_ALBUMENTATIONS_UPDATE"] = "1"
    probe = """
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import EMBODIMENT_TAG_TO_PROJECTOR_INDEX
from gr00t.policy.server_client import PolicyServer
assert EmbodimentTag.resolve('unitree_g1_sonic').value == 'unitree_g1_sonic'
assert EMBODIMENT_TAG_TO_PROJECTOR_INDEX['unitree_g1_sonic'] == 11
print('Unitree G1 SONIC inference imports verified (projector 11).')
"""
    subprocess.run([args.python, "-c", probe], env=env, check=True)
    subprocess.run([args.python, "-m", "gr00t.eval.run_gr00t_server", "--help"], env=env, check=True)


if __name__ == "__main__":
    main()

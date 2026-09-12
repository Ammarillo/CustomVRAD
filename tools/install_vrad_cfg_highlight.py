#!/usr/bin/env python3
"""Copy the PathRAD .cfg highlighter into Cursor's extensions folder."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = Path(__file__).resolve().parent / "vscode-vrad-cfg"
GEN = Path(__file__).resolve().parent / "_gen_vrad_cfg_grammar.py"


def ensure_extension() -> dict:
    if not (SRC / "package.json").is_file():
        subprocess.check_call([sys.executable, str(GEN)])
    return json.loads((SRC / "package.json").read_text(encoding="utf-8"))


def install_into(ext_root: Path, pkg: dict) -> Path:
    ext_root.mkdir(parents=True, exist_ok=True)
    prefix = f"{pkg['publisher']}.{pkg['name']}-"
    for child in ext_root.iterdir():
        if child.is_dir() and child.name.startswith(prefix):
            shutil.rmtree(child)
    dest = ext_root / f"{pkg['publisher']}.{pkg['name']}-{pkg['version']}"
    shutil.copytree(SRC, dest)
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--also-vscode", action="store_true")
    args = ap.parse_args()

    pkg = ensure_extension()
    targets = [Path.home() / ".cursor" / "extensions"]
    if args.also_vscode:
        targets.append(Path.home() / ".vscode" / "extensions")

    for root in targets:
        dest = install_into(root, pkg)
        print(f"Installed: {dest}")

    print()
    print("Reload Cursor: Ctrl+Shift+P then Developer: Reload Window")
    print("Open configs/full.cfg - language mode should be VRAD Config.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Check PathRAD -config files against tools/vrad_cfg_flags.json."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CATALOG = Path(__file__).resolve().parent / "vrad_cfg_flags.json"

BANNED_FALLBACK = (
    "-config",
    "-cfg",
    "-game",
    "-vproject",
    "-insert_search_path",
    "-novconfig",
)


def load_catalog() -> dict:
    return json.loads(CATALOG.read_text(encoding="utf-8"))


def strip_comment(line: str) -> str:
    in_quote = False
    i = 0
    while i < len(line):
        c = line[i]
        if c == '"':
            in_quote = not in_quote
        if not in_quote:
            if c == "#":
                return line[:i]
            if c == "/" and i + 1 < len(line) and line[i + 1] == "/":
                return line[:i]
        i += 1
    return line


def tokenize(s: str) -> list[str]:
    return [t for t in re.findall(r'"[^"]*"|\S+', s.strip())]


def emit(path: Path, line_no: int, col: int, severity: str, message: str) -> None:
    print(f"{path}({line_no},{col}): {severity}: {message}")


def collect_cfg_files(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    files: list[Path] = []
    for pat in ("*.cfg", "*.txt"):
        files.extend(sorted(target.rglob(pat)))
    return files


def lint_file(path: Path, catalog: dict, style: bool) -> tuple[int, int]:
    flags_meta = {f["name"].lower(): f for f in catalog.get("flags", [])}
    banned = {b.lower() for b in catalog.get("bannedInConfig", BANNED_FALLBACK)}
    conflicts = catalog.get("conflicts", [])

    errors = 0
    warnings = 0
    enabled: dict[str, int] = {}

    text = path.read_text(encoding="utf-8", errors="replace")
    for line_no, raw in enumerate(text.splitlines(), start=1):
        if style and raw.rstrip("\n\r") != raw.rstrip():
            warnings += 1
            emit(path, line_no, 1, "warning", "Trailing whitespace")

        code = strip_comment(raw)
        if not code.strip():
            continue

        tokens = tokenize(code)
        if not tokens:
            continue

        flag = tokens[0]
        col = raw.find(flag) + 1 if flag in raw else 1
        if not flag.startswith("-"):
            errors += 1
            emit(path, line_no, col, "error", f"Expected a -flag, got '{flag}'")
            continue

        key = flag.lower()
        if key in banned:
            errors += 1
            emit(
                path,
                line_no,
                col,
                "error",
                f"{flag} belongs on the Hammer / command line, not in a config file",
            )
            continue

        meta = flags_meta.get(key)
        if meta is None:
            sev = "error" if style else "warning"
            if sev == "error":
                errors += 1
            else:
                warnings += 1
            emit(path, line_no, col, sev, f"Unknown flag {flag}")
            continue

        if meta.get("debugOnly"):
            warnings += 1
            emit(
                path,
                line_no,
                col,
                "warning",
                f"{flag} is debug-build only; Release VRAD ignores it",
            )

        arity = int(meta.get("arity", 0))
        values = tokens[1:]
        if style and len(tokens) > 1 + max(arity, 0):
            errors += 1
            emit(
                path,
                line_no,
                col,
                "error",
                f"{flag} has extra tokens (expected at most {arity} value(s))",
            )
        elif len(values) > arity:
            errors += 1
            emit(
                path,
                line_no,
                col,
                "error",
                f"{flag} takes {arity} value(s), got {len(values)}",
            )
        else:
            enums = meta.get("enum")
            if enums and values:
                got = values[0].strip('"').lower()
                allowed = [e.lower() for e in enums]
                if got not in allowed:
                    errors += 1
                    emit(
                        path,
                        line_no,
                        col,
                        "error",
                        f"{flag} value '{values[0]}' is not one of {', '.join(enums)}",
                    )
            vmin = meta.get("min")
            vmax = meta.get("max")
            vtype = meta.get("value")
            if values and vtype in ("int", "float"):
                try:
                    num = float(values[0]) if vtype == "float" else int(values[0], 10)
                    if vmin is not None and num < vmin:
                        errors += 1
                        emit(path, line_no, col, "error", f"{flag} {num} is below min {vmin}")
                    if vmax is not None and num > vmax:
                        errors += 1
                        emit(path, line_no, col, "error", f"{flag} {num} is above max {vmax}")
                except ValueError:
                    errors += 1
                    emit(path, line_no, col, "error", f"{flag} wants a {vtype}, got '{values[0]}'")
            warn = meta.get("warn")
            if warn and values:
                warnings += 1
                emit(path, line_no, col, "warning", f"{flag}: {warn}")

        if key not in enabled:
            enabled[key] = line_no

    for rule in conflicts:
        names = [n.lower() for n in rule.get("flags", [])]
        present = [n for n in names if n in enabled]
        if len(present) < 2:
            continue
        sev = rule.get("severity", "warning")
        msg = rule.get("message", "Conflicting flags")
        line = enabled[present[0]]
        if sev == "error":
            errors += 1
        else:
            warnings += 1
        emit(path, line, 1, sev, msg)

    return errors, warnings


def main() -> int:
    ap = argparse.ArgumentParser(description="Lint PathRAD -config files.")
    ap.add_argument(
        "--style",
        action="store_true",
        help="Stricter: unknown flags are errors, extra tokens are errors.",
    )
    ap.add_argument(
        "paths",
        nargs="*",
        default=["configs"],
        help="Files or folders (default: configs)",
    )
    args = ap.parse_args()

    if not CATALOG.is_file():
        print(f"error: missing catalog {CATALOG}", file=sys.stderr)
        return 2

    catalog = load_catalog()
    files: list[Path] = []
    for p in args.paths:
        target = Path(p)
        if not target.is_absolute():
            target = (Path.cwd() / target).resolve()
        if not target.exists():
            print(f"error: path not found: {target}", file=sys.stderr)
            return 2
        files.extend(collect_cfg_files(target))

    if not files:
        print("error: no .cfg / .txt files found", file=sys.stderr)
        return 2

    errors = 0
    warnings = 0
    for f in files:
        e, w = lint_file(f, catalog, args.style)
        errors += e
        warnings += w

    print(f"{len(files)} file(s), {errors} error(s), {warnings} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

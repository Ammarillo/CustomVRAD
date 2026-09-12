#!/usr/bin/env python3
"""Build the VS Code / Cursor grammar from tools/vrad_cfg_flags.json."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CATALOG = Path(__file__).resolve().parent / "vrad_cfg_flags.json"
EXT = Path(__file__).resolve().parent / "vscode-vrad-cfg"


def alt(names: list[str]) -> str:
    names = sorted(names, key=lambda s: (-len(s), s.lower()))
    return "|".join(names)


def main() -> None:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    flags = sorted({e["name"] for e in catalog["flags"]}, key=str.lower)
    banned = sorted(catalog["bannedInConfig"], key=str.lower)

    (EXT / "syntaxes").mkdir(parents=True, exist_ok=True)

    grammar = {
        "$schema": "https://raw.githubusercontent.com/martinring/tmlanguage/master/tmlanguage.json",
        "name": "VRAD Config",
        "scopeName": "source.vrad-cfg",
        "patterns": [
            {"include": "#comments"},
            {"include": "#bannedFlags"},
            {"include": "#knownFlags"},
            {"include": "#unknownFlags"},
            {"include": "#strings"},
            {"include": "#numbers"},
        ],
        "repository": {
            "comments": {
                "patterns": [
                    {
                        "name": "comment.line.number-sign.vrad-cfg",
                        "begin": "(^[ \\t]+)?(?=#)",
                        "end": "$",
                        "beginCaptures": {
                            "1": {
                                "name": "punctuation.whitespace.comment.leading.vrad-cfg"
                            }
                        },
                        "patterns": [
                            {
                                "name": "markup.heading.section.vrad-cfg",
                                "match": "={3,}.*$",
                            },
                            {
                                "name": "markup.heading.subsection.vrad-cfg",
                                "match": "-{3,}.*$",
                            },
                            {
                                "name": "keyword.flag.commented.vrad-cfg",
                                "match": f"(?i)(?<!\\w)(?:{alt(flags)})(?!\\w)",
                            },
                            {
                                "name": "constant.numeric.commented.vrad-cfg",
                                "match": "\\b\\d+(\\.\\d+)?\\b",
                            },
                        ],
                    },
                    {
                        "name": "comment.line.double-slash.vrad-cfg",
                        "begin": "(^[ \\t]+)?(?=//)",
                        "end": "$",
                        "beginCaptures": {
                            "1": {
                                "name": "punctuation.whitespace.comment.leading.vrad-cfg"
                            }
                        },
                    },
                ]
            },
            "bannedFlags": {
                "name": "invalid.illegal.banned-flag.vrad-cfg",
                "match": f"(?i)(?<!\\w)(?:{alt(banned)})(?!\\w)",
            },
            "knownFlags": {
                "name": "keyword.control.flag.vrad-cfg",
                "match": f"(?i)(?<!\\w)(?:{alt(flags)})(?!\\w)",
            },
            "unknownFlags": {
                "name": "entity.name.tag.flag.unknown.vrad-cfg",
                "match": "(?<!\\w)-[A-Za-z_][\\w.\\-]*(?!\\w)",
            },
            "strings": {
                "name": "string.quoted.double.vrad-cfg",
                "begin": '"',
                "end": '"',
                "patterns": [
                    {
                        "name": "constant.character.escape.vrad-cfg",
                        "match": "\\\\.",
                    }
                ],
            },
            "numbers": {
                "patterns": [
                    {
                        "name": "constant.numeric.float.vrad-cfg",
                        "match": "(?<![\\w.])-?\\d+\\.\\d+(?!\\w)",
                    },
                    {
                        "name": "constant.numeric.integer.vrad-cfg",
                        "match": "(?<![\\w.])-?\\d+(?!\\w)",
                    },
                ]
            },
        },
    }

    (EXT / "syntaxes" / "vrad-cfg.tmLanguage.json").write_text(
        json.dumps(grammar, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    package = {
        "name": "vrad-cfg",
        "displayName": "PathRAD Config",
        "description": "Syntax highlighting for PathRAD -config presets (configs/*.cfg).",
        "version": "0.1.0",
        "publisher": "pathrad",
        "engines": {"vscode": "^1.74.0"},
        "categories": ["Programming Languages"],
        "contributes": {
            "languages": [
                {
                    "id": "vrad-cfg",
                    "aliases": ["VRAD Config", "vrad-cfg"],
                    "configuration": "./language-configuration.json",
                    "filenamePatterns": ["**/configs/*.cfg"],
                }
            ],
            "grammars": [
                {
                    "language": "vrad-cfg",
                    "scopeName": "source.vrad-cfg",
                    "path": "./syntaxes/vrad-cfg.tmLanguage.json",
                }
            ],
        },
    }
    (EXT / "package.json").write_text(
        json.dumps(package, indent=2) + "\n", encoding="utf-8"
    )

    lang_cfg = {
        "comments": {"lineComment": "#"},
        "brackets": [["\"", "\""]],
        "autoClosingPairs": [{"open": "\"", "close": "\""}],
        "surroundingPairs": [["\"", "\""]],
    }
    (EXT / "language-configuration.json").write_text(
        json.dumps(lang_cfg, indent=2) + "\n", encoding="utf-8"
    )

    (EXT / "README.md").write_text(
        """# PathRAD Config highlighter

Cursor / VS Code TextMate grammar for `configs/*.cfg`.

## Install

From repo root:

```powershell
.\\tools\\install_vrad_cfg_highlight.ps1
```

Then **Developer: Reload Window**.

## Scopes

| Scope | Meaning |
|-------|---------|
| `keyword.control.flag.vrad-cfg` | Known VRAD flags |
| `keyword.flag.commented.vrad-cfg` | Flags inside `#` comments |
| `invalid.illegal.banned-flag.vrad-cfg` | Hammer-only flags that must not be in configs |
| `entity.name.tag.flag.unknown.vrad-cfg` | Unknown `-flag` |
| `constant.numeric.*.vrad-cfg` | Values |
| `comment.line.*.vrad-cfg` | Comments |
| `markup.heading.*.vrad-cfg` | Section banners |

Workspace colors live in `.vscode/settings.json`.

Regenerate after catalog edits:

```powershell
python tools/_gen_vrad_cfg_grammar.py
.\\tools\\install_vrad_cfg_highlight.ps1
```
""",
        encoding="utf-8",
    )

    print(f"wrote {EXT} ({len(flags)} flags, {len(banned)} banned)")


if __name__ == "__main__":
    main()

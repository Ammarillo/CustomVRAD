# PathRAD Config highlighter

Cursor / VS Code TextMate grammar for `configs/*.cfg`.

## Install

From repo root:

```powershell
.\tools\install_vrad_cfg_highlight.ps1
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
.\tools\install_vrad_cfg_highlight.ps1
```

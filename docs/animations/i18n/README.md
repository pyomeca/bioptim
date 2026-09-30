# Translations of the animation texts

`fr.json` (and any `fr_*.json`, merged in alphabetical order, so several people can work in parallel without
merge conflicts) maps a **key** to a French **template**. The series layer (`series_style.py`) looks every
`Text` / `MarkupText` up here when rendering with `--lang fr` (or `SERIES_LANG=fr`).

## Keys

The key is the English string where every numeric token (`-0.87`, `3.14`, `1e-3`, `12`, the `100` of `100 N`) is
replaced by `{0}`, `{1}`, ... in order of appearance. Newlines are part of the key. Leading / trailing spaces are
ignored. For `MarkupText` the key keeps the markup (`<b>..</b>`), only the text outside the tags is normalised.

```json
"IPOPT cost {0}  ·  {1} iterations": "coût IPOPT {0}  ·  {1} itérations"
```

The template may reorder the numbers (`{1}` before `{0}`). Numbers are re-inserted verbatim; in French the decimal
point becomes a comma (`SERIES_FR_COMMA=0` or `--fr-dot` to keep the point).

## Never translated (skipped silently)

* text in the code font (`Consolas` / `DejaVu Sans Mono`): code panels, `method='...'`
* strings without a real word (three consecutive letters): `t (s)`, `12`, symbols
* single tokens that look like identifiers or acronyms: `OdeSolver`, `ObjectiveFcn.Lagrange`, `IPOPT`, `n_shooting`

Keep Bioptim / API identifiers untranslated inside sentences too.

## Workflow

1. Collect the keys of a scene (or of all scenes, construct-only, a few minutes):
   `python docs/animations/render_series.py anim_controls.py ControlTypes --dry --collect keys.jsonl`
   `python docs/animations/render_series.py --all --dry --collect keys.jsonl --jobs 4`
   Each line: `{"scene", "key", "count", "file", "line", "markup"}`.
2. Add the translations to `fr_<topic>.json`.
3. Render in French; strings still missing are logged (`<out>/logs/<Scene>_fr_missing.jsonl`) and stay in English.
   `--strict` makes the render exit non-zero if any is missing.

## Glossary

| English | Français |
| --- | --- |
| OCP | problème de commande optimale (OCP) |
| multiple shooting | tir multiple |
| direct collocation | collocation directe |
| node | nœud |
| controls | commandes |
| states | états |
| constraint | contrainte |
| bounds | bornes |
| cost | coût |
| warm start | initialisation à chaud |
| initial guess | estimation initiale |
| phase | phase |
| weight | poids |

French text is about 15-20 % longer: keep sentences short. When a translated text is much wider than the English
one it is scaled down automatically (never below 0.8) and logged in the audit json (`fr_autofit`).

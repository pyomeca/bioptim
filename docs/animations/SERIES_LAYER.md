# Series layer (logo, pacing, plain text transitions, EN/FR, end card, audit)

A shared layer applied **at render time** to every scene of `docs/animations/`, without editing any scene,
`generate_*` or notes file. Scene authors keep writing plain Manim; the layer patches Manim before the scene module
is imported (so `from manim import *` already sees the patched names).

| File | Role |
| --- | --- |
| `series_style.py` | the layer itself (`install()`) |
| `render_series.py` | CLI: patch first, load the scene by file path, render with Manim's config |
| `catalog.json` | one entry per scene: titles, descriptions, links of the end card, logo corner |
| `i18n/fr.json`, `i18n/README.md` | French templates, translation workflow and glossary |
| `assets/fetch_logo.py` | downloads `assets/bioptim_logo.png` (git-ignored, never commit it) |

## Quick start

```bash
python docs/animations/assets/fetch_logo.py          # once (also done automatically on first render)
python docs/animations/render_series.py anim_controls.py ControlTypes --lang en
python docs/animations/render_series.py anim_controls.py ControlTypes --lang fr --strict
python docs/animations/render_series.py --all --lang both --jobs 4 --out my_videos
python docs/animations/render_series.py --all --dry --collect keys.jsonl --jobs 4     # no video, list every text key
python docs/animations/render_series.py --gen-catalog                                 # add new scenes to catalog.json
```

Output: `<out>/<Scene>_<lang>.mp4` (default `docs/animations/media/series/out`, git-ignored), audit and
missing-translation logs in `<out>/logs` (`<Scene>_<lang>_audit.json`, `<Scene>_fr_missing.jsonl`,
`batch_summary.json`). `--quality` is `<height>p<fps>`: `1080p30` (default), `1080p60`, `480p15`.
Caching is always disabled (`always_redraw` closures break Manim's hashing in some scenes).

Options: `--no-endcard`, `--no-logo`, `--logo-corner auto|br|bl|tr|tl`, `--slow F`, `--min-wait S`, `--fr-dot`,
`--strict`, `--catalog PATH`, `--log-dir DIR`, `--media-dir DIR`, `--jobs N`. Everything is also configurable through
environment variables (used by `series_style.install()`): `SERIES_LANG`, `SERIES_SLOW` (1.6), `SERIES_MIN_WAIT` (0.5),
`SERIES_LOGO`, `SERIES_LOGO_CORNER`, `SERIES_ENDCARD`, `SERIES_CATALOG`, `SERIES_COLLECT`, `SERIES_MISSING`,
`SERIES_LOG_DIR`, `SERIES_FR_COMMA` (1), `SERIES_FIT_X` (0.15), `SERIES_FIT_MIN` (0.8).

To use the layer from another script: set the variables, `import series_style; series_style.install()`, then import
the scene module and render as usual.

## 1. Logo

`assets/bioptim_logo.png` (from `pyomeca/biorbd_design`, `logo_png/bioptim_full.png`) is composited on **every
frame** on its way to the movie file (`CairoRenderer.add_frame`), 0.35 scene units high, 75 % opaque. It is not a
mobject, so it can never be removed by `FadeOut(*self.mobjects)`, `self.clear()` or `self.remove(...)`, and it costs
nothing in the scene graph. The large logo of the end card is a regular `ImageMobject`.

Corner: `--logo-corner` > `logo_corner` of the catalog entry > **automatic**: a construct-only probe pass audits the
end of every `play()` against the four corners (order `br`, `tl`, `tr`, `bl`) and picks the first corner that is free
during the whole scene (otherwise the least often covered one). Collisions of the chosen corner are reported in the
audit json.

If the file is missing and cannot be downloaded, the videos are rendered without logo (warning on stderr).

## 2. Slower, same content

`SERIES_SLOW` (default 1.6) multiplies the `run_time` of every animation played (also of `wait()`, since Manim
implements it as a `Wait` animation). It is applied to the top level animation after Manim merged the `run_time=`
keyword of `play()`, so `AnimationGroup`, `LaggedStart` and `Succession` are stretched uniformly (their internal
timings are relative). `always_redraw`, updaters and `ValueTracker` animations are unaffected: only the clock is
stretched.

Reading pause: after a `play()` of at least 0.25 s (before scaling) that introduces or changes text (any
`Text`/`MarkupText` in a non-removing animation), a pause of `SERIES_MIN_WAIT` (0.5 s) is guaranteed: if the next call
is a `wait()` it is lengthened to at least that value, otherwise a wait is inserted before the next `play()`. Very
short plays (loops of 0.1-0.3 s updates) are not padded.

Measured: the video is exactly `1.6 x` the original plus the added pauses plus the 4.5 s end card.

## 3. Plain text transitions

Names are replaced in the `manim` namespace; only **text** is simplified (a Text/MarkupText or a group made only of
them), everything else keeps the original class:

| Original call on text | Becomes |
| --- | --- |
| `Write`, `AddTextLetterByLetter`, `AddTextWordByWord`, `TypeWithCursor` (any text inside) | `FadeIn` |
| `Unwrite`, `RemoveTextLetterByLetter`, `UntypeWithCursor` | `FadeOut` |
| `Create`, `DrawBorderThenFill`, `ShowIncreasingSubsets`, `ShowSubmobjectsOneByOne` (text only) | `FadeIn` |
| `Uncreate` (text only) | `FadeOut` |
| `Transform(a, b)` | cross-fade, **`a` still exists and now shows `b`** (Transform semantics) |
| `ReplacementTransform`, `TransformMatchingShapes`, `TransformMatchingTex` | cross-fade, `a` leaves the scene, `b` takes its place |
| `Indicate`, `Circumscribe` | colour pulse without scaling or box |
| `Wiggle`, `ApplyWave` | nothing (a `Wait` of the same duration) |

`Transform` between curves / polygons / shapes, `Create` on plots, `Write` on non-text are untouched. The cross-fade
copies the old text, fades it out at its place while the target content (assigned into the original object) fades in.
The number of replacements per name is in the audit json (`text_animation_replacements`).

## 4. EN / FR

`Text.__init__` and `MarkupText.__init__` are patched (so `Paragraph`, which builds `Text` lines, is covered too).
In `fr` mode the string is looked up in `i18n/fr*.json` **before** construction:

1. the string is normalised: every numeric token (`-0.87`, `3.14`, `1e-3`, `12`, the `100` of `100 N`) becomes `{0}`,
   `{1}`, ... in order of appearance (numbers glued to letters such as `x1`, and the inside of markup tags/entities,
   are left alone); newlines are part of the key; leading/trailing blanks are ignored;
2. the template of `fr.json` is filled with the original numbers, verbatim, except that `.` becomes `,` in FR
   (`SERIES_FR_COMMA=0` to disable). Templates may reorder `{0}`, `{1}`;
3. no translation: the string stays English and is logged (json lines `scene, key, file, line, count`) to
   `<out>/logs/<Scene>_fr_missing.jsonl`; `--strict` exits with status 3 if any.

Never translated: text in a monospace font (`Consolas`, `DejaVu Sans Mono`, ... : all code panels), strings without a
real word (3 consecutive letters), single tokens that look like identifiers/acronyms (`OdeSolver`,
`ObjectiveFcn.Lagrange.X`, `IPOPT`, `n_shooting`).

`SERIES_COLLECT=<file.jsonl>` (or `--collect`) logs, for every scene built, each translatable key with its count and
`file:line` of the scene method that created it (both languages). With `--dry` nothing is rendered.

FR auto-fit: after construction, a translated text wider than `(1 + SERIES_FIT_X)` x the width of the English one (or
than 13.6 units) is scaled down, never below `SERIES_FIT_MIN` = 0.8, and logged in `fr_autofit`. Rule of thumb for
translators: French is 15-20 % longer, keep sentences short. `t2c=` colour maps keyed on English words do not match
translated text (none is used today).

## 5. End card

After `construct()`, `tear_down` appends about 4.5 s: everything fades out (0.5 s), the card fades in (0.8 s) and holds
(3.2 s), with no other effect. Content from `catalog.json` (entry id `<scene_file>.py:<SceneClass>`): large logo,
"Learn more" / "Pour aller plus loin", the scene title (`title_en` / `title_fr`), up to 5 links (label + repo-relative
path in the code font, optional `:lines`) and `github.com/pyomeca/bioptim`. Without links only the logo, the title
and the repository line are shown. `--no-endcard` disables it.

### Catalog schema

```json
{
  "id": "anim_controls.py:ControlTypes",
  "slug": "control-types",
  "title_en": "Control interpolation", "title_fr": "Interpolation des commandes",
  "description_en": "", "description_fr": "",
  "video_basename": "ControlTypes",
  "duration_note": "",
  "links": [
    {"label_en": "Example", "label_fr": "Exemple",
     "path": "bioptim/examples/getting_started/custom_dynamics.py", "lines": "40-90"}
  ],
  "notes_file": "notes/controls.md",
  "logo_corner": "tl"
}
```

`logo_corner` and `lines` are optional. `python render_series.py --gen-catalog` adds the skeleton of any new scene
class (found by AST, no Manim import) and never overwrites existing entries.

## 6. Frame audit

At the end of every `play()` (not of waits) the layer inspects everything on screen and appends to the audit json
(`warnings`, each with the time in the video, the play index and an occurrence count; deduplicated):

* `logo_overlap`: a visible shape or glyph of the chosen corner box (0.03 margin; filled shapes count from 30 %
  opacity, strokes are sampled along their Bezier curves, not by bounding box);
* `text_out_of_frame`: a Text/MarkupText box beyond x = +-7.1 or y = +-4.0 (0.05 tolerance);
* `text_overlap`: two texts whose boxes overlap by more than 25 % of the smaller one;
* `endcard`: informational.

`corner_scores_end_of_play` and `corner_overlaps` give, for the 4 corners, the number of plays with a collision and
what collided. The audit samples the state at the end of each play, not every frame: a moving object crossing the logo
mid-animation is not detected.

## Known limitations

* Only Cairo is supported. The audit is sampled at the end of plays. Text created and animated but never in
  `scene.mobjects` is not audited.
* `Tex`/`MathTex` (LaTeX) are not translated or simplified (none is used).
* `t2c`/`t2w`... maps and index-based slicing (`text[3:7]`) of translated text refer to the English string.
* Dynamic texts whose numbers are not the only variable part (e.g. an f-string inserting words) give one key per
  variant.
* `Transform` between a text and a non-text mobject keeps Manim's morph.
* A translated `Text` is scaled down on construction; scene code that later uses `.width` sees the scaled value.

Repository notes: the root `.gitignore` ignores `*.json` and `*.png`; `docs/animations/.gitignore` re-includes
`catalog.json` and `i18n/*.json` and ignores `assets/*.png`, `*.jsonl` and `series_out/`. The logo PNG must never be
committed (`fetch_logo.py` recreates it).

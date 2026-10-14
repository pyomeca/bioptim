# Standard for a Bioptim teaching video

*Version française : [STANDARD.fr.md](STANDARD.fr.md).*

The authoritative specification for the videos of `docs/animations/`. A new video, written by a person or by an agent,
must follow it so that it is indistinguishable from the 48 existing ones. Everything here is checkable; section 7 is the
checklist to tick before a video is accepted. Start from [templates/scene_template.py](templates/scene_template.py)
(a working skeleton that follows every rule) and use [SERIES_LAYER.md](SERIES_LAYER.md) for what the render layer does
for you.

Where the older documents disagree with this file (see "Decisions" at the end of section 7), this file wins.

## 1. Purpose, audience, scope

* **Audience**: students and researchers in biomechanics and optimal control who know some Python and mechanics, not
  necessarily direct multiple shooting. The video is a door into the Bioptim example and code, not a replacement.
* **One idea per video.** A viewer must be able to say in one sentence what it taught (that sentence is the
  `description_en` of the catalog entry). If you need "and also", make a second video.
* **Length**: 10-20 s of content at native Manim speed. The layer multiplies every `run_time` by 1.6 (plus 0.5 s reading
  pauses) and appends a 4.5 s end card, so the final video is about 16-32 s + 4.5 s. At most **2 beats** (a beat is a
  self-contained sequence, e.g. "the node grid" then "real solves"; a second beat starts by fading everything but the
  title). The oldest scenes are longer (median 31 s in total); do not use them as a length reference.
* **Language**: scenes are written in English; French is produced by the layer from `i18n/fr_<topic>.json` (section 5).
* **Level** (catalog): 1 introductory, 2 intermediate, 3 advanced (needs earlier videos).

## 2. Data honesty rules

1. **Every curve and every number comes from a REAL bioptim solve** (or a real library object: a real `OdeSolver`,
   `VectorLayout`, `Solution`...), produced by `generate_<topic>_data.py` and stored in `data/<topic>_*.npz`
   (small: a few kB to a few tens of kB). Nothing is typed by hand, extrapolated, smoothed or "sketched".
2. **Synthetic input data** (simulated measurements, noisy markers, a made-up reference) is allowed only when the
   scene says so on screen (subtitle or legend: "synthetic data: simulated motion + Gaussian noise (sigma = 0.05 rad)"),
   with the noise level and the seed in the generator and the notes.
3. **Re-drawn interfaces** (IPOPT output, a plot window, a table of a Bioptim object) are labelled "re-drawn, not a
   screen capture" on screen, and their content is real.
4. **Report the solver honestly**: IPOPT status and iteration count are stored (`status`, `iterations`, `converged`)
   and shown (`ipopt_line`). Say on screen or in the notes when a run is warm started or uses continuation (each solve
   starts from the previous one), and when several local minima exist (a non-convex problem returns *a* local
   minimum). Never show a non-converged solution as a result; if a failure is the point (e.g. `CONTINUOUS` instead of
   `IMPACT` is infeasible), label it ("last IPOPT iterate, not a solution").
5. **If the topic cannot be made to work** (no convergence, an API that does not do what the docs say in this
   version), stop, keep the failing script, and report it. Do not fake a result and do not change the claim to fit
   the numbers.
6. **Numbers quoted on screen are computed from the npz** in the scene (`float(np.abs(tau).max())`, then
   `f"{peak:.1f}"`), never read off a plot by eye and never copied into the string as a literal. The same holds for the
   notes: they quote the values printed by the generator.
7. **The code shown is the code that ran.** Every line of the code panel is (a simplification of) a line of
   `generate_<topic>_data.py`, and every identifier, keyword argument and default is verified against the source of
   THIS version (`grep -n "def name\|class Name" bioptim/...`), not against the docs or memory. A simplified call
   (bounds or imports omitted, `...`) is allowed only if the linked example is the reference and the notes say so.
8. **Simplifications are listed** in the notes ("Honest caveats"): simplified model, fixed durations, missing
   constraints, choices of N, T, tolerances.

## 3. Visual anatomy (1920x1080, Manim frame 14.2 x 8 units)

```
+--------------------------------------------------------------------------------+
|                          TITLE (34 bold)  /  subtitle (22 gray)                |  y = +3.7
| axis label (20)                              Bioptim code   (caption, 20 gray) |
|  +-----------------------+                   objectives.add(...   (code, 17-20)|  CODE_X = 0.15
|  |  plot 1 (ghost dashed |                   ...                              |  CODE_W = 6.75
|  |  + new curve)         |                                                     |
|  +-----------------------+                   readout: weight = 10 . peak = 0.67 |
|  |  plot 2: difference   |                   IPOPT: 22 iterations, converged   |
|  +-----------------------+   t (s)           remark (yellow, 17-19)            |
| footer sentence (16, at most 2 lines, ends before the logo)            [logo]  |  y = -3.9
+--------------------------------------------------------------------------------+
```

* **Title**: `scene_title(title, subtitle)`: title 34 bold, subtitle 22 gray (`GRAY_B`), centred at the top. The title
  names the concept ("Penalty on the derivative of a control"), the subtitle states the setting with its numbers
  ("swing-up, N = 30, T = 1 s"). Synthetic data is declared here.
* **Left = visuals, right = code.** Plots in x from `PLOT_X0 = -6.6` to `PLOT_X1 = -0.5` (`make_axes` centred at
  x = -3.55, width 5.6); the code panel starts at `CODE_X = 0.15` and is at most `CODE_W = 6.75` wide, top at y = 2.3;
  other right-hand texts are at most `TEXT_W = 5.9` wide (`place`).
* **"Bioptim code" caption**: every code panel has the caption `Bioptim code` (size 20, `GRAY_B`) **above** the code
  lines. Use `code_panel(lines)` of `features_scenes.py`: it adds the caption, indents `(level, text, colour)` lines by
  0.3 units per level, scales to `CODE_W`. If a panel needs a longer caption, keep the start (`Bioptim code (M =
  walk_hopper.bioMod)`). The code lines are in the code font at size 17-20 (never below 15), one call per line.
  Colour the arguments that change, and animate the change (`Transform` of the line).
* **Readout block** (right, below the code): numbers computed from the data, size 19, `GRAY_A`, `line_spacing=0.9`,
  built as ONE `Text` with newlines: `weight = 10 · peak |y| = 0.67` / `IPOPT: 22 iterations, converged`
  (`ipopt_line(iterations, converged)`). Put it through `place(mob, CODE_X, y)`.
* **Remark** (optional, right): `say(sentence, size 17-19)` in `YELLOW_C`: one whole sentence, only about something
  that is plotted. Wrapping is done by `para`.
* **Footer**: `footer(sentence)`, size 16, `YELLOW_C` (or `GRAY_B` for a plain note), at the bottom left
  (x = -6.9, y = -3.92): one sentence, at most two lines, width at most 10.4 so that it ends before the bottom-right
  logo. Use it for the honest caveat of the video ("Schematic grid with N = 10; the real problem uses N = 30").
* **Ghost reference and difference axis.** When a quantity *changes* (a weight, a bound, a solver, a model), make the
  change visible: keep the reference curve on screen as a dashed grey ghost
  (`DashedVMobject(poly(...), num_dashes=40).set_opacity(0.8)`, colour `GRAY_B`) and add a second, smaller axis with the
  difference `new - reference` (colour `PURPLE_B`, zero line drawn), with its own label ("Δθ = θ − θ_free (rad)").
  Never narrate something that is not plotted: if the text says "the peak falls", the peak must be visible and the
  number in the readout.
* **Colour roles** (constants of `features_scenes.py`; reuse them, add new ones only for a new role and name them
  `C_<ROLE>`):

  | Role | Constant | Colour |
  | --- | --- | --- |
  | controls, Lagrange | `C_CTRL`, `C_LAG` | `GREEN_C` |
  | states, the changing curve | `C_STATE` | `YELLOW_C` |
  | Mayer, second phase | `C_MAY`, `C_PH1` | `ORANGE` |
  | first phase | `C_PH0` | `BLUE_C` |
  | bounds, forbidden region, failure | `C_BOUND` | `RED_C` (band: `band(...)`, opacity 0.22) |
  | time, duration | `C_TIME` | `TEAL_C` |
  | parameters, differences | `C_PAR` | `PURPLE_B` |
  | ghost, axis labels, secondary text | - | `GRAY_B` |
  | readouts | - | `GRAY_A` |
  | remarks, footer | - | `YELLOW_C` |

  A curve keeps the same colour in all plots and in the code lines that create it.
* **Fonts**: `FONT` (Segoe UI on Windows, DejaVu Sans elsewhere) for all text, `MONO` (Consolas / DejaVu Sans Mono)
  for code; both are set as defaults by importing `features_scenes`. Sizes: title 34, subtitle 22, axis labels and caption
  20, readout 19, code 17-20, remark 17-19, footer 16, ticks and `t (s)` 16, smallest annotation 14.
* **Axes and units**: `make_axes` (no ticks drawn), then `axis_label` (above the y axis, with the unit in parentheses:
  `angle (rad)`, `actuated force (N)`), `time_label(ax)` (`t (s)`), `x_ticks` / `y_ticks` for tick labels (they call
  `dec()`). One idea per axis label; symbol first when useful: `θ(t)  angle (rad)`.
* **Decimals**: numbers formatted in Python inside a *sentence* are plain (`f"{x:.2f}"`); the layer turns the `.` into a
  `,` in French. Numeric-only labels that you build yourself and that contain no word (ticks, bar values) go through
  `dec(...)`, as `x_ticks` / `y_ticks` do. Never use `dec()` on code-font text (a French code panel keeps its points).
* **Frame safety**: nothing beyond x = ±7.1 or y = ±4.0 (the audit flags any `Text` box outside, with a 0.05
  tolerance), texts must not overlap (audit: overlap above 25 % of the smaller box), the logo corner (auto,
  0.35 units high, 75 % opaque, bottom right by default) must stay free. French is about 20 % longer: design every
  right-hand text to still end before x = 7.1 with +20 % width.
* **Ending**: the last statement of `construct` is `self.wait(2.5)`. The layer then adds the end card.

## 4. Text rules

* **Whole sentences** in one `Text` object (or one multi-line `Text` made by `para`/`say`, where the newlines are part
  of the translation key). Never assemble a sentence from several `Text` objects: French word order differs.
* **Numbers inside the same string** as the sentence, as a plain number (`f"peak |τ| = {peak:.1f} N"`): the layer turns
  each number into `{0}`, `{1}` of the translation key. Do not put a number in a separate `Text` next to its words.
* **No abbreviation without expansion** the first time (`OCP` -> "optimal control problem (OCP)"; `NLP`, `DMS`, `RK4` are
  spelled out in the subtitle or the footer of the video that introduces them). Symbols (`θ`, `τ`, `∫`) are fine when the
  axis label also says the word.
* **Plain transitions only**: `FadeIn`, `FadeOut`, `Create` (plots), `Transform` (curves, numbers, code lines). Do not
  write text letter by letter, no `Wiggle`, no `Indicate` on text; the layer converts them anyway (SERIES_LAYER.md, section 3)
  but the scene must read well without.
* **Code is never translated**: code-font text, identifiers (`ObjectiveFcn.Lagrange.MINIMIZE_CONTROL`, `n_shooting`,
  `IPOPT`), file names stay English, also inside French sentences. The caption `Bioptim code` is a normal key
  (`Code Bioptim` in French).
* **No `Paragraph`** (Manim cuts translated lines to the English glyph count; the layer works around it but the rule
  avoids the trap): use one `Text` per line (`Lines(...)` helper of `anim_mhe.py`, or `say`/`para`).
* **No `MathTex`/`Tex`/LaTeX** (not installed, not translated). Write formulas with `Text`/`MarkupText` and Unicode
  (`∫ L(x, u) dt`, `t<sub>N−1</sub>` with `MarkupText`).
* No `t2c`/`t2w` maps or index slicing on text: they refer to the English string; colour whole objects instead.

## 5. Files, naming, catalog, translation

| What | Where | Notes |
| --- | --- | --- |
| Scene classes | `docs/animations/anim_<topic>.py` | `<topic>` lower case, one file per video; class name CamelCase = mp4 name |
| Data generator | `docs/animations/generate_<topic>_data.py` | real bioptim solve, prints status/iterations, writes the npz; docstring explains what is stored; usage line `PYTHONPATH=. python docs/animations/generate_<topic>_data.py` |
| Data | `docs/animations/data/<topic>_*.npz` | small; committed (the only binaries allowed) |
| Models | `docs/animations/models/<topic>_*.bioMod` | only if no model of `bioptim/examples/models` fits |
| Notes | `docs/animations/notes/<topic>.md` | template below |
| Catalog entry | `docs/animations/catalog.json` | fields below |
| Translation | `docs/animations/i18n/fr_<topic>.json` | key -> template with `{0}`, `{1}` |

The scene file starts with a docstring (what it shows, data file, generator, scene name and length, render command),
imports `from manim import *` then the helpers from `features_scenes` (`scene_title`, `code_panel`, `make_axes`, `poly`,
`steps`, `axis_label`, `time_label`, `x_ticks`, `y_ticks`, `say`, `footer`, `place`, `ipopt_line`, `hline`, `band`,
`DATA_DIR`, colour constants). Do not copy helpers into the scene file; if one is missing, add it to
`features_scenes.py` in a separate commit. Format with `black -t py311 -l120`.

**Notes file** (`notes/<topic>.md`), in this order:
1. `# Title (anim_<topic>.py)`, then `## What the scene teaches`: the idea, the setting (model, N, T, solver), the
   numbers shown, taken from the generator output;
2. `## Commands`: environment, generator, render (section 6);
3. `## Honest caveats`: local minima, warm starts, simplifications, timings, what is *not* shown;
4. `## Exercises`: 2-3 numbered, concrete tasks that change one argument of the code shown.

**Catalog entry** (all fields; `id` is `<file>.py:<Class>`):

```json
{
  "id": "anim_<topic>.py:MyScene", "slug": "my-scene",
  "title_en": "...", "title_fr": "...",
  "description_en": "one idea, the setting, the result (with numbers)", "description_fr": "...",
  "video_basename": "MyScene", "duration_note": "",
  "links": [
    {"label_en": "Example: ...", "label_fr": "Exemple : ...", "path": "bioptim/examples/.../x.py", "lines": "40-90"},
    {"label_en": "Scene data generator", "label_fr": "Générateur des données de la scène",
     "path": "docs/animations/generate_<topic>_data.py", "lines": "46-72"}
  ],
  "notes_file": "notes/<topic>.md", "level": 2,
  "section": "Objectives and constraints", "section_fr": "Objectifs et contraintes"
}
```

* `section` is one of the eight: Fundamentals (Fondamentaux), Discretization (Discrétisation), Objectives and
  constraints (Objectifs et contraintes), Phases and time (Phases et temps), Solvers and numerics (Solveurs et
  numérique), Models and biomechanics (Modèles et biomécanique), Advanced control (Commande avancée), Library overview
  (Vue d'ensemble de la bibliothèque). `section_fr` must be the French name exactly as given.
* `links`: at most 5 (end card limit), first an **example** of `bioptim/examples/...`, then **library lines**, last the
  generator. Every `path` exists and every `lines` range (`"40-90"`) is checked by opening the file in this version and
  making sure it covers the definition it names; `build_readme_tables.py --check` checks existence only, not line numbers.
* `logo_corner` (`tl`, `tr`, `bl`, `br`) only if the automatic choice still collides (audit json, `logo_overlap`).
* Add the entry with `python docs/animations/render_series.py --gen-catalog` (adds a skeleton for every new
  `class X(Scene)` of `anim_*.py`, never overwrites), then fill every field, both languages. Templates in
  `templates/` are not scanned.
* Then `python docs/animations/build_readme_tables.py` and update the counts written in prose in `README.md` /
  `README.fr.md` (number of videos, duration statistics).

**Translation workflow** (details and glossary in [i18n/README.md](i18n/README.md)):
1. `python docs/animations/render_series.py anim_<topic>.py MyScene --dry --collect keys.jsonl` lists every key
   (`file:line` of the origin);
2. write `i18n/fr_<topic>.json`: same key with numbers replaced by `{0}`, `{1}` in order of appearance; the template
   may reorder them; keep newlines of multi-line keys as the same number of lines;
3. glossary: OCP = problème de commande optimale, multiple shooting = tir multiple, direct collocation = collocation
   directe, node = nœud, controls = commandes, states = états, constraint = contrainte, bounds = bornes, cost = coût,
   warm start = initialisation à chaud, initial guess = estimation initiale, weight = poids. Identifiers stay English;
4. `--lang fr --strict` must pass (exit status 3 = untranslated key).

**Commit hygiene**: only files under `docs/animations/` (scene, generator, npz, model, notes, `catalog.json`, `i18n`,
README tables). Never commit `*.mp4`, `*.png`, `assets/bioptim_logo.png`, `media/`, `*.jsonl`, `p.out` or any output
file dropped in the repository root; check `git status` before committing.

## 6. Commands

Rendering environment (no bioptim needed): Python 3.11 venv with `manim==0.21.*`, `numpy`, `black`; fonts Segoe UI and
Consolas on Windows. Data environment: conda env with bioptim, biorbd, casadi and IPOPT; on Windows, when calling its
`python.exe` without `conda activate`:

```bash
E=/c/Users/<you>/miniconda3/envs/captury_biobuddy
export PATH="$E:$E/Library/bin:$E/Library/mingw-w64/bin:$E/Library/usr/bin:$E/Scripts:$PATH"   # else "Plugin 'ipopt' is not found"
export PYTHONIOENCODING=utf-8
```

From the repository root:

```bash
PYTHONPATH=. python docs/animations/generate_<topic>_data.py                      # real solve -> data/<topic>_*.npz
python docs/animations/render_series.py anim_<topic>.py MyScene --dry              # construct only (seconds)
python docs/animations/render_series.py anim_<topic>.py MyScene --lang en          # 1080p30 (default quality)
python docs/animations/render_series.py anim_<topic>.py MyScene --lang fr --strict # French, fails on a missing key
python docs/animations/render_series.py anim_<topic>.py MyScene --lang both --quality 480p15   # quick preview
python docs/animations/render_series.py --all --lang both --jobs 4 --out my_videos # the whole series
black -t py311 -l120 docs/animations/anim_<topic>.py docs/animations/generate_<topic>_data.py
python docs/animations/build_readme_tables.py                                      # README tables + link check
```

Outputs: `<out>/<Scene>_<lang>.mp4` (default `docs/animations/media/series/out`, git-ignored) and, in `<out>/logs`,
`<Scene>_<lang>_audit.json` and `<Scene>_fr_missing.jsonl`. Template check:
`python docs/animations/render_series.py templates/scene_template.py TemplateScene --dry`
(data: `python docs/animations/templates/generate_template_data.py`).

## 7. QA and definition of done

**Frames to look at**, in English and in French, from the 1080p30 mp4 (or a 480p15 one): at ~1 s, at 35 %, at 65 %,
just before the end card, and in the middle of the end card. Extract them with PyAV (bundled with Manim) or any player.
Look for: logo not overlapping anything, no text overlapping or cut by the frame, no text unreadable at the size of a
laptop screen, plots drawn before the sentence that talks about them.

**Audit json** (`<Scene>_<lang>_audit.json`): `text_overlap`, `text_out_of_frame` and `logo_overlap` must be absent from
`warnings`; `fr_autofit` (French texts scaled down, never below 0.8) should be empty or mild: shorten the sentence rather than rely on it; `corner_overlaps` of the chosen corner
empty. Only `endcard` (informational) may remain.

**Checklist** (all boxes before merging):

- [ ] One idea, at most 2 beats, 10-20 s of content at native speed; the subtitle states the setting.
- [ ] Every curve and number from a real bioptim solve stored in `data/<topic>_*.npz` by `generate_<topic>_data.py`;
      IPOPT status 0 for every solution shown; warm start / continuation / local minima disclosed.
- [ ] Synthetic input is labelled synthetic on screen; re-drawn UI is labelled "re-drawn, not a screen capture".
- [ ] Numbers on screen computed from the npz in the scene (no literals); notes quote the generator output.
- [ ] Code panel: caption `Bioptim code` above, lines verified by `grep` against this version and identical to the
      generator; changing arguments coloured and animated.
- [ ] A changing quantity has its ghost reference and its difference axis; nothing narrated that is not plotted.
- [ ] Colour roles, fonts, sizes, `CODE_X` / `CODE_W` / `TEXT_W` as in section 3; units in parentheses; `t (s)`.
- [ ] Whole sentences, numbers inside the string, no Paragraph, no LaTeX, no abbreviation without expansion, plain fades.
- [ ] Footer of at most two lines ending before the logo; `self.wait(2.5)` last.
- [ ] Dry pass OK (`--dry`); EN render: audit clean; frames checked at the five moments.
- [ ] `i18n/fr_<topic>.json` written; `--lang fr --strict` exits 0 (0 missing); glossary respected (commandes, états,
      nœud, tir multiple, collocation directe, bornes, coût...); decimals with commas; FR frames checked, audit clean.
- [ ] Links resolve (`build_readme_tables.py --check`) and each line range was opened and checked.
- [ ] `notes/<topic>.md` written with honest caveats and 2-3 exercises.
- [ ] Catalog entry complete (both languages, level, section pair, links <= 5), `--gen-catalog` not needed again, README
      tables regenerated and prose counts updated.
- [ ] Formatted with `black -t py311 -l120`; `git status` shows only `docs/animations/` text files and small npz (no mp4,
      png, `p.out`, output folder).

### Decisions (where the existing conventions were ambiguous)

1. **Code panel helper**: the scenes use either `code_panel()` (features_scenes) or a local `code_block` + caption with the
   same look. New scenes use `code_panel()`; a hand-made panel is allowed only for a layout it cannot express, with the
   caption above.
2. **Caption translation**: README.md says the panel caption is "never translated"; in fact `Bioptim code` has the key
   `Code Bioptim` in `i18n/fr.json`. Rule: the caption follows the normal translation, the code lines never do.
3. **Multi-line text**: a wrapped sentence is one `Text` (`say`/`para`), a stack of independent lines is one `Text` per line
   (`Lines` in the `anim_mhe.py` family); `Paragraph` is not used in new scenes although `anim_markers.py` still has one.
4. **`dec()`**: for word-less labels built by the scene; sentences and code are not passed through it.
5. **Length target**: 10-20 s content / 2 beats for new videos; older ones (up to 106 s in total) are not a model.
6. **Beat transition**: fade every mobject except the title (`self.play(*[FadeOut(m) for m in self.mobjects if m is not
   title])`, then `self.add(title)`), as in `ObjectivesNodes`.
7. **Templates outside the scanned files**: `templates/` is not scanned by `--gen-catalog` or `--all`; a real scene is copied to
   `docs/animations/anim_<topic>.py`. The template's placeholder data `data/template_demo.npz` is git-ignored; the template
   has no French translation file, so `--lang fr --strict` fails on it by design.

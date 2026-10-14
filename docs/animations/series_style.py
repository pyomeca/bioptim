"""
Series layer for the bioptim Manim videos: a shared, non-invasive layer applied at RENDER time to every scene.

It never edits a scene file. ``install()`` monkey-patches manim (before the scene modules are imported, so that
``from manim import *`` picks up the patched names) and provides

    * a small Bioptim logo composited on every frame (survives ``FadeOut(*self.mobjects)`` because it is not a mobject)
    * a global slow-down (``SERIES_SLOW``, default 1.6) and a minimum reading pause after text changes
    * plain text transitions (no letter-by-letter Write, Transform of text becomes a cross-fade, ...)
    * EN / FR (``SERIES_LANG``) through template matching of every Text / MarkupText (``i18n/fr*.json``)
    * a collect mode (``SERIES_COLLECT``) listing every translatable key, and a missing-translation log
    * an "end card" (Learn more / Pour aller plus loin) built from ``catalog.json``
    * a frame audit (logo overlap, text out of the frame, text overlapping text) logged as JSON

Use it through ``render_series.py``. See SERIES_LAYER.md for the full documentation.
"""

from __future__ import annotations

import atexit
import functools
import html
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LOGO_PATH = HERE / "assets" / "bioptim_logo.png"
I18N_DIR = HERE / "i18n"
DEFAULT_CATALOG = HERE / "catalog.json"

FRAME_W, FRAME_H = 14.2222222, 8.0
TEXT_X_LIMIT, TEXT_Y_LIMIT, TEXT_TOL = 7.1, 4.0, 0.05
LOGO_HEIGHT = 0.35  # scene units
LOGO_OPACITY = 0.75
LOGO_MARGIN = (0.2, 0.15)
CORNER_ORDER = ("br", "tl", "tr", "bl")
CORNER_ALIASES = {
    "bottom-right": "br",
    "bottom_right": "br",
    "br": "br",
    "top-left": "tl",
    "top_left": "tl",
    "tl": "tl",
    "top-right": "tr",
    "top_right": "tr",
    "tr": "tr",
    "bottom-left": "bl",
    "bottom_left": "bl",
    "bl": "bl",
}
FR_FRAME_MAX_WIDTH = 13.6
FADE_PART = 0.45  # text cross-fade: fraction of the time used by each of the fade-out and the fade-in
TEXT_FONT = "Segoe UI" if sys.platform == "win32" else "DejaVu Sans"
MONO_FONT = "Consolas" if sys.platform == "win32" else "DejaVu Sans Mono"
MONO_RE = re.compile(r"mono|consolas|courier|menlo|fira ?code|source code", re.I)

NUM_RE = re.compile(r"(?<![\w.])(?:[-−](?=\d))?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
PROTECT_RE = re.compile(r"(<[^>]*>|&#?\w+;)")  # markup tags and entities are never normalised
PLACEHOLDER_RE = re.compile(r"\{(\d+)\}")


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() not in ("0", "false", "no", "off")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


class Config:
    """Read from the environment by ``install()`` (render_series.py sets these variables from its CLI flags)."""

    def load(self) -> "Config":
        self.slow = _env_float("SERIES_SLOW", 1.6)
        self.min_wait = _env_float("SERIES_MIN_WAIT", 0.5)
        self.lang = os.environ.get("SERIES_LANG", "en").strip().lower() or "en"
        self.fr_comma = _env_bool("SERIES_FR_COMMA", True)
        self.collect = os.environ.get("SERIES_COLLECT", "")
        self.missing = os.environ.get("SERIES_MISSING", "")
        self.log_dir = os.environ.get("SERIES_LOG_DIR", "")
        self.logo = _env_bool("SERIES_LOGO", True)
        self.logo_corner = os.environ.get("SERIES_LOGO_CORNER", "auto").strip().lower() or "auto"
        self.endcard = _env_bool("SERIES_ENDCARD", True)
        self.catalog = os.environ.get("SERIES_CATALOG", "") or str(DEFAULT_CATALOG)
        self.fit_x = _env_float("SERIES_FIT_X", 0.15)  # FR text may be this much wider than the EN one
        self.fit_min = _env_float("SERIES_FIT_MIN", 0.8)  # never scale a text below this factor
        self.small_max = _env_float("SERIES_SMALL_MAX", 40.0)  # texts below this font size are built larger...
        self.small_k = _env_float("SERIES_SMALL_K", 4.0)  # ...by this factor, then scaled back (Pango hinting)
        self.i18n_extra = os.environ.get("SERIES_I18N_EXTRA", "")  # extra json files (os.pathsep separated)
        return self


CFG = Config().load()


# --------------------------------------------------------------------------------------------------------------------
# State (one scene at a time, one process per render)
# --------------------------------------------------------------------------------------------------------------------
class State:
    def __init__(self):
        self.reset_scene()
        self.installed = False
        self.probe = False  # True during the probe pass (dry, no logs written except the corner scores)
        self.dry = False  # construct only, no frames
        self.no_endcard = False
        self.overlay = True
        self.corner = "br"
        self.scene_name = "?"
        self.scene_id = "?"
        self.guard = False  # translation bypass (temporary English text for the width comparison, end card)
        self.fr = {}
        self.fr_loaded = False
        self.catalog = None
        self.logo_cache = {}
        self.last_probe_scores = None

    def reset_scene(self):
        self.keys = {}  # key -> dict(count, file, line, markup)
        self.missing = {}
        self.fits = []
        self.warnings = []
        self.warn_seen = {}
        self.corner_scores = {c: 0 for c in CORNER_ORDER}
        self.corner_details = {c: [] for c in CORNER_ORDER}
        self.counts = {}
        self.play_index = 0
        self.finished = False
        self.t_wait_added = 0.0
        self.text_plays_padded = 0


S = State()


def _write_lines(path: str, rows: list):
    if not path or not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(payload)


def norm_corner(value) -> str | None:
    return CORNER_ALIASES.get(str(value).strip().lower()) if value else None


# --------------------------------------------------------------------------------------------------------------------
# i18n
# --------------------------------------------------------------------------------------------------------------------
def load_translations(lang: str) -> dict:
    """Merge i18n/<lang>.json and i18n/<lang>_*.json (several translators can work on separate files)."""
    table = {}
    files = sorted(I18N_DIR.glob(f"{lang}.json")) + sorted(I18N_DIR.glob(f"{lang}_*.json"))
    if lang == "fr" and CFG.i18n_extra:  # SERIES_I18N_EXTRA: extra files (tests), applied last
        files += [Path(f) for f in CFG.i18n_extra.split(os.pathsep) if f.strip()]
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"[series] cannot read {path}: {exc}", file=sys.stderr)
            continue
        for key, value in data.items():
            if not key.startswith("_") and isinstance(value, str) and value != "":
                table[key] = value
    return table


def is_mono(font) -> bool:
    return bool(font) and bool(MONO_RE.search(str(font)))


def is_translatable(core: str, markup: bool) -> bool:
    s = html.unescape(re.sub(r"<[^>]*>", "", core)) if markup else core
    if not re.search(r"[^\W\d_]{3,}", s):  # needs a real word (skips symbols, "t (s)", numbers...)
        return False
    if not re.search(r"\s", s.strip()):  # single token: API identifier / acronym / camelCase -> keep
        if "_" in s or "." in s or re.search(r"[a-z][A-Z]", s) or s.isupper() or "(" in s:
            return False
    return True


def normalise(core: str, markup: bool = False):
    """Return (key, numbers): every numeric token replaced by {0}, {1}, ... in order of appearance."""
    numbers = []

    def repl_segment(seg: str) -> str:
        def sub(match):
            numbers.append(match.group(0))
            return "{%d}" % (len(numbers) - 1)

        return NUM_RE.sub(sub, seg)

    if markup:
        parts = PROTECT_RE.split(core)
        key = "".join(p if i % 2 else repl_segment(p) for i, p in enumerate(parts))
    else:
        key = repl_segment(core)
    return key, numbers


# a decimal number (digits.digits[e+-n]) that is not part of a version / file name / identifier (v1.2.3, a1.5, 3.14.py)
DEC_RE = re.compile(r"(?<![\w.])\d+\.\d+(?:[eE][-+]?\d+)?(?!\.\w)(?![A-Za-z]*_)")


def decimal_comma(text: str, markup: bool = False) -> str:
    """FR: 0.25 -> 0,25 in any text (markup tags and entities are left alone)."""
    if "." not in text:
        return text
    if markup:
        parts = PROTECT_RE.split(text)
        return "".join(
            p if i % 2 else DEC_RE.sub(lambda m: m.group(0).replace(".", ","), p) for i, p in enumerate(parts)
        )
    return DEC_RE.sub(lambda m: m.group(0).replace(".", ","), text)


def render_template(template: str, numbers: list, comma: bool) -> str:
    def sub(match):
        idx = int(match.group(1))
        if idx >= len(numbers):
            return match.group(0)
        value = numbers[idx]
        return value.replace(".", ",") if comma else value

    return PLACEHOLDER_RE.sub(sub, template)


def _source_location():
    """file:line of the nearest frame executing a method of a Scene (fallback: nearest frame in docs/animations)."""
    try:
        from manim import Scene

        frame = sys._getframe(2)
        fallback = None
        while frame is not None:
            fname = frame.f_code.co_filename
            if fname != __file__ and Path(fname).parent.resolve() == HERE:
                if fallback is None:
                    fallback = (Path(fname).name, frame.f_lineno)
                if isinstance(frame.f_locals.get("self"), Scene):
                    return Path(fname).name, frame.f_lineno
            frame = frame.f_back
        return fallback or ("?", 0)
    except Exception:
        return ("?", 0)


def translate(text: str, mono: bool, markup: bool):
    """Return (text_to_build, status) with status in None (skipped), 'seen', 'translated', 'missing'."""
    if S.guard or mono or not isinstance(text, str):
        return text, None
    core = text.strip()
    if not core or not is_translatable(core, markup):
        if CFG.lang == "fr" and CFG.fr_comma:  # numeric labels are never translated but still get the decimal comma
            return decimal_comma(text, markup), None
        return text, None
    lead = text[: len(text) - len(text.lstrip())]
    trail = text[len(text.rstrip()) :]
    key, numbers = normalise(core, markup)
    entry = S.keys.get(key)
    if entry is None:
        file, line = _source_location()
        entry = S.keys[key] = {"count": 0, "file": file, "line": line, "markup": markup}
    entry["count"] += 1
    if CFG.lang != "fr":
        return text, "seen"
    template = S.fr.get(key)
    if template is None:
        miss = S.missing.get(key)
        if miss is None:
            S.missing[key] = {"count": 1, "file": entry["file"], "line": entry["line"]}
        else:
            miss["count"] += 1
        return text, "missing"
    return lead + render_template(template, numbers, CFG.fr_comma) + trail, "translated"


def _small_text_factor(args, kwargs) -> float:
    """Build factor for small texts: Pango hints/kerns badly at the tiny size Manim asks for (font_size / 4.8 pt)."""
    if CFG.small_k <= 1 or kwargs.get("height") is not None or kwargs.get("width") is not None:
        return 1.0
    size = kwargs.get("font_size", args[3] if len(args) > 3 else 48)
    try:
        return CFG.small_k if 0 < float(size) < CFG.small_max else 1.0
    except (TypeError, ValueError):
        return 1.0


def _install_text_patches():
    from manim import MarkupText, Paragraph, Text

    for cls in (Text, MarkupText):
        if getattr(cls.__init__, "_series_patched", False):
            continue
        original = cls.__init__
        markup = cls is MarkupText

        @functools.wraps(original)
        def init(self, text=None, *args, __orig=original, __markup=markup, **kwargs):
            if text is None and "text" in kwargs:  # text passed by keyword
                text = kwargs.pop("text")
            font = kwargs.get("font", args[5] if len(args) > 5 else None)
            new_text, status = translate(text, is_mono(font), __markup)
            k = _small_text_factor(args, kwargs)
            if k != 1.0:  # build at k x the size, then scale back: same size on screen, clean glyph spacing
                size = float(kwargs.get("font_size", args[3] if len(args) > 3 else 48))
                if "font_size" in kwargs or len(args) <= 3:
                    kwargs["font_size"] = size * k
                    b_args = args
                else:
                    b_args = args[:3] + (size * k,) + args[4:]
                __orig(self, new_text, *b_args, **kwargs)
                self.scale(1.0 / k)
            else:
                __orig(self, new_text, *args, **kwargs)
            self._series_text = new_text
            if status == "translated":
                _autofit(self, type(self), text, args, kwargs, __orig)

        init._series_patched = True
        cls.__init__ = init

    if not getattr(Paragraph.__init__, "_series_patched", False):
        par_original = Paragraph.__init__

        @functools.wraps(par_original)
        def par_init(self, *text, **kwargs):
            # Manim splits the Paragraph glyphs by the ENGLISH line lengths: translate the joined text first (one key,
            # newlines included), then hand the translated string to Manim, which splits it by its own newline-separated lines.
            joined = chr(10).join(text)
            new_text, status = translate(joined, is_mono(kwargs.get("font")), False)
            prev, S.guard = S.guard, True
            try:
                par_original(self, new_text, **kwargs)
                ref_w = None
                if status == "translated":
                    try:
                        ref_w = Paragraph(joined, **kwargs).width
                    except Exception:
                        pass
            finally:
                S.guard = prev
            self._series_text = new_text
            if status == "translated":
                _fit_width(self, ref_w, new_text)
                self.lines_initial_positions = [line.get_center() for line in self.lines_chars]

        par_init._series_patched = True
        Paragraph.__init__ = par_init


def _autofit(mob, cls, original_text, args, kwargs, orig_init):
    """FR text is longer: scale it down (never below CFG.fit_min) when it is much wider than the EN one."""
    S.guard = True
    try:
        ref = cls(original_text, *args, **kwargs)
        ref_w = ref.width
    except Exception:
        ref_w = None
    finally:
        S.guard = False
    _fit_width(mob, ref_w, getattr(mob, "_series_text", "") or "")


def _fit_width(mob, ref_w, text: str):
    width = mob.width
    target = min(ref_w * (1 + CFG.fit_x), FR_FRAME_MAX_WIDTH) if ref_w else FR_FRAME_MAX_WIDTH
    if width > target > 0:
        scale = max(CFG.fit_min, target / width)
        mob.scale(scale)
        S.fits.append(
            {
                "kind": "fr_autofit",
                "text": text[:70],
                "en_width": round(ref_w, 2) if ref_w else None,
                "fr_width": round(width, 2),
                "scale": round(scale, 3),
            }
        )


# --------------------------------------------------------------------------------------------------------------------
# Text-aware animation patches
# --------------------------------------------------------------------------------------------------------------------
def is_text_leaf(m) -> bool:
    from manim import MarkupText, Text

    return isinstance(m, (Text, MarkupText))


def all_text(m) -> bool:
    """True when the mobject is a Text/MarkupText, or a (nested) group made only of them."""
    from manim import Mobject

    if not isinstance(m, Mobject):
        return False
    if is_text_leaf(m):
        return True
    if m.has_points() or not m.submobjects:
        return False
    return all(all_text(s) for s in m.submobjects)


def any_text(m) -> bool:
    from manim import Mobject

    return isinstance(m, Mobject) and any(is_text_leaf(x) for x in m.get_family())


def _anim_kwargs(kwargs: dict) -> dict:
    return {k: v for k, v in kwargs.items() if k in ("run_time", "rate_func", "name")}


def _first_mobject(args, kwargs, name="mobject"):
    if args:
        return args[0]
    return kwargs.get(name)


def _make_text_crossfade():
    from manim import Animation, linear, smooth

    class TextCrossFade(Animation):
        """
        Two-step fade between two texts: the old one fades out during the first 45 % of the time, the new one fades in
        during the last 45 % (a 10 % gap in between), so both are never readable together. Between two texts (or groups of text). ``replace=False`` keeps Transform semantics: the
        object ``mobject`` still exists afterwards and now shows the target text. ``replace=True`` keeps
        ReplacementTransform semantics: ``mobject`` leaves the scene and ``target`` takes its place.
        """

        def __init__(self, mobject, target, replace=False, **kwargs):
            kwargs.setdefault("rate_func", linear)
            self.target_text = target
            self.replace = replace
            super().__init__(mobject, use_override=False, **kwargs)

        @staticmethod
        def _record(mob):
            rec = []
            for m in mob.get_family():
                rec.append((m, float(m.get_fill_opacity()), float(m.get_stroke_opacity())))
            return rec

        def _setup_scene(self, scene):
            self._scene = scene
            super()._setup_scene(scene)

        def begin(self):
            scene = getattr(self, "_scene", None)
            a, b = self.mobject, self.target_text
            if self.replace:
                self.fade_out, self.fade_in = a, b
                self.rec_out = self._record(a)
                self.rec_in = self._record(b)
                self.temp = None
            else:
                new = b.copy()
                old = a.copy()
                a.submobjects = []
                a.points = new.points.copy()
                for sm in list(new.submobjects):
                    a.add(sm)
                if hasattr(new, "text"):
                    a.text = new.text
                self.fade_out, self.fade_in = old, a
                self.rec_out = self._record(old)
                self.rec_in = self._record(a)
                self.temp = old
            if scene is not None:
                if self.temp is not None:
                    scene.add(self.temp)
                if self.replace and b not in scene.get_mobject_family_members():
                    scene.add(b)
            super().begin()

        def get_all_mobjects(self):
            return [m for m in (self.mobject, self.fade_out, self.fade_in, self.target_text) if m is not None]

        def create_starting_mobject(self):
            return self.mobject.copy() if not hasattr(self, "fade_out") else self.fade_out

        def interpolate_mobject(self, alpha):
            a = min(max(self.rate_func(alpha), 0.0), 1.0)
            k_out = 1.0 - smooth(min(a / FADE_PART, 1.0))  # old text: 1 -> 0 over the first 45 %
            k_in = smooth(
                min(max((a - (1.0 - FADE_PART)) / FADE_PART, 0.0), 1.0)
            )  # new text: 0 -> 1 over the last 45 %
            for m, fo, so in self.rec_out:
                m.set_fill(opacity=fo * k_out)
                m.set_stroke(opacity=so * k_out)
            for m, fo, so in self.rec_in:
                m.set_fill(opacity=fo * k_in)
                m.set_stroke(opacity=so * k_in)

        def clean_up_from_scene(self, scene):
            super().clean_up_from_scene(scene)
            if self.temp is not None:
                scene.remove(self.temp)
            if self.replace:
                scene.remove(self.mobject)
                scene.add(self.target_text)
                for m, fo, so in self.rec_out:  # leave the removed object pristine
                    m.set_fill(opacity=fo)
                    m.set_stroke(opacity=so)

    return TextCrossFade


def _fade_out_all(mobs):
    """Fade out everything (robust: manim's FadeOut scales the mobjects, which breaks on empty Line/Arrow objects)."""
    from manim import Animation, Group, ImageMobject, VMobject

    class FadeOutAll(Animation):
        def __init__(self, mobjects, **kwargs):
            super().__init__(Group(*mobjects), remover=True, **kwargs)

        def begin(self):
            self.rec = [
                (m, float(m.get_fill_opacity()), float(m.get_stroke_opacity()))
                for m in self.mobject.get_family()
                if isinstance(m, VMobject)
            ]
            self.images = [(m, m.pixel_array.copy()) for m in self.mobject.get_family() if isinstance(m, ImageMobject)]
            super().begin()

        def interpolate_mobject(self, alpha):
            k = 1 - min(max(self.rate_func(alpha), 0.0), 1.0)
            for m, fo, so in self.rec:
                m.set_fill(opacity=fo * k)
                m.set_stroke(opacity=so * k)
            for m, px in self.images:
                m.pixel_array[..., 3] = (px[..., 3] * k).astype(px.dtype)

        def clean_up_from_scene(self, scene):
            scene.remove(*self.mobject.submobjects)
            scene.clear()

    return FadeOutAll(mobs)


def _install_animation_patches():
    import manim

    if getattr(manim, "_series_anim_patched", False):
        return
    TextCrossFade = _make_text_crossfade()
    FadeIn, FadeOut, Wait = manim.FadeIn, manim.FadeOut, manim.Wait
    Indicate = manim.Indicate
    patched = {}

    def patch(name, builder):
        orig = getattr(manim, name, None)
        if orig is None:
            return

        def __new__(cls, *args, **kwargs):
            if cls is patched[name]:
                result = builder(args, kwargs)
                if result is not None:
                    S.counts[name] = S.counts.get(name, 0) + 1
                    return result
            return orig.__new__(cls, *args, **kwargs)

        new_cls = type(name, (orig,), {"__new__": __new__, "__module__": orig.__module__, "__doc__": orig.__doc__})
        patched[name] = new_cls
        setattr(manim, name, new_cls)

    def fade_in_if(pred):
        def build(args, kwargs):
            mob = _first_mobject(args, kwargs)
            if mob is not None and pred(mob):
                return FadeIn(mob, **_anim_kwargs(kwargs))
            return None

        return build

    def fade_out_if(pred):
        def build(args, kwargs):
            mob = _first_mobject(args, kwargs)
            if mob is not None and pred(mob):
                return FadeOut(mob, **_anim_kwargs(kwargs))
            return None

        return build

    for name in ("Write", "AddTextLetterByLetter", "AddTextWordByWord", "TypeWithCursor"):
        patch(name, fade_in_if(any_text))
    for name in ("Unwrite", "RemoveTextLetterByLetter", "UntypeWithCursor"):
        patch(name, fade_out_if(any_text))
    for name in ("Create", "DrawBorderThenFill", "ShowIncreasingSubsets", "ShowSubmobjectsOneByOne"):
        patch(name, fade_in_if(all_text))
    patch("Uncreate", fade_out_if(all_text))

    def transform_builder(replace):
        def build(args, kwargs):
            a = args[0] if args else kwargs.get("mobject")
            b = args[1] if len(args) > 1 else kwargs.get("target_mobject")
            if a is not None and b is not None and a is not b and all_text(a) and all_text(b):
                return TextCrossFade(a, b, replace=replace, **_anim_kwargs(kwargs))
            return None

        return build

    patch("Transform", transform_builder(False))
    patch("ReplacementTransform", transform_builder(True))
    patch("TransformMatchingShapes", transform_builder(True))
    patch("TransformMatchingTex", transform_builder(True))

    def indicate_like(args, kwargs):
        mob = _first_mobject(args, kwargs)
        if mob is not None and all_text(mob):
            color = kwargs.get("color", manim.YELLOW)
            return Indicate(mob, scale_factor=1.0, color=color, **_anim_kwargs(kwargs))
        return None

    def nothing_like(args, kwargs):
        mob = _first_mobject(args, kwargs)
        if mob is not None and all_text(mob):
            return Wait(run_time=kwargs.get("run_time", 1))
        return None

    patch("Circumscribe", indicate_like)
    patch("Wiggle", nothing_like)
    patch("ApplyWave", nothing_like)
    # Indicate itself: keep the colour pulse but drop the scaling on text
    orig_indicate = Indicate

    def indicate_builder(args, kwargs):
        mob = _first_mobject(args, kwargs)
        if mob is not None and all_text(mob):
            kw = dict(kwargs)
            kw["scale_factor"] = 1.0
            if len(args) > 1:  # scale_factor given positionally
                args = (args[0],) + args[2:]
            return orig_indicate(*args, **kw)
        return None

    patch("Indicate", indicate_builder)
    manim._series_anim_patched = True


# --------------------------------------------------------------------------------------------------------------------
# Logo (composited on the frames, never a mobject)
# --------------------------------------------------------------------------------------------------------------------
def logo_aspect() -> float:
    from PIL import Image

    with Image.open(LOGO_PATH) as im:
        return im.width / im.height


def logo_box(corner: str, aspect: float | None = None):
    """(x0, x1, y0, y1) of the logo in scene units."""
    aspect = aspect or (logo_aspect() if LOGO_PATH.exists() else 2.85)
    w, h = LOGO_HEIGHT * aspect, LOGO_HEIGHT
    mx, my = LOGO_MARGIN
    x1 = FRAME_W / 2 - mx
    y0 = -FRAME_H / 2 + my
    if corner == "br":
        return x1 - w, x1, y0, y0 + h
    if corner == "bl":
        return -x1, -x1 + w, y0, y0 + h
    if corner == "tr":
        return x1 - w, x1, FRAME_H / 2 - my - h, FRAME_H / 2 - my
    return -x1, -x1 + w, FRAME_H / 2 - my - h, FRAME_H / 2 - my  # tl


def _logo_pixels(pixel_h: int):
    cached = S.logo_cache.get(pixel_h)
    if cached is not None:
        return cached
    from PIL import Image

    with Image.open(LOGO_PATH) as im:
        im = im.convert("RGBA")
        h_px = max(2, int(round(LOGO_HEIGHT / FRAME_H * pixel_h)))
        w_px = max(2, int(round(h_px * im.width / im.height)))
        im = im.resize((w_px, h_px), Image.LANCZOS)
    arr = np.asarray(im, dtype=np.float32)
    alpha = arr[..., 3:4] / 255.0 * LOGO_OPACITY
    S.logo_cache[pixel_h] = (arr[..., :3], alpha)
    return S.logo_cache[pixel_h]


def composite_logo(frame):
    if not (CFG.logo and S.overlay and LOGO_PATH.exists()):
        return frame
    H, W = frame.shape[:2]
    rgb, alpha = _logo_pixels(H)
    h_px, w_px = rgb.shape[:2]
    x0, x1, y0, y1 = logo_box(S.corner)
    px = int(round((x0 + FRAME_W / 2) / FRAME_W * W))
    py = int(round((FRAME_H / 2 - y1) / FRAME_H * H))
    px, py = max(0, min(px, W - w_px)), max(0, min(py, H - h_px))
    region = frame[py : py + h_px, px : px + w_px, :3].astype(np.float32)
    frame = frame.copy()
    frame[py : py + h_px, px : px + w_px, :3] = (region * (1 - alpha) + rgb * alpha).astype(np.uint8)
    return frame


# --------------------------------------------------------------------------------------------------------------------
# Frame audit
# --------------------------------------------------------------------------------------------------------------------
def _describe(owner, mob) -> str:
    if owner is not None:
        return "Text '%s'" % (getattr(owner, "_series_text", "") or "")[:50].replace("\n", " ")
    return type(mob).__name__


def _collect(scene):
    """Return shape records and text records of everything currently on screen (logo excluded)."""
    from manim import ImageMobject, VMobject

    shapes, texts = [], []
    t = np.linspace(0, 1, 7)[None, :, None]

    def walk(mob, owner):
        if is_text_leaf(mob):
            fam = [m for m in mob.get_family() if m.has_points()]
            if not fam:
                return
            if max(max(m.get_fill_opacity(), m.get_stroke_opacity()) for m in fam) < 0.05:
                return
            pts = np.concatenate([m.points for m in fam])
            texts.append((mob, pts.min(0), pts.max(0)))
            owner = mob
        if isinstance(mob, ImageMobject):
            pts = mob.points
            shapes.append((owner, mob, pts.min(0), pts.max(0), 1.0, pts))
        elif isinstance(mob, VMobject) and mob.has_points():
            fill, stroke = mob.get_fill_opacity(), mob.get_stroke_opacity()
            if stroke > 0.05 and mob.get_stroke_width() <= 0:
                stroke = 0
            if max(fill, stroke) >= 0.05:
                P = mob.points[: len(mob.points) // 4 * 4].reshape(-1, 4, 3)
                if len(P):
                    s = (
                        (1 - t) ** 3 * P[:, 0:1]
                        + 3 * (1 - t) ** 2 * t * P[:, 1:2]
                        + 3 * (1 - t) * t**2 * P[:, 2:3]
                        + t**3 * P[:, 3:4]
                    ).reshape(-1, 3)
                    shapes.append((owner, mob, s.min(0), s.max(0), fill, s if stroke >= 0.05 else None))
        for sm in mob.submobjects:
            walk(sm, owner)

    from manim import Mobject

    seen = set()
    for m in list(scene.mobjects) + list(getattr(scene, "foreground_mobjects", [])):
        if id(m) in seen or not isinstance(m, Mobject):
            continue
        seen.add(id(m))
        walk(m, None)
    return shapes, texts


def _warn(kind: str, detail: dict, key: str, scene):
    if key in S.warn_seen:
        S.warn_seen[key]["count"] += 1
        return
    entry = {"kind": kind, "time": round(float(scene.renderer.time), 2), "play": S.play_index, "count": 1, **detail}
    S.warn_seen[key] = entry
    S.warnings.append(entry)


def audit(scene):
    """Called at the end of every play (not for waits): logo overlap for the 4 corners, text vs frame, text vs text."""
    if S.finished:
        return
    S.play_index += 1
    try:
        shapes, texts = _collect(scene)
    except Exception as exc:  # the audit must never break a render
        _warn("audit_error", {"error": repr(exc)}, "audit_error", scene)
        return
    aspect = logo_aspect() if LOGO_PATH.exists() else 2.85
    for corner in CORNER_ORDER:
        x0, x1, y0, y1 = logo_box(corner, aspect)
        x0, x1, y0, y1 = x0 - 0.03, x1 + 0.03, y0 - 0.03, y1 + 0.03
        hits = []
        for owner, mob, lo, hi, fill, samples in shapes:
            if hi[0] < x0 or lo[0] > x1 or hi[1] < y0 or lo[1] > y1:
                continue
            area = (hi[0] - lo[0]) * (hi[1] - lo[1])
            hit = False
            if fill >= 0.3 and area < 0.4 * FRAME_W * FRAME_H:
                hit = True
            elif samples is not None:
                hit = bool(
                    np.any(
                        (samples[:, 0] >= x0) & (samples[:, 0] <= x1) & (samples[:, 1] >= y0) & (samples[:, 1] <= y1)
                    )
                )
            if hit:
                hits.append(_describe(owner, mob))
        if hits:
            S.corner_scores[corner] += 1
            uniq = sorted(set(hits))[:4]
            for h in uniq:
                if h not in S.corner_details[corner] and len(S.corner_details[corner]) < 12:
                    S.corner_details[corner].append(h)
            if corner == S.corner and not S.probe:
                for h in uniq:
                    _warn("logo_overlap", {"corner": corner, "what": h}, f"logo|{corner}|{h}", scene)
    for mob, lo, hi in texts:
        if (
            lo[0] < -TEXT_X_LIMIT - TEXT_TOL
            or hi[0] > TEXT_X_LIMIT + TEXT_TOL
            or lo[1] < -TEXT_Y_LIMIT - TEXT_TOL
            or hi[1] > TEXT_Y_LIMIT + TEXT_TOL
        ):
            what = (getattr(mob, "_series_text", "") or "")[:60].replace("\n", " ")
            _warn(
                "text_out_of_frame",
                {
                    "text": what,
                    "x": [round(float(lo[0]), 2), round(float(hi[0]), 2)],
                    "y": [round(float(lo[1]), 2), round(float(hi[1]), 2)],
                },
                f"oof|{what}",
                scene,
            )
    if len(texts) > 1:
        los = np.array([t[1][:2] for t in texts])
        his = np.array([t[2][:2] for t in texts])
        for i in range(len(texts)):
            ov = np.minimum(his[i], his[i + 1 :]) - np.maximum(los[i], los[i + 1 :])
            area = np.clip(ov[:, 0], 0, None) * np.clip(ov[:, 1], 0, None)
            small = np.minimum(np.prod(his[i] - los[i]), np.prod(his[i + 1 :] - los[i + 1 :], axis=1))
            for off in np.nonzero(area > 0.25 * np.maximum(small, 1e-9))[0]:
                j = i + 1 + int(off)
                a = (getattr(texts[i][0], "_series_text", "") or "")[:40].replace("\n", " ")
                b = (getattr(texts[j][0], "_series_text", "") or "")[:40].replace("\n", " ")
                _warn("text_overlap", {"a": a, "b": b}, f"tov|{a}|{b}", scene)


def best_corner(scores: dict | None = None) -> str:
    scores = scores or S.corner_scores
    for corner in CORNER_ORDER:
        if scores.get(corner, 0) == 0:
            return corner
    return min(CORNER_ORDER, key=lambda c: (scores.get(c, 0), CORNER_ORDER.index(c)))


# --------------------------------------------------------------------------------------------------------------------
# Catalog + end card
# --------------------------------------------------------------------------------------------------------------------
def load_catalog(path: str | None = None) -> dict:
    path = Path(path or CFG.catalog)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {entry["id"]: entry for entry in data if isinstance(entry, dict) and "id" in entry}


def scene_id_of(scene) -> str:
    cls = type(scene)
    module = sys.modules.get(cls.__module__)
    file = getattr(module, "__file__", None)
    return f"{Path(file).name if file else cls.__module__}:{cls.__name__}"


def _split_camel(name: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", name)


def _entry_field(entry: dict, base: str, default: str = "") -> str:
    if CFG.lang == "fr" and entry.get(base + "_fr"):
        return entry[base + "_fr"]
    return entry.get(base + "_en") or default


def build_end_card(scene):
    from manim import (
        DOWN,
        GRAY_B,
        GRAY_C,
        GREEN_C,
        LEFT,
        UP,
        WHITE,
        FadeIn,
        FadeOut,
        ImageMobject,
        Text,
        Group,
        VGroup,
    )

    entry = S.catalog.get(S.scene_id) or {}
    title = _entry_field(entry, "title", _split_camel(S.scene_name))
    links = [l for l in (entry.get("links") or []) if isinstance(l, dict) and l.get("path")][:5]
    header = "Pour aller plus loin" if CFG.lang == "fr" else "Learn more"

    S.guard = True  # end card strings are written for the right language directly: no translation, no collect
    try:
        mobs = list(scene.mobjects)
        if mobs:
            S._raw_play(scene, _fade_out_all(mobs), run_time=0.5)
        scene.clear()
        S.overlay = False
        group = Group()
        if LOGO_PATH.exists():
            from PIL import Image

            im = np.asarray(Image.open(LOGO_PATH).convert("RGBA"))
            logo = ImageMobject(im)
            logo.set_height(1.5 if links else 2.0)
            group.add(logo)
        else:
            logo = None
        title_mob = Text(title, font=TEXT_FONT, font_size=30, color=GRAY_B)
        if title_mob.width > 12.5:
            title_mob.scale_to_fit_width(12.5)
        footer = Text("github.com/pyomeca/bioptim", font=MONO_FONT, font_size=26, color=GREEN_C)
        if links:
            head = Text(header, font=TEXT_FONT, font_size=42, weight="BOLD")
            labels, paths = [], []
            for link in links:
                label = (link.get("label_fr") if CFG.lang == "fr" else None) or link.get("label_en") or ""
                path = link["path"] + (f":{link['lines']}" if link.get("lines") else "")
                labels.append(Text(label, font=TEXT_FONT, font_size=22, color=GRAY_C))
                paths.append(Text(path, font=MONO_FONT, font_size=20, color=WHITE))
            label_w = max(l.width for l in labels)
            rows = VGroup()
            for l, p in zip(labels, paths):
                l.align_to([0, 0, 0], LEFT)
                p.next_to(l, buff=0).align_to(l, DOWN)
                p.shift(RIGHT_ * (label_w - l.width + 0.35))
                rows.add(VGroup(l, p))
            rows.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
            if rows.width > 12.8:
                rows.scale_to_fit_width(12.8)
            if logo is not None:
                logo.move_to([0, 2.85, 0])
            head.move_to([0, 1.55, 0])
            title_mob.move_to([0, 0.95, 0])
            rows.move_to([0, -0.85, 0])
            footer.move_to([0, -3.35, 0])
            group.add(head, title_mob, rows, footer)
        else:
            if logo is not None:
                logo.move_to([0, 1.2, 0])
            title_mob.move_to([0, -0.5, 0])
            footer.move_to([0, -1.3, 0])
            group.add(title_mob, footer)
        S._raw_play(scene, FadeIn(group), run_time=0.8)
        S._raw_play(scene, __import__("manim").Wait(run_time=3.2))
        S.warnings.append({"kind": "endcard", "links": len(links), "title": title, "count": 1})
    finally:
        S.guard = False


RIGHT_ = np.array([1.0, 0.0, 0.0])


# --------------------------------------------------------------------------------------------------------------------
# Scene patches
# --------------------------------------------------------------------------------------------------------------------
def _anim_has_text(anim) -> bool:
    from manim import AnimationGroup, Wait

    if isinstance(anim, Wait):
        return False
    if isinstance(anim, AnimationGroup):
        return any(_anim_has_text(s) for s in anim.animations)
    if anim.is_remover():
        return False
    mob = getattr(anim, "mobject", None)
    return mob is not None and any_text(mob)


def _install_scene_patches():
    import manim
    from manim import Scene, Wait
    from manim.renderer.cairo_renderer import CairoRenderer

    if getattr(Scene, "_series_patched", False):
        return
    orig_init = Scene.__init__
    orig_compile = Scene.compile_animations
    orig_play = Scene.play
    orig_tear_down = Scene.tear_down

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        S.begin_scene(self)

    def compile_animations(self, *args, **kwargs):
        anims = orig_compile(self, *args, **kwargs)
        if getattr(self, "_series_raw", False):
            return anims
        is_wait = len(anims) == 1 and isinstance(anims[0], Wait)
        text_play = (not is_wait) and any(_anim_has_text(a) for a in anims)
        pre = max((a.run_time or 0) for a in anims)
        for a in anims:
            if a.run_time is not None:
                a.run_time = a.run_time * CFG.slow
        if is_wait:
            pending = getattr(self, "_series_pending", 0.0)
            if pending > anims[0].run_time and anims[0].stop_condition is None:
                S.t_wait_added += pending - anims[0].run_time
                anims[0].run_time = pending
        self._series_text_play = text_play and pre >= 0.25
        return anims

    def play(self, *args, **kwargs):
        if getattr(self, "_series_raw", False):
            return orig_play(self, *args, **kwargs)
        is_wait = len(args) == 1 and isinstance(args[0], Wait)
        pending = getattr(self, "_series_pending", 0.0)
        if pending > 0 and not is_wait:
            self._series_pending = 0.0
            S.t_wait_added += pending
            S.text_plays_padded += 1
            S._raw_play(self, Wait(run_time=pending))
        self._series_text_play = False
        orig_play(self, *args, **kwargs)
        if is_wait:
            self._series_pending = 0.0
        else:
            self._series_pending = CFG.min_wait if self._series_text_play else 0.0
            audit(self)

    def tear_down(self):
        try:
            if CFG.endcard and not S.no_endcard and not S.probe:
                build_end_card(self)
        finally:
            orig_tear_down(self)
            S.finish_scene(self)

    Scene.__init__ = __init__
    Scene.compile_animations = compile_animations
    Scene.play = play
    Scene.tear_down = tear_down
    Scene._series_patched = True

    # frames: composite the logo on the way to the movie file
    orig_add_frame = CairoRenderer.add_frame
    orig_skip_status = CairoRenderer.update_skipping_status

    def add_frame(self, frame, num_frames=1):
        if not self.skip_animations:
            frame = composite_logo(frame)
        return orig_add_frame(self, frame, num_frames)

    def update_skipping_status(self):
        orig_skip_status(self)
        if S.dry or S.probe:
            self.skip_animations = True

    CairoRenderer.add_frame = add_frame
    CairoRenderer.update_skipping_status = update_skipping_status

    def _raw_play(scene, *anims, **kwargs):
        scene._series_raw = True
        try:
            orig_play(scene, *anims, **kwargs)
        finally:
            scene._series_raw = False

    S._raw_play = _raw_play


def _begin_scene(self, scene):
    self.reset_scene()
    self.scene_name = type(scene).__name__
    self.scene_id = scene_id_of(scene)
    self.catalog = load_catalog()
    self.overlay = True
    scene._series_pending = 0.0
    scene._series_raw = False
    entry = self.catalog.get(self.scene_id) or {}
    forced = norm_corner(CFG.logo_corner) or norm_corner(entry.get("logo_corner"))
    self.corner = forced or self.corner
    if CFG.lang == "fr" and not self.fr_loaded:
        self.fr = load_translations("fr")
        self.fr_loaded = True


def _finish_scene(self, scene):
    if self.finished:
        return
    self.finished = True
    if self.probe:
        self.last_probe_scores = dict(self.corner_scores)
        return
    rows = [
        {
            "scene": self.scene_id,
            "key": k,
            "count": v["count"],
            "file": v["file"],
            "line": v["line"],
            "markup": v["markup"],
        }
        for k, v in self.keys.items()
    ]
    _write_lines(CFG.collect, rows)
    if CFG.lang == "fr":
        _write_lines(
            CFG.missing,
            [
                {"scene": self.scene_id, "key": k, "count": v["count"], "file": v["file"], "line": v["line"]}
                for k, v in self.missing.items()
            ],
        )
    if CFG.log_dir:
        out = Path(CFG.log_dir)
        out.mkdir(parents=True, exist_ok=True)
        report = {
            "scene": self.scene_id,
            "lang": CFG.lang,
            "slow": CFG.slow,
            "logo_corner": self.corner,
            "corner_scores_end_of_play": self.corner_scores,
            "corner_overlaps": self.corner_details,
            "plays": self.play_index,
            "extra_wait_s": round(self.t_wait_added, 2),
            "duration_s": round(float(scene.renderer.time), 2),
            "translatable_keys": len(self.keys),
            "missing_translations": len(self.missing),
            "text_animation_replacements": self.counts,
            "fr_autofit": self.fits,
            "warnings": self.warnings,
        }
        (out / f"{type(scene).__name__}_{CFG.lang}_audit.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
        )


State.begin_scene = _begin_scene
State.finish_scene = _finish_scene


def finish_scene_if_needed(scene):
    """Call from the runner when construct() raised, so the logs collected so far are not lost."""
    try:
        S.finish_scene(scene)
    except Exception:
        pass


def install():
    """Patch manim. Must be called BEFORE the scene modules are imported. Idempotent."""
    if S.installed:
        return
    CFG.load()
    if CFG.lang == "fr":
        S.fr = load_translations("fr")
        S.fr_loaded = True
    if CFG.logo and not LOGO_PATH.exists():
        try:
            sys.path.insert(0, str(HERE / "assets"))
            from fetch_logo import fetch_logo  # noqa: E402

            fetch_logo(verbose=False)
        except Exception as exc:
            print(f"[series] logo unavailable ({exc}); rendering without it", file=sys.stderr)
    _install_text_patches()
    _install_animation_patches()
    _install_scene_patches()
    S.installed = True

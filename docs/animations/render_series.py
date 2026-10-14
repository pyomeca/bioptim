"""
Render bioptim animation scenes through the series layer (see SERIES_LAYER.md).

    python docs/animations/render_series.py anim_controls.py ControlTypes --lang fr
    python docs/animations/render_series.py --all --lang both --jobs 4 --out my_videos
    python docs/animations/render_series.py --all --dry --collect keys.jsonl     # construct only, list translatable keys
    python docs/animations/render_series.py --gen-catalog                        # (re)generate the catalog skeleton

The layer is patched in BEFORE the scene module is imported, so no scene file has to be edited.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CATALOG = HERE / "catalog.json"
SCENE_FILES = ["features_scenes.py", "dms_vs_dc.py"]


# --------------------------------------------------------------------------------------------------------------------
# Catalog skeleton (AST only, manim is not imported)
# --------------------------------------------------------------------------------------------------------------------
def scan_scenes(directory: Path = HERE) -> list:
    """[(file_name, ClassName)] for every ``class X(Scene)`` (any base whose name ends with 'Scene')."""
    files = [directory / f for f in SCENE_FILES] + sorted(directory.glob("anim_*.py"))
    found = []
    for path in files:
        if not path.exists():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                bases = [b.id if isinstance(b, ast.Name) else getattr(b, "attr", "") for b in node.bases]
                if any(b.endswith("Scene") for b in bases):
                    found.append((path.name, node.name))
    return found


def camel_to_words(name: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", name)


def slugify(name: str) -> str:
    return camel_to_words(name).lower().replace(" ", "-")


def skeleton_entry(file: str, cls: str) -> dict:
    stem = file[:-3]
    notes = ""
    if stem.startswith("anim_"):
        candidate = HERE / "notes" / f"{stem[5:]}.md"
        if candidate.exists():
            notes = f"notes/{candidate.name}"
    return {
        "id": f"{file}:{cls}",
        "slug": slugify(cls),
        "title_en": camel_to_words(cls),
        "title_fr": "",
        "description_en": "",
        "description_fr": "",
        "video_basename": cls,
        "duration_note": "",
        "links": [],
        "notes_file": notes,
    }


def generate_catalog(path: Path = CATALOG) -> list:
    """Merge with the existing catalog: existing entries (edited by hand) are never overwritten."""
    existing = {}
    if path.exists():
        existing = {e["id"]: e for e in json.loads(path.read_text(encoding="utf-8"))}
    entries = []
    for file, cls in scan_scenes():
        entries.append(existing.pop(f"{file}:{cls}", None) or skeleton_entry(file, cls))
    entries.extend(existing.values())  # entries of scenes that disappeared: kept, flagged by the caller
    path.write_text(json.dumps(entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return entries


def load_catalog_entries(path: Path) -> list:
    entries = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    known = {e["id"] for e in entries}
    for file, cls in scan_scenes():  # scenes missing from the catalog are still rendered
        if f"{file}:{cls}" not in known:
            entries.append(skeleton_entry(file, cls))
    return entries


# --------------------------------------------------------------------------------------------------------------------
# One render, in this process
# --------------------------------------------------------------------------------------------------------------------
def parse_quality(text: str):
    m = re.fullmatch(r"(\d+)p(\d+)", text)
    if not m:
        raise SystemExit(f"bad --quality '{text}' (expected e.g. 1080p30, 1080p60, 480p15)")
    h, fps = int(m.group(1)), int(m.group(2))
    return int(round(h * 16 / 9 / 2)) * 2 if h != 480 else 854, h, fps


def set_env(args, lang: str, scene_name: str, log_dir: Path):
    os.environ["SERIES_LANG"] = lang
    os.environ["SERIES_LOG_DIR"] = str(log_dir)
    os.environ["SERIES_MISSING"] = str(log_dir / f"{scene_name}_{lang}_missing.jsonl")
    os.environ["SERIES_ENDCARD"] = "0" if args.no_endcard else "1"
    os.environ["SERIES_LOGO"] = "0" if args.no_logo else "1"
    os.environ["SERIES_LOGO_CORNER"] = args.logo_corner
    os.environ["SERIES_SLOW"] = str(args.slow)
    os.environ["SERIES_MIN_WAIT"] = str(args.min_wait)
    os.environ["SERIES_FR_COMMA"] = "0" if args.fr_dot else "1"
    if args.catalog:
        os.environ["SERIES_CATALOG"] = str(Path(args.catalog).resolve())
    if args.collect:
        os.environ["SERIES_COLLECT"] = str(Path(args.collect).resolve())
    else:
        os.environ.pop("SERIES_COLLECT", None)


def load_scene_class(scene_file: str, scene_name: str):
    path = Path(scene_file)
    if not path.exists():
        path = HERE / scene_file
    path = path.resolve()
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return getattr(module, scene_name), path


def run_job(args, scene_file: str, scene_name: str, lang: str) -> int:
    out = Path(args.out).resolve()
    media = Path(args.media_dir).resolve()
    log_dir = Path(args.log_dir).resolve() if args.log_dir else out / "logs"
    out.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    missing_path = log_dir / f"{scene_name}_{lang}_missing.jsonl"
    missing_path.unlink(missing_ok=True)
    set_env(args, lang, scene_name, log_dir)

    sys.path.insert(0, str(HERE))
    import series_style as ss  # noqa: E402  (imports manim)

    ss.install()  # BEFORE the scene module is imported
    from manim import config  # noqa: E402

    cls, path = load_scene_class(scene_file, scene_name)
    width, height, fps = parse_quality(args.quality)
    config.media_dir = str(media)
    config.input_file = str(path)
    config.pixel_width, config.pixel_height, config.frame_rate = width, height, fps
    config.disable_caching = True  # always_redraw closures crash the hashing in some scenes
    config.progress_bar = "none"
    config.verbosity = "WARNING"
    config.output_file = f"{scene_name}_{lang}"
    config.write_to_movie = True
    config.dry_run = False

    scene_id = f"{path.name}:{scene_name}"
    entry = ss.load_catalog(os.environ.get("SERIES_CATALOG")).get(scene_id) or {}
    forced = ss.norm_corner(args.logo_corner) or ss.norm_corner(entry.get("logo_corner"))
    t0 = time.time()

    def build_and_render():
        scene = cls()
        try:
            scene.render()
        except BaseException:
            ss.finish_scene_if_needed(scene)
            raise
        return scene

    if not args.dry and not args.no_logo and not forced:
        ss.S.probe = True  # cheap construct-only pass: which corner of the frame stays free for the whole scene?
        config.dry_run, config.write_to_movie = True, False
        build_and_render()
        ss.S.probe = False
        ss.S.corner = ss.best_corner(ss.S.last_probe_scores)
        config.dry_run, config.write_to_movie = False, True
    elif forced:
        ss.S.corner = forced

    if args.dry:
        ss.S.dry = True
        config.dry_run, config.write_to_movie = True, False
        scene = build_and_render()
        print(f"[dry] {scene_id} ({lang}) constructed in {time.time() - t0:.1f}s")
        return check_strict(args, missing_path, lang)

    scene = build_and_render()
    movie = Path(scene.renderer.file_writer.movie_file_path)
    target = out / f"{scene_name}_{lang}.mp4"
    if movie.resolve() != target.resolve():
        shutil.copy2(movie, target)
    print(f"[ok] {scene_id} ({lang}) -> {target} in {time.time() - t0:.0f}s (logo corner {ss.S.corner})")
    return check_strict(args, missing_path, lang)


def check_strict(args, missing_path: Path, lang: str) -> int:
    if lang != "fr" or not missing_path.exists():
        return 0
    rows = [json.loads(l) for l in missing_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if rows:
        print(f"[fr] {len(rows)} string(s) without translation (see {missing_path})", file=sys.stderr)
        if args.strict:
            for r in rows[:20]:
                print(f"   {r['file']}:{r['line']}  {r['key']!r}", file=sys.stderr)
            return 3
    return 0


# --------------------------------------------------------------------------------------------------------------------
# Batch (subprocess per job, so that every render starts from a clean manim)
# --------------------------------------------------------------------------------------------------------------------
def child_command(args, scene_file, scene_name, lang, collect_path=None) -> list:
    cmd = [sys.executable, str(Path(__file__).resolve()), scene_file, scene_name, "--lang", lang]
    cmd += [
        "--quality",
        args.quality,
        "--out",
        args.out,
        "--media-dir",
        str(Path(args.media_dir) / f"{scene_name}_{lang}"),
        "--slow",
        str(args.slow),
    ]
    cmd += ["--min-wait", str(args.min_wait), "--logo-corner", args.logo_corner]
    if args.log_dir:
        cmd += ["--log-dir", args.log_dir]
    if args.catalog:
        cmd += ["--catalog", args.catalog]
    for flag, on in (
        ("--no-endcard", args.no_endcard),
        ("--no-logo", args.no_logo),
        ("--strict", args.strict),
        ("--dry", args.dry),
        ("--fr-dot", args.fr_dot),
    ):
        if on:
            cmd.append(flag)
    if collect_path:
        cmd += ["--collect", str(collect_path)]
    return cmd


def run_batch(args, jobs_list: list) -> int:
    tmp = Path(tempfile.mkdtemp(prefix="series_collect_"))
    results = []

    def one(job):
        file, cls, lang = job
        partial = tmp / f"{file[:-3]}__{cls}__{lang}.jsonl" if args.collect else None
        t0 = time.time()
        proc = subprocess.run(child_command(args, file, cls, lang, partial), capture_output=True, text=True)
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-4:]
        return {
            "scene": f"{file}:{cls}",
            "lang": lang,
            "rc": proc.returncode,
            "seconds": round(time.time() - t0, 1),
            "error": " | ".join(tail) if proc.returncode else "",
            "partial": partial,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for res in pool.map(one, jobs_list):
            results.append(res)
            print(
                f"[{'ok' if res['rc'] == 0 else 'FAIL'}] {res['scene']} {res['lang']} ({res['seconds']}s) {res['error']}"
            )

    unique = set()
    if args.collect:
        target = Path(args.collect).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as fh:
            for res in results:
                p = res["partial"]
                if p and p.exists():
                    text = p.read_text(encoding="utf-8")
                    fh.write(text)
                    for line in text.splitlines():
                        unique.add(json.loads(line)["key"])
        print(f"collect: {len(unique)} unique keys -> {target}")
    failed = [r for r in results if r["rc"] != 0]
    log_dir = Path(args.log_dir).resolve() if args.log_dir else Path(args.out).resolve() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "jobs": len(results),
        "failed": [{k: r[k] for k in ("scene", "lang", "rc", "error")} for r in failed],
        "unique_keys": len(unique),
    }
    (log_dir / "batch_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"{len(results) - len(failed)}/{len(results)} jobs succeeded")
    for r in failed:
        print(f"  FAILED {r['scene']} ({r['lang']}): {r['error']}", file=sys.stderr)
    return 1 if failed else 0


def main(argv=None) -> int:
    default_media = HERE / "media" / "series"
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scene_file", nargs="?", help="scene file, e.g. anim_controls.py")
    p.add_argument("scene_class", nargs="?", help="scene class, e.g. ControlTypes")
    p.add_argument("--lang", default="en", choices=["en", "fr", "both"])
    p.add_argument("--quality", default="1080p30", help="e.g. 1080p30 (default), 1080p60, 480p15")
    p.add_argument("--out", default=str(default_media / "out"), help="where <Scene>_<lang>.mp4 are copied")
    p.add_argument("--media-dir", default=str(default_media), help="manim media directory (temporary files)")
    p.add_argument("--log-dir", default="", help="audit / missing-translation logs (default: <out>/logs)")
    p.add_argument("--catalog", default="", help="catalog json (default docs/animations/catalog.json)")
    p.add_argument("--no-endcard", action="store_true")
    p.add_argument("--no-logo", action="store_true")
    p.add_argument("--logo-corner", default="auto", help="auto (default), br, bl, tr, tl")
    p.add_argument("--slow", type=float, default=float(os.environ.get("SERIES_SLOW", 1.6)))
    p.add_argument("--min-wait", type=float, default=float(os.environ.get("SERIES_MIN_WAIT", 0.5)))
    p.add_argument("--fr-dot", action="store_true", help="keep the decimal point in FR (default: convert to a comma)")
    p.add_argument("--strict", action="store_true", help="fr: exit non-zero if a string has no translation")
    p.add_argument("--collect", default="", help="jsonl file receiving every translatable key")
    p.add_argument("--dry", action="store_true", help="construct only (no video): fast layer check / collect")
    p.add_argument("--all", action="store_true", help="every scene of the catalog")
    p.add_argument("--jobs", type=int, default=1, help="parallel processes for --all / --lang both")
    p.add_argument("--gen-catalog", action="store_true", help="scan the scene files and (re)generate catalog.json")
    args = p.parse_args(argv)

    if args.gen_catalog:
        entries = generate_catalog()
        print(f"{len(entries)} scenes in {CATALOG}")
        return 0

    langs = ["en", "fr"] if args.lang == "both" else [args.lang]
    if args.all:
        catalog = Path(args.catalog) if args.catalog else CATALOG
        jobs = []
        for entry in load_catalog_entries(catalog):
            file, cls = entry["id"].split(":")
            jobs += [(file, cls, lang) for lang in langs]
        if args.collect:
            Path(args.collect).unlink(missing_ok=True)
        return run_batch(args, jobs)
    if not args.scene_file or not args.scene_class:
        p.error("give <scene_file> <SceneClass>, or --all / --gen-catalog")
    if len(langs) > 1:
        return run_batch(args, [(args.scene_file, args.scene_class, lang) for lang in langs])
    if args.collect:
        Path(args.collect).unlink(missing_ok=True)
    return run_job(args, args.scene_file, args.scene_class, langs[0])


if __name__ == "__main__":
    sys.exit(main())

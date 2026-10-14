"""Regenerate the video tables of README.md / README.fr.md from catalog.json, and check their relative links.

    python docs/animations/build_readme_tables.py           # rewrite the marked regions (idempotent)
    python docs/animations/build_readme_tables.py --check   # only verify that every relative link resolves

Marked regions: ``<!-- BEGIN TABLE -->`` ... ``<!-- END TABLE -->`` (all videos by section) and
``<!-- BEGIN START -->`` ... ``<!-- END START -->`` (the "start here" videos). Text outside is left untouched.
"""

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SECTION_ORDER = [
    "Fundamentals",
    "Discretization",
    "Objectives and constraints",
    "Phases and time",
    "Solvers and numerics",
    "Models and biomechanics",
    "Advanced control",
    "Library overview",
]
START_HERE = [
    "OCPStatement",
    "FirstOCP",
    "BoundsInitialGuess",
    "MultipleShooting",
    "ArchitecturePath",
    "ObjectivesNodes",
    "SolutionTour",
    "TrackState",
]
LEVELS = {
    "en": {1: "1 - introductory", 2: "2 - intermediate", 3: "3 - advanced"},
    "fr": {1: "1 - introduction", 2: "2 - intermédiaire", 3: "3 - avancé"},
}
TEXT = {
    "en": {
        "cols": ["Scene (mp4)", "Title and content", "Level", "Example and code", "Sources"],
        "notes": "notes",
        "data": "data generator",
        "scene": "scene",
        "start_cols": ["Scene (mp4)", "Title", "Why start here"],
    },
    "fr": {
        "cols": ["Scène (mp4)", "Titre et contenu", "Niveau", "Exemple et code", "Sources"],
        "notes": "notes",
        "data": "générateur de données",
        "scene": "scène",
        "start_cols": ["Scène (mp4)", "Titre", "Pourquoi commencer ici"],
    },
}
START_WHY = {
    "en": {
        "OCPStatement": "The vocabulary: state, control, dynamics, cost, constraints, bounds.",
        "FirstOCP": "Your first OCP built line by line, with the real solution.",
        "BoundsInitialGuess": "Bounds and initial guesses, and how they change the solve.",
        "MultipleShooting": "How Bioptim turns an OCP into an NLP (nodes, defects).",
        "ArchitecturePath": "The map of the library: from your inputs to the Solution.",
        "ObjectivesNodes": "Lagrange versus Mayer terms and the Node enum.",
        "SolutionTour": "How to read and post-process a solved problem.",
        "TrackState": "A first realistic objective: tracking a reference.",
    },
    "fr": {
        "OCPStatement": "Le vocabulaire : état, commande, dynamique, coût, contraintes, bornes.",
        "FirstOCP": "Votre premier OCP construit ligne par ligne, avec la vraie solution.",
        "BoundsInitialGuess": "Bornes et estimations initiales, et leur effet sur la résolution.",
        "MultipleShooting": "Comment Bioptim transforme un OCP en NLP (nœuds, défauts).",
        "ArchitecturePath": "La carte de la bibliothèque : de vos entrées à la Solution.",
        "ObjectivesNodes": "Termes de Lagrange et de Mayer, énumération Node.",
        "SolutionTour": "Lire et post-traiter un problème résolu.",
        "TrackState": "Un premier objectif réaliste : suivre une référence.",
    },
}


def link(label, target, lines=None):
    """Markdown link to a repo path (relative to docs/animations/), with GitHub line anchors."""
    href = target if target.startswith(("http", "#")) else "../../" + target
    if lines:
        a, _, b = str(lines).partition("-")
        href += f"#L{a}" + (f"-L{b}" if b else "")
    return f"[{label}]({href})"


def local_link(label, name, anchor=""):
    return f"[{label}]({name}{anchor})"


def generator_of(entry):
    scene = entry["id"].split(":")[0]
    stem = scene[:-3]
    special = {"features_scenes": "generate_features_data.py", "dms_vs_dc": "generate_pendulum_data.py"}
    name = special.get(stem, "generate_" + stem.replace("anim_", "") + "_data.py")
    return name if (HERE / name).exists() else None


def notes_of(entry):
    scene = entry["id"].split(":")[0]
    if entry.get("notes_file"):
        return entry["notes_file"]
    return {"features_scenes.py": "FEATURES.md", "dms_vs_dc.py": "README.md"}.get(scene)


def first_sentence(text, limit=220):
    text = re.sub(r"\s+", " ", text).strip()
    m = re.search(r"(?<=[a-z\)\.0-9])\. (?=[A-Z])", text)
    sentence = text[: m.start() + 1] if m else text
    if len(sentence) > limit:
        sentence = sentence[: limit - 1].rsplit(" ", 1)[0] + "…"
    return sentence


def esc(text):
    return text.replace("|", "\|")


def build_table(catalog, lang):
    t = TEXT[lang]
    sec_key = "section" if lang == "en" else "section_fr"
    names = {e["section"]: e[sec_key] for e in catalog}
    out = []
    for section in SECTION_ORDER:
        entries = sorted((e for e in catalog if e["section"] == section), key=lambda e: e["level"])
        if not entries:
            continue
        out += [f"#### {names[section]}", "", "| " + " | ".join(t["cols"]) + " |", "|" + " --- |" * len(t["cols"])]
        for e in entries:
            scene_file, cls = e["id"].split(":")
            gen, notes = generator_of(e), notes_of(e)
            links = [
                link(l[f"label_{lang}"], l["path"], l.get("lines"))
                for l in e["links"]
                if l["path"] != f"docs/animations/{gen}"
            ]
            src = [local_link(t["scene"], scene_file)]
            if gen:
                src.append(local_link(t["data"], gen))
            if notes:
                src.append(local_link(t["notes"], notes))
            cells = [
                f"`{e['video_basename']}`",
                f"**{esc(e['title_' + lang])}**<br>{esc(first_sentence(e['description_' + lang]))}",
                LEVELS[lang][e["level"]],
                "<br>".join(links),
                " · ".join(src),
            ]
            out.append("| " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out).rstrip()


def build_start(catalog, lang):
    t = TEXT[lang]
    by = {e["video_basename"]: e for e in catalog}
    out = ["| " + " | ".join(t["start_cols"]) + " |", "|" + " --- |" * len(t["start_cols"])]
    for name in START_HERE:
        e = by[name]
        out.append(f"| `{name}` | **{esc(e['title_' + lang])}** | {START_WHY[lang][name]} |")
    return "\n".join(out)


def fill(text, tag, body):
    pat = re.compile(rf"(<!-- BEGIN {tag} -->)(.*?)(<!-- END {tag} -->)", re.S)
    if not pat.search(text):
        raise SystemExit(f"marker {tag} not found")
    return pat.sub(lambda m: f"{m.group(1)}\n\n{body}\n\n{m.group(3)}", text)


def check_links(path):
    text = path.read_text(encoding="utf-8")
    total, broken = 0, []
    for m in re.finditer(r"\]\(([^)\s]+)\)", text):
        href = m.group(1)
        if href.startswith(("http://", "https://", "#", "mailto:")):
            continue
        total += 1
        if not (path.parent / href.split("#")[0]).resolve().exists():
            broken.append(href)
    return total, broken


def main():
    catalog = json.loads((HERE / "catalog.json").read_text(encoding="utf-8"))
    files = {"en": HERE / "README.md", "fr": HERE / "README.fr.md"}
    if "--check" not in sys.argv:
        for lang, path in files.items():
            text = path.read_text(encoding="utf-8")
            text = fill(text, "TABLE", build_table(catalog, lang))
            text = fill(text, "START", build_start(catalog, lang))
            path.write_text(text, encoding="utf-8", newline="\n")
    status = 0
    for lang, path in files.items():
        total, broken = check_links(path)
        print(f"{path.name}: {total} relative links checked, {len(broken)} broken")
        for b in broken:
            print("  BROKEN", b)
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())

"""
TEMPLATE scene for the Bioptim animation series (read ../STANDARD.md first). Copy it to
``docs/animations/anim_<topic>.py``, rename the class, replace the data file, the texts and the code lines.

What it shows (about 14 s of content at native speed; x1.6 with the series layer, plus the 4.5 s end card):
title -> one plot with a GHOST reference curve (dashed grey) -> the new curve and the DIFFERENCE plot below (a change is
made visible, not narrated) -> code panel with the caption "Bioptim code" ABOVE the lines -> numeric readout computed
from the data -> one footer sentence -> final hold ``self.wait(2.5)``.

DATA: ``data/template_demo.npz`` is produced by ``templates/generate_template_data.py``, a PLACEHOLDER without bioptim.
In a real video every curve and number comes from a real bioptim solve stored in ``data/<topic>_*.npz`` by
``generate_<topic>_data.py``.

Dry check (no video), from the repository root:
    python docs/animations/render_series.py templates/scene_template.py TemplateScene --dry
Real videos: see STANDARD.md, section 6. The template lives in templates/ on purpose: ``--gen-catalog`` and ``--all``
only scan docs/animations/anim_*.py (plus features_scenes.py and dms_vs_dc.py), so it never enters catalog.json.
"""

import sys
from pathlib import Path

import numpy as np
from manim import *

# When the file sits in docs/animations/ this line is not needed (render_series.py puts that folder on sys.path).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features_scenes import (  # noqa: E402  (importing it also sets the default fonts Segoe UI / Consolas)
    C_CTRL,
    C_STATE,
    CODE_X,
    DATA_DIR,
    axis_label,
    code_panel,
    footer,
    hline,
    ipopt_line,
    make_axes,
    place,
    poly,
    say,
    scene_title,
    time_label,
    x_ticks,
    y_ticks,
)

C_GHOST = GRAY_B  # the reference ("before") curve: dashed, grey
C_NEW = C_STATE  # the curve that changes
C_DIFF = PURPLE_B  # the difference curve


class TemplateScene(Scene):
    def construct(self):
        # ---------------------------------------------------------------- data: everything shown comes from the npz
        d = np.load(DATA_DIR / "template_demo.npz")
        t, y_ref, y_new = d["t"], d["y_ref"], d["y_new"]
        T, n, weight = float(d["final_time"]), int(d["n_shooting"]), float(d["weight"])
        diff = y_new - y_ref
        # numbers quoted on screen are computed here, never typed by hand
        peak_ref, peak_new = float(np.abs(y_ref).max()), float(np.abs(y_new).max())
        peak_diff = float(np.abs(diff).max())

        title = scene_title("Template: one idea per video", f"N = {n} intervals, T = {T:g} s, weight = {weight:g}")
        self.play(FadeIn(title), run_time=0.4)

        # ---------------------------------------------------------------- left: two stacked plots (x in [-6.6, -0.5])
        y_lim, d_lim = 1.25, 0.6
        ax_y = make_axes([-3.55, 0.85, 0], 5.6, 2.5, [0, T], [-0.1, y_lim], 0.5, 0.5)
        ax_d = make_axes([-3.55, -2.15, 0], 5.6, 1.6, [0, T], [-d_lim, d_lim], 0.5, 0.5)
        decos = VGroup(
            axis_label("output y (units)", ax_y, C_NEW),  # units in parentheses
            axis_label("Δy = y − y_ref (units)", ax_d, C_DIFF),
            time_label(ax_d),  # "t (s)"
            x_ticks(ax_d, [0, 0.5, 1.0], "{:.1f}"),  # tick labels go through dec(): decimal comma in French
            y_ticks(ax_y, [0, 1]),
            y_ticks(ax_d, [-0.5, 0, 0.5]),
        )
        self.play(Create(ax_y), Create(ax_d), FadeIn(decos), run_time=0.8)

        # ghost = the reference (here: the run without the change), dashed and grey, drawn first and kept on screen
        ghost = DashedVMobject(poly(ax_y, t, y_ref, C_GHOST, 3), num_dashes=40).set_opacity(0.8)
        self.play(Create(ghost), run_time=1.0)

        # ---------------------------------------------------------------- right: code panel, caption ABOVE the code
        panel = code_panel(
            [
                (0, "objectives = ObjectiveList()", WHITE),
                (0, "objectives.add(", C_CTRL),
                (1, "ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,", C_CTRL),
                (1, 'key="tau", weight=w)', C_CTRL),  # the changing argument gets its own colour
                (0, "# placeholder: show the code of generate_<topic>_data.py", GRAY_B),
            ],
            size=19,
        )  # default caption "Bioptim code" (FR: "Code Bioptim", key already in i18n/fr.json); the code lines are never translated
        self.play(FadeIn(panel), run_time=0.6)

        # ---------------------------------------------------------------- the change, and its difference made visible
        curve = poly(ax_y, t, y_new, C_NEW, 5)
        curve_d = poly(ax_d, t, diff, C_DIFF, 5)
        zero = hline(ax_d, 0, T, 0, GRAY_D, dashed=False)

        # ONE templated string: the numbers become {0}, {1}... in the French key; the IPOPT status comes from the npz
        body = (
            f"weight = {weight:g}  ·  peak |y| = {peak_new:.2f} (was {peak_ref:.2f})\n"
            f"max |Δy| = {peak_diff:.2f}\n" + ipopt_line(int(d["iterations"]), bool(d["converged"]))
        )
        info = place(Text(body, font_size=19, color=GRAY_A, line_spacing=0.9), CODE_X, -0.6)
        self.play(Create(curve), Create(zero), Create(curve_d), FadeIn(info), run_time=1.6)
        self.wait(0.8)

        # one remark of the right panel: a whole sentence, yellow, only about something that IS plotted
        remark = place(say("A larger weight lowers the peak: the dashed curve is the run without it."), CODE_X, -1.9)
        self.play(FadeIn(remark))

        # ---------------------------------------------------------------- footer: one sentence, at most two lines
        foot = footer("Placeholder data (no bioptim solve): replace generate_template_data.py by a real solve.")
        self.play(FadeIn(foot))
        self.wait(2.5)  # final hold before the end card, in every scene

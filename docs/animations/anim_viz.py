"""
Manim CE scene: visualising a solution with ``sol.animate(...)`` (pyorerun or bioviz). REAL data stored in
``data/viz_pendulum.npz`` by ``generate_viz_data.py`` (double pendulum, N = 30, T = 1 s, IPOPT).

Beat 1: pyorerun really run headlessly (``show_now=False`` then ``rr.save``); the stick figure is RE-DRAWN from the marker
positions read back from the .rrd file written by pyorerun (not a screen capture, no GUI was captured).
Beat 2: how many frames each viewer receives (pyorerun ignores ``n_frames``; bioviz would get the frames returned by
the library's ``interpolate_data``); bioviz is not installed here, so it was NOT run (the RuntimeError is shown).

Scene: Visualization (about 17 s of content at native speed).
Render:  python docs/animations/render_series.py anim_viz.py Visualization --lang both --quality 1080p30
"""

import sys
from pathlib import Path

import numpy as np
from manim import *

# When the file sits in docs/animations/ this line is not needed (render_series.py puts that folder on sys.path).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from features_scenes import (  # noqa: E402  (importing it also sets the default fonts Segoe UI / Consolas)
    C_CTRL,
    C_PAR,
    C_STATE,
    CODE_X,
    DATA_DIR,
    MONO,
    code_panel,
    dec,
    footer,
    ipopt_line,
    place,
    say,
    scene_title,
)

C_GHOST = GRAY_B


class Visualization(Scene):
    def construct(self):
        # ---------------------------------------------------------------- data: everything shown comes from the npz
        d = np.load(DATA_DIR / "viz_pendulum.npz")
        t, mk = d["t"], d["markers_rrd"]  # mk: (3, 4 markers, 31 frames) as logged by pyorerun, read back from the .rrd
        T, n = float(d["final_time"]), int(d["n_shooting"])
        n_frames = mk.shape[2]
        n_stamps, n_stamps_0 = int(d["n_stamps_200"]), int(d["n_stamps_0"])
        n_bv_0, n_bv_200 = int(d["n_bioviz_0"]), int(d["n_bioviz_200"])
        n_ent, kb = int(d["n_entities"]), float(d["rrd_bytes"]) / 1000
        msg = str(d["bioviz_message"])
        assert n_frames == n + 1 == n_stamps == n_stamps_0 == n_bv_0 and n_bv_200 == 200
        assert float(d["gap_markers"]) < 1e-6  # the .rrd markers agree with biorbd's

        title = scene_title("Visualising a solution", f"double pendulum, N = {n}, T = {T:g} s: sol.animate()")
        self.play(FadeIn(title), run_time=0.4)

        # ---------------------------------------------------------------- beat 1: the stick figure, re-drawn from the .rrd
        yy, zz = mk[1][[0, 1, 3]], mk[2][[0, 1, 3]]  # pivot, elbow, tip (marker_1, marker_2, marker_4)
        box_x, box_y, box_w, box_h = -3.55, 0.05, 5.6, 4.2
        scale = min(box_w / np.ptp(yy), box_h / np.ptp(zz))
        cy_, cz_ = (yy.max() + yy.min()) / 2, (zz.max() + zz.min()) / 2

        def p(k, j):
            return np.array([box_x + scale * (mk[1, j, k] - cy_), box_y + scale * (mk[2, j, k] - cz_), 0])

        tip_path = [p(k, 3) for k in range(n_frames)]
        trace = DashedVMobject(
            VMobject(color=C_GHOST, stroke_width=3).set_points_as_corners(tip_path), num_dashes=60
        ).set_opacity(0.8)
        node_dots = VGroup(*[Dot(pt, radius=0.04, color=C_GHOST) for pt in tip_path])
        idx = ValueTracker(0)

        def figure():
            k = int(round(min(idx.get_value(), n_frames - 1)))
            piv, elb, tip = p(k, 0), p(k, 1), p(k, 3)
            return VGroup(
                Line(piv, elb, color=C_STATE, stroke_width=8),
                Line(elb, tip, color=C_STATE, stroke_width=8),
                Dot(piv, radius=0.09, color=GRAY_B),
                Dot(elb, radius=0.09, color=WHITE),
                Dot(tip, radius=0.11, color=WHITE),
            )

        fig = always_redraw(figure)
        self.play(FadeIn(trace), FadeIn(node_dots), FadeIn(fig), run_time=1.0)

        # timeline of the frames: one tick per time stamp of the recording
        x0, x1, y_tl = -6.0, -1.1, -2.65

        def tx(tt):
            return x0 + (x1 - x0) * tt / T

        ticks = VGroup(
            *[Line([tx(tt), y_tl - 0.07, 0], [tx(tt), y_tl + 0.07, 0], color=GRAY_B, stroke_width=2) for tt in t]
        )
        axis = Line([x0, y_tl, 0], [x1, y_tl, 0], color=GRAY_B, stroke_width=2)
        tlabels = VGroup(
            *[
                Text(dec(f"{v:g}"), font_size=16, color=GRAY_B).next_to([tx(v), y_tl - 0.07, 0], DOWN, buff=0.06)
                for v in (0, T / 2, T)
            ],
            Text("t (s)", font_size=16, color=GRAY_B).move_to([x1 + 0.55, y_tl - 0.3, 0]),
        )
        cursor = always_redraw(
            lambda: Dot([tx(t[int(round(min(idx.get_value(), n_frames - 1)))]), y_tl, 0], radius=0.09, color=C_STATE)
        )
        self.play(Create(axis), FadeIn(ticks), FadeIn(tlabels), FadeIn(cursor), run_time=0.6)

        # right: the code that produced the recording, caption "Bioptim code" above
        panel = code_panel(
            [
                (0, "sol.animate(", WHITE),
                (1, 'viewer="pyorerun",', C_STATE),
                (1, "show_now=False,", C_CTRL),
                (1, "n_frames=200)", WHITE),
                (0, "import rerun as rr", WHITE),
                (0, 'rr.save("solution.rrd")', C_CTRL),
            ],
            size=19,
        )
        self.play(FadeIn(panel), run_time=0.6)

        # play the animation: the viewer shows the latest logged frame (one per node), so the figure steps
        self.play(idx.animate.set_value(n_frames - 1), run_time=2.6, rate_func=linear)
        info = place(
            Text(
                f"the recording holds {n_stamps} time stamps, one per node\n"
                f"{n_ent} entities · {kb:.0f} kB file\n" + ipopt_line(int(d["iterations"]), bool(d["converged"])),
                font_size=19,
                color=GRAY_A,
                line_spacing=0.9,
            ),
            CODE_X,
            -1.05,
        )
        self.play(FadeIn(info), run_time=0.5)
        remark = place(
            say("With show_now=False no window opens: the animation is only written to the file."), CODE_X, -2.15
        )
        self.play(FadeIn(remark), run_time=0.5)
        foot1 = footer(
            "Stick figure re-drawn from the marker positions that pyorerun wrote to the file, not a screen capture."
        )
        self.play(FadeIn(foot1))
        self.wait(0.5)

        # ---------------------------------------------------------------- beat 2: the frames each viewer receives
        self.play(*[FadeOut(m) for m in self.mobjects if m is not title], run_time=0.5)
        self.add(title)
        rows = [
            (1.55, f"pyorerun ignores n_frames: {n_stamps_0} frames, one per node", C_STATE, t, 0.2),
            (0.3, f"bioviz with n_frames=0: {n_bv_0} frames, one per node", C_STATE, t, 0.2),
            (
                -0.95,
                f"bioviz with n_frames=200: {n_bv_200} interpolated frames",
                C_PAR,
                np.linspace(0, T, n_bv_200),
                0.14,
            ),
        ]
        strips = VGroup()
        for y, label, color, times, hgt in rows:
            lab = place(Text(label, font_size=19, color=GRAY_A), -6.6, y + 0.4, max_width=6.1)
            base = Line([x0, y - 0.1, 0], [x1, y - 0.1, 0], color=GRAY_D, stroke_width=1.5)
            bars = VGroup(
                *[Line([tx(tt), y - 0.1, 0], [tx(tt), y - 0.1 + hgt, 0], color=color, stroke_width=2) for tt in times]
            )
            strips.add(VGroup(lab, base, bars))
        for strip in strips:
            self.play(FadeIn(strip[0]), FadeIn(strip[1]), Create(strip[2]), run_time=0.7)
        tl2 = VGroup(
            *[
                Text(dec(f"{v:g}"), font_size=16, color=GRAY_B).next_to([tx(v), -1.1, 0], DOWN, buff=0.06)
                for v in (0, T / 2, T)
            ],
            Text("t (s)", font_size=16, color=GRAY_B).move_to([x1 + 0.55, -1.4, 0]),
        )
        self.play(FadeIn(tl2), run_time=0.3)

        panel2 = code_panel(
            [
                (0, "sol.animate(", WHITE),
                (1, 'viewer="bioviz",', C_STATE),
                (1, "show_now=False,", C_CTRL),
                (1, "n_frames=200)", WHITE),
            ],
            size=19,
        )
        self.play(FadeIn(panel2), run_time=0.5)
        err_head = Text("Here bioviz is not installed, so the call raises:", font_size=19, color=GRAY_A)
        err_l1, err_l2 = (Text(s, font=MONO, font_size=17, color=RED_C) for s in msg.replace(": ", ":\n").split("\n"))
        err = VGroup(err_head, err_l1, err_l2).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        place(err, CODE_X, -0.85)
        remark2 = place(
            say("bioviz would play the frames returned by interpolate_data, pyorerun always plays the nodes."),
            CODE_X,
            -2.15,
        )
        self.play(FadeIn(err), run_time=0.5)
        self.play(FadeIn(remark2), run_time=0.5)
        foot2 = footer("bioviz was not run here: its frame counts come from the library function interpolate_data.")
        self.play(FadeIn(foot2))
        self.wait(2.5)  # final hold before the end card, in every scene

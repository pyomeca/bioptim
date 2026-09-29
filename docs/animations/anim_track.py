"""
Manim CE scene: tracking a reference with ObjectiveFcn.Lagrange.TRACK_STATE, driven by REAL solves
(data/track_pendulum.npz, see ``generate_track_data.py``): pendulum pole angle following a sine, 3 weights, then the
hard-constraint alternative ConstraintFcn.TRACK_STATE.

Render (from docs/animations):  manim render -qh anim_track.py TrackState
"""

import sys
from pathlib import Path

import numpy as np
from manim import *

sys.path.insert(0, str(Path(__file__).parent))
from features_scenes import (  # noqa: E402  (also sets the default fonts)
    CODE_X,
    axis_label,
    code,
    code_panel,
    make_axes,
    place,
    poly,
    scene_title,
    steps,
    time_label,
    x_ticks,
    y_ticks,
)

DATA = Path(__file__).parent / "data" / "track_pendulum.npz"
C_REF, C_ACH, C_ERR, C_CTRL = WHITE, YELLOW_C, RED_C, GREEN_C
TAGS = ["w0", "w1", "w2", "hard"]


class TrackState(Scene):
    def construct(self):
        d = np.load(DATA)
        n, T = int(d["n_shooting"]), float(d["final_time"])
        t = np.linspace(0, T, n + 1)
        W = d["weights"]
        ref = d["target"]

        self.play(FadeIn(scene_title("Tracking a reference", "TRACK_STATE: weight vs error vs effort")), run_time=0.5)

        ax_q = make_axes([-3.6, 1.15, 0], 5.6, 1.7, (0, T), (-0.4, 0.4))
        ax_e = make_axes([-3.6, -0.75, 0], 5.6, 1.3, (0, T), (-0.45, 0.45))
        ax_u = make_axes([-3.6, -2.55, 0], 5.6, 1.3, (0, T), (-6.5, 6.5))
        decos = VGroup(
            axis_label("θ (rad)", ax_q),
            axis_label("e = θ − θ_ref (rad)", ax_e),
            axis_label("τ (N)", ax_u),
            time_label(ax_u),
            x_ticks(ax_u, [0, 1, 2], "{:.0f}"),
            y_ticks(ax_q, [-0.3, 0.3]),
            y_ticks(ax_e, [-0.4, 0.4]),
            y_ticks(ax_u, [-5, 5]),
        )
        zeros = VGroup(
            *[DashedLine(a.c2p(0, 0), a.c2p(T, 0), color=GRAY_D, stroke_width=2) for a in (ax_q, ax_e, ax_u)]
        )
        self.play(Create(ax_q), Create(ax_e), Create(ax_u), FadeIn(decos), Create(zeros), run_time=0.8)

        ghost = poly(ax_q, t, ref, C_REF, 3)
        ghost.set_stroke(opacity=0.9)
        ref_lab = Text("θ_ref", font_size=18, color=C_REF).next_to(ax_q.c2p(0.25, 0.3), UP, buff=0.02)
        ach_lab = Text("achieved θ", font_size=18, color=C_ACH).next_to(ax_q.c2p(1.75, -0.32), DOWN, buff=0.02)

        def theta(i):
            return poly(ax_q, t, d[f"{TAGS[i]}_theta"], C_ACH, 4)

        def err(i):
            return poly(ax_e, t, d[f"{TAGS[i]}_err"], C_ERR, 4)

        def tau(i):
            return steps(ax_u, t, d[f"{TAGS[i]}_tau"], C_CTRL, 4)

        lines = [
            (0, "obj.add(ObjectiveFcn.Lagrange.TRACK_STATE,", WHITE),
            (1, 'key="q", index=[1], node=Node.ALL,', WHITE),
            (1, "target=q_ref,  # shape (1, n_shooting + 1)", WHITE),
            (1, f"weight={W[0]:g})", C_ACH),
            (0, "obj.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,", WHITE),
            (1, 'key="tau", weight=1)', WHITE),
        ]
        panel = code_panel(lines, size=19, top=2.35, caption="Bioptim code")
        w_line = panel[1][3]

        def w_mob(i):
            return code(f"weight={W[i]:g})", 19, C_ACH, max_width=6.2).move_to(w_line, aligned_edge=LEFT)

        def readout(i):
            tg = TAGS[i]
            head = f"weight {W[i]:g}" if i < 3 else "hard constraint"
            body = (
                f"{head}\n"
                f"rms error   {float(d[tg + '_rms_err']):.3f} rad   (max {float(d[tg + '_max_err']):.3f})\n"
                f"effort ∫τ² dt   {float(d[tg + '_effort']):.4g}\n"
                f"IPOPT status {int(d[tg + '_status'])}, {int(d[tg + '_iterations'])} iterations"
            )
            return place(Text(body, font_size=18, color=GRAY_A, line_spacing=0.95), CODE_X, -1.55)

        comments = [
            "small weight: the controller saves effort\nand barely follows the reference.",
            "a larger weight buys tracking\nwith a larger torque.",
            "weight 1000: error under 0.03 rad,\nat about 13x the effort of weight 30.",
        ]

        def comment(i):
            return place(Text(comments[i], font_size=19, color=YELLOW_C), CODE_X, -2.95)

        c_ach, c_err, c_tau = theta(0), err(0), tau(0)
        info, cmt = readout(0), comment(0)
        self.play(FadeIn(panel), Create(ghost), FadeIn(ref_lab), run_time=0.9)
        self.play(Create(c_ach), Create(c_err), Create(c_tau), FadeIn(ach_lab), run_time=1.0)
        self.play(FadeIn(info), FadeIn(cmt), run_time=0.4)
        self.wait(1.2)
        for i in (1, 2):
            self.play(
                Transform(c_ach, theta(i)),
                Transform(c_err, err(i)),
                Transform(c_tau, tau(i)),
                Transform(w_line, w_mob(i)),
                Transform(info, readout(i)),
                Transform(cmt, comment(i)),
                run_time=1.4,
            )
            self.wait(1.5)

        # ---- second beat: hard constraint ----
        hard_lines = [
            (0, "cons.add(ConstraintFcn.TRACK_STATE,", WHITE),
            (1, 'key="q", index=[1], node=Node.ALL,', WHITE),
            (1, "target=q_ref)", C_ACH),
        ]
        hard = code_panel(hard_lines, size=19, top=2.35, caption="Hard-constraint alternative")
        hard_cmt = place(
            Text(
                "the error is forced to 0 at every node:\nmore effort, chattering τ, no trade-off to tune.",
                font_size=19,
                color=YELLOW_C,
            ),
            CODE_X,
            -2.95,
        )
        self.play(FadeOut(panel), FadeOut(cmt), FadeIn(hard), run_time=0.5)
        self.play(
            Transform(c_ach, theta(3)),
            Transform(c_err, err(3)),
            Transform(c_tau, tau(3)),
            Transform(info, readout(3)),
            FadeIn(hard_cmt),
            run_time=1.4,
        )
        self.wait(2.0)

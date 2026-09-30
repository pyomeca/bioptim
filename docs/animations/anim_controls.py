"""
Manim CE scene: control interpolation in bioptim, driven by REAL solves (data/controls_types.npz, see
``generate_controls_data.py``). Same pendulum swing-up solved with
    ControlType.CONSTANT, ControlType.LINEAR_CONTINUOUS, ControlType.CONSTANT_WITH_LAST_NODE.

Render (from docs/animations):  manim render -qh anim_controls.py ControlTypes
"""

import sys
from pathlib import Path

import numpy as np
from manim import *

sys.path.insert(0, str(Path(__file__).parent))
from features_scenes import (  # noqa: E402  (also sets the default fonts)
    CODE_X,
    code,
    code_panel,
    make_axes,
    place,
    poly,
    scene_title,
    steps,
    axis_label,
    time_label,
    x_ticks,
    y_ticks,
)

DATA = Path(__file__).parent / "data" / "controls_types.npz"
C_CTRL, C_STATE, C_NODE, C_UNUSED = GREEN_C, YELLOW_C, WHITE, GRAY_C
TAGS = ["constant", "linear", "last"]
CT = ["ControlType.CONSTANT", "ControlType.LINEAR_CONTINUOUS", "ControlType.CONSTANT_WITH_LAST_NODE"]


class ControlTypes(Scene):
    def construct(self):
        d = np.load(DATA)
        n, T = int(d["n_shooting"]), float(d["final_time"])
        t = np.linspace(0, T, n + 1)

        self.play(FadeIn(scene_title("Control interpolation", "same swing-up, three control types")), run_time=0.5)

        ax_u = make_axes([-3.6, 0.55, 0], 5.6, 2.9, (0, T), (-25, 8), y_step=10)
        ax_q = make_axes([-3.6, -2.55, 0], 5.6, 1.5, (0, T), (0, 3.4))
        decos = VGroup(
            axis_label("τ (N)", ax_u),
            axis_label("θ (rad)", ax_q),
            time_label(ax_q),
            x_ticks(ax_q, [0, 0.5, 1.0], "{:.1f}"),
            y_ticks(ax_u, [-20, 0]),
            y_ticks(ax_q, [0, 3]),
        )
        zero = DashedLine(ax_u.c2p(0, 0), ax_u.c2p(T, 0), color=GRAY_D, stroke_width=2)
        self.play(Create(ax_u), Create(ax_q), FadeIn(decos), Create(zero), run_time=0.8)

        def values(i):
            return d[f"{TAGS[i]}_tau_raw"]

        def ctrl_curve(i):
            u = values(i)
            if i == 1:
                return poly(ax_u, t, u, C_CTRL, 4)
            return steps(ax_u, t, u[:n], C_CTRL, 4)

        def nodes(i):
            u = values(i)
            dots = VGroup()
            for k in range(n + 1):
                if k < len(u):
                    used = not (i == 2 and k == n)
                    dots.add(Dot(ax_u.c2p(t[k], u[k]), radius=0.05, color=C_NODE if used else C_UNUSED))
                else:
                    dots.add(Dot(ax_u.c2p(t[k], 0), radius=0.0001, fill_opacity=0))
            return dots

        def theta(i):
            return poly(ax_q, t, d[f"{TAGS[i]}_theta"], C_STATE, 4)

        lines = [
            (0, "ocp = OptimalControlProgram(", WHITE),
            (1, "bio_model, n_shooting=20, phase_time=1.0,", WHITE),
            (1, "dynamics=dynamics, x_bounds=x_bounds,", WHITE),
            (1, "u_bounds=u_bounds, objective_functions=obj,", WHITE),
            (1, f"control_type={CT[0]},", C_CTRL),
            (0, ")", WHITE),
        ]
        panel = code_panel(lines, size=19, top=2.35, caption="Bioptim code")
        ct_line = panel[1][4]

        def ct_mob(i):
            return code(f"control_type={CT[i]},", 19, C_CTRL, max_width=6.2).move_to(ct_line, aligned_edge=LEFT)

        def readout(i):
            tg = TAGS[i]
            cols = d[f"{tg}_ncols"]
            colstr = "1" if i != 1 else "2  (1 at the last node)"
            body = (
                f"nodes holding a control: {int(d[tg + '_n_ctrl_nodes'])}   ·   columns per node: {colstr}\n"
                f"decision vector: {int(d[tg + '_n_vars_sol'])} variables\n"
                f"IPOPT cost {float(d[tg + '_cost']):.2f}  ·  {int(d[tg + '_iterations'])} iterations"
                f"{'' if int(d[tg + '_status']) == 0 else '  (NOT converged)'}"
            )
            return place(Text(body, font_size=19, color=GRAY_A, line_spacing=0.9), CODE_X, -1.0)

        comments = [
            "CONSTANT: staircase, one τ per interval.\nThe last node has no control.",
            "LINEAR_CONTINUOUS: piecewise-linear τ,\none extra value at t = T, and more freedom.",
            "CONSTANT_WITH_LAST_NODE: same staircase,\nplus a control at t = T that nothing uses here.",
        ]

        def comment(i):
            return place(Text(comments[i], font_size=19, color=YELLOW_C), CODE_X, -2.4)

        def legend():
            return place(
                Text("white dot = free decision variable   ·   grey = unused", font_size=16, color=GRAY_B),
                CODE_X,
                -3.25,
            )

        curve_u, dots, curve_q = ctrl_curve(0), nodes(0), theta(0)
        info, cmt = readout(0), comment(0)
        self.play(FadeIn(panel), Create(curve_u), FadeIn(dots), Create(curve_q), run_time=1.5)
        self.play(FadeIn(info), FadeIn(cmt), FadeIn(legend()), run_time=0.5)
        self.wait(1.8)
        for i in (1, 2):
            self.play(
                Transform(curve_u, ctrl_curve(i)),
                Transform(dots, nodes(i)),
                Transform(curve_q, theta(i)),
                Transform(ct_line, ct_mob(i)),
                Transform(info, readout(i)),
                Transform(cmt, comment(i)),
                run_time=1.6,
            )
            self.wait(2.2 if i == 1 else 2.5)

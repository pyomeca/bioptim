"""
Manim CE scene: VariableScaling in bioptim, driven by REAL solves (data/scaling_pendulum.npz, see
``generate_scaling_data.py``): the pendulum of example_variable_scaling.py solved with all scaling factors = 1 and with
x_scaling / u_scaling. Part 1: what the optimiser sees (x / scale). Part 2: IPOPT primal infeasibility history.

Render (from docs/animations):  manim render -qh anim_scaling.py Scaling
"""

import sys
from pathlib import Path

import numpy as np
from manim import *

sys.path.insert(0, str(Path(__file__).parent))
from features_scenes import (  # noqa: E402  (also sets the default fonts)
    CODE_X,
    MONO,
    code_panel,
    make_axes,
    place,
    scene_title,
    axis_label,
)

DATA = Path(__file__).parent / "data" / "scaling_pendulum.npz"
C_PHYS, C_SCAL, C_UNS = YELLOW_C, GREEN_C, ORANGE


class Scaling(Scene):
    def construct(self):
        d = np.load(DATA)
        self.play(
            FadeIn(scene_title("Variable scaling", "the optimiser sees x / scale, you still see x")), run_time=0.4
        )

        lines = [
            (0, "x_scaling = VariableScalingList()", WHITE),
            (0, 'x_scaling["q"] = [1, 3]', C_SCAL),
            (0, 'x_scaling["qdot"] = [85, 85]', C_SCAL),
            (0, "u_scaling = VariableScalingList()", WHITE),
            (0, 'u_scaling["tau"] = [900, 1]', C_SCAL),
            (0, "ocp = OptimalControlProgram(", WHITE),
            (1, "..., x_scaling=x_scaling,", C_SCAL),
            (1, "u_scaling=u_scaling, ...)", C_SCAL),
        ]
        panel = code_panel(lines, size=19, top=2.3, caption="Bioptim code")
        self.play(FadeIn(panel), run_time=0.5)

        # ---------------------------------------------------------------- part 1: magnitudes, physical vs scaled
        names = ["q  (rad, m)", "qdot", "tau  (N)"]
        phys = [abs(d["scaled_q"]).max(), abs(d["scaled_qdot"]).max(), abs(d["scaled_tau"]).max()]
        # per-component max of |x| / factor, then the largest one (what the optimiser variable reaches)
        scal = [
            (abs(d["scaled_q"]).max(1) / d["factor_q"]).max(),
            (abs(d["scaled_qdot"]).max(1) / d["factor_qdot"]).max(),
            (abs(d["scaled_tau"]).max(1) / d["factor_tau"]).max(),
        ]
        x0, per_dec, ys = -5.3, 1.05, [1.5, 0.3, -0.9]  # log axis from 0.1 to 1000 (4 decades)

        def bar_w(v):
            return (np.log10(v) + 1) * per_dec

        def bar(v, y, color):
            return Rectangle(width=bar_w(v), height=0.42, stroke_width=0, fill_color=color, fill_opacity=0.9).move_to(
                [x0 + bar_w(v) / 2, y, 0]
            )

        def val_txt(v, y, color):
            return Text((f"{v:.0f}" if v >= 100 else f"{v:.3g}"), font_size=19, color=color).move_to(
                [x0 + bar_w(v) + 0.45, y, 0]
            )

        axis = Line([x0, 2.1, 0], [x0, -1.5, 0], color=GRAY_B, stroke_width=2)
        ticks = VGroup()
        for e in (-1, 0, 1, 2, 3):
            xx = x0 + (e + 1) * per_dec
            ticks.add(
                Line([xx, -1.5, 0], [xx, -1.6, 0], color=GRAY_B, stroke_width=2),
                Text(f"{10.0 ** e:g}", font_size=16, color=GRAY_B).move_to([xx, -1.85, 0]),
            )
        labels = VGroup(
            *[
                Text(n, font_size=19, color=GRAY_A).move_to([x0 + 0.05, y + 0.45, 0], aligned_edge=LEFT)
                for n, y in zip(names, ys)
            ]
        )
        head = Text("largest |value| in the solution (log scale)", font_size=19, color=GRAY_B).move_to([-3.4, 2.3, 0])
        bars = VGroup(*[bar(v, y, C_PHYS) for v, y in zip(phys, ys)])
        vals = VGroup(*[val_txt(v, y, C_PHYS) for v, y in zip(phys, ys)])
        cap1 = Text("what you see: x", font_size=20, color=C_PHYS).move_to([-3.4, -2.55, 0])
        self.play(Create(axis), FadeIn(ticks), FadeIn(labels), FadeIn(head), run_time=0.5)
        self.play(GrowFromEdge(bars, LEFT), FadeIn(vals), FadeIn(cap1), run_time=1.0)
        self.wait(0.8)
        bars2 = VGroup(*[bar(v, y, C_SCAL) for v, y in zip(scal, ys)])
        vals2 = VGroup(*[val_txt(v, y, C_SCAL) for v, y in zip(scal, ys)])
        cap2 = Text("what IPOPT sees: x / scale, all near 1", font_size=20, color=C_SCAL).move_to([-3.4, -2.55, 0])
        self.play(Transform(bars, bars2), Transform(vals, vals2), Transform(cap1, cap2), run_time=1.4)
        self.wait(1.4)
        self.play(FadeOut(VGroup(axis, ticks, labels, head, bars, vals, cap1)), run_time=0.4)

        # ---------------------------------------------------------------- part 2: IPOPT inf_pr history
        h_u, h_s = d["unscaled_hist"], d["scaled_hist"]
        n_u, n_s = int(d["unscaled_iterations"]), int(d["scaled_iterations"])
        lo, hi = -13.0, 2.5
        ax = make_axes([-3.6, -0.1, 0], 5.4, 3.7, (0, n_u), (lo, hi), y_step=4)
        ax.get_x_axis().shift(0)  # x axis drawn at the bottom of the box
        ax.x_axis.move_to(ax.c2p(n_u / 2, lo))
        ticks2 = VGroup(
            *[
                Text(f"1e{e}" if e else "1", font_size=16, color=GRAY_B).next_to(ax.c2p(0, e), LEFT, buff=0.08)
                for e in (-12, -8, -4, 0)
            ],
            *[
                Text(str(v), font_size=16, color=GRAY_B).next_to(ax.c2p(v, lo), DOWN, buff=0.08)
                for v in (0, 100, 200, 300, n_u)
            ],
        )
        ylab = axis_label("IPOPT inf_pr (log scale)", ax)
        xlab = Text("iteration", font_size=16, color=GRAY_B).next_to(ax.c2p(n_u, lo), DOWN, buff=0.4).shift(LEFT * 0.4)
        self.play(Create(ax), FadeIn(ticks2), FadeIn(ylab), FadeIn(xlab), run_time=0.5)

        it = ValueTracker(0)

        def curve(h, color):
            def make():
                k = int(min(it.get_value(), len(h) - 1)) + 1
                pts = [ax.c2p(r[0], np.log10(max(r[2], 1e-14))) for r in h[:k]]
                m = VMobject(color=color, stroke_width=4)
                if len(pts) > 1:
                    m.set_points_as_corners(pts)
                return m

            return always_redraw(make)

        def counter(h, n, color, y, label):
            def make():
                k = int(min(it.get_value(), n))
                done = " (converged)" if k >= n else ""
                return place(Text(f"{label}: iteration {k}{done}", font_size=20, color=color), CODE_X, y)

            return always_redraw(make)

        c_u, c_s = curve(h_u, C_UNS), curve(h_s, C_SCAL)
        t_u = counter(h_u, n_u, C_UNS, -1.65, "no scaling")
        t_s = counter(h_s, n_s, C_SCAL, -2.1, "with scaling")
        self.add(c_u, c_s, t_u, t_s)
        self.play(it.animate.set_value(n_u), run_time=5.0, rate_func=linear)
        dtau = abs(d["unscaled_tau"] - d["scaled_tau"]).max()
        note = place(
            Paragraph(
                f"Same problem, same optimum: cost {float(d['scaled_cost']):.1f} in both.",
                f"Largest torque difference between the two solutions: {dtau:.0e} N.",
                f"{n_u} vs {n_s} iterations  ({n_u / n_s:.1f}× fewer).",
                font_size=19,
                color=YELLOW_C,
                line_spacing=0.9,
            ),
            CODE_X,
            -3.0,
        )
        self.play(FadeIn(note), run_time=0.5)
        self.wait(2.5)

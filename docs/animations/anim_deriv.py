"""
Manim CE scene: penalty on the derivative of a control, ``Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau",
derivative=True, weight=w)``. REAL bioptim / IPOPT solves stored in ``data/deriv_pendulum.npz`` (see
``generate_deriv_data.py``): cart-pendulum swing-up, N = 30, T = 1 s, ``ControlType.LINEAR_CONTINUOUS``, w = 0, 1, 10, 100.

Scene: DerivativePenalty (about 20 s).  Render (from docs/animations):  manim render -qh anim_deriv.py DerivativePenalty
"""

import numpy as np
from manim import *

from features_scenes import (
    CODE_W,
    DATA_DIR,
    M,
    code,
    fit,
    make_axes,
    poly,
    scene_title,
    steps,
    time_label,
    x_ticks,
    y_ticks,
)

CODE_X0 = 0.15
W = WHITE
C_PLAIN = GRAY_B
C_W = {1: BLUE_C, 10: GREEN_C, 100: ORANGE}
CAP = 250.0  # top of the |dtau/dt| axis


def caption(text, size=19, color=GRAY_B):
    return Text(text, font_size=size, color=color)


def code_block(lines, size=17):
    block = VGroup(*[code(text, size, color) for _, text, color in lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.11)
    for line, (level, _, _) in zip(block, lines):
        line.shift(RIGHT * 0.3 * level)
    return block


class DerivativePenalty(Scene):
    def construct(self):
        d = np.load(DATA_DIR / "deriv_pendulum.npz")
        n, horizon = int(d["N"]), float(d["T"])
        t = np.linspace(0, horizon, n + 1)
        dt = horizon / n

        def tau(w):
            return d[f"w{w}_tau"]

        def dtau(w):
            return np.abs(np.diff(tau(w))) / dt

        self.play(
            FadeIn(scene_title("Penalty on the derivative of a control", f"swing-up, N = {n}, T = {horizon:g} s")),
            run_time=0.4,
        )

        # ------------------------------------------------------------- axes (left)
        ax1 = make_axes([-3.6, 1.05, 0], 6.0, 2.5, (0, horizon), (-40, 12))
        ax2 = make_axes([-3.6, -2.25, 0], 6.0, 1.75, (0, horizon), (0, CAP))
        lab1 = caption("torque (N)", 17).next_to(ax1.get_y_axis(), UP, buff=0.05).align_to(ax1.get_y_axis(), LEFT)
        lab2 = caption("torque rate |dτ/dt| (N/s)", 17)
        lab2.next_to(ax2.get_y_axis(), UP, buff=0.05).align_to(ax2.get_y_axis(), LEFT)
        ticks = VGroup(
            y_ticks(ax1, [-30, -15, 0, 10]),
            y_ticks(ax2, [0, 100, 200]),
            x_ticks(ax2, [0, 0.5, 1]),
        )
        tlab = time_label(ax2)
        plain1 = poly(ax1, t, tau(0), C_PLAIN, 4)
        plain2 = steps(ax2, t, np.minimum(dtau(0), CAP), C_PLAIN, 4)
        clip = caption(f"clipped, peak {d['w0_dtau_max']:.0f} N/s", 15, C_PLAIN).move_to(ax2.c2p(0.5, CAP * 0.9))
        clip.shift(RIGHT * 0.9)
        self.play(FadeIn(VGroup(ax1, ax2, lab1, lab2, ticks, tlab)), run_time=0.5)

        # ------------------------------------------------------------- code (right)
        cap1 = caption("Bioptim code")
        code1 = code_block(
            [
                (0, "objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,", W),
                (1, 'key="tau")', W),
                (0, "objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,", C_W[1]),
                (1, 'key="tau", derivative=True, weight=w)', C_W[1]),
                (0, "OptimalControlProgram(...,", GRAY_A),
                (1, "control_type=ControlType.LINEAR_CONTINUOUS)", GRAY_A),
            ]
        )
        panel = VGroup(cap1, code1).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        fit(panel, CODE_W)
        panel.move_to([CODE_X0, 2.3, 0], aligned_edge=UL)
        new_line = VGroup(code1[2], code1[3])
        new_line.set_opacity(0)
        self.play(FadeIn(panel), Create(plain1), run_time=1.2)
        self.play(Create(plain2), FadeIn(clip), run_time=1.0)

        # ------------------------------------------------------------- table of readouts (right)
        header = code("w    ∫tau² dt   peak|dtau/dt|   IPOPT", 16, GRAY_B)
        rows = []

        def row(w, color):
            r = d[f"w{w}_effort"], d[f"w{w}_dtau_max"], int(d[f"w{w}_iterations"])
            assert int(d[f"w{w}_status"]) == 0
            return code(f"{w:<4d} {r[0]:8.1f} {r[1]:13.0f} {r[2]:8d} it", 16, color)

        table_top = [CODE_X0, 0.2, 0]
        header.move_to(table_top, aligned_edge=UL)
        first = row(0, C_PLAIN).next_to(header, DOWN, buff=0.14, aligned_edge=LEFT)
        rows.append(first)
        self.play(FadeIn(header), FadeIn(first), run_time=0.5)
        self.wait(0.4)

        # ------------------------------------------------------------- the derivative penalty at 3 weights
        self.play(new_line.animate.set_opacity(1), run_time=0.6)

        plain1.set_stroke(opacity=0.55)
        plain2.set_stroke(opacity=0.55)
        wlab = None
        for w in (1, 10, 100):
            color = C_W[w]
            n1 = poly(ax1, t, tau(w), color, 5)
            n2 = steps(ax2, t, np.minimum(dtau(w), CAP), color, 5)
            tag = code(f"w = {w}", 20, color).move_to(ax1.c2p(0.2, -34))
            new_row = row(w, color).next_to(rows[-1], DOWN, buff=0.12, aligned_edge=LEFT)
            rows.append(new_row)
            if w == 1:
                self.play(Create(n1), Create(n2), FadeIn(tag), FadeIn(new_row), run_time=1.2)
                c1, c2, wlab = n1, n2, tag
            else:
                self.play(
                    Transform(c1, n1),
                    Transform(c2, n2),
                    Transform(wlab, tag),
                    FadeIn(new_row),
                    new_line.animate.set_color(color),
                    run_time=1.2,
                )
            self.wait(0.7)

        # ------------------------------------------------------------- the CONSTANT gotcha
        c0, c100 = float(d["const_w0_cost"]), float(d["const_w100_cost"])
        assert abs(float(np.abs(d["const_w0_tau"] - d["const_w100_tau"]).max())) == 0.0
        note = VGroup(
            M("<b>With the default ControlType.CONSTANT</b>", 20, W),
            M(
                "the derivative term is exactly 0 (u_end = u_start),\n"
                f"so the cost does not depend on w:\n"
                f"{c0:.4f} with w = 0, {c100:.4f} with w = 100.\n"
                "Use LINEAR_CONTINUOUS to make the penalty work.",
                18,
                GRAY_B,
            ),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        fit(note, CODE_W)
        note.next_to(rows[-1], DOWN, buff=0.3, aligned_edge=LEFT)
        self.play(FadeIn(note), run_time=0.6)
        self.wait(2.5)

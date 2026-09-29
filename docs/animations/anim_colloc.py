"""
Manim CE scene: direct collocation, effect of ``polynomial_degree`` and of the point family (``method="legendre"`` or
``"radau"``) on accuracy. REAL bioptim / IPOPT solves stored in ``data/colloc_pendulum.npz`` (see
``generate_colloc_data.py``): cart-pendulum swing, N = 30 intervals, T = 1 s, degrees 2..6, both families.

Scene: CollocationDegree (about 16 s).  Render (from docs/animations):  manim render -qh anim_colloc.py CollocationDegree
"""

import numpy as np
from manim import *

from features_scenes import CODE_W, DATA_DIR, M, code, fit, scene_title

CODE_X0 = 0.15
C_LEG = BLUE_C
C_RAD = ORANGE
DEGREES = [2, 3, 4, 5, 6]
FAMILIES = [("legendre", C_LEG), ("radau", C_RAD)]
W = WHITE


def code_block(lines, size=17):
    block = VGroup(*[code(text, size, color) for _, text, color in lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.11)
    for line, (level, _, _) in zip(block, lines):
        line.shift(RIGHT * 0.3 * level)
    return block


def caption(text, size=19, color=GRAY_B):
    return Text(text, font_size=size, color=color)


class CollocationDegree(Scene):
    def construct(self):
        d = np.load(DATA_DIR / "colloc_pendulum.npz")
        n = int(d["legendre2_tau"].shape[1])

        def val(fam, deg, key):
            return d[f"{fam}{deg}_{key}"]

        title = scene_title(
            "Direct collocation: polynomial degree", f"same problem, N = {n} intervals, only the scheme changes"
        )
        self.play(FadeIn(title), run_time=0.4)

        # ---------------------------------------------------------------- code (right)
        cap1 = caption("Bioptim code")
        code1 = code_block(
            [
                (0, "DynamicsOptions(ode_solver=", W),
                (1, "OdeSolver.COLLOCATION(", W),
                (2, 'polynomial_degree=d, method="legendre"))', W),
            ]
        )
        cap2 = caption("points used by the integrator (integrator.py)")
        code2 = code_block([(0, "[0] + collocation_points(degree, method)", GRAY_A)])
        panel = VGroup(cap1, code1, cap2, code2).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        panel[2].shift(DOWN * 0.15)
        fit(panel, CODE_W)
        panel.move_to([CODE_X0, 2.3, 0], aligned_edge=UL)

        # ---------------------------------------------------------------- beat 1: the points in one interval
        col_x0 = {"legendre": -6.0, "radau": -2.75}
        width = 2.45
        ys = {deg: 1.55 - i * 0.7 for i, deg in enumerate(DEGREES)}
        heads = VGroup(
            *[code(f'method="{fam}"', 17, color).move_to([col_x0[fam] + width / 2, 2.5, 0]) for fam, color in FAMILIES]
        )
        row_groups = []
        for deg in DEGREES:
            y = ys[deg]
            g = VGroup(code(f"d={deg}", 17, GRAY_B).move_to([-6.95, y, 0], aligned_edge=LEFT))
            for fam, color in FAMILIES:
                x0 = col_x0[fam]
                pts = val(fam, deg, "points")
                line = Line([x0, y, 0], [x0 + width, y, 0], color=GRAY_D, stroke_width=3)
                tick1 = Line([x0 + width, y - 0.13, 0], [x0 + width, y + 0.13, 0], color=GRAY_B, stroke_width=3)
                start = Dot([x0, y, 0], radius=0.09).set_fill(BLACK, 1).set_stroke(GRAY_B, 3)
                dots = VGroup(*[Dot([x0 + p * width, y, 0], radius=0.085, color=color) for p in pts[1:]])
                g.add(line, tick1, start, dots)
            row_groups.append(g)
        ends = VGroup(
            *[
                code(lab, 15, GRAY_B).move_to([col_x0[fam] + dx, 2.05, 0])
                for fam, _ in FAMILIES
                for lab, dx in (("t_k", 0), ("t_k+1", width))
            ]
        )
        note1 = caption("one interval, rescaled to [0, 1]:  hollow = shooting node, dots = collocation points", 18)
        note1.move_to([-6.95, -2.2, 0], aligned_edge=LEFT)
        note2 = M(
            "<b>legendre</b>: all points inside the interval    <b>radau</b>: the last point is the next node", 19, W
        )
        note2.move_to([-6.95, -2.75, 0], aligned_edge=LEFT)
        self.play(FadeIn(heads), FadeIn(ends), FadeIn(panel), FadeIn(note1), run_time=0.6)
        for g in row_groups:
            self.play(FadeIn(g), run_time=0.45)
        self.play(FadeIn(note2), run_time=0.4)
        self.wait(1.2)

        # ---------------------------------------------------------------- beat 2: error vs degree
        self.play(
            FadeOut(VGroup(heads, ends, note1, note2, *row_groups)),
            FadeOut(VGroup(cap2, code2)),
            run_time=0.5,
        )
        lo, hi = -4.5, 0.0
        ax_x0, ax_w = -5.6, 4.6
        base_y, height = -0.8, 2.7

        def col(deg):
            return ax_x0 + (deg - 2) / 4 * ax_w

        def y_of(v):
            return base_y + height * (np.log10(v) - lo) / (hi - lo)

        axis = Line([ax_x0 - 0.5, base_y, 0], [ax_x0 + ax_w + 0.5, base_y, 0], color=GRAY_B)
        yaxis = Line([ax_x0 - 0.5, base_y, 0], [ax_x0 - 0.5, base_y + height, 0], color=GRAY_B)
        decades = [-4, -3, -2, -1, 0]
        ticks = VGroup(
            *[
                Text(f"1e{e}" if e else "1", font_size=15, color=GRAY_B).move_to([ax_x0 - 0.95, y_of(10.0**e), 0])
                for e in decades
            ]
        )
        guides = VGroup(
            *[
                DashedLine(
                    [ax_x0 - 0.5, y_of(10.0**e), 0],
                    [ax_x0 + ax_w + 0.5, y_of(10.0**e), 0],
                    color=GRAY_D,
                    stroke_width=1.5,
                )
                for e in decades
            ]
        )
        ylab = caption("‖x(T) re-integrated − x(T) optimised‖, log scale", 17).move_to(
            [ax_x0 - 1.1, base_y + height + 0.3, 0], aligned_edge=LEFT
        )
        xt = VGroup(*[Text(str(deg), font_size=18, color=W).move_to([col(deg), base_y - 0.25, 0]) for deg in DEGREES])
        xlab = code("polynomial_degree", 15, GRAY_B).move_to([ax_x0 + ax_w / 2, base_y - 0.6, 0])
        # rows under the axis: real vector size and IPOPT effort (legendre)
        y1, y2 = base_y - 1.0, base_y - 1.4
        r1 = caption("variables", 15).move_to([-6.95, y1, 0], aligned_edge=LEFT)
        r2 = caption("IPOPT", 15).move_to([-6.95, y2, 0], aligned_edge=LEFT)
        nv = VGroup(
            *[
                Text(f"{int(val('legendre', deg, 'n_variables'))}", font_size=16, color=W).move_to([col(deg), y1, 0])
                for deg in DEGREES
            ]
        )
        it = VGroup(
            *[
                Text(
                    f"{int(val('legendre', deg, 'iterations'))} it, {float(val('legendre', deg, 'solve_time')):.1f} s",
                    font_size=14,
                    color=GRAY_A,
                ).move_to([col(deg), y2, 0])
                for deg in DEGREES
            ]
        )
        note_rows = caption("(IPOPT row: legendre, one run each)", 13).move_to([-6.95, y2 - 0.35, 0], aligned_edge=LEFT)
        self.play(FadeIn(VGroup(axis, yaxis, ticks, guides, ylab, xt, xlab, r1, r2, note_rows)), run_time=0.5)

        cap_code = caption("Bioptim code")
        code_b = code_block([(0, "OdeSolver.COLLOCATION(", W), (1, "polynomial_degree=d, method=m)", W)])
        code_b2 = code_block([(0, "sol = ocp.solve(Solver.IPOPT())", W)])
        panel2 = VGroup(cap_code, code_b, code_b2).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        fit(panel2, CODE_W)
        panel2.move_to([CODE_X0, 2.3, 0], aligned_edge=UL)
        self.play(FadeOut(VGroup(cap1, code1)), FadeIn(panel2), run_time=0.4)

        err = {fam: [float(val(fam, deg, "drift_final")) for deg in DEGREES] for fam, _ in FAMILIES}
        curves = {}
        for fam, color in FAMILIES:
            pts = [[col(deg), y_of(e), 0] for deg, e in zip(DEGREES, err[fam])]
            line = VMobject(color=color, stroke_width=5).set_points_as_corners(pts)
            dots = VGroup(*[Dot(p, radius=0.07, color=color) for p in pts])
            curves[fam] = (line, dots)
        self.play(Create(curves["legendre"][0]), FadeIn(curves["legendre"][1]), FadeIn(nv), FadeIn(it), run_time=1.3)
        self.play(Create(curves["radau"][0]), FadeIn(curves["radau"][1]), run_time=1.1)
        leg = code('method="legendre"', 15, C_LEG).move_to([col(4.6), y_of(err["legendre"][3]) - 0.4, 0])
        rad = code('method="radau"', 15, C_RAD).move_to([col(3.3), y_of(err["radau"][1]) + 0.45, 0])
        self.play(FadeIn(leg), FadeIn(rad), run_time=0.4)

        # numeric readout, computed from the data
        e2, e5 = err["legendre"][0], err["legendre"][3]
        v2, v5 = int(val("legendre", 2, "n_variables")), int(val("legendre", 5, "n_variables"))
        msg = VGroup(
            M("<b>legendre, degree 2 to 5</b>", 22, C_LEG),
            M(f"error  {e2:.1e} to {e5:.1e}   (÷ {e2 / e5:.0f})", 22, W),
            M(f"variables  {v2} to {v5}   (× {v5 / v2:.2f}, +{n * 4} per degree)", 22, W),
            M("radau: one order lower (2d−1 vs 2d):", 19, GRAY_B),
            M("legendre wins from degree 4; beyond degree 5", 19, GRAY_B),
            M("the optimizer, not the scheme, limits the error", 19, GRAY_B),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        msg[3:].shift(DOWN * 0.15)
        fit(msg, CODE_W)
        msg.move_to([CODE_X0, 0.3, 0], aligned_edge=UL)
        self.play(FadeIn(msg[:3]), run_time=0.7)
        self.wait(1.0)
        self.play(FadeIn(msg[3:]), run_time=0.6)
        self.wait(2.2)

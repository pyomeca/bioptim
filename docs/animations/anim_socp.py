"""
Manim Community animation: stochastic optimal control (StochasticOptimalControlProgram, SocpType.COLLOCATION), driven
by REAL bioptim/IPOPT solves stored in ``data/socp_results.npz`` (see ``generate_socp_data.py``).

Problem (bioptim/examples/toy_examples/stochastic_optimal_control/obstacle_avoidance_direct_collocation.py, Gillis 2013):
a mass point makes a periodic, time-optimal loop around two super-ellipse obstacles, with motor noise.  Two SOCPs are
compared: the path constraint applied to the MEAN trajectory only, and the robustified one, where a safe guard
gamma * sqrt(dh/dx P dh/dx') built from the optimised covariance P is subtracted.

Scene: RobustPath (about 18 s).  Render (from docs/animations):  manim render -qh anim_socp.py RobustPath
"""

import numpy as np
from manim import *

from features_scenes import CODE_W, DATA_DIR, code, fit, make_axes, poly, scene_title, time_label, x_ticks, y_ticks

C_NOM = ORANGE  # mean trajectory, constraint on the mean only
C_ROB = GREEN_C  # robustified
C_OBS = "#DA1984"
CODE_X0 = 0.0
C_BAD = "#E01010"

Z_MAX = 3
CX, CY, A, B, NEXP = [0, 1], [0, 0.5], [1, 0.5], [1, 2], 4  # obstacles (MassPointModel)


def h_and_grad(q, i):
    hh = ((q[0] - CX[i]) / A[i]) ** NEXP + ((q[1] - CY[i]) / B[i]) ** NEXP - 1
    g = np.array(
        [NEXP * ((q[0] - CX[i]) / A[i]) ** (NEXP - 1) / A[i], NEXP * ((q[1] - CY[i]) / B[i]) ** (NEXP - 1) / B[i]]
    )
    return hh, g


def analyse(d, name, poly_deg):
    """Per node: mean position, 2x2 position covariance, min over obstacles of h and of z = h / sqrt(dh P dh')."""
    q = d[f"{name}_q"][:, :: poly_deg + 2]
    cov = d[f"{name}_cov"]
    n = q.shape[1]
    P = np.array([cov[:, k].reshape(4, 4, order="F")[:2, :2] for k in range(n)])
    h = np.zeros(n)
    hs = np.zeros(n)
    for k in range(n):
        vals = []
        for i in range(2):
            hh, g = h_and_grad(q[:, k], i)
            vals.append((hh, hh / np.sqrt(g @ P[k] @ g)))
        h[k] = min(v[0] for v in vals)
        hs[k] = min(v[1] for v in vals)
    return q, P, h, hs


def superellipse(i, n=200):
    th = np.linspace(0, 2 * np.pi, n)
    c, s = np.cos(th), np.sin(th)
    x = CX[i] + A[i] * np.sign(c) * np.abs(c) ** (2 / NEXP)
    y = CY[i] + B[i] * np.sign(s) * np.abs(s) ** (2 / NEXP)
    return x, y


def code_block(lines, size=14.5):
    block = VGroup(*[code(text, size, color) for _, text, color in lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
    for line, (level, _, _) in zip(block, lines):
        line.shift(RIGHT * 0.3 * level)
    return block


class RobustPath(Scene):
    def construct(self):
        d = np.load(DATA_DIR / "socp_results.npz")
        n, poly_deg, noise = int(d["n"]), int(d["poly"]), float(d["noise"][0])
        q0, P0, h0, hs0 = analyse(d, "nonrobust", poly_deg)
        q1, P1, h1, hs1 = analyse(d, "robust", poly_deg)
        tf0, tf1 = float(d["nonrobust_tf"]), float(d["robust_tf"])
        t0, t1 = np.linspace(0, tf0, n + 1), np.linspace(0, tf1, n + 1)
        t_end = 1.5

        title = scene_title(
            "Stochastic optimal control: robust path constraint",
            f"time-optimal loop around 2 obstacles, motor noise magnitude {noise:g}, SocpType.COLLOCATION",
        )
        self.play(FadeIn(title), run_time=0.5)

        # ------------------------------------------------------------------------------------------ xy plane
        sc = 0.93
        ax_xy = Axes(
            x_range=[-2.0, 2.4, 1],
            y_range=[-2.3, 3.1, 1],
            x_length=4.4 * sc,
            y_length=5.4 * sc,
            tips=False,
            axis_config={"color": GRAY_D, "stroke_width": 1.5, "include_ticks": False},
        ).move_to([-4.6, -0.3, 0])
        unit = np.linalg.norm(ax_xy.c2p(1, 0) - ax_xy.c2p(0, 0))
        obstacles = VGroup()
        for i in range(2):
            x, y = superellipse(i)
            pts = [ax_xy.c2p(a, b) for a, b in zip(x, y)]
            obstacles.add(
                VMobject(stroke_color=C_OBS, stroke_width=2, fill_color=C_OBS, fill_opacity=0.55).set_points_as_corners(
                    pts
                )
            )
        lab_xy = Text("position of the mass point (m)", font_size=18, color=GRAY_B).next_to(ax_xy, UP, buff=0.05)

        def ellipses(q, P, color, bad=None):
            grp = VGroup()
            for k in range(q.shape[1]):
                w, v = np.linalg.eigh(P[k])
                ang = np.arctan2(v[1, 1], v[0, 1])
                el = Ellipse(width=2 * np.sqrt(w[1]) * unit, height=2 * np.sqrt(w[0]) * unit)
                col = C_BAD if (bad is not None and bad[k] < 0.99) else color
                el.rotate(ang).move_to(ax_xy.c2p(*q[:, k])).set_stroke(col, 1.5).set_fill(col, 0.25)
                grp.add(el)
            return grp

        # ------------------------------------------------------------------------------------------ clearance plot
        ax_c = make_axes([3.05, -2.75, 0], 6.0, 1.55, [0, t_end], [0, Z_MAX], 0.5, 1)
        lab_c = Text("distance to obstacle in sigmas, z = h / sqrt(dh P dh')", font_size=17, color=GRAY_B)
        lab_c.next_to(ax_c.get_y_axis(), UP, buff=0.06).align_to(ax_c.get_y_axis(), LEFT)
        zero = DashedLine(ax_c.c2p(0, 1), ax_c.c2p(t_end, 1), color=GRAY_B).set_stroke(width=2)
        gam = Text("gamma = 1", font_size=15, color=GRAY_B).next_to(ax_c.c2p(t_end, 1), UP, buff=0.04)
        gam.align_to(ax_c.c2p(t_end, 1), RIGHT)
        deco = VGroup(
            lab_c, gam, time_label(ax_c), x_ticks(ax_c, [0, 0.5, 1.0, 1.5], "{:g}"), y_ticks(ax_c, [0, 1, 2, 3])
        )

        # ------------------------------------------------------------------------------------------ code
        W = WHITE
        setup = code_block(
            [
                (0, 'socp_type = SocpType.COLLOCATION(polynomial_degree=5, method="legendre")', W),
                (0, "bio_model = StochasticMassPointDynamicsModel(", W),
                (1, "problem_type=socp_type,", W),
                (1, f"motor_noise_magnitude=np.array([{noise:g}, {noise:g}]))", YELLOW_C),
                (0, "phase_transitions.add(PhaseTransitionFcn.COVARIANCE_CYCLIC)", W),
                (0, "socp = StochasticOptimalControlProgram(bio_model, 40, 4, problem_type=socp_type, ...)", W),
            ]
        )
        cap1 = Text("SOCP: model, noise, covariance P as a control", font_size=17, color=GRAY_B)
        g1 = VGroup(cap1, setup).arrange(DOWN, aligned_edge=LEFT, buff=0.12)

        flag_false = code("is_robustified=False", 14.5, C_NOM)
        flag_true = code("is_robustified=True", 14.5, C_ROB)
        cons_head = code("constraints.add(path_constraint, node=Node.ALL,", 14.5, W)
        cons_a = code("min_bound=0, max_bound=cas.inf,", 14.5, W)
        cons_b = VGroup(cons_a, flag_false).arrange(RIGHT, buff=0.15)
        cons = VGroup(cons_head, cons_b).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        cons_b.shift(RIGHT * 0.3)
        inner = code_block(
            [
                (0, "if is_robustified:  # inside path_constraint, gamma = 1", W),
                (1, "dh_dx = cas.jacobian(h, controller.states.cx)", W),
                (1, "safe_guard = gamma * cas.sqrt(dh_dx @ cov @ dh_dx.T)", W),
                (1, "out -= safe_guard", W),
            ]
        )
        cap2 = Text("h(q) >= 0 outside the obstacle, at every node", font_size=17, color=GRAY_B)
        g2 = VGroup(cap2, cons, inner).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        panel = VGroup(g1, g2).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        fit(panel, 6.5)
        panel.move_to([CODE_X0 + 0.25, 2.45, 0], aligned_edge=UL)
        cap3 = Text(
            "Bioptim adds ConstraintFcn.STOCHASTIC_COVARIANCE_MATRIX_CONTINUITY_COLLOCATION between nodes",
            font_size=13,
            color=GRAY_B,
        )
        fit(cap3, 6.5)
        cap3.next_to(panel, DOWN, buff=0.12, aligned_edge=LEFT)

        self.play(
            FadeIn(VGroup(ax_xy, lab_xy)),
            FadeIn(obstacles),
            FadeIn(g1),
            FadeIn(ax_c),
            FadeIn(zero),
            FadeIn(deco),
            run_time=0.8,
        )
        self.play(FadeIn(g2), FadeIn(cap3), run_time=0.6)

        def legend_item(color, text, size=15.5):
            mark = Line(LEFT * 0.3, RIGHT * 0.3, color=color).set_stroke(width=7)
            return VGroup(mark, Text(text, font_size=size)).arrange(RIGHT, buff=0.12)

        # ------------------------------------------------------------------------------------------ solution 1
        loop0 = poly(ax_xy, d["nonrobust_q"][0], d["nonrobust_q"][1], C_NOM, 4)
        ell0 = ellipses(q0, P0, C_NOM, hs0)
        curve0 = poly(ax_c, t0, np.minimum(hs0, Z_MAX), C_NOM, 4)
        box = SurroundingRectangle(flag_false, color=YELLOW, buff=0.05, stroke_width=2.5)
        n_bad = int(np.sum(hs0 < 0.99))
        row0 = legend_item(
            C_NOM,
            f"mean only: min {hs0.min():.2f} sigma, {n_bad}/{n + 1} nodes overlap (red)",
        )
        row0.move_to([-6.85, -3.2, 0], aligned_edge=LEFT)
        self.play(Create(box), Create(loop0), FadeIn(row0), run_time=1.2)
        self.play(
            LaggedStart(*[FadeIn(e) for e in ell0], lag_ratio=0.03), Create(curve0), run_time=2.2, rate_func=linear
        )
        self.wait(1.2)

        # ------------------------------------------------------------------------------------------ solution 2
        loop1 = poly(ax_xy, d["robust_q"][0], d["robust_q"][1], C_ROB, 4)
        ell1 = ellipses(q1, P1, C_ROB)
        curve1 = poly(ax_c, t1, np.minimum(hs1, Z_MAX), C_ROB, 4)
        row1 = legend_item(
            C_ROB,
            f"robustified: min {hs1.min():.2f} sigma, loop {tf0:.3f} -> {tf1:.3f} s (+{100 * (tf1 / tf0 - 1):.1f} %)",
        )
        row1.move_to([-6.85, -3.5, 0], aligned_edge=LEFT)
        box2 = SurroundingRectangle(flag_true, color=YELLOW, buff=0.05, stroke_width=2.5)
        flag_true.move_to(flag_false, aligned_edge=LEFT)
        box2.move_to(flag_true)
        self.play(
            ReplacementTransform(flag_false, flag_true),
            ReplacementTransform(box, box2),
            loop0.animate.set_stroke(opacity=0.3),
            ell0.animate.set_stroke(opacity=0.15).set_fill(opacity=0.05),
            FadeIn(row1),
            run_time=0.8,
        )
        self.play(Create(loop1), run_time=0.9)
        self.play(
            LaggedStart(*[FadeIn(e) for e in ell1], lag_ratio=0.03), Create(curve1), run_time=2.2, rate_func=linear
        )
        foot = Text(
            f"2 real IPOPT solves, both converged (status 0, {int(d['nonrobust_iterations'])} and "
            f"{int(d['robust_iterations'])} iterations)",
            font_size=15,
            color=GRAY_B,
        )
        foot.move_to([-6.85, -3.8, 0], aligned_edge=LEFT)
        self.play(FadeIn(foot), run_time=0.4)
        self.wait(2.5)

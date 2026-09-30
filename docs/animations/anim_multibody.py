"""
Manim CE scene: MultiBiorbdModel (here MultiTorqueBiorbdModel), two independent bodies optimised in ONE OCP.
REAL bioptim / IPOPT solve stored in ``data/multibody_results.npz`` (see ``generate_multibody_data.py``):
body A = one pendulum link (1 DoF), body B = double pendulum (2 DoF); the tips must meet at the last node.

Scene: MultiBody (about 18 s).  Render (from docs/animations):  manim render -qh anim_multibody.py MultiBody
"""

import numpy as np
from manim import *

from features_scenes import CODE_W, DATA_DIR, M, code, fit, make_axes, scene_title, time_label, x_ticks

C_A = BLUE_C
C_B = ORANGE
CODE_X0 = 0.15
W = WHITE
PIV_A, PIV_B = -0.8, 0.8
LA, LB1, LB2 = 1.0, 0.6, 0.6


def caption(text, size=19, color=GRAY_B):
    return Text(text, font_size=size, color=color)


def code_block(lines, size=16):
    block = VGroup(*[code(text, size, color) for _, text, color in lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
    for line, (level, _, _) in zip(block, lines):
        line.shift(RIGHT * 0.3 * level)
    return block


def joints(q):
    """Planar forward kinematics (y, z) of both bodies, rotation about x: point (0, 0, -L) -> (L sin q, -L cos q)."""
    a = np.array([PIV_A + LA * np.sin(q[0]), -LA * np.cos(q[0])])
    e = np.array([PIV_B + LB1 * np.sin(q[1]), -LB1 * np.cos(q[1])])
    t = e + np.array([LB2 * np.sin(q[1] + q[2]), -LB2 * np.cos(q[1] + q[2])])
    return a, e, t


class MultiBody(Scene):
    def construct(self):
        d = np.load(DATA_DIR / "multibody_results.npz")
        t, q, qdot, tau = d["t"], d["q"], d["qdot"], d["tau"]
        gap, n, T = d["gap"], int(d["n"]), float(d["T"])
        idx0, idx1 = list(d["idx_q0"]), list(d["idx_q1"])
        nq = q.shape[0]
        # dense q so the drawing is smooth; forward kinematics checked against the markers computed by biorbd
        tf = np.linspace(0, T, 240)
        qf = np.array([np.interp(tf, t, q[i]) for i in range(nq)])
        kin = [joints(qf[:, k]) for k in range(len(tf))]
        for k in range(n + 1):
            a, _, tp = joints(q[:, k])
            assert np.allclose(a, d["markers"][k][0][1:], atol=1e-6)
            assert np.allclose(tp, d["markers"][k][2][1:], atol=1e-6)

        title = scene_title(
            "MultiBiorbdModel: two bodies, one problem", "each body keeps its own model; q, qdot and tau are stacked"
        )
        self.play(FadeIn(title), run_time=0.4)

        # ---------------------------------------------------------------- drawing (left)
        ox, oy, sc = -3.55, 1.55, 2.0  # world (y, z) -> screen

        def P(p):
            return np.array([ox + sc * p[0], oy + sc * p[1], 0])

        floor_line = Line(P([-1.3, 0]), P([1.3, 0]), color=GRAY_D, stroke_width=2)
        piv = VGroup(*[Dot(P([y, 0]), radius=0.07, color=GRAY_B) for y in (PIV_A, PIV_B)])
        lab_a = Text("body A", font_size=20, color=C_A).next_to(P([PIV_A, 0]), UP, buff=0.12)
        lab_b = Text("body B", font_size=20, color=C_B).next_to(P([PIV_B, 0]), UP, buff=0.12)
        tr = ValueTracker(0)

        def kidx():
            return int(round(tr.get_value()))

        def node_of_frame():
            return min(int(round(tr.get_value() * n / (len(tf) - 1))), n)

        def bodies():
            a, e, tp = kin[kidx()]
            return VGroup(
                Line(P([PIV_A, 0]), P(a), color=C_A, stroke_width=9),
                Line(P([PIV_B, 0]), P(e), color=C_B, stroke_width=9),
                Line(P(e), P(tp), color=C_B, stroke_width=9),
                Dot(P(a), radius=0.09, color=W),
                Dot(P(tp), radius=0.09, color=W),
            )

        body_mob = always_redraw(bodies)
        trace_a = always_redraw(
            lambda: VMobject(color=C_A, stroke_width=2, stroke_opacity=0.7).set_points_as_corners(
                [P(kin[i][0]) for i in range(max(kidx(), 1) + 1)]
            )
        )
        trace_b = always_redraw(
            lambda: VMobject(color=C_B, stroke_width=2, stroke_opacity=0.7).set_points_as_corners(
                [P(kin[i][2]) for i in range(max(kidx(), 1) + 1)]
            )
        )

        # ---------------------------------------------------------------- code (right)
        cap1 = caption("Bioptim code")
        code1 = code_block(
            [
                (0, "bio_model = MultiTorqueBiorbdModel(", W),
                (1, "(model_a_path, model_b_path))", W),
            ]
        )
        cap2 = caption("stacked layout of q (model.variable_index)")
        code2 = code_block(
            [
                (0, f"variable_index('q', 0) = range({idx0[0]}, {idx0[-1] + 1})", C_A),
                (0, f"variable_index('q', 1) = range({idx1[0]}, {idx1[-1] + 1})", C_B),
            ]
        )
        panel = VGroup(cap1, code1, cap2, code2).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
        panel[2].shift(DOWN * 0.1)
        fit(panel, CODE_W)
        panel.move_to([CODE_X0, 2.3, 0], aligned_edge=UL)

        # one node of x = [q | qdot] and u = [tau], values read from the solution
        cell_w, cell_h = 0.7, 0.5
        n_cells = nq * 3
        names = [f"q_{i}" for i in range(nq)] + [f"qd_{i}" for i in range(nq)] + [f"tau_{i}" for i in range(nq)]
        owner = [0 if i in idx0 else 1 for i in range(nq)] * 3
        row = VGroup(
            *[
                Rectangle(width=cell_w, height=cell_h, stroke_color=C_A if owner[j] == 0 else C_B, stroke_width=2.5)
                for j in range(n_cells)
            ]
        ).arrange(RIGHT, buff=0.0)
        for j in (nq, 2 * nq):
            row[j:].shift(RIGHT * 0.2)
        row.move_to([CODE_X0 + row.width / 2 + 0.05, -0.6, 0])
        head = VGroup(
            *[Text(nm, font_size=15, color=GRAY_B).next_to(row[j], UP, buff=0.06) for j, nm in enumerate(names)]
        )
        lab_x = Text("x_k = [q | qdot]", font_size=16, color=GRAY_B).next_to(row[0], DOWN, buff=0.12, aligned_edge=LEFT)
        lab_u = Text("u_k = tau", font_size=16, color=GRAY_B).next_to(row[2 * nq], DOWN, buff=0.12, aligned_edge=LEFT)
        cap3 = caption("one node of the decision vector (A blue, B orange)", 17).move_to(
            [CODE_X0, 0.15, 0], aligned_edge=LEFT
        )

        def vals_row():
            k = node_of_frame()
            v = [f"{x:+.2f}" for x in list(q[:, k]) + list(qdot[:, k])]
            v += [f"{x:+.2f}" for x in tau[:, k]] if k < n else ["-"] * nq  # no control at the last node
            return VGroup(*[Text(x, font_size=15, color=W).move_to(row[j].get_center()) for j, x in enumerate(v)])

        vals = always_redraw(vals_row)
        node_lab = always_redraw(
            lambda: Text(f"node k = {node_of_frame()} of {n}", font_size=17, color=GRAY_A).move_to(
                [CODE_X0 + 5.6, -1.4, 0]
            )
        )

        self.play(FadeIn(floor_line), FadeIn(piv), FadeIn(lab_a), FadeIn(lab_b), FadeIn(body_mob), run_time=0.6)
        self.play(FadeIn(panel), run_time=0.7)
        self.play(FadeIn(VGroup(row, head, cap3, lab_x, lab_u, vals, node_lab)), run_time=0.7)
        self.wait(1.0)

        # ---------------------------------------------------------------- coupling: constraint + gap plot
        cap4 = caption("Bioptim code: coupling between the two models (last node)")
        code4 = code_block(
            [
                (0, "constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS,", W),
                (1, 'node=Node.END, first_marker="A_tip",', W),
                (1, 'second_marker="B_tip")', W),
            ]
        )
        panel2 = VGroup(cap4, code4).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
        fit(panel2, CODE_W)
        panel2.move_to([CODE_X0, 2.3, 0], aligned_edge=UL)
        self.play(FadeOut(panel), FadeIn(panel2), run_time=0.5)

        ax = make_axes([-3.55, -2.25, 0], 5.6, 1.6, [0, T], [0, 2.2], 0.5, 0.5)
        lab = Text("distance A_tip to B_tip (m)", font_size=18, color=GRAY_B).next_to(ax.get_y_axis(), UP, buff=0.08)
        lab.align_to(ax.get_y_axis(), LEFT)
        xt = x_ticks(ax, [0, 0.5, 1.0, 1.5], "{:g}")
        yt = VGroup(
            *[Text(f"{v:g}", font_size=16, color=GRAY_B).next_to(ax.c2p(0, v), LEFT, buff=0.08) for v in (0, 1)]
        )
        tlab = time_label(ax)
        self.play(FadeIn(VGroup(ax, lab, xt, yt, tlab)), FadeIn(trace_a), FadeIn(trace_b), run_time=0.5)

        gap_f = np.array([np.linalg.norm(k_[0] - k_[2]) for k_ in kin])
        gcurve = always_redraw(
            lambda: VMobject(color=YELLOW_C, stroke_width=5).set_points_as_corners(
                [ax.c2p(tf[i], gap_f[i]) for i in range(max(kidx(), 1) + 1)]
            )
        )
        self.add(gcurve)
        self.play(tr.animate.set_value(len(tf) - 1), run_time=8.0, rate_func=linear)
        self.wait(0.3)

        msg = VGroup(
            M(f"distance at t = 0:  {gap[0]:.2f} m", 21, W),
            M(f"distance at t = T:  {gap[-1]:.0e} m", 21, YELLOW_C),
            M(f"IPOPT status {int(d['status'])} (optimal), {int(d['iterations'])} iterations", 19, GRAY_B),
            M("the dynamics are block-diagonal: only the constraint couples A and B", 19, GRAY_B),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
        fit(msg, 5.9)
        msg.move_to([CODE_X0, -1.85, 0], aligned_edge=UL)
        self.play(FadeIn(msg), run_time=0.6)
        self.wait(2.2)

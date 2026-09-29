"""
Holonomic (closed-loop) constraints in bioptim: two single pendulums glued into a double pendulum.
Data: data/holonomic_two_pendulums.npz, REAL IPOPT solution of bioptim/examples/toy_examples/holonomic_constraints/two_pendulums.py
(generate_holonomic_data.py). One scene, ~20 s.

Render (from docs/animations):  manim render -qh anim_holonomic.py HolonomicDoublePendulum
"""

import numpy as np
from manim import *

from features_scenes import (  # helpers only (also sets the default fonts)
    CODE_W,
    DATA_DIR,
    M,
    axis_label,
    code,
    code_panel,
    fit,
    make_axes,
    poly,
    scene_title,
    time_label,
)

C_U = BLUE_C  # independent coordinates
C_V = ORANGE  # dependent coordinates
C_RES = GREEN_C


class HolonomicDoublePendulum(Scene):
    def construct(self):
        d = np.load(DATA_DIR / "holonomic_two_pendulums.npz")
        t, q, joint1, tip, res = d["t"], d["q"], d["joint1"][:, 1:], d["tip"][:, 1:], d["residual"][:, 1:]
        T = float(t[-1])
        max_res = float(np.abs(res).max())

        title = scene_title("Holonomic constraint", "two pendulums, one closed kinematic condition")
        self.play(FadeIn(title), run_time=0.6)

        # ------------------------------------------------------------ beat 1: partition q = (u | v) + code
        parts = VGroup(M("<b>θ₀</b>", 30, C_U), M("<b>y₁  z₁</b>", 30, C_V), M("<b>θ₁</b>", 30, C_U)).arrange(
            RIGHT, buff=0.7
        )
        q_lab = M("q =", 30).next_to(parts, LEFT, buff=0.3)
        row = VGroup(q_lab, parts).move_to([-3.4, 1.3, 0])
        u_lab = M("independent  u  (rotations)", 22, C_U).move_to([-3.4, 0.35, 0])
        v_lab = M("dependent  v  (translations of segment 1)", 22, C_V).next_to(u_lab, DOWN, buff=0.2)
        cons = M("v is solved from  φ(q) = marker₁ − marker₃ = 0", 22).next_to(v_lab, DOWN, buff=0.35)
        cons_box = SurroundingRectangle(cons, color=GRAY_B, buff=0.12, stroke_width=2)
        for grp in (VGroup(u_lab, v_lab, cons), row):
            fit(grp, 6.2)
        code_lines = [
            (0, "holonomic_constraints = HolonomicConstraintsList()", WHITE),
            (0, "holonomic_constraints.add(", WHITE),
            (1, '"holonomic_constraints",', GRAY_B),
            (1, "HolonomicConstraintsFcn.superimpose_markers,", C_V),
            (1, 'marker_1="marker_1", marker_2="marker_3",', C_V),
            (1, "index=slice(1, 3), local_frame_index=0)", C_V),
            (0, "bio_model = HolonomicTorqueBiorbdModel(path,", WHITE),
            (1, "holonomic_constraints=holonomic_constraints,", WHITE),
            (1, "independent_joint_index=[0, 3],", C_U),
            (1, "dependent_joint_index=[1, 2])", C_V),
        ]
        panel = code_panel(code_lines, 19, top=2.35)
        self.play(FadeIn(row), run_time=0.5)
        self.play(FadeIn(u_lab), FadeIn(v_lab), run_time=0.6)
        self.play(FadeIn(cons), Create(cons_box), run_time=0.5)
        self.play(FadeIn(panel), run_time=0.8)
        self.wait(2.2)

        # ------------------------------------------------------------ beat 2: the mechanism moves, residual stays 0
        self.play(
            FadeOut(row), FadeOut(u_lab), FadeOut(v_lab), FadeOut(cons), FadeOut(cons_box), FadeOut(panel), run_time=0.5
        )

        sc = 1.2  # metres -> scene units
        origin = np.array([-3.6, 1.6, 0.0])

        def pt(p):  # (y, z) in the model frame -> scene (z is up)
            return origin + np.array([p[0] * sc, p[1] * sc, 0.0])

        pivot = Dot(origin, radius=0.09, color=GRAY_B)
        ceiling = Line(origin + LEFT * 0.5, origin + RIGHT * 0.5, color=GRAY_B, stroke_width=3)
        tr = ValueTracker(0)

        def idx():
            return min(int(round(tr.get_value() / T * (len(t) - 1))), len(t) - 1)

        seg0 = always_redraw(lambda: Line(origin, pt(joint1[idx()]), color=C_U, stroke_width=9))
        seg1 = always_redraw(lambda: Line(pt(joint1[idx()]), pt(tip[idx()]), color=C_U, stroke_width=9))
        # the two markers that the constraint superimposes: marker_1 (tip of seg 0) and marker_3 (origin of seg 1)
        m1 = always_redraw(lambda: Dot(pt(joint1[idx()] - res[idx()]), radius=0.11, color=YELLOW_C))
        m3 = always_redraw(lambda: Dot(pt(joint1[idx()]), radius=0.17, color=C_V, fill_opacity=0.55, stroke_width=3))
        trace = TracedPath(lambda: pt(tip[idx()]), stroke_color=GRAY_B, stroke_width=2, stroke_opacity=0.7)
        lab_m = M("<span foreground='#FFFF00'>marker_1</span> = <span foreground='#FF862F'>marker_3</span>", 20)
        lab_m.move_to([-3.6, -1.45, 0])
        cap = M("dependent joints (y₁, z₁) follow the rotations (θ₀, θ₁)", 20, GRAY_B).move_to([-3.6, -2.35, 0])
        fit(cap, 6.4)
        self.play(
            FadeIn(pivot),
            FadeIn(ceiling),
            FadeIn(seg0),
            FadeIn(seg1),
            FadeIn(m1),
            FadeIn(m3),
            FadeIn(lab_m),
            FadeIn(cap),
            run_time=0.6,
        )
        self.add(trace)

        # residual plot (right)
        ax = make_axes([3.2, 1.0, 0], 5.0, 2.4, [0, T], [-1.0, 1.0], 0.2, 1)
        zero = DashedLine(ax.c2p(0, 0), ax.c2p(T, 0), color=GRAY_B, stroke_width=2)
        ylab = axis_label("‖marker₁ − marker₃‖  (m)", ax, C_RES)
        tl = Text("t (0 to %.1f s)" % T, font_size=16, color=GRAY_B).next_to(ax, DOWN, buff=0.15).align_to(ax, LEFT)
        rng = Text(f"axis ±1e−9 m", font_size=16, color=GRAY_B).next_to(ax, DOWN, buff=0.15).align_to(ax, RIGHT)
        norm = np.linalg.norm(res, axis=1)
        curve = poly(ax, t, np.clip(norm / 1e-9, 0, 1), C_RES, 6)  # scaled: axis top = 1e-9
        code_res = code("q, qdot, qddot, lambdas =", 18, WHITE)
        code_res2 = code("  bio_model.compute_all_states_from_u_iterative(", 18, C_V)
        code_res3 = code("      q_u, qdot_u, tau)", 18, C_U)
        cb = (
            VGroup(code_res, code_res2, code_res3)
            .arrange(DOWN, aligned_edge=LEFT, buff=0.1)
            .move_to([0.3, -2.25, 0], aligned_edge=LEFT)
        )
        fit(cb, 6.6)
        cb.move_to([0.3, -2.3, 0], aligned_edge=LEFT)
        self.play(FadeIn(ax), FadeIn(zero), FadeIn(ylab), FadeIn(tl), FadeIn(rng), FadeIn(cb), run_time=0.5)
        mover = always_redraw(
            lambda: (
                poly(ax, t[: idx() + 1], np.clip(norm[: idx() + 1] / 1e-9, 0, 1), C_RES, 6) if idx() > 0 else VMobject()
            )
        )
        self.add(mover)
        self.play(tr.animate.set_value(T), run_time=6, rate_func=linear)
        verdict = M(f"max ‖residual‖ = {max_res:.0e} m  (IPOPT status: optimal)", 22, C_RES).move_to([3.6, -1.35, 0])
        fit(verdict, 6.0)
        self.play(FadeIn(verdict), run_time=0.5)
        self.wait(2.0)

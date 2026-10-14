"""
Manim Community animation: a muscle-driven reaching OCP (bioptim/examples/toy_examples/muscle_driven_ocp/static_arm.py).
REAL data from ``data/muscle_arm.npz`` (see ``generate_muscle_data.py``): arm26 (2 dof, 6 muscles), 30 nodes, 0.5 s.

Scene MuscleReaching (~18 s): code difference TorqueBiorbdModel -> MusclesBiorbdModel, the arm reaching the target and
the 6 muscle activations (controls) staying in [0, 1].

    manim render -qh anim_muscle.py MuscleReaching
"""

import sys
from pathlib import Path

import numpy as np
from manim import *

sys.path.insert(0, str(Path(__file__).parent))
from features_scenes import (  # noqa: E402  (read-only reuse of the helpers; also sets the default fonts)
    CODE_X,
    CODE_W,
    MONO,
    band,
    code,
    code_panel,
    fit,
    hline,
    make_axes,
    scene_title,
    steps,
    time_label,
    y_ticks,
    x_ticks,
    axis_label,
)

D = np.load(Path(__file__).parent / "data" / "muscle_arm.npz")
T, ACT, NAMES = D["t"], D["act"], [str(n) for n in D["muscle_names"]]
SHO, ELB, HAND, TGT = D["shoulder"], D["elbow"], D["hand"], D["target"][0]
MUSCLE_COLORS = [RED_C, ORANGE, YELLOW_C, GREEN_C, TEAL_C, BLUE_C]
SCALE = 4.5
ARM_C = np.array([-4.6, 1.75])  # screen position of the point (0.13, -0.075) [m]


def scr(p):
    return np.array([ARM_C[0] + SCALE * (p[0] - 0.13), ARM_C[1] + SCALE * (p[1] + 0.075), 0])


def at(arr, s):
    """Linear interpolation of a (N+1, 3) trajectory at fractional node s."""
    i = int(min(np.floor(s), len(arr) - 2))
    return arr[i] + (s - i) * (arr[i + 1] - arr[i])


class MuscleReaching(Scene):
    def construct(self):
        title = scene_title("Muscle-driven optimal control: reach a target")
        self.play(FadeIn(title), run_time=0.4)
        n = ACT.shape[1]

        # ---------------- left top: the arm ----------------
        tgt = VGroup(
            Circle(radius=0.12, color=RED_C, stroke_width=4),
            Dot(radius=0.04, color=RED_C),
        ).move_to(scr(TGT))
        tgt_lab = Text("target", font_size=18, color=RED_C).next_to(tgt, RIGHT, buff=0.12)
        s = ValueTracker(0.0)

        def arm():
            k = s.get_value()
            sh, el, ha = scr(SHO[0]), scr(at(ELB, k)), scr(at(HAND, k))
            return VGroup(
                Line(sh, el, color=GRAY_A, stroke_width=10),
                Line(el, ha, color=GRAY_A, stroke_width=8),
                Dot(sh, radius=0.09, color=WHITE),
                Dot(el, radius=0.08, color=WHITE),
                Dot(ha, radius=0.09, color=YELLOW_C),
            )

        trail = VMobject(color=YELLOW_C, stroke_width=3, stroke_opacity=0.6)
        trail.add_updater(
            lambda m: m.set_points_as_corners(
                [scr(at(HAND, j)) for j in np.linspace(0, s.get_value(), 1 + max(1, int(s.get_value() * 3)))]
            )
        )
        dist = Text("", font_size=20)

        def dist_updater(m):
            d = np.linalg.norm(at(HAND, s.get_value()) - TGT) * 100
            new = Text(f"hand to target: {d:.1f} cm", font_size=20, color=YELLOW_C)
            new.move_to([-1.7, 1.6, 0])
            m.become(new)

        dist.add_updater(dist_updater)
        arm_mob = always_redraw(arm)

        # ---------------- left bottom: activations ----------------
        ax = make_axes([-3.3, -1.55, 0], 5.4, 2.5, [0, 0.5], [-0.15, 1.15], 0.1, 0.5)
        xt, yt = x_ticks(ax, [0, 0.25, 0.5]), y_ticks(ax, [0, 1])
        lab = axis_label("muscle activation a(t)", ax)
        tl = time_label(ax)
        b0, b1 = hline(ax, 0, 0.5, 0, RED_C), hline(ax, 0, 0.5, 1, RED_C)
        forbidden = VGroup(band(ax, 0, 0.5, 1, 1.15, RED_C, 0.18), band(ax, 0, 0.5, -0.15, 0, RED_C, 0.18))
        bnd_txt = Text("bounds [0, 1]", font_size=18, color=RED_C).next_to(ax.c2p(0.5, 1.15), LEFT, buff=0.05)
        curves = [steps(ax, T, ACT[i], MUSCLE_COLORS[i], 3.5) for i in range(len(NAMES))]
        legend = VGroup(
            *[
                VGroup(Line(ORIGIN, RIGHT * 0.25, color=c, stroke_width=5), Text(nm, font_size=16, color=c)).arrange(
                    RIGHT, buff=0.08
                )
                for nm, c in zip(NAMES, MUSCLE_COLORS)
            ]
        ).arrange_in_grid(rows=2, cols=3, buff=(0.3, 0.1), col_alignments="lll")
        legend.move_to([CODE_X + 2.6, -1.9, 0])

        # ---------------- right: code ----------------
        cap = Text("Bioptim code: from torque-driven to muscle-driven", font_size=20, color=GRAY_B)
        old = code("bio_model = TorqueBiorbdModel(path)", 19, RED_C)
        new = code("bio_model = MusclesBiorbdModel(path, with_residual_torque=True)", 19, GREEN_C)
        model_blk = VGroup(cap, old).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        fit(model_blk, CODE_W)
        model_blk.move_to([CODE_X, 2.45, 0], aligned_edge=UL)
        fit(new, CODE_W)
        new.move_to(old, aligned_edge=LEFT)
        rest = code_panel(
            [
                (0, "objective_functions.add(", WHITE),
                (1, "ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,", GREEN_C),
                (1, 'key="muscles")', GREEN_C),
                (0, "objective_functions.add(", WHITE),
                (1, "ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS,", ORANGE),
                (1, 'first_marker="target", second_marker="COM_hand",', ORANGE),
                (1, "weight=1000)", ORANGE),
                (0, "u_bounds = BoundsList()", WHITE),
                (0, 'u_bounds["muscles"] = [0.0] * nb_muscles, [1.0] * nb_muscles', RED_C),
            ],
            size=19,
            top=1.4,
            caption=None,
        )
        rest_lines = rest[0]
        note = Text("6 muscles: a_i(t) are the controls, tau is only a small residual", font_size=17, color=GRAY_B)
        note.move_to([CODE_X + 0.25, -3.05, 0], aligned_edge=LEFT)
        status = Text(
            f"IPOPT: converged, {int(D['iterations'])} iterations, "
            f"final error {float(D['marker_error'])*100:.1f} cm",
            font_size=17,
            color=GRAY_B,
        )
        status.move_to([CODE_X + 0.25, -3.4, 0], aligned_edge=LEFT)

        # ---------------- timeline (~18 s) ----------------
        self.play(FadeIn(model_blk, shift=UP * 0.1), run_time=0.8)
        self.wait(0.7)
        self.play(ReplacementTransform(old, new), run_time=0.9)
        self.play(FadeIn(tgt), FadeIn(tgt_lab), FadeIn(arm_mob), Write(rest_lines[:7]), run_time=1.2)
        self.play(
            FadeIn(ax),
            FadeIn(xt),
            FadeIn(yt),
            FadeIn(lab),
            FadeIn(tl),
            FadeIn(b0),
            FadeIn(b1),
            FadeIn(forbidden),
            FadeIn(bnd_txt),
            Write(rest_lines[7:]),
            FadeIn(legend),
            run_time=1.0,
        )
        self.add(trail, dist)
        self.play(
            s.animate.set_value(float(n)),
            *[Create(c) for c in curves],
            run_time=6.5,
            rate_func=linear,
        )
        trail.clear_updaters()
        dist.clear_updaters()
        self.play(FadeIn(note), FadeIn(status), run_time=0.6)
        self.wait(2.5)

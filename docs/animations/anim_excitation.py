"""
Manim CE scene: muscle excitation -> activation dynamics in bioptim, driven by a REAL solve
(``data/excitation_arm.npz``, see ``generate_excitation_data.py``). Arm26 reaching (2 dof, 6 muscles, 30 nodes,
0.5 s) with MusclesWithExcitationsBiorbdModel: the excitation e(t) is a control, the activation a(t) a state.

Render (from docs/animations):  manim render -qh anim_excitation.py ExcitationActivation
"""

import sys
from pathlib import Path

import numpy as np
from manim import *

sys.path.insert(0, str(Path(__file__).parent))
from features_scenes import (  # noqa: E402  (also sets the default fonts)
    CODE_X,
    code_panel,
    hline,
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

D = np.load(Path(__file__).parent / "data" / "excitation_arm.npz")
T, EXC, ACT = D["t"], D["exc"], D["act"]
TD, ACT_D, Q = D["t_dense"], D["act_dense"], D["q"]
NAMES = [str(n) for n in D["muscle_names"]]
M_IDX = NAMES.index("BICshort")
FT = float(D["final_time"])
C_EXC, C_ACT = GREEN_C, YELLOW_C
EVENTS = [(7, True), (23, False)]  # node where the BICshort excitation crosses 0.5 (up / down)


def clipped_steps(ax, w0, w1, color, width=4):
    pts = []
    for k in range(len(EXC[M_IDX])):
        if T[k + 1] > w0 and T[k] < w1:
            e = EXC[M_IDX][k]
            pts += [ax.c2p(max(T[k], w0), e), ax.c2p(min(T[k + 1], w1), e)]
    return VMobject(color=color, stroke_width=width).set_points_as_corners(pts)


def lag_event(k, up):
    """(t of the excitation change, t at which the dense activation crosses 0.5)."""
    a = ACT_D[M_IDX]
    i0 = int(round(T[k] / TD[1]))
    seg = a[i0 : i0 + 200]
    idx = np.where(seg >= 0.5)[0] if up else np.where(seg <= 0.5)[0]
    return T[k], TD[i0 + idx[0]]


class ExcitationActivation(Scene):
    def construct(self):
        lags = []
        for k, up in EVENTS:
            t0, t1 = lag_event(k, up)
            lags.append((t0, t1, (t1 - t0) * 1000))

        self.play(
            FadeIn(scene_title("Excitation → activation", f"MusclesWithExcitationsBiorbdModel, {NAMES[M_IDX]}")),
            run_time=0.5,
        )

        # ---------------- top left: full time course ----------------
        ax = make_axes([-3.6, 1.05, 0], 5.6, 2.25, (0, FT), (-0.1, 1.15), y_step=0.5)
        deco = VGroup(
            axis_label("BICshort", ax),
            time_label(ax),
            x_ticks(ax, [0, 0.25, 0.5], "{:g}"),
            y_ticks(ax, [0, 1]),
        )
        e_curve = steps(ax, T, EXC[M_IDX], C_EXC, 4)
        a_curve = poly(ax, TD, ACT_D[M_IDX], C_ACT, 4)
        leg = VGroup(
            Text("e(t) excitation = control", font_size=17, color=C_EXC),
            Text("a(t) activation = state", font_size=17, color=C_ACT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        leg.next_to(ax.c2p(FT, 1.15), UP, buff=0.05, aligned_edge=RIGHT)

        # ---------------- right bottom: joints ----------------
        axq = make_axes([3.45, -2.2, 0], 5.2, 1.6, (0, FT), (-1.2, 2.8), y_step=1)
        decq = VGroup(
            axis_label("joint angles q (rad)", axq),
            time_label(axq),
            x_ticks(axq, [0, 0.25, 0.5]),
            y_ticks(axq, [-1, 0, 1, 2]),
        )
        q_c = [poly(axq, T, Q[0], BLUE_C, 4), poly(axq, T, Q[1], ORANGE, 4)]
        q_leg = VGroup(Text("shoulder", font_size=15, color=BLUE_C), Text("elbow", font_size=15, color=ORANGE)).arrange(
            RIGHT, buff=0.3
        )
        q_leg.move_to(axq.c2p(0.25, 2.55))

        # ---------------- right top: code ----------------
        nm = "nb_muscles"
        panel = code_panel(
            [
                (0, "bio_model = MusclesWithExcitationsBiorbdModel(", WHITE),
                (1, "path, with_residual_torque=True)", WHITE),
                (0, "# activation a(t) = STATE", C_ACT),
                (0, f'x_bounds["muscles"] = [0.0] * {nm}, [1.0] * {nm}', C_ACT),
                (0, "# excitation e(t) = CONTROL", C_EXC),
                (0, f'u_bounds["muscles"] = [0.0] * {nm}, [1.0] * {nm}', C_EXC),
                (0, "objective_functions.add(", WHITE),
                (1, 'ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscles")', C_EXC),
            ],
            size=19,
            top=2.35,
        )
        info = VGroup(
            Text("da/dt = f(e, a): biorbd activationDot (De Groote)", font_size=16, color=GRAY_A),
            Text("nominal τ_act = 10 ms, τ_deact = 40 ms", font_size=16, color=GRAY_A),
            Text("vs MusclesBiorbdModel: a is the control, no lag", font_size=16, color=GRAY_B),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.05)
        info.move_to([CODE_X, -0.55, 0], aligned_edge=LEFT)
        status = place(
            Text(
                f"IPOPT: converged, {int(D['iterations'])} iterations, final hand-target error "
                f"{float(D['marker_error'])*100:.1f} cm",
                font_size=15,
                color=GRAY_B,
            ),
            CODE_X,
            -3.55,
        )

        self.play(Create(ax), FadeIn(deco), FadeIn(leg), Write(panel), run_time=1.3)
        self.play(FadeIn(info), Create(axq), FadeIn(decq), FadeIn(q_leg), run_time=0.7)
        self.play(Create(e_curve), Create(a_curve), Create(q_c[0]), Create(q_c[1]), run_time=4.5, rate_func=linear)

        # ---------------- bottom left: zoom on one step ----------------
        axz_c = [-3.6, -2.35, 0]

        def zoom(i):
            t0, t1, lag = lags[i]
            up = EVENTS[i][1]
            w0, w1 = t0 - 0.03, t0 + 0.09
            az = make_axes(axz_c, 5.6, 1.7, (w0, w1), (-0.1, 1.15), y_step=0.5)
            cur_e = clipped_steps(az, w0, w1, C_EXC, 5)
            m = (TD >= w0) & (TD <= w1)
            cur_a = poly(az, TD[m], ACT_D[M_IDX][m], C_ACT, 5)
            nodes = VGroup(
                *[Dot(az.c2p(T[k], ACT[M_IDX][k]), radius=0.05, color=WHITE) for k in range(len(T)) if w0 <= T[k] <= w1]
            )
            y_half = hline(az, w0, w1, 0.5, GRAY_C)
            g0 = DashedLine(az.c2p(t0, -0.1), az.c2p(t0, 1.15), color=C_EXC, stroke_width=2)
            g1 = DashedLine(az.c2p(t1, -0.1), az.c2p(t1, 1.15), color=C_ACT, stroke_width=2)
            arrow = DoubleArrow(az.c2p(t0, 0.5), az.c2p(t1, 0.5), buff=0, stroke_width=3, tip_length=0.12, color=WHITE)
            txt = Text(f"lag to a = 0.5: {lag:.0f} ms", font_size=18, color=WHITE)
            txt.move_to(az.c2p(t1 + 0.004, 0.25 if up else 0.75), aligned_edge=LEFT)
            ttl = Text(
                "e steps up (0 → 0.67, then 1)" if up else "e steps down (1 → 0)",
                font_size=16,
                color=C_EXC,
            )
            ttl.next_to(az.c2p(w0, 1.15), UP, buff=0.06).align_to(az.get_y_axis(), LEFT)
            tk = x_ticks(az, [round(w0, 3), round(t0, 3), round(w1, 3)], "{:.3f}")
            tl = time_label(az)
            yt = y_ticks(az, [0, 1])
            grp = VGroup(az, y_half, cur_e, cur_a, nodes, g0, g1, arrow, txt, ttl, tk, tl, yt)
            b = Rectangle(
                width=abs(ax.c2p(w1, 0)[0] - ax.c2p(w0, 0)[0]),
                height=abs(ax.c2p(0, 1.15)[1] - ax.c2p(0, -0.1)[1]),
                color=WHITE,
                stroke_width=2,
            ).move_to((ax.c2p(w0, 1.15) + ax.c2p(w1, -0.1)) / 2)
            return grp, b

        z0, b0 = zoom(0)
        self.play(Create(b0), FadeIn(z0), run_time=1.0)
        self.wait(2.0)
        z1, b1 = zoom(1)
        self.play(ReplacementTransform(z0, z1), ReplacementTransform(b0, b1), run_time=1.2)
        self.wait(2.0)
        self.play(FadeIn(status), run_time=0.5)
        self.wait(1.5)

"""
Manim CE scene: full muscle-driven reach A -> B (arm26, 2 dof, 6 muscles, rest at both ends) compared with the
torque-driven reach of the same model. REAL IPOPT solves stored in ``data/muscfull_arm.npz`` (see
``generate_muscfull_data.py``): N = 30, T = 0.8 s, RK4, both solves IPOPT status 0.

Scenes (about 16 s in total), render from docs/animations:
    manim render -qh anim_muscfull.py MuscFullPaths
    manim render -qh anim_muscfull.py MuscFullActivations
"""

import sys
from pathlib import Path

import numpy as np
from manim import *

sys.path.insert(0, str(Path(__file__).parent))
from features_scenes import (  # noqa: E402  (read-only reuse of the helpers; also sets the default fonts)
    CODE_W,
    CODE_X,
    DATA_DIR,
    axis_label,
    code_panel,
    fit,
    make_axes,
    poly,
    scene_title,
    steps,
    x_ticks,
    y_ticks,
)

D = np.load(DATA_DIR / "muscfull_arm.npz")
T = D["t"]
N = len(T) - 1
DT = float(T[-1] - T[0]) / N
NAMES = [str(n) for n in D["muscle_names"]]
ACT = D["mus_act"]  # (6, N)
C_MUS, C_TOR = GREEN_C, ORANGE
MUSCLE_COLORS = [RED_C, ORANGE, YELLOW_C, GREEN_C, TEAL_C, BLUE_C]
FLEX, EXT = [1, 2, 5], [0, 3, 4]  # BIClong, BICshort, BRA  /  TRIlong, TRIlat, TRImed


def legend_item(color, text):
    return VGroup(
        Line(ORIGIN, RIGHT * 0.35, color=color, stroke_width=6), Text(text, font_size=20, color=color)
    ).arrange(RIGHT, buff=0.12)


class MuscFullPaths(Scene):
    def construct(self):
        title = scene_title(
            "Muscles or torques: the same reach", f"arm26, N = {N}, T = {T[-1]:.1f} s, at rest in A and in B"
        )
        self.play(FadeIn(title), run_time=0.4)

        # ------------------------------------------------ hand paths (left)
        x0, x1, y0, y1 = -0.30, 0.40, -0.48, 0.02
        w = 5.8
        h = w * (y1 - y0) / (x1 - x0)
        ax = make_axes([-3.7, -0.75, 0], w, h, [x0, x1], [y0, y1])
        box = SurroundingRectangle(ax, buff=0.0, color=GRAY_D, stroke_width=2)
        lab = Text("hand (COM_hand) in the plane of the arm, m", font_size=18, color=GRAY_B).next_to(box, UP, buff=0.1)
        mus = D["mus_hand"][:, :2]
        tor = D["tor_hand"][:, :2]
        p_mus = poly(ax, mus[:, 0], mus[:, 1], C_MUS, 6)
        p_tor = poly(ax, tor[:, 0], tor[:, 1], C_TOR, 6)
        a_dot = Dot(ax.c2p(*mus[0]), radius=0.11, color=WHITE)
        b_dot = Dot(ax.c2p(*mus[-1]), radius=0.11, color=WHITE)
        a_lab = Text("A", font_size=24).next_to(a_dot, DOWN, buff=0.12)
        b_lab = Text("B", font_size=24).next_to(b_dot, RIGHT, buff=0.12)
        lg = VGroup(legend_item(C_MUS, "muscles"), legend_item(C_TOR, "torques")).arrange(
            DOWN, aligned_edge=LEFT, buff=0.1
        )
        lg.move_to(ax.c2p(-0.29, -0.06), aligned_edge=UL)

        # ------------------------------------------------ code (right)
        panel = code_panel(
            [
                (0, "MusclesBiorbdModel(path, with_residual_torque=False)", C_MUS),
                (0, "TorqueBiorbdModel(path)", C_TOR),
                (0, 'x_bounds["q"][:, 0] = qA', WHITE),
                (0, 'x_bounds["q"][:, -1] = qB', WHITE),
                (0, 'x_bounds["qdot"][:, [0, -1]] = 0', WHITE),
                (0, "objective_functions.add(", WHITE),
                (1, "ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,", GREEN_C),
                (1, 'key="muscles")    # or key="tau"', GREEN_C),
            ],
            size=17,
            top=2.3,
            caption="Bioptim code (same OCP, only the model and the key change)",
        )
        self.play(FadeIn(VGroup(ax, box, lab, a_dot, b_dot, a_lab, b_lab, lg)), FadeIn(panel), run_time=0.7)

        # both paths grow together with a moving hand
        s = ValueTracker(0)

        def hand_dot(path, color):
            def upd(m):
                k = s.get_value()
                i = int(min(np.floor(k), N - 1))
                p = path[i] + (k - i) * (path[i + 1] - path[i])
                m.move_to(ax.c2p(*p))

            d = Dot(radius=0.12, color=color)
            d.add_updater(upd)
            return d

        dm, dtq = hand_dot(mus, C_MUS), hand_dot(tor, C_TOR)
        self.add(dm, dtq)
        self.play(
            Create(p_mus, rate_func=linear),
            Create(p_tor, rate_func=linear),
            s.animate.set_value(N),
            run_time=3.2,
            rate_func=linear,
        )
        dm.clear_updaters()
        dtq.clear_updaters()

        def length(p):
            return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())

        direct = float(np.linalg.norm(mus[-1] - mus[0]))
        st = (
            f"IPOPT status {int(D['mus_status'])} ({int(D['mus_iters'])} it) / "
            f"status {int(D['tor_status'])} ({int(D['tor_iters'])} it)"
        )
        msg = VGroup(
            Text(f"path length  muscles {length(mus):.2f} m,  torques {length(tor):.2f} m", font_size=20),
            Text(f"straight line A-B: {direct:.2f} m; no hand-path cost, so both detour", font_size=18, color=GRAY_B),
            Text(st, font_size=18, color=GRAY_B),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
        fit(msg, CODE_W)
        msg.move_to([CODE_X, -0.9, 0], aligned_edge=UL)
        self.play(FadeIn(msg), run_time=0.5)
        self.wait(1.6)


class MuscFullActivations(Scene):
    def construct(self):
        title = scene_title("Inside the muscle solution", "activations and the joint torque they produce")
        self.play(FadeIn(title), run_time=0.4)
        s = ValueTracker(0.0)  # node index, 0..N-1 (controls are constant on each interval)

        # ------------------------------------------------ activation bars (left)
        base_y, hgt, bx0, gap = -1.8, 3.0, -6.3, 0.85
        floor = Line([bx0 - 0.4, base_y, 0], [bx0 + 5 * gap + 0.4, base_y, 0], color=GRAY_B, stroke_width=2)
        top_line = DashedLine(
            [bx0 - 0.4, base_y + hgt, 0], [bx0 + 5 * gap + 0.4, base_y + hgt, 0], color=RED_C, stroke_width=2
        )
        top_lab = Text("a = 1", font_size=16, color=RED_C).next_to(top_line, UP, buff=0.05).align_to(top_line, LEFT)
        names = VGroup(
            *[
                Text(n, font_size=15, color=c).move_to([bx0 + i * gap, base_y - 0.25, 0])
                for i, (n, c) in enumerate(zip(NAMES, MUSCLE_COLORS))
            ]
        )
        cap = Text("muscle activations a(t), the controls", font_size=19, color=GRAY_B).move_to(
            [bx0 - 0.4, 2.05, 0], aligned_edge=LEFT
        )

        def k_now():
            return int(min(round(s.get_value()), N - 1))

        def bars():
            k = k_now()
            g = VGroup()
            for i, c in enumerate(MUSCLE_COLORS):
                hh = max(ACT[i, k] * hgt, 0.02)
                g.add(
                    Rectangle(width=0.55, height=hh, stroke_width=0, fill_color=c, fill_opacity=0.9).move_to(
                        [bx0 + i * gap, base_y + hh / 2, 0]
                    )
                )
                pk = base_y + ACT[i, : k + 1].max() * hgt
                g.add(Line([bx0 + i * gap - 0.32, pk, 0], [bx0 + i * gap + 0.32, pk, 0], color=WHITE, stroke_width=3))
            return g

        bars_mob = always_redraw(bars)
        clock = always_redraw(
            lambda: Text(f"t = {T[k_now()]:.2f} s", font_size=20, color=WHITE).move_to(
                [bx0 + 4.1, 2.05, 0], aligned_edge=LEFT
            )
        )

        # ------------------------------------------------ joint torques (right)
        axes, curves = [], []
        specs = [("shoulder torque, N m", 0, (-17, 11)), ("elbow torque, N m", 1, (-5, 9))]
        for row, (name, j, (lo, hi)) in enumerate(specs):
            cy = 0.55 - row * 2.3
            ax = make_axes([2.4, cy, 0], 5.0, 1.55, [0, T[-1]], [lo, hi])
            zero = Line(ax.c2p(0, 0), ax.c2p(T[-1], 0), color=GRAY_D, stroke_width=2)
            lab = axis_label(name, ax)
            yt = y_ticks(ax, [-15, 0, 10] if row == 0 else [0, 5])
            xt = x_ticks(ax, [0, 0.4, 0.8]) if row == 1 else VGroup()
            c1 = steps(ax, T, D["mus_tau"][j], C_MUS, 4)
            c2 = steps(ax, T, D["tor_tau"][j], C_TOR, 4)
            cur = always_redraw(
                lambda ax=ax, lo=lo, hi=hi: Line(
                    ax.c2p(T[k_now()], lo), ax.c2p(T[k_now()], hi), color=WHITE, stroke_width=2, stroke_opacity=0.6
                )
            )
            axes.append(VGroup(ax, zero, lab, yt, xt, cur))
            curves.append((c1, c2))
        leg = VGroup(
            Text("muscle-implied", font_size=17, color=C_MUS),
            Text("tau of the torque-driven solve", font_size=17, color=C_TOR),
        ).arrange(RIGHT, buff=0.35)
        leg.move_to([2.4, -3.0, 0])
        note = Text("the two trajectories differ (previous scene)", font_size=15, color=GRAY_B).move_to([2.4, -3.45, 0])
        code_lines = code_panel(
            [
                (0, "muscle_joint_torque = bio_model.muscle_joint_torque()", C_MUS),
                (0, "tau = muscle_joint_torque(act, q, qdot, [])", C_MUS),
            ],
            size=16,
            top=3.0,
        )
        code_lines.move_to([-0.2, 2.15, 0], aligned_edge=LEFT)
        self.play(FadeIn(VGroup(floor, top_line, top_lab, names, cap, *axes, leg, note, code_lines)), run_time=0.6)
        self.add(bars_mob, clock)
        self.play(
            s.animate.set_value(N - 1),
            *[Create(c, rate_func=linear) for pair in curves for c in pair],
            run_time=4.5,
            rate_func=linear,
        )

        # ------------------------------------------------ readouts computed from the data
        peak = int(np.argmax(ACT.max(axis=1)))
        both = int(((ACT[FLEX].max(0) > 0.1) & (ACT[EXT].max(0) > 0.1)).sum())
        effort = float((ACT**2).sum() * DT)
        readout = VGroup(
            Text(f"peak activation {ACT.max():.2f} ({NAMES[peak]});  effort  ∫Σa² dt = {effort:.3f}", font_size=19),
            Text(
                f"flexors and extensors both above 0.1 in {both} of {N} intervals",
                font_size=17,
                color=GRAY_B,
            ),
            Text("(co-contraction, partly biarticular muscles)", font_size=17, color=GRAY_B),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        fit(readout, 6.4)
        readout.move_to([bx0 - 0.4, -3.1, 0], aligned_edge=LEFT)
        self.play(FadeIn(readout), run_time=0.5)
        self.wait(1.8)

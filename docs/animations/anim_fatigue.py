"""
Manim CE scene: fatigue model (XiaTauFatigue) on the joint torques of the pendulum swing-up, driven by a REAL solve
(data/fatigue_xia.npz, see ``generate_fatigue_data.py``, based on examples/toy_examples/fatigue/pendulum_with_fatigue.py).

Render (from docs/animations):  manim render -qh anim_fatigue.py FatigueXia
"""

import sys
from pathlib import Path

import numpy as np
from manim import *

sys.path.insert(0, str(Path(__file__).parent))
from features_scenes import (  # noqa: E402  (also sets the default fonts)
    CODE_X,
    axis_label,
    code_panel,
    make_axes,
    place,
    poly,
    scene_title,
    time_label,
    x_ticks,
    y_ticks,
)

DATA = Path(__file__).parent / "data" / "fatigue_xia.npz"
C_TAU, C_Q = GREEN_C, YELLOW_C
C_ACT_P, C_ACT_M, C_FAT_P, C_FAT_M = GREEN_C, TEAL_C, RED_C, ORANGE


class FatigueXia(Scene):
    def construct(self):
        d = np.load(DATA)
        n, T = int(d["n_shooting"]), float(d["final_time"])
        t = np.linspace(0, T, n + 1)
        tau = np.append(d["tau"][0], d["tau"][0][-1])  # last node has no control: repeat for plotting

        self.play(
            FadeIn(scene_title("Fatigue model on the torque", "XiaTauFatigue: active · fatigued · resting")),
            run_time=0.5,
        )

        ax_u = make_axes([-3.6, 1.35, 0], 5.6, 1.5, (0, T), (-25, 8), y_step=10)
        ax_f = make_axes([-3.6, -0.55, 0], 5.6, 1.5, (0, T), (0, 0.22))
        ax_q = make_axes([-3.6, -2.55, 0], 5.6, 1.2, (0, T), (0, 3.4))
        decos = VGroup(
            axis_label("τ on the sliding joint (N)", ax_u),
            axis_label("fractions", ax_f),
            axis_label("θ (rad)", ax_q),
            time_label(ax_q),
            x_ticks(ax_q, [0, 0.5, 1.0], "{:.1f}"),
            y_ticks(ax_u, [-20, 0]),
            y_ticks(ax_f, [0, 0.1, 0.2], "{:.1f}"),
            y_ticks(ax_q, [0, 3]),
        )
        zero = DashedLine(ax_u.c2p(0, 0), ax_u.c2p(T, 0), color=GRAY_D, stroke_width=2)
        self.play(Create(ax_u), Create(ax_f), Create(ax_q), FadeIn(decos), Create(zero), run_time=0.8)

        def f(key):
            return d[key][0]

        curves = VGroup(
            poly(ax_u, t, tau, C_TAU, 4),
            poly(ax_f, t, f("tau_plus_ma"), C_ACT_P, 4),
            poly(ax_f, t, f("tau_minus_ma"), C_ACT_M, 4),
            poly(ax_f, t, f("tau_plus_mf"), C_FAT_P, 4),
            poly(ax_f, t, f("tau_minus_mf"), C_FAT_M, 4),
            poly(ax_q, t, d["q"][1], C_Q, 4),
        )
        cursor = Line(ax_u.c2p(0, -25), ax_q.c2p(0, 0), color=WHITE, stroke_width=1.5, stroke_opacity=0.5)

        legend = VGroup(
            Text("active τ+", font_size=15, color=C_ACT_P),
            Text("active τ−", font_size=15, color=C_ACT_M),
            Text("fatigued τ+", font_size=15, color=C_FAT_P),
            Text("fatigued τ−", font_size=15, color=C_FAT_M),
        ).arrange(RIGHT, buff=0.22)
        legend.scale(0.85).next_to(ax_f, UP, buff=0.05).align_to(ax_f, RIGHT)

        lines = [
            (0, "fatigue_dynamics = FatigueList()", WHITE),
            (0, "for i in range(n_tau):", WHITE),
            (1, "fatigue_dynamics.add(XiaTauFatigue(", WHITE),
            (2, "XiaFatigue(LD=100, LR=100, F=5, R=10, scaling=tau_min),", WHITE),
            (2, "XiaFatigue(LD=100, LR=100, F=5, R=10, scaling=tau_max),", WHITE),
            (2, "state_only=False, split_controls=False))", WHITE),
            (0, "bio_model = TorqueBiorbdModel(path, fatigue=fatigue_dynamics)", WHITE),
            (0, "x_bounds.concatenate(", WHITE),
            (1, "FatigueBounds(fatigue_dynamics, fix_first_frame=True))", WHITE),
            (0, "x_init.concatenate(", WHITE),
            (1, "FatigueInitialGuess(fatigue_dynamics))", WHITE),
        ]
        panel = code_panel(lines, size=19, top=2.35, caption="Bioptim code")

        max_fat = max(f("tau_plus_mf").max(), f("tau_minus_mf").max())
        min_rest = min(f("tau_plus_mr").min(), f("tau_minus_mr").min())
        peak = float(np.abs(tau).max())
        info = place(
            Text(
                f"peak |τ| = {peak:.0f} N  (limit 100)\n"
                f"fatigued fraction never exceeds {100 * max_fat:.1f} %\n"
                f"resting fraction never below {100 * min_rest:.0f} %\n"
                f"IPOPT: status {int(d['status'])}, {int(d['iterations'])} iterations",
                font_size=19,
                color=GRAY_A,
                line_spacing=0.9,
            ),
            CODE_X,
            -2.0,
        )
        cmt = place(
            Text(
                "Torque demand recruits active fibres, some of which\n"
                "become fatigued and only slowly recover.\n"
                "Resting fraction = 1 − active − fatigued.",
                font_size=19,
                color=YELLOW_C,
            ),
            CODE_X,
            -3.3,
        )

        self.play(FadeIn(panel), FadeIn(legend), FadeIn(cursor), run_time=0.8)
        dx = ax_u.c2p(T, 0)[0] - ax_u.c2p(0, 0)[0]
        self.play(
            *[Create(c, rate_func=linear) for c in curves],
            cursor.animate(rate_func=linear).shift(RIGHT * dx),
            run_time=5,
        )
        self.play(FadeIn(info), FadeIn(cmt), run_time=0.6)
        self.wait(2.5)

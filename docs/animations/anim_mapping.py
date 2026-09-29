"""
Manim CE scene: BiMapping / BiMappingList, driven by REAL solves (data/mapping_double_pendulum.npz, see
``generate_mapping_data.py``). Double pendulum, same task (reach q = [1, 1] at rest in 3 s), solved with two
independent torques and with ``BiMappingList.add("tau", to_second=[0, 0], to_first=[0])`` (one shared torque).

Render (from docs/animations):  manim render -qh anim_mapping.py Mapping
"""

import sys
from pathlib import Path

import numpy as np
from manim import *

sys.path.insert(0, str(Path(__file__).parent))
from features_scenes import (  # noqa: E402  (also sets the default fonts)
    CODE_X,
    axis_label,
    code,
    code_panel,
    make_axes,
    place,
    poly,
    scene_title,
    steps,
    time_label,
    x_ticks,
    y_ticks,
)

DATA = Path(__file__).parent / "data" / "mapping_double_pendulum.npz"
C1, C2, C_DIFF = GREEN_C, ORANGE, RED_C


class Mapping(Scene):
    def construct(self):
        d = np.load(DATA)
        n, T = int(d["n_shooting"]), float(d["final_time"])
        t = np.linspace(0, T, n + 1)
        dt = T / n
        free_tau, map_tau = d["free_tau"], d["mapped_tau"]
        phys = {k: dt * float((d[f"{k}_tau"] ** 2).sum()) for k in ("free", "mapped")}
        gap = {k: float(np.abs(d[f"{k}_tau"][:, 0] - d[f"{k}_tau"][:, 1]).max()) for k in ("free", "mapped")}

        self.play(FadeIn(scene_title("Mapping", "two actuators forced to share one torque")), run_time=0.5)

        ax = make_axes([-3.6, 0.6, 0], 5.6, 2.7, (0, T), (-10, 11), y_step=5)
        ax_d = make_axes([-3.6, -2.5, 0], 5.6, 1.1, (0, T), (-10, 10), y_step=10)
        decos = VGroup(
            axis_label("τ (N·m)", ax),
            axis_label("τ1 − τ2", ax_d),
            time_label(ax_d),
            x_ticks(ax_d, [0, 1, 2, 3], "{:g}"),
            y_ticks(ax, [-5, 0, 5]),
            y_ticks(ax_d, [-8, 0, 8]),
        )
        zero = DashedLine(ax.c2p(0, 0), ax.c2p(T, 0), color=GRAY_D, stroke_width=2)
        zero_d = DashedLine(ax_d.c2p(0, 0), ax_d.c2p(T, 0), color=GRAY_D, stroke_width=2)
        self.play(Create(ax), Create(ax_d), FadeIn(decos), Create(zero), Create(zero_d), run_time=0.6)

        def curve(tau, j, color):
            return steps(ax, t, tau[:, j], color, 4)

        def diff(tau):
            return steps(ax_d, t, tau[:, 0] - tau[:, 1], C_DIFF, 4)

        leg = VGroup(
            Text("τ1 (joint 1)", font_size=18, color=C1),
            Text("τ2 (joint 2)", font_size=18, color=C2),
        ).arrange(RIGHT, buff=0.4)
        leg.next_to(ax, UP, buff=0.05).align_to(ax, RIGHT)

        code_lines = [
            (0, 'u_bounds["tau"] = [-60] * 2, [60] * 2', WHITE),
            (0, "ocp = OptimalControlProgram(", WHITE),
            (1, "bio_model, n_shooting=30, phase_time=3.0,", WHITE),
            (1, "dynamics=dynamics, x_bounds=x_bounds,", WHITE),
            (1, "u_bounds=u_bounds, objective_functions=obj,", WHITE),
            (1, "variable_mappings=None,", GRAY_B),
            (0, ")", WHITE),
        ]
        panel = code_panel(code_lines, size=19, top=1.75, caption="Bioptim code")
        map_lines = [
            (0, "mappings = BiMappingList()", C1),
            (0, 'mappings.add("tau", to_second=[0, 0], to_first=[0])', C1),
        ]
        map_panel = code_panel(map_lines, size=19, top=2.65)

        def readout(k):
            tag = "free" if k == "free" else "mapped"
            n_tau = 2 * n if k == "free" else n
            body = (
                f"decision vector: {int(d[tag + '_n_vars'])} variables   (τ: {n_tau} of them)\n"
                f"IPOPT cost {float(d[tag + '_cost']):.1f}   ·   Σ(τ1² + τ2²)·dt = {phys[tag]:.1f}\n"
                f"max |τ1 − τ2| = {gap[tag]:.1f} N·m   ·   {int(d[tag + '_iterations'])} iterations"
                f"{'' if int(d[tag + '_status']) == 0 else '  (NOT converged)'}"
            )
            return place(Text(body, font_size=19, color=GRAY_A, line_spacing=0.9), CODE_X, -1.9)

        comment_free = place(
            Text("No mapping: 2 torques per node,\neach joint has its own curve.", font_size=19, color=YELLOW_C),
            CODE_X,
            -2.9,
        )
        comment_map = place(
            Text(
                "to_second=[0, 0]: the one optimised value feeds both joints.\n"
                "to_first=[0]: only joint 1's torque is kept as a variable.",
                font_size=19,
                color=YELLOW_C,
            ),
            CODE_X,
            -2.9,
        )
        note = place(
            Text(
                "Costs are not comparable: the objective sees the reduced tau\n"
                "(1 curve), so the mapped IPOPT cost counts τ once.",
                font_size=16,
                color=GRAY_B,
            ),
            CODE_X,
            -3.6,
        )

        c1, c2, dd = curve(free_tau, 0, C1), curve(free_tau, 1, C2), diff(free_tau)
        info = readout("free")
        self.play(FadeIn(panel), Create(c1), Create(c2), Create(dd), FadeIn(leg), run_time=1.4)
        self.play(FadeIn(info), FadeIn(comment_free), run_time=0.4)
        self.wait(2.4)

        # the mapping is declared, u_bounds shrink to one torque, the OCP receives it
        ub_old, vm_old = panel[1][0], panel[1][5]
        ub_new = code('u_bounds["tau"] = [-60] * 1, [60] * 1', 19, C1)
        ub_new.scale(ub_old.height / ub_new.height).move_to(ub_old, aligned_edge=LEFT)
        vm_new = code("variable_mappings=mappings,", 19, C1)
        vm_new.scale(vm_old.height / vm_new.height).move_to(vm_old, aligned_edge=LEFT)
        self.play(FadeIn(map_panel), Transform(ub_old, ub_new), Transform(vm_old, vm_new), run_time=0.7)
        self.play(
            Transform(c2, curve(map_tau, 1, C2)),
            Transform(c1, curve(map_tau, 0, C1)),
            Transform(dd, diff(map_tau)),
            Transform(info, readout("mapped")),
            Transform(comment_free, comment_map),
            FadeIn(note),
            run_time=2.0,
        )
        self.wait(4.5)

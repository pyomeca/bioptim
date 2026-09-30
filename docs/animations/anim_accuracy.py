"""
Manim CE scene: is the optimised trajectory consistent with the dynamics? Re-integrate the optimal controls from the
initial state with ``sol.integrate(shooting_type=Shooting.SINGLE, integrator=SolutionIntegrator.SCIPY_DOP853)`` and
look at the drift. Four REAL solves stored in ``data/accuracy_pendulum.npz`` (see ``generate_accuracy_data.py``).

Render (from docs/animations):  manim -qh anim_accuracy.py AccuracyCheck
"""

import numpy as np
from manim import *

from features_scenes import (
    DATA_DIR,
    CODE_X,
    axis_label,
    code,
    code_panel,
    fit,
    make_axes,
    M,
    poly,
    scene_title,
    time_label,
    x_ticks,
    y_ticks,
)

ROT = 1
TAGS = ["rk4_coarse", "rk4_fine", "col3", "col5"]
NAMES = ["RK4, 1 step", "RK4, 5 steps", "Collocation, degree 3", "Collocation, degree 5"]
ODE = [
    "OdeSolver.RK4(n_integration_steps=1)",
    "OdeSolver.RK4(n_integration_steps=5)",
    "OdeSolver.COLLOCATION(polynomial_degree=3)",
    "OdeSolver.COLLOCATION(polynomial_degree=5)",
]
C_OPT = YELLOW_C
C_INT = RED_C
C_BAR = [ORANGE, GREEN_C, BLUE_C, TEAL_C]


class AccuracyCheck(Scene):
    def construct(self):
        d = np.load(DATA_DIR / "accuracy_pendulum.npz")
        T = 1.0
        title = scene_title("Does the solution respect the dynamics?", "re-integrate the optimal controls, N = 30")
        self.play(FadeIn(title), run_time=0.5)

        # ------------------------------------------------------------------ beat 1: optimised vs re-integrated
        ax = make_axes([-3.55, -0.3, 0], 5.6, 4.2, [0, T], [-1, 8], 0.5, 1)
        decos = VGroup(
            axis_label("θ(t)  pendulum angle (rad)", ax),
            time_label(ax),
            x_ticks(ax, [0, 0.5, 1.0], "{:.1f}"),
            y_ticks(ax, [0, 3, 6]),
        )
        opt_lbl = Text("optimized (IPOPT)", font_size=20, color=C_OPT).move_to([-6.4, -2.85, 0], aligned_edge=LEFT)
        int_lbl = Text("re-integrated from x₀", font_size=20, color=C_INT).move_to([-3.0, -2.85, 0], aligned_edge=LEFT)

        code_lines = [
            (0, "# ode_solver of the problem:", GRAY_B),
            (0, ODE[0], WHITE),
            (0, "sol = ocp.solve(Solver.IPOPT())", WHITE),
            (0, "out = sol.integrate(", C_INT),
            (1, "shooting_type=Shooting.SINGLE,", C_INT),
            (1, "integrator=SolutionIntegrator.SCIPY_DOP853)", C_INT),
        ]
        panel = code_panel(code_lines, size=19, top=2.3, caption="Bioptim code")
        first = panel[1][1]
        # the ode_solver line is replaced for each case, with the same scale as the panel lines
        scale = first.height / code(code_lines[1][1], 19).height

        def ode_line(i):
            m = code(ODE[i], 19 * scale, WHITE)
            return m.move_to(first.get_left(), aligned_edge=LEFT)

        def curves(i):
            tag = TAGS[i]
            t = d[f"{tag}_t_nodes"]
            opt = poly(ax, t, d[f"{tag}_q_opt"][ROT], C_OPT, 6)
            integ = poly(ax, t, d[f"{tag}_x_nodes_bioptim_dop853"][ROT], C_INT, 4)
            dots = VGroup(*[Dot(ax.c2p(a, b), radius=0.035, color=C_OPT) for a, b in zip(t, d[f"{tag}_q_opt"][ROT])])
            return opt, integ, dots

        def readout(i):
            err = float(d[f"{TAGS[i]}_max_theta_error"])
            txt = M(f"<b>{NAMES[i]}</b>    max |Δθ| = <b>{err:.1e}</b> rad", 24, C_INT if err > 0.05 else GREEN_C)
            return txt.move_to([-6.9, -3.5, 0], aligned_edge=LEFT)

        opt, integ, dots = curves(0)
        read = readout(0)
        self.play(FadeIn(ax), FadeIn(decos), FadeIn(panel), FadeIn(opt_lbl), run_time=0.7)
        self.play(Create(opt), FadeIn(dots), run_time=0.8)
        self.play(FadeIn(int_lbl), Create(integ), FadeIn(read), run_time=1.2)
        self.wait(0.6)
        for i in range(1, 4):
            opt_n, integ_n, dots_n = curves(i)
            self.play(
                Transform(opt, opt_n),
                Transform(integ, integ_n),
                Transform(dots, dots_n),
                Transform(first, ode_line(i)),
                Transform(read, readout(i)),
                run_time=0.7,
            )
            self.wait(1.1)

        # ------------------------------------------------------------------ beat 2: error vs cost
        self.play(*[FadeOut(m) for m in [ax, decos, opt, integ, dots, opt_lbl, int_lbl, read]], run_time=0.5)
        errs = [float(d[f"{t}_max_theta_error"]) for t in TAGS]
        costs = [float(d[f"{t}_solve_time"]) for t in TAGS]
        lo, hi = -4.0, 1.0  # log10 range
        base_y, height = -2.0, 3.4
        base_x, dx = -5.7, 1.35

        def y_of(v):
            return base_y + height * (np.log10(v) - lo) / (hi - lo)

        axis = Line([base_x - 0.4, base_y, 0], [base_x + 4 * dx, base_y, 0], color=GRAY_B)
        yl = Line([base_x - 0.4, base_y, 0], [base_x - 0.4, base_y + height, 0], color=GRAY_B)
        ticks = VGroup(
            *[
                Text(f"1e{e}" if e else "1", font_size=16, color=GRAY_B).move_to([base_x - 0.75, y_of(10.0**e), 0])
                for e in range(-3, 1)
            ]
        )
        guides = VGroup(
            *[
                DashedLine(
                    [base_x - 0.4, y_of(10.0**e), 0],
                    [base_x + 4 * dx, y_of(10.0**e), 0],
                    color=GRAY_D,
                    stroke_width=1.5,
                )
                for e in range(-3, 1)
            ]
        )
        ylab = Text("max |Δθ| (rad), log scale", font_size=20, color=GRAY_B).move_to(
            [base_x - 0.4, base_y + height + 0.3, 0], aligned_edge=LEFT
        )
        self.play(FadeIn(axis), FadeIn(yl), FadeIn(ticks), FadeIn(guides), FadeIn(ylab), run_time=0.4)
        for i, (e, c) in enumerate(zip(errs, costs)):
            x = base_x + (i + 0.5) * dx
            h = y_of(e) - base_y
            bar = Rectangle(width=0.9, height=h, stroke_width=0, fill_color=C_BAR[i], fill_opacity=0.9)
            bar.move_to([x, base_y + h / 2, 0])
            val = Text(f"{e:.1e}" if e < 0.01 else f"{e:.2f}", font_size=18).next_to(bar, UP, buff=0.06)
            name = Text(NAMES[i].replace(", ", "\n"), font_size=15, color=C_BAR[i], line_spacing=0.8)
            name.scale_to_fit_width(min(name.width, 1.2))
            name.move_to([x, base_y - 0.42, 0])
            cst = Text(f"solve {c:.2f} s", font_size=15, color=GRAY_B)
            cst.scale_to_fit_width(min(cst.width, 1.25))
            cst.move_to([x, base_y - 0.85, 0])
            self.play(GrowFromEdge(bar, DOWN), FadeIn(val), FadeIn(name), FadeIn(cst), run_time=0.45)

        # right panel: what to conclude
        notes = VGroup(
            M("Same problem, same N = 30, same controls.", 22, WHITE),
            M("Only the transcription changes:", 22, WHITE),
            M("•  RK4: more integration steps  →  drift shrinks 1000x", 20, C_BAR[1]),
            M("•  COLLOCATION: degree 5 drifts less than degree 3", 20, C_BAR[3]),
            M(
                "•  the optimizer exploits a coarse scheme:\n   a small defect is not a small error in reality",
                20,
                GRAY_B,
            ),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        fit(notes, 6.6)
        notes.move_to([CODE_X, 0.9, 0], aligned_edge=LEFT)
        final_code = code_panel(
            [
                (0, "out = sol.integrate(", C_INT),
                (1, "shooting_type=Shooting.SINGLE,", C_INT),
                (1, "integrator=SolutionIntegrator.SCIPY_DOP853)", C_INT),
            ],
            size=19,
            top=3.0,
            caption="Bioptim code",
        )
        final_code.next_to(notes, DOWN, buff=0.4).align_to(notes, LEFT)
        self.play(FadeOut(panel), run_time=0.3)
        self.play(FadeIn(final_code), FadeIn(notes), run_time=0.6)
        self.wait(2.5)

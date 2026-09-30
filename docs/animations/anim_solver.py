"""
Manim Community animations on IPOPT, driven by REAL bioptim/IPOPT solves stored in ``data/solver_*.npz``
(see ``generate_solver_data.py``). Pendulum swing-up, N = 30, T = 1 s.

Scenes:
    1. IpoptIterates     the trajectory after k IPOPT iterations (each one a real solve with set_maximum_iterations(k)),
                         with objective and primal infeasibility read out from IPOPT's own iteration history
    2. IpoptMultiStart   different initial guesses end in different local minima, with their (real) final costs

Helpers are imported read-only from ``features_scenes.py``. No LaTeX. Render: see notes/solver.md.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from manim import *

from features_scenes import (  # noqa: E402  (also sets the default fonts)
    CODE_X,
    CODE_W,
    axis_label,
    code_panel,
    fit,
    make_axes,
    poly,
    scene_title,
    steps,
    time_label,
    x_ticks,
    y_ticks,
)

DATA_DIR = Path(__file__).parent / "data"
ROT = 1
C_STATE = YELLOW_C
C_CTRL = GREEN_C
C_INF = RED_C


def plots(T, theta_range, tau_range, tau_ticks):
    ax_q = make_axes([-3.55, 0.9, 0], 5.6, 2.7, [0, T], theta_range, 0.5, 1)
    ax_u = make_axes([-3.55, -2.2, 0], 5.6, 1.7, [0, T], tau_range, 0.5, tau_range[1])
    decos = VGroup(
        axis_label("pendulum angle θ (rad)", ax_q, C_STATE),
        axis_label("actuated force τ (N)", ax_u, C_CTRL),
        time_label(ax_u),
        x_ticks(ax_u, [0, 0.5, 1.0], "{:.1f}"),
        y_ticks(ax_q, [0, 1, 2, 3]),
        y_ticks(ax_u, tau_ticks),
    )
    return ax_q, ax_u, decos


# ====================================================================================================================
# Scene 1 - the iterates
# ====================================================================================================================
class IpoptIterates(Scene):
    def construct(self):
        d = np.load(DATA_DIR / "solver_iterates.npz")
        t, T = d["t"], float(d["final_time"])
        ks = [int(k) for k in d["ks"]]
        show = [0, 1, 3, 8, 12, 40, ks[-1]]  # iterates shown (all are real solves)
        idx = {k: i for i, k in enumerate(ks)}
        hist_pr, hist_obj = d["hist_inf_pr"], d["hist_obj"]

        title = scene_title("How IPOPT converges", "the trajectory after k iterations")
        ax_q, ax_u, decos = plots(T, [-1, 3.6], [-30, 30], [-30, 0, 30])
        self.play(FadeIn(title), Create(ax_q), Create(ax_u), FadeIn(decos), run_time=0.8)

        # Solver.IPOPT options verified in bioptim/interfaces/ipopt_options.py (set_maximum_iterations, set_tol)
        panel = code_panel(
            [
                (0, "solver = Solver.IPOPT()", WHITE),
                (0, "solver.set_maximum_iterations(k)", YELLOW_C),
                (0, "solver.set_tol(1e-8)", GRAY_B),
                (0, "sol = ocp.solve(solver)", WHITE),
                (0, "sol.iterations, sol.cost", GRAY_B),
            ],
            size=18,
            top=2.3,
            caption="Bioptim code",
        )
        self.play(FadeIn(panel), run_time=0.5)

        # convergence history: log10(primal infeasibility) versus iteration, right panel
        n_hist = len(hist_pr)
        ax_h = Axes(
            x_range=[0, n_hist - 1, 10],
            y_range=[-12, 1, 4],
            x_length=5.4,
            y_length=1.5,
            tips=False,
            axis_config={"color": GRAY_B, "stroke_width": 2, "include_ticks": False},
        ).move_to([3.7, -2.65, 0])
        curve_h = VMobject(color=C_INF, stroke_width=3).set_points_as_corners(
            [ax_h.c2p(i, np.log10(max(v, 1e-12))) for i, v in enumerate(hist_pr)]
        )
        h_lab = Text("primal infeasibility (log10)", font_size=17, color=C_INF).next_to(ax_h, UP, buff=0.08)
        h_lab.align_to(ax_h, LEFT)
        h_x = Text("iteration", font_size=16, color=GRAY_B).next_to(ax_h, DOWN, buff=0.08)
        h_ticks = VGroup(
            *[
                Text(f"{v}", font_size=16, color=GRAY_B).next_to(ax_h.c2p(0, v), LEFT, buff=0.08)
                for v in (-12, -8, -4, 0)
            ]
        )
        self.play(Create(ax_h), FadeIn(h_lab), FadeIn(h_x), FadeIn(h_ticks), Create(curve_h), run_time=0.7)
        curve_h.set_stroke(opacity=0.35)

        def idx_frac(k):
            return show.index(k) / (len(show) - 1)

        def frame_mobs(k):
            i = idx[k]
            col = interpolate_color(BLUE_C, ORANGE, idx_frac(k))
            c_q = poly(ax_q, t, d[f"q_{i}"][ROT], col, 5)
            c_u = steps(ax_u, t, d[f"tau_{i}"][0], col, 4)
            dot = Dot(ax_h.c2p(k, np.log10(max(hist_pr[k], 1e-12))), radius=0.09, color=col)
            read = VGroup(
                Text(f"iteration {k}", font_size=26, weight=BOLD, color=col),
                Text(f"objective = {hist_obj[k]:.3g}", font_size=22, color=WHITE),
                Text(f"primal infeasibility = {hist_pr[k]:.1e}", font_size=22, color=C_INF),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
            read.move_to([CODE_X, -0.1, 0], aligned_edge=UL)
            return c_q, c_u, dot, read

        c_q, c_u, dot, read = frame_mobs(show[0])
        self.play(Create(c_q), Create(c_u), FadeIn(dot), FadeIn(read), run_time=0.6)
        self.wait(0.4)
        for k in show[1:]:
            n_q, n_u, n_dot, n_read = frame_mobs(k)
            self.play(
                Transform(c_q, n_q),
                Transform(c_u, n_u),
                Transform(dot, n_dot),
                Transform(read, n_read),
                run_time=0.9,
            )
            self.wait(0.35)
        done = Text(
            f"Solve_Succeeded: {ks[-1]} iterations, cost {float(d['full_cost']):.2f}", font_size=21, color=GREEN_C
        ).move_to([CODE_X, -1.3, 0], aligned_edge=LEFT)
        self.play(FadeIn(done))
        self.wait(2.0)


# ====================================================================================================================
# Scene 2 - multi-start
# ====================================================================================================================
class IpoptMultiStart(Scene):
    def construct(self):
        it = np.load(DATA_DIR / "solver_iterates.npz")
        ms = np.load(DATA_DIR / "solver_multistart.npz")
        t, T = ms["t"], 1.0

        # distinct converged minima (status 0), plus the zero initial guess of the previous scene
        last = len(it["ks"]) - 1
        runs = [(float(it["full_cost"]), it[f"q_{last}"], it[f"tau_{last}"], "zero guess")]
        seen = [runs[0][0]]
        for i in range(int(ms["n_starts"])):
            c = float(ms[f"cost_{i}"])
            if int(ms[f"status_{i}"]) != 0 or any(abs(c - s) < 1e-3 for s in seen):
                continue
            seen.append(c)
            runs.append((c, ms[f"q_{i}"], ms[f"tau_{i}"], f"random guess {i}"))
        runs.sort(key=lambda r: r[0])
        picks = runs[:4] + runs[-2:] if len(runs) > 6 else runs
        colors = [GREEN_C, TEAL_C, BLUE_C, PURPLE_B, ORANGE, RED_C][: len(picks)]

        title = scene_title("Multi-start", "different initial guesses, different local minima")
        ax_q, ax_u, decos = plots(T, [-1, 3.8], [-80, 80], [-80, 0, 80])
        self.play(FadeIn(title), Create(ax_q), Create(ax_u), FadeIn(decos), run_time=0.7)

        panel = code_panel(
            [
                (0, "for q0, tau0 in guesses:", WHITE),
                (1, "x_init = InitialGuessList()", GRAY_B),
                (1, 'x_init.add("q", q0, interpolation=', WHITE),
                (3, "InterpolationType.EACH_FRAME)", WHITE),
                (1, "ocp = prepare_ocp(x_init=x_init, ...)", GRAY_B),
                (1, "sol = ocp.solve(solver)   # IPOPT", YELLOW_C),
            ],
            size=18,
            top=2.3,
            caption="Bioptim code",
        )
        self.play(FadeIn(panel), run_time=0.5)

        rows = VGroup()
        for j, ((c, q, tau, name), col) in enumerate(zip(picks, colors)):
            cq, cu = poly(ax_q, t, q[ROT], col, 4), steps(ax_u, t, tau[0], col, 3)
            name_t = Text(name, font_size=19, color=col)
            cost_t = Text(f"cost = {c:.2f}" + (" (best)" if j == 0 else ""), font_size=19, color=col)
            cost_t.next_to(name_t, RIGHT, buff=0.2).shift(RIGHT * (2.3 - name_t.width))
            tag = VGroup(name_t, cost_t)
            rows.add(tag)
            rows.arrange(DOWN, aligned_edge=LEFT, buff=0.12)
            fit(rows, CODE_W)
            rows.move_to([CODE_X, -0.85, 0], aligned_edge=UL)
            self.play(Create(cq), Create(cu), FadeIn(tag), run_time=0.85)
        note = Text("every run: Solve_Succeeded, yet IPOPT is a local method", font_size=17, color=GRAY_B)
        fit(note, CODE_W)
        note.move_to([CODE_X, -3.65, 0], aligned_edge=LEFT)
        self.play(FadeIn(note))
        self.wait(2.0)

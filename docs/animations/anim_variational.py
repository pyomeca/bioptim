"""
Manim Community animation: discrete mechanics (variational integrator) in bioptim, driven by REAL computations stored in
``data/variational_pendulum.npz`` (see ``generate_variational_data.py``).

Beat 1: free pendulum (tau = 0, dt = 0.1 s, 600 s = 6000 steps). Total-energy error of three schemes: the model's own
        discrete Euler-Lagrange equations (VariationalTorqueBiorbdModel), explicit Euler (RK1) and RK4.
Beat 2: a real VariationalOptimalControlProgram solve (pendulum swing-up, IPOPT).

Render (from docs/animations):  manim render -qh anim_variational.py DiscreteMechanics
"""

import sys
from pathlib import Path

import numpy as np
from manim import *

sys.path.insert(0, str(Path(__file__).parent))
from features_scenes import (  # noqa: E402  (also sets the default fonts)
    CODE_W,
    DATA_DIR,
    code,
    fit,
    make_axes,
    poly,
    scene_title,
    time_label,
    x_ticks,
    y_ticks,
)

C_VAR, C_RK4, C_RK1, C_Q, C_TAU = GREEN_C, ORANGE, RED_C, YELLOW_C, GREEN_C
CODE_X0 = 0.15


def code_block(lines, size=18):
    block = VGroup(*[code(text, size, WHITE) for _, text in lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.09)
    for line, (level, _) in zip(block, lines):
        line.shift(RIGHT * 0.3 * level)
    return block


def captioned(caption, lines, size=18):
    cap = Text(caption, font_size=19, color=GRAY_B)
    return VGroup(cap, code_block(lines, size)).arrange(DOWN, aligned_edge=LEFT, buff=0.12)


def box(mob, color=YELLOW):
    return SurroundingRectangle(mob, color=color, buff=0.06, stroke_width=2.5)


def to_points(ax, x, y):
    p0 = ax.c2p(0, 0)
    sx, sy = ax.c2p(1, 0) - p0, ax.c2p(0, 1) - p0
    return p0[None, :] + np.asarray(x)[:, None] * sx[None, :] + np.asarray(y)[:, None] * sy[None, :]


def swept(ax, x, y, color, width):
    """Curve revealed progressively: alpha in [0, 1] -> the first alpha * len(x) points."""
    pts = to_points(ax, x, y)
    mob = VMobject(color=color, stroke_width=width)
    mob.set_points_as_corners(pts[:2])

    def show(alpha):
        n = max(2, int(round(alpha * (len(pts) - 1))) + 1)
        mob.set_points_as_corners(pts[:n])

    return mob, show


def legend_row(color, text):
    line = Line(LEFT * 0.3, RIGHT * 0.3, color=color, stroke_width=5)
    return VGroup(line, Text(text, font_size=18)).arrange(RIGHT, buff=0.15)


class DiscreteMechanics(Scene):
    def construct(self):
        d = np.load(DATA_DIR / "variational_pendulum.npz")
        dt, t_end = float(d["dt"]), float(d["t_end"])
        t = d["t"]
        e0 = float(d["e0"])
        de = {k: d["e_" + k] - e0 for k in ("var", "rk4", "rk1")}
        n_steps = len(t) - 1
        y_lo, y_hi = -0.5, 0.15
        i_off = int(np.argmax(de["rk1"] > y_hi))  # first RK1 sample that leaves the plot
        var_lo, var_hi = float(de["var"].min()), float(de["var"].max())
        i_cross = int(np.argmax(np.abs(de["rk4"]) > max(abs(var_lo), abs(var_hi))))

        title = scene_title(
            "Discrete mechanics", "the variational integrator replaces the ordinary differential equation (ODE)"
        )
        self.play(FadeIn(title), run_time=0.5)

        # ------------------------------------------------------------------------------------------------ beat 1
        ax = make_axes([-3.55, 0.35, 0], 5.6, 3.4, [0, t_end], [y_lo, y_hi], 200, 0.1)
        lab = Text("energy error E(t) − E(0) (J), free pendulum, τ = 0", font_size=19, color=GRAY_B)
        lab.next_to(ax.get_y_axis(), UP, buff=0.08).align_to(ax.get_y_axis(), LEFT)
        decos = VGroup(
            lab,
            time_label(ax),
            x_ticks(ax, [0, 200, 400, 600], "{:g}"),
            y_ticks(ax, [-0.4, -0.2, 0.0], "{:g}"),
        )
        zero = DashedLine(ax.c2p(0, 0), ax.c2p(t_end, 0), color=GRAY_D, stroke_width=2)

        blk_a = captioned(
            "Bioptim code: variational dynamics of the problem",
            [
                (0, "dynamics.add(skip_continuity=True,"),
                (1, "ode_solver=OdeSolver.VARIATIONAL())"),
                (0, "multinode_constraints.add("),
                (1, "self.variational_integrator_three_nodes,"),
                (1, "nodes_phase=(0, 0, 0), nodes=(i, i+1, i+2))"),
            ],
        )
        blk_b = captioned(
            "each triplet q(k-1), q(k), q(k+1) obeys (= 0):",
            [
                (0, "model.discrete_euler_lagrange_equations("),
                (1, "time_step, q_prev, q_cur, q_next,"),
                (1, "control_prev, control_cur, control_next)"),
            ],
        )
        blk_c = captioned(
            "reference schemes, same Δt, one step each:",
            [
                (0, "x = x + dt * f(x)                 # RK1"),
                (0, "f(x) = [qdot, model.forward_dynamics(...)]"),
            ],
        )
        panel = VGroup(blk_a, blk_b, blk_c).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        fit(panel, CODE_W)
        panel.move_to([CODE_X0, 2.35, 0], aligned_edge=UL)

        rows = VGroup(
            legend_row(C_VAR, "variational (discrete Euler-Lagrange)"),
            legend_row(C_RK4, "RK4"),
            legend_row(C_RK1, "explicit Euler (RK1)"),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        rows.move_to([-6.6, -2.95, 0], aligned_edge=LEFT)

        self.play(FadeIn(VGroup(ax, decos, zero)), FadeIn(panel), FadeIn(rows), run_time=0.7)

        mv, sv = swept(ax, t, np.clip(de["var"], y_lo, y_hi), C_VAR, 2.5)
        m4, s4 = swept(ax, t, np.clip(de["rk4"], y_lo, y_hi), C_RK4, 4)
        m1, s1 = swept(ax, t[: i_off + 1], np.clip(de["rk1"][: i_off + 1], y_lo, y_hi), C_RK1, 4)
        self.add(mv, m4, m1)

        seg = 2.2
        cur = None
        for j, blk in enumerate((blk_a, blk_b, blk_c)):
            bx = box(blk)
            a0, a1 = j / 3, (j + 1) / 3

            def upd(alpha, a0=a0, a1=a1):
                a = a0 + (a1 - a0) * alpha
                sv(a)
                s4(a)
                s1(min(1.0, a * n_steps / max(i_off, 1)))

            if cur is None:
                self.play(Create(bx), run_time=0.3)
            else:
                self.play(ReplacementTransform(cur, bx), run_time=0.3)
            cur = bx
            self.play(UpdateFromAlphaFunc(VGroup(), lambda m, alpha, u=upd: u(alpha)), run_time=seg, rate_func=linear)
        self.play(FadeOut(cur), run_time=0.3)

        # Euler leaves the plot: arrow and numbers
        arrow = Arrow(ax.c2p(t[i_off], y_hi - 0.1), ax.c2p(t[i_off], y_hi), color=C_RK1, buff=0, stroke_width=5)
        arr_txt = Text(
            f"off scale: +{de['rk1'][100]:.0f} J at t = {t[100]:g} s, +{de['rk1'][-1]:.0f} J at {t_end:g} s",
            font_size=17,
            color=C_RK1,
        )
        arr_txt.next_to(ax.c2p(t[i_off] + 10, y_hi - 0.05), RIGHT, buff=0.15)
        final = VGroup(
            legend_row(
                C_VAR, f"variational: stays in [{var_lo:.2f}, {var_hi:.2f}] J for {n_steps} steps (Δt = {dt:g} s)"
            ),
            legend_row(
                C_RK4,
                f"RK4: {de['rk4'][-1]:.2f} J at {t_end:g} s and still drifting; leaves that band at {t[i_cross]:.0f} s",
            ),
            legend_row(C_RK1, "explicit Euler (RK1): +%.0f J at %g s" % (de["rk1"][-1], t_end)),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        final.move_to([-6.6, -2.95, 0], aligned_edge=LEFT)
        fit(final, 13.2)
        self.play(FadeIn(arrow), FadeIn(arr_txt), FadeOut(rows), FadeIn(final), run_time=0.6)
        self.wait(2.0)

        # ------------------------------------------------------------------------------------------------ beat 2
        tq = np.linspace(0, float(d["su_final_time"]), int(d["su_n"]) + 1)
        tau = np.r_[d["su_tau"][0], d["su_tau"][1::2]][: int(d["su_n"])]  # value at nodes 0 .. N-1 (last node unused)
        q = d["su_q"]
        T = float(d["su_final_time"])
        ax_q = make_axes([-3.55, 1.15, 0], 5.6, 2.3, [0, T], [-0.8, 3.4], 0.5, 1)
        ax_u = make_axes([-3.55, -1.95, 0], 5.6, 1.9, [0, T], [-10, 10], 0.5, 10)
        lab_q = Text("θ (rad)", font_size=19, color=GRAY_B).next_to(ax_q.get_y_axis(), UP, buff=0.08)
        lab_q.align_to(ax_q.get_y_axis(), LEFT)
        lab_u = Text("τ (N·m)", font_size=19, color=GRAY_B).next_to(ax_u.get_y_axis(), UP, buff=0.08)
        lab_u.align_to(ax_u.get_y_axis(), LEFT)
        decos2 = VGroup(
            lab_q,
            lab_u,
            time_label(ax_u),
            x_ticks(ax_u, [0, 1, 2], "{:g}"),
            y_ticks(ax_q, [0, 3], "{:g}"),
            y_ticks(ax_u, [-8, 0, 8], "{:g}"),
        )
        pi_line = DashedLine(ax_q.c2p(0, np.pi), ax_q.c2p(T, np.pi), color=GRAY_D, stroke_width=2)
        zero_u = DashedLine(ax_u.c2p(0, 0), ax_u.c2p(T, 0), color=GRAY_D, stroke_width=2)

        panel2 = captioned(
            "Bioptim code: variational swing-up problem",
            [
                (0, 'q_bounds["q"][:, 0] = 0'),
                (0, 'q_bounds["q"][:, -1] = np.pi'),
                (0, 'qdot_bounds.add("qdot_start", min_bound=[0],'),
                (1, "max_bound=[0],"),
                (1, "interpolation=InterpolationType.CONSTANT)"),
                (0, "ocp = VariationalOptimalControlProgram("),
                (1, "bio_model, n_shooting, final_time,"),
                (1, "q_bounds=q_bounds, u_bounds=u_bounds,"),
                (1, "qdot_bounds=qdot_bounds, q_init=q_init,"),
                (1, "objective_functions=objective_functions)"),
                (0, "sol = ocp.solve(Solver.IPOPT())"),
            ],
        )
        fit(panel2, CODE_W)
        panel2.move_to([CODE_X0, 2.35, 0], aligned_edge=UL)
        title2 = scene_title(
            "Variational optimal control problem", f"pendulum swing-up, {int(d['su_n'])} intervals, {T:g} s"
        )
        self.play(
            FadeOut(VGroup(ax, decos, zero, panel, m1, mv, m4, arrow, arr_txt, final, title)),
            FadeIn(VGroup(ax_q, ax_u, decos2, pi_line, zero_u, panel2, title2)),
            run_time=0.7,
        )
        cq = poly(ax_q, tq, q, C_Q, 5)
        cu = poly(ax_u, tq[: len(tau)], tau, C_TAU, 5)
        self.play(Create(cq), Create(cu), run_time=2.6)
        ok = "converged" if int(d["su_status"]) == 0 else "FAILED"
        status = Text(
            f"real IPOPT solve: {int(d['su_iterations'])} iterations, status {int(d['su_status'])} ({ok}); "
            f"no velocity state, only q at each node",
            font_size=18,
            color=GRAY_B,
        )
        fit(status, 13.2)
        status.to_edge(DOWN, buff=0.15)
        self.play(FadeIn(status), run_time=0.4)
        self.wait(2.0)

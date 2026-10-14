"""
Manim CE scene: a custom objective and a custom constraint. A user function ``f(controller, **extra)`` that returns a
casadi expression is passed to ``objectives.add(..., custom_type=...)`` / ``constraints.add(...)`` in place of an
``ObjectiveFcn`` / ``ConstraintFcn``. Beat 1: a custom Lagrange objective (tip height, read through ``controller.model``)
changes a pendulum swing-up. Beat 2: a custom constraint (torque x velocity, read from ``controller.controls`` and
``controller.states``) keeps the mechanical power inside a limit.
REAL bioptim / IPOPT solves stored in ``data/custompen_pendulum.npz`` (see ``generate_custompen_data.py``): three
independent solves (plain, custom objective, custom constraint), same initial guess, no warm start.

Scene: CustomPenalty (about 16 s of content at native speed).  Render, from the repository root:
    python docs/animations/render_series.py anim_custompen.py CustomPenalty --lang en
"""

import numpy as np
from manim import *

from features_scenes import (
    C_BOUND,
    C_CTRL,
    C_LAG,
    C_PAR,
    C_STATE,
    CODE_X,
    DATA_DIR,
    axis_label,
    band,
    code_panel,
    footer,
    hline,
    ipopt_line,
    make_axes,
    place,
    poly,
    say,
    scene_title,
    time_label,
    x_ticks,
    y_ticks,
)

C_GHOST = GRAY_B  # the plain solve, dashed
C_DIFF = C_PAR  # difference with the plain solve


def ghost_of(ax, t, y):
    return DashedVMobject(poly(ax, t, y, C_GHOST, 3), num_dashes=40).set_opacity(0.8)


class CustomPenalty(Scene):
    def construct(self):
        # ---------------------------------------------------------------- data: everything shown comes from the npz
        d = np.load(DATA_DIR / "custompen_pendulum.npz")
        t = d["plain_t"]
        T, n = float(d["final_time"]), int(d["n_shooting"])
        z0, z1, z2 = d["plain_z"], d["obj_z"], d["con_z"]
        p0, p2 = d["plain_power"], d["con_power"]
        pmax = float(d["p_max"])
        w_height = float(d["w_height"])

        title = scene_title(
            "Custom objective and custom constraint", f"pendulum swing-up, N = {n} intervals, T = {T:g} s"
        )
        self.play(FadeIn(title), run_time=0.4)

        # ================================================================ beat 1: a custom Lagrange objective
        m0, m1 = float(z0.mean()), float(z1.mean())
        tau0, tau1 = float(np.abs(d["plain_tau"]).max()), float(np.abs(d["obj_tau"]).max())
        dz = z1 - z0
        ax_z = make_axes([-3.55, 0.85, 0], 5.6, 2.5, [0, T], [-1.25, 1.25], 0.5, 0.5)
        ax_d = make_axes([-3.55, -2.15, 0], 5.6, 1.6, [0, T], [-1.6, 1.6], 0.5, 0.5)
        decos = VGroup(
            axis_label("tip height z (m)", ax_z, C_STATE),
            axis_label("Δz = z − z_plain (m)", ax_d, C_DIFF),
            time_label(ax_d),
            x_ticks(ax_d, [0, 1, 2]),
            y_ticks(ax_z, [-1, 0, 1]),
            y_ticks(ax_d, [-1, 0, 1]),
        )
        self.play(Create(ax_z), Create(ax_d), FadeIn(decos), run_time=0.8)
        ghost = ghost_of(ax_z, t, z0)
        self.play(Create(ghost), run_time=1.0)

        panel = code_panel(
            [
                (0, "def tip_height(controller: PenaltyController, marker: str) -> MX:", WHITE),
                (1, 'q = controller.states["q"].cx', C_STATE),
                (1, "tip = controller.model.markers()(q, controller.parameters.cx)", WHITE),
                (1, "return tip[2, controller.model.marker_index(marker)]", WHITE),
                (0, "objectives.add(tip_height, custom_type=ObjectiveFcn.Lagrange,", C_LAG),
                (1, 'weight=1.0, quadratic=False, marker="tip")', C_LAG),
            ],
            size=19,
        )
        self.play(FadeIn(panel), run_time=0.6)

        curve = poly(ax_z, t, z1, C_STATE, 5)
        curve_d = poly(ax_d, t, dz, C_DIFF, 5)
        zero = hline(ax_d, 0, T, 0, GRAY_D, dashed=False)
        body = (
            f"weight = {w_height:g}  ·  mean tip height = {m1:.2f} m\n"
            f"plain solve: mean tip height = {m0:.2f} m\n"
            f"peak |τ| = {tau1:.1f} N·m (plain solve: {tau0:.1f} N·m)\n"
            + ipopt_line(int(d["obj_iterations"]), bool(d["obj_converged"]))
        )
        info = place(Text(body, font_size=19, color=GRAY_A, line_spacing=0.9), CODE_X, -0.9)
        self.play(Create(curve), Create(zero), Create(curve_d), FadeIn(info), run_time=1.6)
        self.wait(0.6)
        remark = place(
            say("The height penalty keeps the tip low as long as possible, so the swing-up is late and fast."),
            CODE_X,
            -2.35,
        )
        foot = footer(
            "Bioptim sets ObjectiveFcn.Lagrange.CUSTOM by itself when a function is passed; it must return a casadi expression."
        )
        self.play(FadeIn(remark), FadeIn(foot))
        self.wait(0.8)

        # ================================================================ beat 2: a custom constraint
        self.play(*[FadeOut(m) for m in self.mobjects if m is not title], run_time=0.5)
        self.add(title)

        pk0, pk2 = float(np.abs(p0[:-1]).max()), float(np.abs(p2[:-1]).max())
        margin = max(pmax - pk2, 0.0)
        n_over = int((np.abs(p0[:-1]) > pmax).sum())
        n_active = int((np.abs(p2[:-1]) >= pmax - 1e-3).sum())
        dz2 = z2 - z0
        y_lo, y_hi = -24.0, 36.0
        ax_p = make_axes([-3.55, 0.85, 0], 5.6, 2.5, [0, T], [y_lo, y_hi], 0.5, 10)
        ax_d2 = make_axes([-3.55, -2.15, 0], 5.6, 1.6, [0, T], [-0.4, 0.4], 0.5, 0.2)
        decos2 = VGroup(
            axis_label("power P = τ · q̇ (W)", ax_p, C_STATE),
            axis_label("Δz = z − z_plain (m)", ax_d2, C_DIFF),
            time_label(ax_d2),
            x_ticks(ax_d2, [0, 1, 2]),
            y_ticks(ax_p, [0, 20]),
            y_ticks(ax_d2, [-0.4, 0, 0.4]),
        )
        forbidden = VGroup(
            band(ax_p, 0, T, pmax, y_hi, C_BOUND),
            band(ax_p, 0, T, y_lo, -pmax, C_BOUND),
            hline(ax_p, 0, T, pmax, C_BOUND),
            hline(ax_p, 0, T, -pmax, C_BOUND),
        )
        self.play(Create(ax_p), Create(ax_d2), FadeIn(decos2), FadeIn(forbidden), run_time=0.8)
        ghost2 = ghost_of(ax_p, t, p0)
        self.play(Create(ghost2), run_time=1.0)

        panel2 = code_panel(
            [
                (0, "def power(controller: PenaltyController) -> MX:", WHITE),
                (1, 'return controller.controls["tau"].cx * controller.states["qdot"].cx', C_CTRL),
                (0, "constraints.add(power, node=Node.ALL_SHOOTING,", C_BOUND),
                (1, "min_bound=-p_max, max_bound=p_max)", C_BOUND),
            ],
            size=19,
        )
        self.play(FadeIn(panel2), run_time=0.6)

        curve2 = poly(ax_p, t, p2, C_STATE, 5)
        curve_d2 = poly(ax_d2, t, dz2, C_DIFF, 5)
        zero2 = hline(ax_d2, 0, T, 0, GRAY_D, dashed=False)
        body2 = (
            f"limit p_max = {pmax:.1f} W  ·  margin = {margin:.2f} W\n"
            f"plain solve: peak |P| = {pk0:.1f} W, over the limit at {n_over} nodes\n"
            f"constrained: peak |P| = {pk2:.1f} W, at the limit at {n_active} nodes\n"
            + ipopt_line(int(d["con_iterations"]), bool(d["con_converged"]))
        )
        info2 = place(Text(body2, font_size=19, color=GRAY_A, line_spacing=0.9), CODE_X, -0.35)
        self.play(Create(curve2), Create(zero2), Create(curve_d2), FadeIn(info2), run_time=1.6)
        self.wait(0.6)
        remark2 = place(
            say("The plain solution enters the red region; the constrained one stays on its limit."),
            CODE_X,
            -1.9,
        )
        foot2 = footer(
            "Bioptim sets ConstraintFcn.CUSTOM by itself. The three solves start from the same guess; each returns a local minimum."
        )
        self.play(FadeIn(remark2), FadeIn(foot2))
        self.wait(2.5)

"""
Manim CE scene CustomDynamics (about 16 s of content at native speed, x1.6 with the series layer + 4.5 s end card):
what a user-defined model must provide to Bioptim (names and sizes of the states and controls, the state derivative),
shown on the pendulum of bioptim/examples/toy_examples/custom_model written directly in CasADi (no biorbd), then two
REAL solves that differ only by a viscous friction term -damping * qdot in the custom dynamics function.

Beat 1: the pieces of the custom class and the state derivative. Beat 2: the torque without friction (ghost), with
friction, their difference, and the energy dissipated, all computed from the solutions.

Data: data/customdyn_pendulum.npz, produced by generate_customdyn_data.py (class DampedPendulum = the code shown).
Render: python docs/animations/render_series.py anim_customdyn.py CustomDynamics --lang both
"""

import numpy as np
from manim import *

from features_scenes import (  # noqa: E402  (importing it also sets the default fonts Segoe UI / Consolas)
    C_CTRL,
    C_PAR,
    C_STATE,
    CODE_X,
    DATA_DIR,
    axis_label,
    code_panel,
    footer,
    hline,
    make_axes,
    place,
    say,
    scene_title,
    steps,
    time_label,
    x_ticks,
    y_ticks,
)

C_GHOST = GRAY_B  # the run without friction: dashed, grey
C_NEW = C_CTRL  # torque curve of the run with friction
C_FRIC = C_PAR  # the added friction term and the difference
CAPTION = "Bioptim code (custom_model example)"


class CustomDynamics(Scene):
    def construct(self):
        # ---------------------------------------------------------------- data: everything shown comes from the npz
        d = np.load(DATA_DIR / "customdyn_pendulum.npz")
        t, T, n = d["t"], float(d["final_time"]), int(d["n_shooting"])
        damping = float(d["damping"])
        tau0, tau1 = d["free_tau"], d["damped_tau"]
        assert bool(d["free_converged"]) and bool(d["damped_converged"])
        diff = tau1 - tau0
        peak_diff = float(np.abs(diff).max())
        dq = float(np.abs(d["damped_q"] - d["free_q"]).max())
        diss, drop = float(d["damped_dissipated"]), float(d["damped_potential_drop"])
        cost0, cost1 = float(d["free_cost"]), float(d["damped_cost"])
        it0, it1 = int(d["free_iterations"]), int(d["damped_iterations"])

        title = scene_title(
            "Custom dynamics: your own model", f"pendulum written in CasADi, not biorbd  ·  N = {n}, T = {T:g} s"
        )
        self.play(FadeIn(title), run_time=0.4)

        # ================================================================ beat 1: what the user provides
        eq_head = Text("state derivative returned by the dynamics function", font_size=20, color=C_GHOST)
        eq1 = Text("d(q)/dt = qdot", font_size=24, color=C_STATE)
        eq2 = Text("d(qdot)/dt = ( tau − g·m·L·sin(q) ) / (I + m·L²)", font_size=24, color=C_STATE)
        eq3 = Text("+ ( −d·qdot ) / (I + m·L²)", font_size=24, color=C_FRIC)
        eq_note = Text(f"the added viscous friction term, d = {damping:g} N·m·s/rad", font_size=19, color=C_FRIC)
        eqs = VGroup(eq_head, eq1, eq2, eq3, eq_note).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        if eqs.width > 6.0:
            eqs.scale_to_fit_width(6.0)
        eqs.move_to([-6.6, 1.9, 0], aligned_edge=UL)

        panel1 = code_panel(
            [
                (0, "class DampedPendulum(StateDynamics):", WHITE),
                (1, 'def name_dofs(self): return ["rotx"]', WHITE),
                (1, "def state_configuration_functions(self):", C_STATE),
                (2, "return [lambda ocp, nlp: ..., States.QDOT]", C_STATE),
                (1, "def control_configuration_functions(self):", C_CTRL),
                (2, "return [Controls.TAU]", C_CTRL),
                (1, "def dynamics(self, time, states, controls, ...):", WHITE),
                (2, "return DynamicsEvaluation(dxdt=vertcat(states[1], qddot))", WHITE),
                (1, "def forward_dynamics(self):  # qddot =", WHITE),
                (2, "1 / (I + m * L**2) * (-self.damping * self.qdot", C_FRIC),
                (3, "- g * m * L * sin(self.q) + self.tau)", C_STATE),
            ],
            size=18,
            caption=CAPTION,
        )
        self.play(FadeIn(panel1), run_time=0.8)
        self.play(FadeIn(eqs), run_time=1.0)
        remark1 = place(
            say("Bounds and objectives refer to the variable names you declare here: q, qdot and tau."), -6.6, -1.5
        )
        self.play(FadeIn(remark1), run_time=0.6)
        self.wait(1.2)

        # ================================================================ beat 2: two real solves
        self.play(*[FadeOut(m) for m in self.mobjects if m is not title], run_time=0.5)
        self.add(title)

        tau_lim = 20
        ax_y = make_axes([-3.55, 0.85, 0], 5.6, 2.5, [0, T], [-tau_lim, tau_lim], 0.5, 10)
        ax_d = make_axes([-3.55, -2.15, 0], 5.6, 1.6, [0, T], [-1, 6], 0.5, 3)
        decos = VGroup(
            axis_label("τ(t)  torque (N·m)", ax_y, C_NEW),
            axis_label("Δτ = τ − τ_free (N·m)", ax_d, C_FRIC),
            time_label(ax_d),
            x_ticks(ax_d, [0, 0.5, 1.0], "{:.1f}"),
            y_ticks(ax_y, [-20, 0, 20]),
            y_ticks(ax_d, [0, 5]),
        )
        self.play(Create(ax_y), Create(ax_d), FadeIn(decos), run_time=0.8)

        ghost = DashedVMobject(steps(ax_y, t, tau0, C_GHOST, 3), num_dashes=60).set_opacity(0.8)
        line_a = f"model = DampedPendulum(damping={float(d['free_damping']):.1f})"
        line_b = f"model = DampedPendulum(damping={damping:.1f})"
        rest = [
            (0, f"ocp = prepare_ocp(model, final_time={T:g}, n_shooting={n})", WHITE),
            (0, "sol = ocp.solve(solver=Solver.IPOPT())", WHITE),
        ]
        panel2 = code_panel([(0, line_a, C_FRIC)] + rest, size=19, caption=CAPTION)
        self.play(FadeIn(panel2), Create(ghost), run_time=1.2)
        self.wait(0.4)

        new_panel = code_panel([(0, line_b, C_FRIC)] + rest, size=19, caption=CAPTION)
        curve = steps(ax_y, t, tau1, C_NEW, 5)
        curve_d = steps(ax_d, t, diff, C_FRIC, 5)
        zero = hline(ax_d, 0, T, 0, GRAY_D, dashed=False)
        body = (
            f"friction dissipates {diss:.1f} J\n"
            f"of the {drop:.1f} J released by gravity\n"
            f"cost: {cost0:.1f} without friction, {cost1:.1f} with\n"
            f"max |Δτ| = {peak_diff:.1f} N·m\n"
            f"IPOPT: {it0} and {it1} iterations, both converged"
        )
        info = place(Text(body, font_size=19, color=GRAY_A, line_spacing=0.9), CODE_X, -0.2)
        self.play(
            Transform(panel2[1][0], new_panel[1][0]),
            Create(curve),
            Create(zero),
            Create(curve_d),
            FadeIn(info),
            run_time=1.8,
        )
        self.wait(0.8)

        remark = place(
            say(
                "Friction brakes the pendulum, so the torque has to brake less: the dashed curve is the run without it."
            ),
            CODE_X,
            -1.9,
        )
        self.play(FadeIn(remark))

        foot = footer(
            f"Same motion in both runs (max |Δq| = {dq:.3f} rad), only the torque changes.\n"
            "One-dof pendulum of the custom_model example; friction is the only change between the two solves."
        )
        self.play(FadeIn(foot))
        self.wait(2.5)

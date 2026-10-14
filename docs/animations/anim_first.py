"""
Manim CE scene: "My first optimal control problem (OCP)". The pendulum swing-up of
``bioptim/examples/getting_started/basic_ocp.py`` (README "A first practical example") built line by line: model,
objective, bounds, initial guess, then the REAL IPOPT solution. Every curve, pose (biorbd markers) and number comes from
``data/first_pendulum.npz`` produced by ``generate_first_data.py`` (N = 30, T = 1 s, RK4 multiple shooting, cold start).

Scene: FirstOCP (about 22 s of content at native speed, 2 beats: the build, then the solve).
Render (repo root):  python docs/animations/render_series.py anim_first.py FirstOCP --lang both
"""

import numpy as np
from manim import *

from features_scenes import (
    C_BOUND,
    C_CTRL,
    C_LAG,
    C_STATE,
    CODE_X,
    DATA_DIR,
    axis_label,
    band,
    code_panel,
    dec,
    footer,
    ipopt_line,
    make_axes,
    place,
    poly,
    say,
    scene_title,
    steps,
    time_label,
    x_ticks,
    y_ticks,
)

C_INIT = WHITE  # role: initial guess (dashed line, its code lines)
C_OLD = GRAY_B  # summary of the code already shown
SZ = 17  # code size
SC = 0.82  # metres -> scene units on the stage
X0S, ZC = -6.5, 1.75  # stage: screen x of y = -1.6 m, screen y of the rail (z = 0)


def P(y, z):
    """Stage coordinates (m) -> scene point."""
    return np.array([X0S + SC * (y + 1.6), ZC + SC * z, 0.0])


def pole(mk, k, color, width=6, dashed=False):
    """The pole: segment marker_1 -> marker_2 of the biorbd model, from the stored marker positions (y, z)."""
    a, b = P(*mk[:, 0, k]), P(*mk[:, 1, k])
    if dashed:
        return DashedLine(a, b, color=color, stroke_width=width, dash_length=0.12)
    return VGroup(
        Line(a, b, color=color, stroke_width=width),
        Dot(a, radius=0.06, color=color),
        Dot(b, radius=0.09, color=color),
    )


def band_xy(y0, y1, z0, z1):
    """Red forbidden band of the stage (metres)."""
    a, b = P(y0, z0), P(y1, z1)
    return Rectangle(
        width=abs(b[0] - a[0]), height=abs(b[1] - a[1]), stroke_width=0, fill_color=C_BOUND, fill_opacity=0.22
    ).move_to((a + b) / 2)


class FirstOCP(Scene):
    def construct(self):
        d = np.load(DATA_DIR / "first_pendulum.npz")
        t, q, tau = d["t"], d["q"], d["tau"]
        n, T = int(d["n_shooting"]), float(d["final_time"])
        mk_rest, mk_tgt, mk_sol = d["markers_rest"], d["markers_target"], d["markers_sol"]
        y_lo, y_hi = float(d["q_min"][0, 1]), float(d["q_max"][0, 1])  # cart range (intermediate nodes)
        q_end = float(d["q_max"][1, -1])  # prescribed final rotation
        f_lim = float(d["tau_max"][0, 0])  # force bound
        cart_min = float(q[0].min())
        assert abs(cart_min - y_lo) < 1e-5, "the scene says the cart touches its lower bound"
        peak = float(np.abs(tau[0]).max())
        assert bool(d["converged"]) and int(d["status"]) == 0

        title = scene_title(
            "My first optimal control problem (OCP)",
            f"pendulum swing-up, N = {n} intervals, T = {T:g} s, multiple shooting, IPOPT",
        )
        self.play(FadeIn(title), run_time=0.4)

        # ------------------------------------------------------------------ left: stage + two plots (built twice)
        def stage_frame():
            rail = Line(P(y_lo, 0), P(y_hi, 0), color=GRAY_B, stroke_width=3)
            base = Line(P(y_lo, -1.12), P(y_hi, -1.12), color=GRAY_B, stroke_width=2)
            ticks = VGroup(
                *[
                    Text(dec(f"{v:g}"), font_size=16, color=GRAY_B).next_to(P(v, -1.12), DOWN, buff=0.06)
                    for v in range(int(y_lo), int(y_hi) + 1)
                ]
            )
            unit = Text("cart position q₀ (m)", font_size=16, color=GRAY_B)
            unit.next_to(P(y_hi, -1.12), DOWN, buff=0.36).align_to(P(y_hi + 0.45, 0), RIGHT)
            return VGroup(rail, base, ticks, unit)

        def q_axes():
            ax = make_axes([-3.55, -0.6, 0], 5.6, 1.3, [0, T], [-1, 4.3], 0.5, 1)
            deco = VGroup(axis_label("q₁(t)  rotation (rad)", ax, C_STATE), y_ticks(ax, [0, q_end], "{:.2f}"))
            return ax, deco

        def tau_axes():
            ax = make_axes([-3.55, -2.5, 0], 5.6, 1.35, [0, T], [-125, 125], 0.5, 100)
            deco = VGroup(
                axis_label("force on the cart (N)", ax, C_CTRL),
                time_label(ax),
                x_ticks(ax, [0, T / 2, T], "{:.1f}"),
                y_ticks(ax, [-f_lim, 0, f_lim]),
            )
            return ax, deco

        def remark_at(text, y):
            return place(say(text), CODE_X, y)

        def guess(ax):
            zero = np.zeros_like(t)
            return DashedVMobject(poly(ax, t, zero, C_INIT, 3), num_dashes=40).set_opacity(0.8)

        def force_bands(ax):
            return VGroup(band(ax, 0, T, f_lim, 125, C_BOUND), band(ax, 0, T, -125, -f_lim, C_BOUND))

        def fixed_dots(ax):
            return VGroup(
                Dot(ax.c2p(0, 0), radius=0.08, color=C_BOUND), Dot(ax.c2p(T, q_end), radius=0.08, color=C_BOUND)
            )

        # ==================================================================== beat 1: build the problem line by line
        A = [
            (0, 'bio_model = TorqueBiorbdModel("pendulum.bioMod")', WHITE),
            (0, "dynamics = DynamicsOptions(ode_solver=OdeSolver.RK4())", WHITE),
        ]
        B = [
            (0, "objective = Objective(", C_LAG),
            (1, 'ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")', C_LAG),
        ]
        C = [
            (0, "x_bounds = BoundsList()", C_BOUND),
            (0, 'x_bounds["q"] = bio_model.bounds_from_ranges("q")', C_BOUND),
            (0, 'x_bounds["q"][:, [0, -1]] = 0', C_BOUND),
            (0, 'x_bounds["q"][1, -1] = 3.14', C_BOUND),
            (0, 'x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")', C_BOUND),
            (0, 'x_bounds["qdot"][:, [0, -1]] = 0', C_BOUND),
            (0, "u_bounds = BoundsList()", C_BOUND),
            (0, 'u_bounds["tau"] = [-100, -100], [100, 100]', C_BOUND),
            (0, 'u_bounds["tau"][1, :] = 0', C_BOUND),
        ]
        D = [
            (0, "x_init = InitialGuessList()", C_INIT),
            (0, 'x_init["q"] = [0, 0]', C_INIT),
            (0, 'x_init["qdot"] = [0, 0]', C_INIT),
            (0, "u_init = InitialGuessList()", C_INIT),
            (0, 'u_init["tau"] = [0, 0]', C_INIT),
        ]

        def old(txt):
            return [(0, txt, C_OLD)]

        panels = [
            code_panel(A, SZ),
            code_panel(A + B, SZ),
            code_panel(old("# bio_model, dynamics, objective: see above") + C, SZ),
            code_panel(old("# bio_model ... u_bounds: see above") + D, SZ),
        ]
        rem_y = min(p.get_bottom()[1] for p in panels) - 0.6

        # ---- step 1: the model (stick figure from the real biorbd markers)
        frame = stage_frame()
        cart = Rectangle(width=0.34, height=0.2, stroke_width=2, color=GRAY_A, fill_color=BLACK, fill_opacity=1)
        cart.move_to(P(*mk_rest[:, 0, 0]))
        pole0 = pole(mk_rest, 0, WHITE)
        self.play(Create(frame), FadeIn(cart), FadeIn(pole0), FadeIn(panels[0]), run_time=1.0)
        rem = remark_at("The model gives the pole: a segment between two markers, on a cart that slides.", rem_y)
        self.play(FadeIn(rem), run_time=0.4)
        self.wait(0.4)

        # ---- step 2: the objective (force axis appears with the cost it accumulates)
        ax_t, deco_t = tau_axes()
        cost_txt = Text("cost = Σ (force)² Δt", font_size=18, color=C_LAG).move_to(ax_t.c2p(T * 0.5, 55))
        rem2 = remark_at("The objective adds up the squared force at every node: less force, lower cost.", rem_y)
        self.play(FadeOut(panels[0]), FadeOut(rem), run_time=0.3)
        self.play(
            Create(ax_t),
            FadeIn(deco_t),
            FadeIn(cost_txt),
            FadeIn(panels[1]),
            FadeIn(rem2),
            run_time=0.9,
        )
        self.wait(0.4)

        # ---- step 3: the bounds (forbidden bands for the cart, fixed poses, force limits)
        ax_q, deco_q = q_axes()
        bnd = VGroup(band_xy(-1.6, y_lo, -1.1, 1.2), band_xy(y_hi, 5.6, -1.1, 1.2))
        ghost_tgt = pole(mk_tgt, 0, GRAY_B, 4, dashed=True)
        rem3 = remark_at(
            f"Bounds: start and end poses are fixed (red dots), and the cart stays between {y_lo:g} and {y_hi:g} m.",
            rem_y,
        )
        self.play(FadeOut(panels[1]), FadeOut(rem2), run_time=0.3)
        self.play(
            Create(ax_q),
            FadeIn(deco_q),
            FadeIn(bnd),
            FadeIn(ghost_tgt),
            FadeIn(fixed_dots(ax_q)),
            FadeIn(force_bands(ax_t)),
            FadeIn(panels[2]),
            FadeIn(rem3),
            run_time=1.1,
        )
        self.wait(0.5)

        # ---- step 4: the initial guess (dashed): a pendulum that does not move and no force
        rem4 = remark_at("The initial guess is where IPOPT starts: here nothing moves and no force is applied.", rem_y)
        self.play(FadeOut(panels[2]), FadeOut(rem3), run_time=0.3)
        self.play(
            Create(guess(ax_q)),
            Create(guess(ax_t)),
            FadeIn(panels[3]),
            FadeIn(rem4),
            run_time=1.0,
        )
        self.wait(0.5)

        # ==================================================================== beat 2: solve and read the result
        self.play(*[FadeOut(m) for m in self.mobjects if m is not title], run_time=0.5)
        self.add(title)

        ax_q2, deco_q2 = q_axes()
        ax_t2, deco_t2 = tau_axes()
        solve_lines = [
            (0, "# bio_model ... u_init: see above", C_OLD),
            (0, "ocp = OptimalControlProgram(", C_STATE),
            (1, "bio_model, 30, 1, dynamics=dynamics,", C_STATE),
            (1, "x_bounds=x_bounds, u_bounds=u_bounds,", C_STATE),
            (1, "x_init=x_init, u_init=u_init,", C_STATE),
            (1, "objective_functions=objective, use_sx=True)", C_STATE),
            (0, "sol = ocp.solve(Solver.IPOPT())", C_STATE),
        ]
        panel2 = code_panel(solve_lines, SZ)
        setup = VGroup(
            deco_q2,
            deco_t2,
            stage_frame(),
            VGroup(band_xy(-1.6, y_lo, -1.1, 1.2), band_xy(y_hi, 5.6, -1.1, 1.2)),
            force_bands(ax_t2),
            fixed_dots(ax_q2),
            guess(ax_q2),
            guess(ax_t2),
            pole(mk_rest, 0, GRAY_B, 4, dashed=True),
            pole(mk_tgt, 0, GRAY_B, 4, dashed=True),
        )
        self.play(Create(ax_q2), Create(ax_t2), FadeIn(setup), FadeIn(panel2), run_time=1.0)

        # the real solution: rotation and force curves, the tip path and some poses on the stage
        c_q = poly(ax_q2, t, q[1], C_STATE, 5)
        c_u = steps(ax_t2, t, tau[0], C_CTRL, 4)
        tip = VMobject(color=C_STATE, stroke_width=3).set_points_as_corners([P(*mk_sol[:, 1, k]) for k in range(n + 1)])
        tip.set_opacity(0.7)
        poses = VGroup(
            *[pole(mk_sol, k, C_STATE, 4).set_opacity(0.35 + 0.65 * k / n) for k in range(0, n, 6)],
            pole(mk_sol, n, C_STATE, 6),
        )
        self.play(Create(tip), FadeIn(poses), Create(c_q), Create(c_u), run_time=2.0)

        body = (
            f"cost = {float(d['cost']):.1f}  ·  peak force = {peak:.1f} N\n"
            + ipopt_line(int(d["iterations"]), bool(d["converged"]))
            + "\ncold start from the initial guess above"
        )
        info = place(Text(body, font_size=19, color=GRAY_A, line_spacing=0.9), CODE_X, -1.55)
        rem5 = remark_at(f"The cart backs up to its bound of {y_lo:g} m, then pushes the pole up.", -2.65)
        self.play(FadeIn(info), FadeIn(rem5), run_time=0.7)
        foot = footer("IPOPT returns a local minimum reached from this initial guess.")
        self.play(FadeIn(foot), run_time=0.4)
        self.wait(2.5)

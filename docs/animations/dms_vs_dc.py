"""
Manim Community animation: what is an optimal control problem, and how does bioptim discretize it with
Direct Multiple Shooting (DMS, ``OdeSolver.RK4``) versus Direct Collocation (DC, ``OdeSolver.COLLOCATION``)?

The example is the bioptim pendulum swing-up (bioptim/examples/toy_examples/sqp_method/pendulum.py, README tutorial).
All trajectories drawn here come from ``data/pendulum_solutions.npz`` (see ``generate_pendulum_data.py``): the
numbers are real bioptim/IPOPT results, not hand-drawn curves.

Scenes (render them one by one or all together, see README.md):
    1. OCPStatement          the OCP (x, u, dynamics, cost, constraints, bounds) and its bioptim API
    2. TimeGrid              N shooting intervals, states at the nodes, piecewise-constant controls
    3. MultipleShooting      RK4 integration inside each interval + continuity constraint (defects)
    4. DirectCollocation     collocation points + polynomial + defects at the collocation points
    5. Comparison            side by side summary, with the exact bioptim lines

No LaTeX needed: only Text / MarkupText with Unicode math.
"""

import sys
from pathlib import Path

import numpy as np
from manim import *

DATA = np.load(Path(__file__).parent / "data" / "pendulum_solutions.npz")
N = int(DATA["n_shooting"])
T_FINAL = float(DATA["final_time"])
H = T_FINAL / N
DEGREE = int(DATA["polynomial_degree"])
ROT = 1  # index of the pendulum rotation in q (index 0 is the sideways translation)

C_DMS = BLUE_C
C_DC = ORANGE
C_DEFECT = RED_C
C_CTRL = GREEN_C
C_STATE = YELLOW_C

# Explicit fonts: the Pango default font renders Unicode math (ẋ, ≤, ₖ) badly on Windows.
FONT = "Segoe UI" if sys.platform == "win32" else "DejaVu Sans"
MONO = "Consolas" if sys.platform == "win32" else "DejaVu Sans Mono"
Text.set_default(font=FONT)
MarkupText.set_default(font=FONT)

# Layout of the "zoom on 5 intervals" scenes (MultipleShooting, DirectCollocation)
WINDOW_WIDTH = 5
WINDOW_CENTER = np.array([-2.6, -0.2, 0.0])
PANEL_X = 2.75  # left edge of the explanation panel on the right


# --------------------------------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------------------------------
def keep_indent(text: str) -> str:
    """Replace leading spaces by non-breaking spaces (Pango strips regular leading spaces)."""
    stripped = text.lstrip(" ")
    return " " * (len(text) - len(stripped)) + stripped


def M(markup: str, size: float = 26, color=WHITE) -> MarkupText:
    """MarkupText with Pango markup (<sub>, <b>, ...). Use it for math with subscripts."""
    return MarkupText(keep_indent(markup), font_size=size, color=color)


def code(text: str, size: float = 20, color=WHITE, max_width: float = None) -> Text:
    """Monospace text for bioptim code, optionally shrunk to fit ``max_width``."""
    mob = Text(keep_indent(text), font=MONO, font_size=size, color=color)
    if max_width is not None and mob.width > max_width:
        mob.scale_to_fit_width(max_width)
    return mob


def code_block(lines: list, size: float = 22) -> VGroup:
    """Left aligned block of code lines given as (indent_level, text, color); indentation is done by shifting."""
    block = VGroup(*[code(text, size, color) for _, text, color in lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
    for line, (level, _, _) in zip(block, lines):
        line.shift(RIGHT * 0.35 * level)
    return block


def fit(mob: Mobject, max_width: float) -> Mobject:
    """Shrink ``mob`` (never enlarge) so that it is at most ``max_width`` wide."""
    if mob.width > max_width:
        mob.scale_to_fit_width(max_width)
    return mob


def scene_title(text: str, subtitle: str = None) -> VGroup:
    title = Text(text, font_size=34, weight=BOLD)
    group = VGroup(title)
    if subtitle:
        group.add(Text(subtitle, font_size=22, color=GRAY_B))
    group.arrange(DOWN, buff=0.12).to_edge(UP, buff=0.3)
    return group


def poly_through(t: np.ndarray, y: np.ndarray, t0: float):
    """
    Polynomial (degree len(t)-1) interpolating the points (t, y) of the interval starting at ``t0``.
    Returns two callables of time: the polynomial and its time derivative.
    """
    s = (t - t0) / H
    coeffs = np.polyfit(s, y, len(t) - 1)
    d_coeffs = np.polyder(coeffs)
    return (lambda tt: np.polyval(coeffs, (tt - t0) / H)), (lambda tt: np.polyval(d_coeffs, (tt - t0) / H) / H)


def collocation_points(degree: int, method: str) -> np.ndarray:
    """
    Collocation points on [0, 1] (same values as casadi.collocation_points used by bioptim, but with numpy only).
    legendre: roots of the shifted Legendre polynomial P_d (interior points only).
    radau: roots of P_(d-1) - P_d (the last point is 1, i.e. the end of the interval).
    """
    from numpy.polynomial import legendre as leg

    c_d = np.zeros(degree + 1)
    c_d[degree] = 1.0
    if method == "legendre":
        roots = leg.legroots(c_d)
    elif method == "radau":
        c_dm1 = np.zeros(degree + 1)
        c_dm1[degree - 1] = 1.0
        roots = leg.legroots(c_d - c_dm1)
    else:
        raise ValueError(method)
    return (np.sort(roots) + 1) / 2


class Window:
    """Axes zoomed on ``width`` consecutive intervals (from interval ``k0``) of a rotation trajectory."""

    def __init__(self, k0: int, width: int, y_min: float, y_max: float, x_length=8.4, y_length=4.0):
        self.k0, self.width = k0, width
        self.t0, self.t1 = k0 * H, (k0 + width) * H
        pad_t = 0.25 * H
        self.ax = Axes(
            x_range=[self.t0 - pad_t, self.t1 + pad_t, H],
            y_range=[y_min, y_max, 1],
            x_length=x_length,
            y_length=y_length,
            tips=False,
            axis_config={"color": GRAY_B, "stroke_width": 2, "include_ticks": False},
        ).move_to(WINDOW_CENTER)
        self.y_min, self.y_max = y_min, y_max

    def p(self, t: float, y: float) -> np.ndarray:
        return self.ax.c2p(t, y)

    def decorations(self) -> VGroup:
        """Node grid lines (dashed), t_k labels and axis captions."""
        group = VGroup()
        for k in range(self.k0, self.k0 + self.width + 1):
            line = DashedLine(self.p(k * H, self.y_min), self.p(k * H, self.y_max), color=GRAY_D, stroke_width=1.5)
            label = M(f"t<sub>{k}</sub>", 20, GRAY_B).next_to(self.p(k * H, self.y_min), DOWN, buff=0.15)
            group.add(line, label)
        ylab = Text("θ (rad)", font_size=22, color=GRAY_B).next_to(self.ax.get_y_axis(), UP, buff=0.1)
        ylab.align_to(self.ax.get_y_axis(), LEFT)
        xlab = Text("time", font_size=22, color=GRAY_B).next_to(self.ax.get_x_axis().get_end(), RIGHT, buff=0.1)
        group.add(ylab, xlab)
        return group

    def build_nodes(self, prefix: str) -> VGroup:
        return VGroup(
            *[
                Dot(self.p(k * H, DATA[f"{prefix}_q_nodes"][ROT, k]), radius=0.09, color=C_STATE)
                for k in range(self.k0, self.k0 + self.width + 1)
            ]
        )

    def node_labels(self, nodes: VGroup) -> VGroup:
        return VGroup(
            *[
                M(f"x<sub>{k}</sub>", 22, C_STATE).next_to(nodes[k - self.k0], UL if k > self.k0 else LEFT, buff=0.08)
                for k in range(self.k0, self.k0 + self.width + 1)
            ]
        )


def y_range_of(*arrays, pad=0.25):
    lo = min(float(np.min(a)) for a in arrays)
    hi = max(float(np.max(a)) for a in arrays)
    return lo - pad, hi + pad


def right_panel(lines: list, size: float = 20, top: float = 2.55) -> VGroup:
    """Explanation panel on the right: ``lines`` are (markup, color) tuples, stacked from the top."""
    panel = VGroup(*[M(markup, size, color) for markup, color in lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.13)
    fit(panel, 4.15)
    panel.move_to([PANEL_X, top, 0], aligned_edge=UL)
    return panel


def right_info(*mobs: Mobject, bottom: float = -2.95) -> VGroup:
    """Small status texts stacked at the bottom right (right aligned)."""
    info = VGroup(*mobs).arrange(DOWN, aligned_edge=RIGHT, buff=0.12)
    info.move_to([6.9, bottom, 0], aligned_edge=DR)
    return info


# --------------------------------------------------------------------------------------------------------------------
# Scene 1 - the OCP and its bioptim API
# --------------------------------------------------------------------------------------------------------------------
class OCPStatement(Scene):
    def construct(self):
        title = scene_title(
            "What is an Optimal Control Problem?", "Find the controls u(t) that drive the system x(t) best"
        )
        self.play(FadeIn(title, shift=DOWN * 0.3))

        # rows: (math markup, meaning, bioptim code, color)
        rows = [
            (
                "min  ∫<sub>0</sub><sup>T</sup> L(x, u) dt",
                "Lagrange cost",
                "ObjectiveFcn.Lagrange.MINIMIZE_CONTROL",
                C_CTRL,
            ),
            ("+  M(x(T))", "Mayer cost", "ObjectiveFcn.Mayer.MINIMIZE_STATE  (node=Node.END)", C_CTRL),
            ("s.t.  ẋ = f(x, u)", "dynamics", "TorqueBiorbdModel + DynamicsOptions(ode_solver=...)", C_STATE),
            ("x(0) = x<sub>0</sub>,  x(T) = x<sub>f</sub>", "boundary values", 'x_bounds["q"][:, [0, -1]] = 0', BLUE_C),
            (
                "x<sub>min</sub> ≤ x ≤ x<sub>max</sub>,  u<sub>min</sub> ≤ u ≤ u<sub>max</sub>",
                "bounds",
                "BoundsList: x_bounds / u_bounds",
                BLUE_C,
            ),
            ("g(x, u) ≤ 0", "path constraints", "Constraint(ConstraintFcn.TRACK_STATE, ...)", C_DEFECT),
        ]
        header_y = 2.2
        col_math, col_name, col_code = -6.9, -1.6, 0.9
        heads = VGroup(
            Text("Mathematics", font_size=20, color=GRAY_B).move_to([col_math, header_y, 0], aligned_edge=LEFT),
            Text("Meaning", font_size=20, color=GRAY_B).move_to([col_name, header_y, 0], aligned_edge=LEFT),
            Text("bioptim", font_size=20, color=GRAY_B).move_to([col_code, header_y, 0], aligned_edge=LEFT),
        )
        rule = Line([-6.95, header_y - 0.3, 0], [6.95, header_y - 0.3, 0], color=GRAY_D)
        self.play(FadeIn(heads), Create(rule))

        row_groups = []
        for i, (math, name, api, color) in enumerate(rows):
            y = header_y - 0.85 - i * 0.68
            m = fit(M(math, 24, color), 5.0).move_to([col_math, y, 0], aligned_edge=LEFT)
            n = Text(name, font_size=20, color=GRAY_A).move_to([col_name, y, 0], aligned_edge=LEFT)
            c = code(api, 18, WHITE, max_width=6.0).move_to([col_code, y, 0], aligned_edge=LEFT)
            row_groups.append(VGroup(m, n, c))
            self.play(FadeIn(m, shift=RIGHT * 0.3), FadeIn(n), FadeIn(c, shift=LEFT * 0.3), run_time=0.9)
            self.wait(0.4)

        legend = (
            VGroup(
                M(
                    "<b>x</b>: state (q, q̇)  ·  <b>u</b>: control (τ)  ·  <b>T</b>: phase_time  ·  <b>N</b>: n_shooting",
                    21,
                    GRAY_A,
                ),
                M(
                    "The problem is infinite-dimensional (x(t), u(t) are functions)  →  discretize it into an NLP.",
                    21,
                    YELLOW_C,
                ),
            )
            .arrange(DOWN, buff=0.15)
            .to_edge(DOWN, buff=0.3)
        )
        self.play(FadeIn(legend, shift=UP * 0.2))
        self.wait(2.5)

        # ---- second beat: the pendulum example, animated with the real solution
        self.play(*[FadeOut(m) for m in [heads, rule, legend, *row_groups]])
        sub = Text(
            "The bioptim pendulum: swing up from hanging (θ = 0) to upright (θ = 3.14), minimum ∫ τ² dt",
            font_size=22,
            color=GRAY_A,
        )
        fit(sub, 13).next_to(title, DOWN, buff=0.3)
        self.play(FadeIn(sub))

        q = DATA["col_q_nodes"]
        t_nodes = np.linspace(0, T_FINAL, N + 1)
        x_scale = 2.0 / max(1e-6, float(np.abs(q[0]).max()))
        rail_y, length, x_origin = -0.4, 1.9, -1.8
        rail = Line([-6.4, rail_y, 0], [3.2, rail_y, 0], color=GRAY_C)
        tracker = ValueTracker(0.0)

        def pendulum():
            t = tracker.get_value()
            y, th = np.interp(t, t_nodes, q[0]), np.interp(t, t_nodes, q[ROT])
            cart_c = np.array([x_origin + y * x_scale, rail_y, 0])
            tip = cart_c + np.array([length * np.sin(th), -length * np.cos(th), 0])
            cart = Rectangle(width=0.8, height=0.35, color=WHITE, fill_opacity=0.9, fill_color=GRAY_D).move_to(cart_c)
            return VGroup(cart, Line(cart_c, tip, color=C_STATE, stroke_width=6), Dot(tip, radius=0.16, color=C_STATE))

        legend2 = VGroup(
            M("q = (y, θ)", 22),
            M("x = (q, q̇)", 22),
            M("u = (F, 0)", 22),
            M("θ(0) = 0,  θ(T) = 3.14", 20, GRAY_A),
            M("q̇(0) = q̇(T) = 0", 20, GRAY_A),
            M("only the sideways force F is actuated", 20, GRAY_A),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        fit(legend2, 3.3).move_to([6.7, 0.3, 0], aligned_edge=RIGHT)
        pend = always_redraw(pendulum)
        self.play(Create(rail), FadeIn(pend), FadeIn(legend2))
        self.play(tracker.animate.set_value(T_FINAL), run_time=5, rate_func=linear)
        self.wait(1.5)


# --------------------------------------------------------------------------------------------------------------------
# Scene 2 - the time grid
# --------------------------------------------------------------------------------------------------------------------
class TimeGrid(Scene):
    def construct(self):
        title = scene_title("Discretizing time: shooting nodes", "n_shooting = N intervals of length Δt = T / N")
        self.play(FadeIn(title))

        t_nodes = np.linspace(0, T_FINAL, N + 1)
        theta = DATA["col_q_nodes"][ROT]
        tau = DATA["col_tau"][0]

        ax_x = Axes(
            x_range=[0, T_FINAL, H],
            y_range=[min(theta.min(), 0) - 0.2, 3.6, 1],
            x_length=10.5,
            y_length=2.3,
            tips=False,
            axis_config={"color": GRAY_B, "stroke_width": 2, "include_ticks": False},
        ).move_to([0.6, 0.5, 0])
        ax_u = Axes(
            x_range=[0, T_FINAL, H],
            y_range=[float(tau.min()) - 3, float(tau.max()) + 3, 10],
            x_length=10.5,
            y_length=1.9,
            tips=False,
            axis_config={"color": GRAY_B, "stroke_width": 2, "include_ticks": False},
        ).move_to([0.6, -1.75, 0])

        lab_x = M("state θ(t<sub>k</sub>)", 22, C_STATE).move_to([-6.9, 0.5, 0], aligned_edge=LEFT)
        lab_u = M("control τ(t)", 22, C_CTRL).move_to([-6.9, -1.75, 0], aligned_edge=LEFT)
        self.play(Create(ax_x), Create(ax_u), FadeIn(lab_x), FadeIn(lab_u))

        grid = VGroup(
            *[
                DashedLine(ax_x.c2p(t, ax_x.y_range[0]), ax_u.c2p(t, ax_u.y_range[0]), color=GRAY_D, stroke_width=1)
                for t in t_nodes
            ]
        )
        self.play(Create(grid), run_time=1.5)

        dots = VGroup(*[Dot(ax_x.c2p(t, th), radius=0.06, color=C_STATE) for t, th in zip(t_nodes, theta)])
        self.play(LaggedStart(*[FadeIn(d, scale=2) for d in dots], lag_ratio=0.08), run_time=2)

        steps = VGroup()
        for k in range(N):
            steps.add(
                Line(ax_u.c2p(t_nodes[k], tau[k]), ax_u.c2p(t_nodes[k + 1], tau[k]), color=C_CTRL, stroke_width=5)
            )
            if k < N - 1:
                steps.add(
                    Line(
                        ax_u.c2p(t_nodes[k + 1], tau[k]),
                        ax_u.c2p(t_nodes[k + 1], tau[k + 1]),
                        color=C_CTRL,
                        stroke_width=2,
                    )
                )
        self.play(LaggedStart(*[Create(s) for s in steps], lag_ratio=0.03), run_time=2.5)

        # annotate one interval
        k = 6
        y_bottom = ax_u.c2p(0, ax_u.y_range[0])[1] - 0.05
        brace = BraceBetweenPoints(
            [ax_u.c2p(t_nodes[k + 1], 0)[0], y_bottom, 0],
            [ax_u.c2p(t_nodes[k], 0)[0], y_bottom, 0],
            direction=DOWN,
            color=WHITE,
        )
        brace_lbl = M("Δt = T / N", 22).next_to(brace, RIGHT, buff=0.15)
        hi_x = Circle(radius=0.16, color=WHITE).move_to(dots[k])
        note_x = M("x<sub>k</sub>  decision variable at node t<sub>k</sub>  (N + 1 nodes)", 22, C_STATE)
        note_u = M("u<sub>k</sub>  held constant on [t<sub>k</sub>, t<sub>k+1</sub>]  (N steps)", 22, C_CTRL)
        notes = (
            VGroup(note_x, note_u).arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to([-6.9, 2.2, 0], aligned_edge=UL)
        )
        self.play(GrowFromCenter(brace), FadeIn(brace_lbl), Create(hi_x), FadeIn(note_x))
        self.play(FadeIn(note_u))
        self.wait(1)
        foot = Text("Same grid for DMS and DC. Next: what happens between two nodes?", font_size=22, color=YELLOW_C)
        foot.to_edge(DOWN, buff=0.2)
        self.play(FadeIn(foot))
        self.wait(2)


# --------------------------------------------------------------------------------------------------------------------
# Scene 3 - Direct Multiple Shooting
# --------------------------------------------------------------------------------------------------------------------
class MultipleShooting(Scene):
    def construct(self):
        title = scene_title("Direct Multiple Shooting", "OdeSolver.RK4(n_integration_steps=5)")
        title[1].set_color(C_DMS)
        self.play(FadeIn(title))

        it, conv = "rk4_it", "rk4"
        k0 = N // 2 - 2  # a window in the middle of the swing-up
        ks = range(k0, k0 + WINDOW_WIDTH)
        y_lo, y_hi = y_range_of(
            DATA[f"{it}_q_steps"][k0 : k0 + WINDOW_WIDTH, ROT],
            DATA[f"{conv}_q_steps"][k0 : k0 + WINDOW_WIDTH, ROT],
            DATA[f"{it}_q_nodes"][ROT, k0 : k0 + WINDOW_WIDTH + 1],
            DATA[f"{conv}_q_nodes"][ROT, k0 : k0 + WINDOW_WIDTH + 1],
        )
        win = Window(k0, WINDOW_WIDTH, y_lo, y_hi)
        self.play(Create(win.ax), FadeIn(win.decorations()))

        def build(prefix):
            nodes = win.build_nodes(prefix)
            segs, ticks = VGroup(), VGroup()
            for k in ks:
                pts = [win.p(t, y) for t, y in zip(DATA[f"{prefix}_t_steps"][k], DATA[f"{prefix}_q_steps"][k, ROT])]
                segs.add(VMobject(color=C_DMS, stroke_width=5).set_points_as_corners(pts))
                ticks.add(*[Dot(pt, radius=0.045, color=WHITE) for pt in pts[1:-1]])
            return nodes, segs, ticks

        nodes, segs, ticks = build(it)
        labels = win.node_labels(nodes)
        side = right_panel(
            [
                ("1  Node states x<sub>k</sub> are decision variables", C_STATE),
                ("2  Inside [t<sub>k</sub>, t<sub>k+1</sub>] integrate ẋ = f(x, u<sub>k</sub>)", C_DMS),
                ("    with 5 RK4 steps  →  F(x<sub>k</sub>, u<sub>k</sub>)", C_DMS),
                ("3  Continuity: F(x<sub>k</sub>, u<sub>k</sub>) = x<sub>k+1</sub>", C_DEFECT),
            ]
        )
        state_txt = Text(
            f"IPOPT iterate no. {int(DATA[f'{it}_iterations'])}: initial guess, not converged",
            font_size=18,
            color=GRAY_A,
        )
        tick_note = Text("white dots: intermediate RK4 steps", font_size=18, color=GRAY_B)
        right_info(tick_note, state_txt)

        self.play(FadeIn(nodes), FadeIn(labels), FadeIn(side[0]), FadeIn(state_txt))
        self.wait(0.5)
        self.play(FadeIn(side[1]), FadeIn(side[2]))
        for seg in segs:
            self.play(Create(seg, rate_func=linear), run_time=0.8)
        self.play(FadeIn(ticks), FadeIn(tick_note), run_time=0.6)
        self.wait(0.5)

        def build_gaps(prefix):
            group = VGroup()
            for k in ks:
                a = win.p((k + 1) * H, DATA[f"{prefix}_q_steps"][k, ROT, -1])
                b = win.p((k + 1) * H, DATA[f"{prefix}_q_nodes"][ROT, k + 1])
                if np.linalg.norm(a - b) > 0.02:
                    group.add(Line(a, b, color=C_DEFECT, stroke_width=6))
            return group

        gap_lines = build_gaps(it)
        self.play(FadeIn(side[3]))
        self.play(LaggedStart(*[Create(g) for g in gap_lines], lag_ratio=0.2))
        defect_lbl = M("defect = F(x<sub>k</sub>, u<sub>k</sub>) − x<sub>k+1</sub> ≠ 0", 22, C_DEFECT)
        defect_lbl.move_to([WINDOW_CENTER[0], -3.0, 0])
        self.play(FadeIn(defect_lbl))
        self.wait(1.5)

        # convergence: the optimizer closes the gaps
        n2, s2, t2 = build(conv)
        end_txt = Text(
            f"IPOPT converged after {int(DATA[f'{conv}_iterations'])} iterations", font_size=18, color=GRAY_A
        )
        end_txt.move_to(state_txt.get_right(), aligned_edge=RIGHT)
        closing = M(
            "The optimizer moves x<sub>k</sub>, u<sub>k</sub> until every defect is 0:  x<sub>k+1</sub> = F(x<sub>k</sub>, u<sub>k</sub>)",
            22,
        )
        fit(closing, 12.8).to_edge(DOWN, buff=0.25)
        counts = M(
            f"Decision variables (whole pendulum): <b>{int(DATA['rk4_n_decision_variables'])}</b>\n"
            "x at N+1 nodes, u at N intervals",
            18,
            C_DMS,
        )
        counts.move_to([6.9, -0.6, 0], aligned_edge=RIGHT)
        self.play(FadeOut(defect_lbl))
        self.play(
            Transform(nodes, n2),
            Transform(segs, s2),
            Transform(ticks, t2),
            FadeOut(gap_lines),
            Transform(state_txt, end_txt),
            *[label.animate.shift(n2[i].get_center() - nodes[i].get_center()) for i, label in enumerate(labels)],
            run_time=3,
        )
        self.play(FadeIn(closing), FadeIn(counts))
        self.wait(3)


# --------------------------------------------------------------------------------------------------------------------
# Scene 4 - Direct Collocation
# --------------------------------------------------------------------------------------------------------------------
class DirectCollocation(Scene):
    def construct(self):
        title = scene_title(
            "Direct Collocation", f"OdeSolver.COLLOCATION(polynomial_degree={DEGREE}, method='legendre')"
        )
        title[1].set_color(C_DC)
        self.play(FadeIn(title))

        self.show_points()  # where are the collocation points?

        # three real IPOPT iterates of the same run: iteration 0 (poor guess), iteration 3, converged
        stages = [
            ("col_it", "IPOPT iterate no. 0: initial guess"),
            ("col_mid", f"IPOPT iterate no. {int(DATA['col_mid_iterations'])}"),
            ("col", f"IPOPT converged after {int(DATA['col_iterations'])} iterations"),
        ]
        prefixes = [name for name, _ in stages]
        tt = np.linspace(0, 1, 40)
        k0 = N // 2 - 2

        # ---- part 1: five intervals, nodes + collocation states + polynomial
        ks = list(range(k0, k0 + WINDOW_WIDTH))
        curves = {
            (prefix, k): poly_through(DATA[f"{prefix}_t_steps"][k], DATA[f"{prefix}_q_steps"][k, ROT], k * H)
            for prefix in prefixes
            for k in ks
        }
        y_lo, y_hi = y_range_of(
            *[curves[("col_it", k)][0](k * H + tt * H) for k in ks], DATA["col_it_q_nodes"][ROT, k0 : k0 + 6]
        )
        win = Window(k0, WINDOW_WIDTH, y_lo, y_hi)
        deco = win.decorations()
        self.play(Create(win.ax), FadeIn(deco))
        nodes = win.build_nodes("col_it")
        polys, colloc = VGroup(), VGroup()
        for k in ks:
            f, _ = curves[("col_it", k)]
            polys.add(
                VMobject(color=C_DC, stroke_width=5).set_points_smoothly(
                    [win.p(k * H + s * H, f(k * H + s * H)) for s in tt]
                )
            )
            for t, y in list(zip(DATA["col_it_t_steps"][k], DATA["col_it_q_steps"][k, ROT]))[1:]:
                colloc.add(Square(0.16, color=WHITE, fill_color=C_DC, fill_opacity=1).move_to(win.p(t, y)))
        labels = win.node_labels(nodes)
        side = right_panel(
            [
                ("1  Node states x<sub>k</sub> and", C_STATE),
                (f"    {DEGREE} collocation states x<sub>k,j</sub> per interval", C_DC),
                (f"2  A degree-{DEGREE} polynomial P<sub>k</sub> passes", C_DC),
                ("    through x<sub>k</sub> and the x<sub>k,j</sub>", C_DC),
                ("3  Defect at each collocation time t<sub>k,j</sub>:", C_DEFECT),
                ("    dP<sub>k</sub>/dt(t<sub>k,j</sub>) − f(x<sub>k,j</sub>, u<sub>k</sub>) = 0", C_DEFECT),
                ("4  Continuity: P<sub>k</sub>(t<sub>k+1</sub>) = x<sub>k+1</sub>", C_DEFECT),
            ],
            size=19,
        )
        state_txt = Text(stages[0][1], font_size=18, color=GRAY_A)
        right_info(state_txt)
        self.play(FadeIn(nodes), FadeIn(labels), FadeIn(side[0:2]), FadeIn(state_txt))
        self.play(FadeIn(colloc, scale=1.5), run_time=1.2)
        self.wait(0.3)
        self.play(FadeIn(side[2:4]))
        self.play(LaggedStart(*[Create(p) for p in polys], lag_ratio=0.3), run_time=2.5)
        poly_lbl = M(
            "P<sub>k</sub>(t) = Σ<sub>j</sub> x<sub>k,j</sub> L<sub>j</sub>(t)   (Lagrange polynomial)", 20, C_DC
        )
        poly_lbl.move_to([WINDOW_CENTER[0], 2.55, 0])
        self.play(FadeIn(poly_lbl))
        self.wait(2)

        # ---- part 2: zoom on ONE interval, defects and continuity gap, iterate 0 -> 3 -> converged
        kk = k0 + 2
        y_lo, y_hi = y_range_of(
            *[curves[(p, kk)][0](kk * H + tt * H) for p in prefixes],
            *[DATA[f"{p}_q_nodes"][ROT, kk : kk + 2] for p in prefixes],
            pad=0.06,
        )
        zoom = Window(kk, 1, y_lo, y_hi, y_length=3.7)
        self.play(FadeOut(VGroup(win.ax, nodes, labels, colloc, polys, poly_lbl, deco)))
        self.play(Create(zoom.ax), FadeIn(zoom.decorations()))
        self.play(FadeIn(side[4:6]))

        x_unit = np.linalg.norm(zoom.p(kk * H + 1.0, 0) - zoom.p(kk * H, 0))  # screen length per second
        y_unit = np.linalg.norm(zoom.p(0, 1.0) - zoom.p(0, 0))  # screen length per rad

        def tangent(t, y, slope, color, width):
            """Segment of fixed screen length 1.3 through (t, y) with the given slope (rad/s)."""
            d = np.array([x_unit, slope * y_unit, 0.0])
            d = d / np.linalg.norm(d) * 0.65
            c = zoom.p(t, y)
            return Line(c - d, c + d, color=color, stroke_width=width)

        def build_zoom(prefix):
            f, df = curves[(prefix, kk)]
            t_j = DATA[f"{prefix}_t_steps"][kk][1:]
            q_j = DATA[f"{prefix}_q_steps"][kk, ROT][1:]
            qd_j = DATA[f"{prefix}_qdot_steps"][kk, ROT][1:]
            nd = zoom.build_nodes(prefix)
            poly = VMobject(color=C_DC, stroke_width=5).set_points_smoothly(
                [zoom.p(kk * H + s * H, f(kk * H + s * H)) for s in tt]
            )
            sq = VGroup(
                *[
                    Square(0.2, color=WHITE, fill_color=C_DC, fill_opacity=1).move_to(zoom.p(t, y))
                    for t, y in zip(t_j, q_j)
                ]
            )
            tang = VGroup()
            for t, y, qd in zip(t_j, q_j, qd_j):
                tang.add(tangent(t, y, float(df(t)), C_DEFECT, 7), tangent(t, y, qd, WHITE, 3))
            gap = Line(
                zoom.p((kk + 1) * H, f((kk + 1) * H)),
                zoom.p((kk + 1) * H, DATA[f"{prefix}_q_nodes"][ROT, kk + 1]),
                color=C_DEFECT,
                stroke_width=8,
            )
            values = df(t_j) - qd_j
            numbers = "   ".join("0.00" if abs(v) < 0.005 else f"{v:+.2f}" for v in values)
            txt = M(
                f"defects dP/dt − q̇ at t<sub>k,1</sub>, t<sub>k,2</sub>, t<sub>k,3</sub> (rad/s):   {numbers}",
                19,
                C_DEFECT,
            )
            txt.move_to([WINDOW_CENTER[0], -3.4, 0])
            gap_val = f((kk + 1) * H) - DATA[f"{prefix}_q_nodes"][ROT, kk + 1]
            gtxt = M(f"P<sub>k</sub>(t<sub>k+1</sub>) − x<sub>k+1</sub> = {gap_val:+.3f} rad", 19, C_DEFECT)
            gtxt.move_to([WINDOW_CENTER[0], 2.55, 0])
            return nd, poly, sq, tang, gap, txt, gtxt

        nd, poly, sq, tang, gap, txt, gtxt = build_zoom(prefixes[0])
        zlabels = zoom.node_labels(nd)
        leg = VGroup(
            Line(ORIGIN, RIGHT * 0.5, color=C_DEFECT, stroke_width=7),
            Text("slope of P(t)", font_size=18),
            Line(ORIGIN, RIGHT * 0.5, color=WHITE, stroke_width=3),
            Text("slope given by dynamics f(x, u)", font_size=18),
        )
        leg[1].next_to(leg[0], RIGHT, buff=0.1)
        leg[2].next_to(leg[1], RIGHT, buff=0.4)
        leg[3].next_to(leg[2], RIGHT, buff=0.1)
        leg.move_to([WINDOW_CENTER[0], -3.0, 0])
        self.play(FadeIn(nd), FadeIn(zlabels), FadeIn(sq, scale=1.5))
        self.play(Create(poly))
        self.play(Create(tang), FadeIn(leg), FadeIn(txt))
        self.wait(1.5)
        self.play(FadeIn(side[6]))
        self.play(Create(gap), FadeIn(gtxt))
        self.wait(2)

        # the optimizer shrinks the defects: iteration 0 -> 3 -> converged (real iterates)
        for prefix, text in stages[1:]:
            n2, p2, s2, t2, g2, x2, gt2 = build_zoom(prefix)
            end_txt = Text(text, font_size=18, color=GRAY_A)
            end_txt.move_to(state_txt.get_right(), aligned_edge=RIGHT)
            self.play(
                Transform(nd, n2),
                Transform(poly, p2),
                Transform(sq, s2),
                Transform(tang, t2),
                Transform(gap, g2),
                Transform(txt, x2),
                Transform(gtxt, gt2),
                Transform(state_txt, end_txt),
                *[label.animate.shift(n2[i].get_center() - nd[i].get_center()) for i, label in enumerate(zlabels)],
                run_time=3,
            )
            self.wait(1.5)

        closing = M(
            "At the solution both slopes match (defects = 0) and the polynomials connect: bigger but sparser NLP", 22
        )
        fit(closing, 12.8).to_edge(DOWN, buff=0.12)
        counts = M(
            f"Decision variables (whole pendulum): <b>{int(DATA['col_n_decision_variables'])}</b>\n"
            f"x at nodes, {DEGREE} x per interval, u at N intervals",
            18,
            C_DC,
        )
        counts.move_to([6.9, -0.6, 0], aligned_edge=RIGHT)
        self.play(FadeIn(closing), FadeIn(counts))
        self.wait(3)

        self.radau_continuity(title)

    def radau_continuity(self, title):
        """Radau: the last collocation point is t(k+1), so the polynomial end IS a collocation state (real solution)."""
        self.play(*[FadeOut(m) for m in self.mobjects if m is not title])
        new_sub = Text(f"OdeSolver.COLLOCATION(polynomial_degree={DEGREE}, method='radau')", font_size=22, color=TEAL_C)
        new_sub.move_to(title[1])
        self.play(Transform(title[1], new_sub))

        prefix = "rad"
        k0, width = N // 2, 2
        ks = list(range(k0, k0 + width))
        tt = np.linspace(0, 1, 40)
        curves = {k: poly_through(DATA[f"{prefix}_t_steps"][k], DATA[f"{prefix}_q_steps"][k, ROT], k * H) for k in ks}
        y_lo, y_hi = y_range_of(
            *[f(k * H + tt * H) for k, (f, _) in curves.items()],
            DATA[f"{prefix}_q_nodes"][ROT, k0 : k0 + width + 1],
        )
        win = Window(k0, width, y_lo, y_hi)
        self.play(Create(win.ax), FadeIn(win.decorations()))
        nodes = win.build_nodes(prefix)
        polys, colloc = VGroup(), VGroup()
        for k in ks:
            f, _ = curves[k]
            polys.add(
                VMobject(color=TEAL_C, stroke_width=5).set_points_smoothly(
                    [win.p(k * H + s * H, f(k * H + s * H)) for s in tt]
                )
            )
            for t, y in list(zip(DATA[f"{prefix}_t_steps"][k], DATA[f"{prefix}_q_steps"][k, ROT]))[1:]:
                colloc.add(Square(0.16, color=WHITE, fill_color=TEAL_C, fill_opacity=1).move_to(win.p(t, y)))
        labels = win.node_labels(nodes)
        side = right_panel(
            [
                ("Radau points: the last one is τ = 1", TEAL_C),
                ("    i.e. the end of the interval t<sub>k+1</sub>", TEAL_C),
                ("The polynomial end P<sub>k</sub>(t<sub>k+1</sub>) is", C_STATE),
                ("itself the collocation state x<sub>k,3</sub>:", C_STATE),
                ("    x<sub>k,3</sub> = x<sub>k+1</sub>", C_DEFECT),
                ("Legendre: P<sub>k</sub>(t<sub>k+1</sub>) is not a", C_DC),
                ("collocation point, continuity is an extra", C_DC),
                ("constraint on the polynomial end", C_DC),
            ],
            size=19,
        )
        info = Text(f"converged solution ({int(DATA['rad_iterations'])} iterations)", font_size=18, color=GRAY_A)
        right_info(info)
        self.play(FadeIn(nodes), FadeIn(labels), FadeIn(info))
        self.play(FadeIn(colloc, scale=1.5), LaggedStart(*[Create(p) for p in polys], lag_ratio=0.3), run_time=2.5)
        self.play(FadeIn(side[0:2]))
        self.wait(1)
        rings = VGroup(
            *[Circle(radius=0.22, color=C_DEFECT, stroke_width=4).move_to(nodes[i + 1]) for i in range(width)]
        )
        self.play(FadeIn(side[2:5]), LaggedStart(*[Create(r) for r in rings], lag_ratio=0.3))
        tag = M("red rings: last collocation point = next node", 20, C_DEFECT)
        tag.move_to([WINDOW_CENTER[0], 2.55, 0])
        self.play(FadeIn(tag))
        self.wait(1.5)
        self.play(FadeIn(side[5:8]))
        closing = M(
            "Radau: order 2d − 1 = 5, continuity built in   ·   Legendre: order 2d = 6, continuity as a constraint", 21
        )
        fit(closing, 12.8).to_edge(DOWN, buff=0.25)
        self.play(FadeIn(closing))
        self.wait(4)

    def show_points(self):
        """A mini number line [0, 1] with the legendre and radau points, computed with numpy."""
        rows = VGroup()
        for i, (method, col) in enumerate((("legendre", C_DC), ("radau", TEAL_C))):
            ln = Line([-3, 0, 0], [3, 0, 0], color=GRAY_B).move_to([0, 1.0 - 2.0 * i, 0])
            pts = collocation_points(DEGREE, method)
            dots = VGroup(*[Dot(ln.point_from_proportion(p), color=col, radius=0.11) for p in pts])
            ends = VGroup(
                Dot(ln.get_start(), color=C_STATE, radius=0.09), Dot(ln.get_end(), color=C_STATE, radius=0.09)
            )
            name = (
                Text(f"method='{method}'", font=MONO, font_size=20, color=col)
                .next_to(ln, UP, buff=0.35)
                .align_to(ln, LEFT)
            )
            vals = Text("τ = " + ",  ".join(f"{p:.3f}" for p in pts), font_size=20, color=GRAY_A).next_to(
                ln, DOWN, buff=0.5
            )
            end0 = M("t<sub>k</sub>", 20, GRAY_B).next_to(ln.get_start(), DOWN, buff=0.12)
            end1 = M("t<sub>k+1</sub>", 20, GRAY_B).next_to(ln.get_end(), DOWN, buff=0.12)
            rows.add(VGroup(ln, dots, ends, name, vals, end0, end1))
        comment = (
            VGroup(
                Text(f"polynomial_degree = {DEGREE}  →  {DEGREE} collocation points per interval", font_size=22),
                Text(
                    "legendre: interior points, highest order (2d = 6)   ·   radau: last point = t(k+1), order 2d − 1 = 5",
                    font_size=19,
                    color=GRAY_A,
                ),
            )
            .arrange(DOWN, buff=0.15)
            .to_edge(DOWN, buff=0.4)
        )
        fit(comment, 13)
        self.play(FadeIn(rows), FadeIn(comment))
        self.wait(3.5)
        self.play(FadeOut(rows), FadeOut(comment))


# --------------------------------------------------------------------------------------------------------------------
# Scene 5 - side-by-side comparison
# --------------------------------------------------------------------------------------------------------------------
class Comparison(Scene):
    def construct(self):
        title = scene_title("DMS vs DC in bioptim", "same OCP, same grid, different transcription")
        self.play(FadeIn(title))

        n_x = int(DATA["rk4_q_nodes"].shape[0] * 2)  # states per node: q and qdot
        rows = [
            ("", "Direct Multiple Shooting", "Direct Collocation"),
            ("OdeSolver", "RK4(n_integration_steps=5)", f"COLLOCATION(polynomial_degree={DEGREE}, method='legendre')"),
            (
                "Extra unknowns",
                "none (integration is hidden in F)",
                f"{DEGREE} × {n_x} = {DEGREE * n_x} collocation states",
            ),
            (
                "Constraint per interval",
                "continuity: x<sub>k+1</sub> = F(x<sub>k</sub>, u<sub>k</sub>)",
                "defects at collocation points + continuity",
            ),
            ("Dynamics evaluated", "4 × 5 = 20 times (RK4 steps)", f"{DEGREE} times (once per collocation point)"),
            ("Approximation order", "4 (RK4)", f"{2 * DEGREE} (legendre)  /  {2 * DEGREE - 1} (radau)"),
            ("NLP structure", "smaller, denser Jacobian", "larger, very sparse Jacobian"),
            (
                f"This pendulum (N = {N})",
                f"{int(DATA['rk4_n_decision_variables'])} variables, {int(DATA['rk4_iterations'])} IPOPT iterations",
                f"{int(DATA['col_n_decision_variables'])} variables, {int(DATA['col_iterations'])} IPOPT iterations",
            ),
        ]
        x_cols = [-6.9, -4.0, 0.2]
        widths = [3.0, 4.0, 6.7]
        y = 2.0
        row_groups = []
        for i, row in enumerate(rows):
            group = VGroup()
            for j, cell in enumerate(row):
                if not cell:
                    continue
                if i == 0:
                    mob = MarkupText(f"<b>{cell}</b>", font_size=22, color=[WHITE, C_DMS, C_DC][j])
                elif i == 1 and j:
                    mob = code(cell, 17, [WHITE, C_DMS, C_DC][j])
                else:
                    mob = M(cell, 18, GRAY_A if j == 0 else WHITE)
                fit(mob, widths[j]).move_to([x_cols[j], y, 0], aligned_edge=LEFT)
                group.add(mob)
            if i == 0:
                group.add(Line([-6.95, y - 0.33, 0], [6.95, y - 0.33, 0], color=GRAY_D))
            row_groups.append(group)
            y -= 0.7 if i == 0 else 0.6
        for group in row_groups:
            self.play(FadeIn(group, shift=UP * 0.15), run_time=0.7)
            self.wait(0.3)

        note = Text(
            "Costs come from two independent IPOPT runs on a non-convex problem (they may end in different local minima):\n"
            "the variable / iteration counts are meaningful, the cost difference is NOT an accuracy ranking.",
            font_size=17,
            color=GRAY_B,
            line_spacing=0.9,
        )
        fit(note, 13).to_edge(DOWN, buff=0.2)
        self.play(FadeIn(note))
        self.wait(4)

        # ---- the exact lines of code
        table = VGroup(*row_groups)
        self.play(FadeOut(table), FadeOut(note))
        code_dms = VGroup(
            Text("Direct Multiple Shooting", font_size=26, color=C_DMS, weight=BOLD),
            code_block(
                [
                    (0, "dynamics = DynamicsOptions(", WHITE),
                    (1, "ode_solver=OdeSolver.RK4(", C_DMS),
                    (2, "n_integration_steps=5", C_DMS),
                    (1, ")", C_DMS),
                    (0, ")", WHITE),
                ]
            ),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        code_dc = VGroup(
            Text("Direct Collocation", font_size=26, color=C_DC, weight=BOLD),
            code_block(
                [
                    (0, "dynamics = DynamicsOptions(", WHITE),
                    (1, "ode_solver=OdeSolver.COLLOCATION(", C_DC),
                    (2, f"polynomial_degree={DEGREE},", C_DC),
                    (2, "method='legendre'", C_DC),
                    (1, ")", C_DC),
                    (0, ")", WHITE),
                ]
            ),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        VGroup(code_dms, code_dc).arrange(RIGHT, aligned_edge=UP, buff=0.8).move_to([0, 0.3, 0])
        takeaway = Text(
            "Only this line changes: the objective, bounds and model stay identical.", font_size=22, color=YELLOW_C
        )
        takeaway.to_edge(DOWN, buff=0.6)
        self.play(FadeIn(code_dms, shift=RIGHT * 0.3), FadeIn(code_dc, shift=LEFT * 0.3))
        self.play(FadeIn(takeaway))
        self.wait(4)

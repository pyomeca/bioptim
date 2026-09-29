"""
Manim Community animations, part 2: four bioptim features, each driven by REAL bioptim/IPOPT solves stored in
``data/features_*.npz`` (see ``generate_features_data.py``). Same pendulum swing-up as in ``dms_vs_dc.py``.

Scenes:
    1. ObjectivesNodes      Lagrange vs Mayer objectives, the Node enum, and what the weights do (5 real solves)
    2. ConstraintsBounds    u_bounds shrinking (4 real solves) and a bound on a state (x_bounds)
    3. MultiphaseTransitions  two phases of different durations, PhaseTransitionFcn.CONTINUOUS vs DISCONTINUOUS
    4. FreeTime             ObjectiveFcn.Mayer.MINIMIZE_TIME: the phase duration becomes an optimization variable

The code lines shown next to the curves are the ones used in generate_features_data.py (names checked against the
bioptim source). No LaTeX needed. Render commands: see FEATURES.md.
"""

import sys
from pathlib import Path

import numpy as np
from manim import *

DATA_DIR = Path(__file__).parent / "data"
ROT = 1  # rotation index in q
TRANS = 0  # sideways translation index in q (the actuated one)

C_LAG = GREEN_C
C_MAY = ORANGE
C_CTRL = GREEN_C
C_STATE = YELLOW_C
C_BOUND = RED_C
C_PH0 = BLUE_C
C_PH1 = ORANGE
C_TIME = TEAL_C

FONT = "Segoe UI" if sys.platform == "win32" else "DejaVu Sans"
MONO = "Consolas" if sys.platform == "win32" else "DejaVu Sans Mono"
Text.set_default(font=FONT)
MarkupText.set_default(font=FONT)

PLOT_X0, PLOT_X1 = -6.6, -0.5  # horizontal extent of the plots (left part of the frame)
CODE_X = 0.15  # left edge of the code panel (right part of the frame)
CODE_W = 6.75


# --------------------------------------------------------------------------------------------------------------------
# Helpers (copied from dms_vs_dc.py, kept here so that the file is self contained)
# --------------------------------------------------------------------------------------------------------------------
def keep_indent(text: str) -> str:
    stripped = text.lstrip(" ")
    return " " * (len(text) - len(stripped)) + stripped


def M(markup: str, size: float = 26, color=WHITE) -> MarkupText:
    return MarkupText(keep_indent(markup), font_size=size, color=color)


def code(text: str, size: float = 20, color=WHITE, max_width: float = None) -> Text:
    mob = Text(keep_indent(text), font=MONO, font_size=size, color=color)
    if max_width is not None and mob.width > max_width:
        mob.scale_to_fit_width(max_width)
    return mob


def fit(mob: Mobject, max_width: float) -> Mobject:
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


def code_panel(lines: list, size: float = 19, top: float = 2.3, caption: str = None) -> VGroup:
    """
    Code lines given as (indent_level, text, color) stacked at the top of the right panel. The whole block is scaled
    to fit CODE_W; an optional caption (Text) is put above.
    """
    block = VGroup(*[code(text, size, color) for _, text, color in lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.13)
    for line, (level, _, _) in zip(block, lines):
        line.shift(RIGHT * 0.3 * level)
    if caption:
        cap = Text(caption, font_size=20, color=GRAY_B)
        group = VGroup(cap, block).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
    else:
        group = VGroup(block)
    fit(group, CODE_W)
    group.move_to([CODE_X, top, 0], aligned_edge=UL)
    return group


def make_axes(center, width, height, x_range, y_range, x_step=None, y_step=None) -> Axes:
    return Axes(
        x_range=[x_range[0], x_range[1], x_step or (x_range[1] - x_range[0])],
        y_range=[y_range[0], y_range[1], y_step or (y_range[1] - y_range[0])],
        x_length=width,
        y_length=height,
        tips=False,
        axis_config={"color": GRAY_B, "stroke_width": 2, "include_ticks": False},
    ).move_to(center)


def poly(ax: Axes, t, y, color, width=5) -> VMobject:
    return VMobject(color=color, stroke_width=width).set_points_as_corners([ax.c2p(a, b) for a, b in zip(t, y)])


def steps(ax: Axes, t, u, color, width=4) -> VMobject:
    """Piecewise constant control: u[k] on [t[k], t[k+1]] (always 2N points, so curves can be morphed)."""
    pts = []
    for k in range(len(u)):
        pts += [ax.c2p(t[k], u[k]), ax.c2p(t[k + 1], u[k])]
    return VMobject(color=color, stroke_width=width).set_points_as_corners(pts)


def axis_label(text: str, ax: Axes, color=GRAY_B) -> Text:
    lab = Text(text, font_size=20, color=color).next_to(ax.get_y_axis(), UP, buff=0.08)
    return lab.align_to(ax.get_y_axis(), LEFT)


def time_label(ax: Axes, text="t (s)") -> Text:
    """Caption right of the last tick (the time axis crosses the plot at y = 0, ticks are at the bottom)."""
    return Text(text, font_size=16, color=GRAY_B).move_to(
        ax.c2p(ax.x_range[1], ax.y_range[0]) + np.array([0.62, -0.24, 0])
    )


def place(mob: Mobject, x: float, y: float, max_width: float = CODE_W) -> Mobject:
    """Shrink to ``max_width`` if needed, then put the left edge at x, the vertical center at y."""
    fit(mob, max_width)
    return mob.move_to([x, y, 0], aligned_edge=LEFT)


def hline(ax: Axes, t0, t1, y, color, dashed=True) -> Mobject:
    a, b = ax.c2p(t0, y), ax.c2p(t1, y)
    return DashedLine(a, b, color=color, stroke_width=3) if dashed else Line(a, b, color=color, stroke_width=3)


def band(ax: Axes, t0, t1, y0, y1, color, opacity=0.22) -> Rectangle:
    a, b = ax.c2p(t0, y0), ax.c2p(t1, y1)
    return Rectangle(
        width=abs(b[0] - a[0]), height=abs(b[1] - a[1]), stroke_width=0, fill_color=color, fill_opacity=opacity
    ).move_to((a + b) / 2)


def y_ticks(ax: Axes, values, fmt="{:g}") -> VGroup:
    return VGroup(
        *[
            Text(fmt.format(v), font_size=16, color=GRAY_B).next_to(ax.c2p(ax.x_range[0], v), LEFT, buff=0.08)
            for v in values
        ]
    )


def x_ticks(ax: Axes, values, fmt="{:g}") -> VGroup:
    return VGroup(
        *[
            Text(fmt.format(v), font_size=16, color=GRAY_B).next_to(ax.c2p(v, ax.y_range[0]), DOWN, buff=0.08)
            for v in values
        ]
    )


# ====================================================================================================================
# Scene 1 - objectives: Lagrange vs Mayer, and the Node enum
# ====================================================================================================================
class ObjectivesNodes(Scene):
    def construct(self):
        d = np.load(DATA_DIR / "features_objectives.npz")
        weights = d["mayer_weights"]
        n, T = int(d["n_shooting"]), float(d["final_time"])

        title = scene_title("Objectives: Lagrange and Mayer", "which nodes does each penalty act on?")
        self.play(FadeIn(title))

        # ---------------------------------------------------------------- beat 1: the Node enum on a time grid
        n_demo = 10
        xs = np.linspace(-2.6, 3.4, n_demo + 1)
        rows = [
            ("Node.START", [0], "first node"),
            ("Node.INTERMEDIATES", list(range(1, n_demo - 1)), "1 … N−2  (bioptim excludes N−1 too)"),
            ("Node.PENULTIMATE", [n_demo - 1], "node N−1"),
            ("Node.END", [n_demo], "last node N"),
            ("Node.ALL_SHOOTING", list(range(n_demo)), "0 … N−1  (all nodes with a control)"),
            ("Node.ALL", list(range(n_demo + 1)), "0 … N"),
        ]
        y0 = 1.85
        header = VGroup(
            *[
                M(f"t<sub>{k}</sub>" if k in (0, 1, n_demo - 1, n_demo) else "…" if k == 2 else "", 17, GRAY_B).move_to(
                    [x, y0 + 0.45, 0]
                )
                for k, x in enumerate(xs)
            ]
        )
        header[n_demo - 1].become(M(f"t<sub>N−1</sub>", 17, GRAY_B).move_to([xs[n_demo - 1], y0 + 0.45, 0]))
        header[n_demo].become(M("t<sub>N</sub>", 17, GRAY_B).move_to([xs[n_demo], y0 + 0.45, 0]))
        self.play(FadeIn(header))
        grid = VGroup()
        row_mobs = []
        for i, (name, idx, comment) in enumerate(rows):
            y = y0 - i * 0.62
            base = VGroup(*[Dot([x, y, 0], radius=0.07, color=GRAY_D) for x in xs])
            on = VGroup(*[Dot([xs[k], y, 0], radius=0.11, color=C_MAY) for k in idx])
            lab = code(name, 18, WHITE).move_to([-6.9, y, 0], aligned_edge=LEFT)
            com = Text(comment, font_size=16, color=GRAY_B).move_to([3.9, y, 0], aligned_edge=LEFT)
            fit(com, 3.0).move_to([3.9, y, 0], aligned_edge=LEFT)
            grid.add(base)
            row_mobs.append((lab, base, on, com))
            self.play(FadeIn(lab), FadeIn(base), FadeIn(on, scale=1.6), FadeIn(com), run_time=0.7)

        # Lagrange integrates over the intervals, Mayer sits on a node
        y_lag, y_may = y0 - 4 * 0.62, y0 - 3 * 0.62
        shades = VGroup(
            *[
                Rectangle(
                    width=xs[1] - xs[0] - 0.04, height=0.3, stroke_width=0, fill_color=C_LAG, fill_opacity=0.45
                ).move_to([(xs[k] + xs[k + 1]) / 2, y_lag, 0])
                for k in range(n_demo)
            ]
        )
        lag_txt = M(
            "<b>Lagrange</b>  ∫ L(x, u) dt : summed over the N intervals   →  Node.ALL_SHOOTING (default)", 20, C_LAG
        )
        may_txt = M("<b>Mayer</b>  M(x) : evaluated at one node   →  Node.END (default)", 20, C_MAY)
        summary = (
            VGroup(lag_txt, may_txt)
            .arrange(DOWN, aligned_edge=LEFT, buff=0.15)
            .move_to([-6.9, -2.05, 0], aligned_edge=LEFT)
        )
        fit(summary, 13.4)
        summary.move_to([-6.9, -2.05, 0], aligned_edge=LEFT)
        self.play(FadeIn(shades), FadeIn(lag_txt))
        self.play(Indicate(row_mobs[3][2], color=C_MAY, scale_factor=1.6), FadeIn(may_txt))
        foot = Text("Schematic grid with N = 10; the real problem below uses N = 30.", font_size=17, color=GRAY_B)
        foot.to_edge(DOWN, buff=0.25)
        self.play(FadeIn(foot))
        self.wait(2.5)

        # ---------------------------------------------------------------- beat 2: weights, with real solves
        self.play(*[FadeOut(m) for m in self.mobjects if m is not title])
        self.add(title)
        ax_q = make_axes([-3.55, 0.9, 0], 5.6, 2.7, [0, T], [-0.6, 3.6], 0.5, 1)
        ax_u = make_axes([-3.55, -2.2, 0], 5.6, 1.7, [0, T], [-40, 40], 0.5, 40)
        decos = VGroup(
            axis_label("θ(t)  pendulum angle (rad)", ax_q, C_STATE),
            axis_label("τ(t)  actuated force (N)", ax_u, C_CTRL),
            time_label(ax_u),
            x_ticks(ax_u, [0, 0.5, 1.0], "{:.1f}"),
            y_ticks(ax_q, [0, 1, 2, 3]),
            y_ticks(ax_u, [-40, 0, 40]),
        )
        target_line = hline(ax_q, 0, T, 3.14, GRAY_B)
        target_lbl = (
            Text("3.14 (upright)", font_size=16, color=GRAY_B)
            .next_to(ax_q.c2p(T, 3.14), UP, buff=0.05)
            .shift(LEFT * 0.7)
        )
        self.play(Create(ax_q), Create(ax_u), FadeIn(decos), Create(target_line), FadeIn(target_lbl))

        t_nodes = np.linspace(0, T, n + 1)

        def curves(i):
            q = d[f"w{i}_p0_q"][ROT]
            tau = d[f"w{i}_p0_tau"][TRANS]
            return poly(ax_q, t_nodes, q, C_STATE), steps(ax_u, t_nodes, tau, C_CTRL)

        panel = code_panel(
            [
                (0, "objectives = ObjectiveList()", WHITE),
                (0, "objectives.add(", C_LAG),
                (1, "ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,", C_LAG),
                (1, 'key="tau", weight=1.0,', C_LAG),
                (1, "node=Node.ALL_SHOOTING)", C_LAG),
                (0, "objectives.add(", C_MAY),
                (1, "ObjectiveFcn.Mayer.TRACK_STATE,", C_MAY),
                (1, 'key="q", index=[1], target=3.14,', C_MAY),
                (1, "node=Node.END,", C_MAY),
                (1, "weight=0)", C_MAY),
            ],
            caption="Bioptim code (Mayer weight is the only change)",
        )
        self.play(FadeIn(panel))
        weight_line = panel[1][-1]
        curve_q, curve_u = curves(0)
        self.play(Create(curve_q), Create(curve_u), run_time=1.5)

        def readout(i):
            w = weights[i]
            q_end = float(d[f"w{i}_p0_q"][ROT, -1])
            tau_max = float(np.abs(d[f"w{i}_p0_tau"][TRANS]).max())
            body = (
                f"Mayer weight = {w:g}\nθ(T) = {q_end:.2f} rad   ·   max |τ| = {tau_max:.0f} N\n"
                f"IPOPT: {int(d[f'w{i}_iterations'])} iterations, status {'converged' if d[f'w{i}_converged'] else 'not converged'}"
            )
            return place(Text(body, font_size=19, color=GRAY_A, line_spacing=0.9), CODE_X, -2.3)

        info = readout(0)
        self.play(FadeIn(info))
        comments = [
            "Only the Lagrange term (weight 1.0):\nthe cheapest control is tau = 0, nothing moves.",
            "Small Mayer weight: reaching 3.14 costs more\ntorque than the terminal error saves.",
            "Larger weight: the pendulum goes further up.",
            "Larger still: almost at the target.",
            "A large Mayer weight makes the final state (almost) a\nhard target; the Lagrange term still shapes the way.",
        ]
        comments = [c.replace("tau", "τ") for c in comments]

        def comment_mob(i):
            return place(Text(comments[i], font_size=19, color=YELLOW_C), CODE_X, -3.3)

        comment = comment_mob(0)
        self.play(FadeIn(comment))
        self.wait(1.5)
        for i in range(1, len(weights)):
            new_line = code(f"weight={weights[i]:g})", 19, C_MAY)
            new_line.move_to(weight_line, aligned_edge=LEFT)
            new_q, new_u = curves(i)
            self.play(
                Transform(curve_q, new_q),
                Transform(curve_u, new_u),
                Transform(weight_line, new_line),
                Transform(info, readout(i)),
                Transform(comment, comment_mob(i)),
                run_time=2.2,
            )
            self.wait(0.8)
        self.wait(3)


# ====================================================================================================================
# Scene 2 - constraints and bounds
# ====================================================================================================================
class ConstraintsBounds(Scene):
    def construct(self):
        d = np.load(DATA_DIR / "features_constraints.npz")
        u_maxs = d["u_maxs"]
        n, T = int(d["n_shooting"]), float(d["final_time"])
        t_nodes = np.linspace(0, T, n + 1)

        title = scene_title(
            "Bounds and constraints", "same swing-up (T = 1 s, minimum ∫ τ² dt), the torque limit shrinks"
        )
        self.play(FadeIn(title))

        y_lim = 35
        ax_u = make_axes([-3.55, 0.85, 0], 5.6, 2.6, [0, T], [-y_lim, y_lim], 0.5, 20)
        ax_q = make_axes([-3.55, -2.2, 0], 5.6, 1.7, [0, T], [-1.2, 3.6], 0.5, 1)
        decos = VGroup(
            axis_label("τ(t)  actuated force (N)", ax_u, C_CTRL),
            axis_label("θ(t)  angle (rad)", ax_q, C_STATE),
            time_label(ax_q),
            x_ticks(ax_q, [0, 0.5, 1.0], "{:.1f}"),
            y_ticks(ax_u, [-30, 0, 30]),
            y_ticks(ax_q, [0, 3]),
        )
        self.play(Create(ax_u), Create(ax_q), FadeIn(decos))

        def bound_mobs(u):
            """Dashed limits and the shaded forbidden region beyond them (clipped to the plot)."""
            g = VGroup()
            for sign in (1, -1):
                if u < y_lim:
                    g.add(hline(ax_u, 0, T, sign * u, C_BOUND))
                    g.add(band(ax_u, 0, T, sign * u, sign * y_lim, C_BOUND))
                else:  # bound outside of the drawn range: keep same number of sub-mobjects, degenerate
                    g.add(hline(ax_u, 0, T, sign * y_lim, C_BOUND).set_opacity(0))
                    g.add(band(ax_u, 0, T, sign * y_lim, sign * y_lim * 0.999, C_BOUND, 0.0))
            return g

        panel = code_panel(
            [
                (0, "u_bounds = BoundsList()", WHITE),
                (0, 'u_bounds["tau"] = (', C_BOUND),
                (1, "[-100] * bio_model.nb_tau,", C_BOUND),
                (1, "[+100] * bio_model.nb_tau,", C_BOUND),
                (0, ")", C_BOUND),
                (0, 'u_bounds["tau"][1, :] = 0   # rotation passive', GRAY_B),
            ],
            caption="Bioptim code: bound on the control",
        )
        self.play(FadeIn(panel))
        lim_lines = [panel[1][2], panel[1][3]]

        def curves(i):
            return (
                steps(ax_u, t_nodes, d[f"u{i}_p0_tau"][TRANS], C_CTRL),
                poly(ax_q, t_nodes, d[f"u{i}_p0_q"][ROT], C_STATE),
            )

        bounds = bound_mobs(float(u_maxs[0]))
        curve_u, curve_q = curves(0)
        cap = Text(
            "|τ| ≤ 100 N is far away: the bound is inactive\n(the unconstrained optimum peaks at 24 N)",
            font_size=19,
            color=YELLOW_C,
        )

        def readout(i):
            u = float(u_maxs[i])
            peak = float(np.abs(d[f"u{i}_p0_tau"][TRANS]).max())
            active = "  (bound active)" if abs(peak - u) < 0.05 * u else ""
            return Text(
                f"|τ| ≤ {u:g} N   ·   peak |τ| = {peak:.1f} N{active}\ncost ∫ τ² dt = {float(d[f'u{i}_cost']):.1f}   ·   "
                f"IPOPT {int(d[f'u{i}_iterations'])} it.",
                font_size=19,
                color=GRAY_A,
                line_spacing=0.9,
            ).move_to([CODE_X, -1.0, 0], aligned_edge=LEFT)

        info = readout(0)
        self.play(Create(curve_u), Create(curve_q), FadeIn(bounds), FadeIn(info), run_time=1.5)
        place(cap, CODE_X, -2.4)
        self.play(FadeIn(cap))
        self.wait(1.2)
        for i in range(1, len(u_maxs)):
            u = int(u_maxs[i])
            new_lines = [
                code(f"[-{u}] * bio_model.nb_tau,", 19, C_BOUND),
                code(f"[+{u}] * bio_model.nb_tau,", 19, C_BOUND),
            ]
            for old, new in zip(lim_lines, new_lines):
                new.move_to(old, aligned_edge=LEFT)
            new_u, new_q = curves(i)
            anims = [
                Transform(curve_u, new_u),
                Transform(curve_q, new_q),
                Transform(bounds, bound_mobs(float(u_maxs[i]))),
                Transform(info, readout(i)),
                *[Transform(o, nw) for o, nw in zip(lim_lines, new_lines)],
            ]
            if i == 1:
                anims.append(FadeOut(cap))
            self.play(*anims, run_time=2.4)
            self.wait(0.8)
        note = Text(
            "Tighter bound → higher cost: the shaded region is\nforbidden, the optimizer squeezes the torque against it.",
            font_size=19,
            color=YELLOW_C,
        )
        place(note, CODE_X, -2.4)
        self.play(FadeIn(note))
        self.wait(2)

        # ------------------------------------------------------------ beat 2: a bound on a state (the cart position)
        self.play(FadeOut(note), FadeOut(bounds), FadeOut(panel), FadeOut(info))
        lim = float(d["cart_limit"])
        ax_y = make_axes([-3.55, 0.85, 0], 5.6, 2.6, [0, T], [-1.2, 1.2], 0.5, 0.5)
        new_deco = VGroup(
            axis_label("y(t)  sideways position (m)", ax_y, C_STATE),
            y_ticks(ax_y, [-1, 0, 1]),
        )
        self.play(FadeOut(curve_u), FadeOut(ax_u), FadeOut(decos[0]), FadeOut(decos[4]))
        # the lower plot now shows the torque, the upper one the position
        y_free = poly(ax_y, t_nodes, d["u0_p0_q"][TRANS], C_STATE)
        self.play(Create(ax_y), FadeIn(new_deco), Create(y_free))
        panel2 = code_panel(
            [
                (0, "x_bounds = BoundsList()", WHITE),
                (0, 'x_bounds["q"] = bio_model.bounds_from_ranges("q")', WHITE),
                (0, 'x_bounds["q"][:, 0] = 0', GRAY_B),
                (0, 'x_bounds["q"][1, -1] = 3.14', GRAY_B),
                (0, 'x_bounds["q"].min[0, 1:] = -0.4', C_BOUND),
                (0, 'x_bounds["q"].max[0, 1:] = +0.4', C_BOUND),
            ],
            caption="Bioptim code: bound on a state (all nodes after the first)",
        )
        cart_lines = [hline(ax_y, 0, T, s * lim, C_BOUND) for s in (1, -1)]
        cart_bands = [band(ax_y, 0, T, s * lim, s * 1.2, C_BOUND) for s in (1, -1)]
        cart_txt = Text(
            f"Free solution: y in [{d['u0_p0_q'][TRANS].min():.2f}, {d['u0_p0_q'][TRANS].max():.2f}] m.\nWith |y| ≤ 0.4 the same task needs more torque\n"
            f"(peak {np.abs(d['cart_p0_tau'][TRANS]).max():.0f} N): cost {float(d['u0_cost']):.0f} → {float(d['cart_cost']):.0f}.",
            font_size=19,
            color=GRAY_A,
            line_spacing=0.9,
        )
        place(cart_txt, CODE_X, -1.2)
        self.play(FadeIn(panel2))
        self.play(*[Create(m) for m in cart_lines], *[FadeIn(m) for m in cart_bands], FadeIn(cart_txt))
        y_cart = poly(ax_y, t_nodes, d["cart_p0_q"][TRANS], C_STATE)
        theta_cart = poly(ax_q, t_nodes, d["cart_p0_q"][ROT], C_STATE)
        self.play(Transform(y_free, y_cart), Transform(curve_q, theta_cart), run_time=3)
        end = Text(
            "Bounds act on decision variables; ConstraintFcn (e.g. TRACK_STATE) handles general path constraints.",
            font_size=17,
            color=YELLOW_C,
        )
        fit(end, 13).to_edge(DOWN, buff=0.2)
        self.play(FadeIn(end))
        self.wait(3)


# ====================================================================================================================
# Scene 3 - multiphase and phase transitions
# ====================================================================================================================
class MultiphaseTransitions(Scene):
    def construct(self):
        d = np.load(DATA_DIR / "features_multiphase.npz")
        t_ph = d["phase_times"]
        n_ph = d["n_shooting"]
        T_total = float(t_ph.sum())

        title = scene_title("Multiphase problems", "two phases with different durations and a phase transition")
        self.play(FadeIn(title))

        ax_q = make_axes([-3.55, 0.95, 0], 5.6, 2.3, [0, T_total], [-0.3, 3.6], 0.5, 1)
        ax_w = make_axes([-3.55, -2.05, 0], 5.6, 1.8, [0, T_total], [-18, 18], 0.5, 10)
        decos = VGroup(
            axis_label("θ(t)  angle (rad)", ax_q, C_STATE),
            axis_label("ω(t) = dθ/dt  angular velocity (rad/s)", ax_w, C_CTRL),
            time_label(ax_w),
            x_ticks(ax_w, [0, 0.5, 1.0, 1.5], "{:.1f}"),
            y_ticks(ax_q, [0, 1.57, 3.14], "{:g}"),
            y_ticks(ax_w, [-10, 0, 10]),
        )
        # coloured time segments: one per phase
        seg = []
        for p, (t0, t1, col) in enumerate(((0, t_ph[0], C_PH0), (t_ph[0], T_total, C_PH1))):
            for ax, (lo, hi) in ((ax_q, (-0.3, 3.6)), (ax_w, (-18, 18))):
                seg.append(band(ax, t0, t1, lo, hi, col, 0.16))
        seg_group = VGroup(*seg)
        ph_labels = VGroup(
            Text(
                f"phase 0\nT = {t_ph[0]:g} s, N = {int(n_ph[0])}", font_size=16, color=C_PH0, line_spacing=0.9
            ).move_to(ax_q.c2p(t_ph[0] / 2, 3.15)),
            Text(
                f"phase 1\nT = {t_ph[1]:g} s, N = {int(n_ph[1])}", font_size=16, color=C_PH1, line_spacing=0.9
            ).move_to(ax_q.c2p(t_ph[0] + t_ph[1] / 2, 3.15)),
        )
        self.play(Create(ax_q), Create(ax_w), FadeIn(decos), FadeIn(seg_group), FadeIn(ph_labels))

        def curves(tag):
            out = []
            for ax, key in ((ax_q, "q"), (ax_w, "qdot")):
                cur = VGroup()
                for p, col in enumerate((C_PH0, C_PH1)):
                    t = d[f"{tag}_p{p}_t"]
                    cur.add(poly(ax, t, d[f"{tag}_p{p}_{key}"][ROT], col))
                out.append(cur)
            return out

        panel = code_panel(
            [
                (0, "n_shooting = (12, 18)", WHITE),
                (0, "phase_time = (0.5, 1.0)", WHITE),
                (0, "phase_transitions = PhaseTransitionList()", C_PH1),
                (0, "phase_transitions.add(", C_PH1),
                (1, "PhaseTransitionFcn.CONTINUOUS,", C_PH1),
                (1, "phase_pre_idx=0)", C_PH1),
                (0, "# x, q̇ at the end of phase 0 == start of phase 1", GRAY_B),
            ],
            caption="Bioptim code (one model, dynamics, objective per phase)",
        )
        self.play(FadeIn(panel))
        cont_q, cont_w = curves("continuous")
        self.play(Create(cont_q), Create(cont_w), run_time=3)
        info = Text(
            f"CONTINUOUS: phase 1 starts exactly where phase 0 ended\n"
            f"ω at the junction = {d['continuous_p0_qdot'][ROT, -1]:.2f} rad/s on both sides\n"
            f"cost = {float(d['continuous_cost']):.1f}, IPOPT {int(d['continuous_iterations'])} it.",
            font_size=18,
            color=GRAY_A,
            line_spacing=0.9,
        )
        place(info, CODE_X, -1.7)
        self.play(FadeIn(info))
        self.wait(2)

        # ---- discontinuous
        new_lines = [
            code("PhaseTransitionFcn.DISCONTINUOUS,", 19, C_PH1),
            code("# no link: phase 1 restarts at rest (bounds)", 19, GRAY_B),
        ]
        old_a, old_b = panel[1][4], panel[1][6]
        new_lines[0].move_to(old_a, aligned_edge=LEFT)
        new_lines[1].scale_to_fit_width(min(new_lines[1].width, CODE_W)).move_to(old_b, aligned_edge=LEFT)
        disc_q, disc_w = curves("discontinuous")
        ang_v = "ω"
        info2 = Text(
            "DISCONTINUOUS: the states are not linked, both jump\n"
            f"y {d['discontinuous_p0_q'][TRANS, -1]:.2f} → {d['discontinuous_p1_q'][TRANS, 0]:.2f} m,   "
            f"{ang_v} {d['discontinuous_p0_qdot'][ROT, -1]:.2f} → {d['discontinuous_p1_qdot'][ROT, 0]:.2f} rad/s\n"
            f"cost = {float(d['discontinuous_cost']):.1f}, IPOPT {int(d['discontinuous_iterations'])} it.",
            font_size=18,
            color=GRAY_A,
            line_spacing=0.9,
        )
        place(info2, CODE_X, -1.7)
        self.play(
            Transform(cont_q, disc_q),
            Transform(cont_w, disc_w),
            Transform(info, info2),
            Transform(old_a, new_lines[0]),
            Transform(old_b, new_lines[1]),
            run_time=3,
        )
        jump = Arrow(
            ax_w.c2p(t_ph[0], float(d["discontinuous_p0_qdot"][ROT, -1])),
            ax_w.c2p(t_ph[0], float(d["discontinuous_p1_qdot"][ROT, 0])),
            buff=0,
            color=C_BOUND,
            stroke_width=6,
            max_tip_length_to_length_ratio=0.15,
        )
        jump_lbl = Text("velocity jump", font_size=16, color=C_BOUND).next_to(ax_w.c2p(t_ph[0], -9), LEFT, buff=0.08)
        self.play(GrowArrow(jump), FadeIn(jump_lbl))
        end = Text(
            "PhaseTransitionFcn.IMPACT also exists (rigid contact): the velocity jump is then given by the impact model.",
            font_size=17,
            color=YELLOW_C,
        )
        fit(end, 13).to_edge(DOWN, buff=0.2)
        self.play(FadeIn(end))
        self.wait(3.5)


# ====================================================================================================================
# Scene 4 - free time
# ====================================================================================================================
class FreeTime(Scene):
    def construct(self):
        d = np.load(DATA_DIR / "features_time.npz")
        u_maxs = d["u_maxs"]
        n = int(d["n_shooting"])
        t_guess = float(d["t_guess"])
        Tmax = 1.15

        title = scene_title("Free-time optimization", "the duration of the phase becomes a decision variable")
        self.play(FadeIn(title))

        ax_q = make_axes([-3.55, 0.85, 0], 5.6, 2.3, [0, Tmax], [-0.3, 6.4], 0.5, 1)
        ax_u = make_axes([-3.55, -2.2, 0], 5.6, 1.75, [0, Tmax], [-115, 115], 0.5, 100)
        decos = VGroup(
            axis_label("θ(t)  angle (rad)", ax_q, C_STATE),
            axis_label("τ(t)  actuated force (N)", ax_u, C_CTRL),
            time_label(ax_u),
            x_ticks(ax_u, [0, 0.5, 1.0], "{:.1f}"),
            y_ticks(ax_q, [0, 1, 2, 3]),
            y_ticks(ax_u, [-100, 0, 100]),
        )
        self.play(Create(ax_q), Create(ax_u), FadeIn(decos))

        # the initial guess of the duration is the value given as phase_time
        guess_bar = Line(ax_q.c2p(0, 6.0), ax_q.c2p(t_guess, 6.0), color=GRAY_B, stroke_width=6)
        guess_lbl = Text(f"phase_time = {t_guess:g} s (initial guess only)", font_size=17, color=GRAY_B).next_to(
            guess_bar, DOWN, buff=0.05
        )
        guess_lbl.align_to(guess_bar, LEFT)
        panel = code_panel(
            [
                (0, "objectives = ObjectiveList()", WHITE),
                (0, "objectives.add(", C_TIME),
                (1, "ObjectiveFcn.Mayer.MINIMIZE_TIME,", C_TIME),
                (1, "min_bound=0.1, max_bound=4.0)", C_TIME),
                (0, "ocp = OptimalControlProgram(", WHITE),
                (1, "bio_model, n_shooting=40, phase_time=1.0, ...", WHITE),
                (0, 'u_bounds["tau"] = [-100]*nb_tau, [100]*nb_tau', C_BOUND),
            ],
            caption="Bioptim code",
        )
        self.play(FadeIn(panel), Create(guess_bar), FadeIn(guess_lbl))
        self.wait(0.8)

        def curves(i):
            t = d[f"u{i}_p0_t"]
            return (
                poly(ax_q, t, d[f"u{i}_p0_q"][ROT], C_STATE),
                steps(ax_u, t, d[f"u{i}_p0_tau"][TRANS], C_CTRL),
            )

        def t_end(i):
            return float(d[f"u{i}_p0_t"][-1])

        def duration_mob(i):
            te = t_end(i)
            bar = Line(ax_q.c2p(0, 4.7), ax_q.c2p(te, 4.7), color=C_TIME, stroke_width=8)
            ticks = VGroup(
                Line(ax_q.c2p(0, 4.4), ax_q.c2p(0, 5.0), color=C_TIME, stroke_width=4),
                Line(ax_q.c2p(te, 4.4), ax_q.c2p(te, 5.0), color=C_TIME, stroke_width=4),
            )
            return VGroup(bar, ticks)

        def bound_mobs(u):
            return VGroup(hline(ax_u, 0, Tmax, u, C_BOUND), hline(ax_u, 0, Tmax, -u, C_BOUND))

        def readout(i):
            mob = Text(
                f"|τ| ≤ {u_maxs[i]:g} N   →   optimal duration T* = {t_end(i):.3f} s\n"
                f"IPOPT: {int(d[f'u{i}_iterations'])} iterations, {'converged' if d[f'u{i}_converged'] else 'acceptable level'}",
                font_size=21,
                color=WHITE,
                line_spacing=0.9,
            )
            return place(mob, CODE_X, -1.55)

        curve_q, curve_u = curves(0)
        dur = duration_mob(0)
        bnd = bound_mobs(float(u_maxs[0]))
        info = readout(0)
        dur_lbl = M(f"T* = {t_end(0):.3f} s", 20, C_TIME).next_to(dur[0], RIGHT, buff=0.12)
        self.play(Create(curve_q), Create(curve_u), Create(dur), Create(bnd), FadeIn(info), FadeIn(dur_lbl), run_time=2)
        info_txt = Text(
            "The torque sits on the bound most of the time (bang-bang like,\nwith some chattering): the fastest swing-up uses all the force.",
            font_size=19,
            color=YELLOW_C,
        )
        place(info_txt, CODE_X, -2.7)
        self.play(FadeIn(info_txt))
        self.wait(1.5)
        for i in range(1, len(u_maxs)):
            u = int(u_maxs[i])
            old = panel[1][6]
            new = code(f'u_bounds["tau"] = [-{u}]*nb_tau, [{u}]*nb_tau', 19, C_BOUND)
            new.move_to(old, aligned_edge=LEFT)
            new_q, new_u = curves(i)
            self.play(
                Transform(curve_q, new_q),
                Transform(curve_u, new_u),
                Transform(dur, duration_mob(i)),
                Transform(bnd, bound_mobs(float(u_maxs[i]))),
                Transform(info, readout(i)),
                Transform(old, new),
                Transform(
                    dur_lbl, M(f"T* = {t_end(i):.3f} s", 20, C_TIME).next_to(ax_q.c2p(t_end(i), 4.7), RIGHT, buff=0.12)
                ),
                run_time=2.5,
            )
            self.wait(1)
        end = Text(
            "Less force → longer T*. One extra decision variable, no fixed duration: this replaces a manual search on phase_time.",
            font_size=17,
            color=YELLOW_C,
        )
        fit(end, 13).to_edge(DOWN, buff=0.2)
        self.play(FadeIn(end))
        self.wait(3)

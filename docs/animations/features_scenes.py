"""
Manim Community animations, part 2: six bioptim features, each driven by REAL bioptim/IPOPT solves stored in
``data/features_*.npz`` (see ``generate_features_data.py``). Same pendulum swing-up as in ``dms_vs_dc.py``.

Scenes:
    1. ObjectivesNodes      Lagrange vs Mayer objectives, the Node enum, and what the weights do (5 real solves)
    2. ConstraintsBounds    u_bounds shrinking (4 real solves) and a bound on a state (x_bounds)
    3. MultiphaseTransitions  two phases of different durations, PhaseTransitionFcn.CONTINUOUS vs DISCONTINUOUS
    4. FreeTime             ObjectiveFcn.Mayer.MINIMIZE_TIME: the phase duration becomes an optimization variable
    5. Parameters           ParameterList: a scalar (max torque) optimized with the trajectory, 4 real solves
    6. Impact               PhaseTransitionFcn.IMPACT on a point mass hitting the floor (vs CONTINUOUS, infeasible)

The code lines shown next to the curves are the ones used in generate_features_data.py (names checked against the
bioptim source). No LaTeX needed. Render commands: see FEATURES.md.
"""

import os
import sys
import textwrap
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
C_PAR = PURPLE_B

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


def code_panel(lines: list, size: float = 19, top: float = 2.3, caption: str = "Bioptim code") -> VGroup:
    """
    Code lines given as (indent_level, text, color) stacked at the top of the right panel. The whole block is scaled
    to fit CODE_W; the caption ("Bioptim code" in the whole series) is always put ABOVE the code lines.
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


def para(text: str, chars: int = 46) -> str:
    """Wrap a whole sentence on short lines (the right panel is ~6 units wide); the lines are part of the FR key."""
    return "\n".join(textwrap.fill(part, width=chars) for part in text.split("\n"))


def say(text: str, size: int = 19, color=YELLOW_C, chars: int = 46) -> Text:
    """Yellow remark of the right panel: one sentence, wrapped by ``para``."""
    return Text(para(text, chars), font_size=size, color=color, line_spacing=0.9)


def footer(text: str, size: int = 16, color=YELLOW_C, max_width: float = 10.4) -> Text:
    """
    One whole sentence wrapped on at most two lines, left aligned at the bottom of the frame. It stops before the
    bottom-right corner (logo) and leaves 15 % of room for the French text.
    """
    mob = Text(text, font_size=size, color=color)
    if mob.width > 1.1 * max_width:
        n_lines = int(np.ceil(mob.width / max_width))
        width = int(len(text) / n_lines * 1.25)
        while True:
            mob = Text("\n".join(textwrap.wrap(text, width=width)), font_size=size, color=color, line_spacing=0.9)
            if mob.width <= max_width or width < 30:
                break
            width -= 2
    fit(mob, max_width)
    return mob.move_to([-6.9, -3.92, 0], aligned_edge=DL)


def ipopt_line(iterations: int, converged: bool) -> str:
    return f"IPOPT: {iterations} iterations, " + ("converged" if converged else "not converged")


def axis_label(text: str, ax: Axes, color=GRAY_B) -> Text:
    lab = Text(text, font_size=20, color=color).next_to(ax.get_y_axis(), UP, buff=0.08)
    return lab.align_to(ax.get_y_axis(), LEFT)


def time_label(ax: Axes, text="t (s)") -> Text:
    """Caption right of the last tick (the time axis crosses the plot at y = 0, ticks are at the bottom)."""
    return Text(text, font_size=16, color=GRAY_B).move_to(
        ax.c2p(ax.x_range[1], ax.y_range[0]) + np.array([0.62, -0.24, 0])
    )


TEXT_W = 5.9  # widest text of the right panel: French is ~15 % longer and must still end before x = 7.1


def place(mob: Mobject, x: float, y: float, max_width: float = TEXT_W) -> Mobject:
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


def dec(text: str) -> str:
    """Decimal comma for numeric-only labels in the French videos (the series layer skips strings without a word)."""
    if os.environ.get("SERIES_LANG", "en").strip().lower() == "fr" and os.environ.get("SERIES_FR_COMMA", "1") != "0":
        return text.replace(".", ",")
    return text


def y_ticks(ax: Axes, values, fmt="{:g}") -> VGroup:
    return VGroup(
        *[
            Text(dec(fmt.format(v)), font_size=16, color=GRAY_B).next_to(ax.c2p(ax.x_range[0], v), LEFT, buff=0.08)
            for v in values
        ]
    )


def x_ticks(ax: Axes, values, fmt="{:g}") -> VGroup:
    return VGroup(
        *[
            Text(dec(fmt.format(v)), font_size=16, color=GRAY_B).next_to(ax.c2p(v, ax.y_range[0]), DOWN, buff=0.08)
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
        xs = np.linspace(-3.0, 3.0, n_demo + 1)
        rows = [
            ("Node.START", [0], "first node"),
            ("Node.INTERMEDIATES", list(range(1, n_demo - 1)), "1 … N−2  (N−1 excluded too)"),
            ("Node.PENULTIMATE", [n_demo - 1], "node N−1"),
            ("Node.END", [n_demo], "last node N"),
            ("Node.ALL_SHOOTING", list(range(n_demo)), "0 … N−1  (nodes with a control)"),
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
            com = Text(comment, font_size=16, color=GRAY_B).move_to([3.55, y, 0], aligned_edge=LEFT)
            fit(com, 3.0).move_to([3.55, y, 0], aligned_edge=LEFT)
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
            "<b>Lagrange</b>  ∫ L(x, u) dt: summed over the N intervals  →  Node.ALL_SHOOTING (default)", 20, C_LAG
        )
        may_txt = M("<b>Mayer</b>  M(x): evaluated at one node  →  Node.END (default)", 20, C_MAY)
        summary = (
            VGroup(lag_txt, may_txt)
            .arrange(DOWN, aligned_edge=LEFT, buff=0.15)
            .move_to([-6.9, -2.05, 0], aligned_edge=LEFT)
        )
        fit(summary, 13.4)
        summary.move_to([-6.9, -2.05, 0], aligned_edge=LEFT)
        self.play(FadeIn(shades), FadeIn(lag_txt))
        self.play(Indicate(row_mobs[3][2], color=C_MAY, scale_factor=1.6), FadeIn(may_txt))
        foot = footer(f"Schematic grid with N = {n_demo} intervals; the real problem below uses N = {n}.", color=GRAY_B)
        self.play(FadeIn(foot))
        self.wait(2.5)

        # ---------------------------------------------------------------- beat 2: weights, with real solves
        self.play(*[FadeOut(m) for m in self.mobjects if m is not title])
        self.add(title)
        ax_q = make_axes([-3.55, 0.9, 0], 5.6, 2.7, [0, T], [-0.6, 3.6], 0.5, 1)
        ax_u = make_axes([-3.55, -2.2, 0], 5.6, 1.7, [0, T], [-40, 40], 0.5, 40)
        decos = VGroup(
            axis_label("θ(t)  angle (rad)", ax_q, C_STATE),
            axis_label("τ(t)  actuated force (N)", ax_u, C_CTRL),
            time_label(ax_u),
            x_ticks(ax_u, [0, 0.5, 1.0], "{:.1f}"),
            y_ticks(ax_q, [0, 1, 2, 3]),
            y_ticks(ax_u, [-40, 0, 40]),
        )
        target_line = hline(ax_q, 0, T, 3.14, GRAY_B)
        target_lbl = (
            Text(f"{3.14} (upright)", font_size=16, color=GRAY_B)
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
            size=17,
        )
        self.play(FadeIn(panel))
        weight_line = panel[1][-1]
        curve_q, curve_u = curves(0)
        self.play(Create(curve_q), Create(curve_u), run_time=1.5)

        def readout(i):
            w = weights[i]
            q_end = float(d[f"w{i}_p0_q"][ROT, -1])
            tau_max = float(np.abs(d[f"w{i}_p0_tau"][TRANS]).max())
            body = f"Mayer weight = {w:g}\nθ(T) = {q_end:.2f} rad  ·  max |τ| = {tau_max:.0f} N\n" + ipopt_line(
                int(d[f"w{i}_iterations"]), bool(d[f"w{i}_converged"])
            )
            return place(Text(body, font_size=19, color=GRAY_A, line_spacing=0.9), CODE_X, -2.15)

        info = readout(0)
        self.play(FadeIn(info))
        comments = [
            "Only the Lagrange term (weight 1.0): the cheapest control is τ = 0, nothing moves.",
            "Small Mayer weight: reaching 3.14 costs more torque than the terminal error saves.",
            "Larger weight: the pendulum goes further up.",
            "Larger still: almost at the target.",
            "A large Mayer weight makes the final state (almost) a hard target; the Lagrange term still shapes the way.",
        ]

        def comment_mob(i):
            return place(say(comments[i], 17, chars=54), CODE_X, -3.1)

        comment = comment_mob(0)
        self.play(FadeIn(comment))
        self.wait(1.5)
        for i in range(1, len(weights)):
            new_line = code(f"weight={weights[i]:g})", 17, C_MAY)
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
        ax_u = make_axes([-3.55, 1.4, 0], 5.6, 1.85, [0, T], [-y_lim, y_lim], 0.5, 20)
        ax_q = make_axes([-3.55, -0.85, 0], 5.6, 1.3, [0, T], [-1.2, 3.6], 0.5, 1)
        dq_lim = 0.6
        ax_dq = make_axes([-3.55, -3.0, 0], 5.6, 1.3, [0, T], [-dq_lim, dq_lim], 0.5, 0.3)
        decos = VGroup(
            axis_label("τ(t)  actuated force (N)", ax_u, C_CTRL),
            axis_label("θ(t)  angle (rad), dashed = unconstrained", ax_q, C_STATE),
            axis_label("Δθ(t) = θ − θ_free  (rad, zoomed)", ax_dq, C_PAR),
            time_label(ax_dq),
            x_ticks(ax_dq, [0, 0.5, 1.0], "{:.1f}"),
            y_ticks(ax_u, [-30, 0, 30]),
            y_ticks(ax_q, [0, 3]),
            y_ticks(ax_dq, [-0.5, 0, 0.5]),
        )
        self.play(Create(ax_u), Create(ax_q), Create(ax_dq), FadeIn(decos))

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
        )
        self.play(FadeIn(panel))
        lim_lines = [panel[1][2], panel[1][3]]

        q_free = d["u0_p0_q"][ROT]
        ghost = DashedVMobject(poly(ax_q, t_nodes, q_free, GRAY_B, 3), num_dashes=40).set_opacity(0.8)

        def dtheta(i):
            return float(np.abs(d[f"u{i}_p0_q"][ROT] - q_free).max())

        def curves(i):
            return (
                steps(ax_u, t_nodes, d[f"u{i}_p0_tau"][TRANS], C_CTRL),
                poly(ax_q, t_nodes, d[f"u{i}_p0_q"][ROT], C_STATE),
                poly(ax_dq, t_nodes, d[f"u{i}_p0_q"][ROT] - q_free, C_PAR),
            )

        bounds = bound_mobs(float(u_maxs[0]))
        curve_u, curve_q, curve_dq = curves(0)
        cap = Text(
            "|τ| ≤ 100 N is far away: the bound is inactive\n(the unconstrained optimum peaks at 24 N)",
            font_size=19,
            color=YELLOW_C,
        )

        def readout(i):
            u = float(u_maxs[i])
            peak = float(np.abs(d[f"u{i}_p0_tau"][TRANS]).max())
            if abs(peak - u) < 0.05 * u:
                first = f"|τ| ≤ {u:g} N  ·  peak |τ| = {peak:.1f} N (bound active)"
            else:
                first = f"|τ| ≤ {u:g} N  ·  peak |τ| = {peak:.1f} N"
            return Text(
                f"{first}\ncost ∫ τ² dt = {float(d[f'u{i}_cost']):.1f}\n"
                + ipopt_line(int(d[f"u{i}_iterations"]), bool(d[f"u{i}_converged"]))
                + f"\nmax |Δθ| = {dtheta(i):.2f} rad (vs unconstrained)",
                font_size=19,
                color=GRAY_A,
                line_spacing=0.9,
            ).move_to([CODE_X, -1.15, 0], aligned_edge=LEFT)

        info = readout(0)
        self.play(
            FadeIn(ghost),
            Create(curve_u),
            Create(curve_q),
            Create(curve_dq),
            FadeIn(bounds),
            FadeIn(info),
            run_time=1.5,
        )
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
            new_u, new_q, new_dq = curves(i)
            anims = [
                Transform(curve_u, new_u),
                Transform(curve_q, new_q),
                Transform(curve_dq, new_dq),
                Transform(bounds, bound_mobs(float(u_maxs[i]))),
                Transform(info, readout(i)),
                *[Transform(o, nw) for o, nw in zip(lim_lines, new_lines)],
            ]
            if i == 1:
                anims.append(FadeOut(cap))
            self.play(*anims, run_time=2.4)
            self.wait(0.8)
        note = say(
            "Tighter bound → higher cost: the shaded region is forbidden. The passive angle θ adapts slightly: "
            "the actuated coordinate absorbs the bound and the rotation follows through the dynamics.",
            18,
            chars=52,
        )
        place(note, CODE_X, -2.75)
        self.play(FadeIn(note))
        self.wait(2)

        # ------------------------------------------------------------ beat 2: a bound on a state (the cart position)
        new_sub = Text(
            "same swing-up, now a bound on a state: the sideways position",
            font_size=22,
            color=GRAY_B,
        ).move_to(title[1])
        self.play(
            *[
                FadeOut(m)
                for m in (note, bounds, panel, info, curve_u, curve_q, curve_dq, ghost, ax_u, ax_q, ax_dq, decos)
            ],
            Transform(title[1], new_sub),
        )
        limits = [float(v) for v in d["cart_limits"]]
        # three stacked plots: position, its velocity, angular velocity (the bound acts on y, the velocities react)
        lim_y, lim_v = 1.2, 30
        ax_y = make_axes([-3.55, 1.4, 0], 5.6, 1.4, [0, T], [-lim_y, lim_y], 0.5, 0.5)
        ax_v = make_axes([-3.55, -0.4, 0], 5.6, 1.4, [0, T], [-lim_v, lim_v], 0.5, 15)
        ax_w = make_axes([-3.55, -2.2, 0], 5.6, 1.4, [0, T], [-lim_v, lim_v], 0.5, 15)
        new_deco = VGroup(
            axis_label("y(t)  sideways position (m)", ax_y, C_STATE),
            axis_label("dy/dt  sideways velocity (m/s)", ax_v, C_CTRL),
            axis_label("dθ/dt  angular velocity (rad/s)", ax_w, C_LAG),
            time_label(ax_w),
            x_ticks(ax_w, [0, 0.5, 1.0], "{:.1f}"),
            y_ticks(ax_y, [-1, 0, 1]),
            y_ticks(ax_v, [-20, 0, 20]),
            y_ticks(ax_w, [-20, 0, 20]),
        )

        def state_curves(tag):
            q, qd = d[f"{tag}_p0_q"], d[f"{tag}_p0_qdot"]
            return (
                poly(ax_y, t_nodes, q[TRANS], C_STATE),
                poly(ax_v, t_nodes, qd[TRANS], C_CTRL),
                poly(ax_w, t_nodes, qd[ROT], C_LAG),
            )

        self.play(Create(ax_y), Create(ax_v), Create(ax_w), FadeIn(new_deco))
        y_c, v_c, w_c = state_curves("u0")
        self.play(Create(y_c), Create(v_c), Create(w_c), run_time=1.5)

        def code_lines(lim):
            return [
                code(f'x_bounds["q"].min[0, 1:] = -{lim:g}', 19, C_BOUND),
                code(f'x_bounds["q"].max[0, 1:] = +{lim:g}', 19, C_BOUND),
            ]

        panel2 = code_panel(
            [
                (0, "x_bounds = BoundsList()", WHITE),
                (0, 'x_bounds["q"] = bio_model.bounds_from_ranges("q")', WHITE),
                (0, 'x_bounds["q"][:, 0] = 0   # start', GRAY_B),
                (0, 'x_bounds["q"][1, -1] = 3.14   # y_final stays free', GRAY_B),
                (0, "# bound on a state: all the nodes after the first", GRAY_B),
                (0, 'x_bounds["q"].min[0, 1:] = -0.9', C_BOUND),
                (0, 'x_bounds["q"].max[0, 1:] = +0.9', C_BOUND),
                (0, "x_init = <previous solution>   # warm start", GRAY_B),
            ],
        )
        lim_lines = [panel2[1][5], panel2[1][6]]

        def y_bound_mobs(lim):
            g = VGroup()
            for s in (1, -1):
                g.add(hline(ax_y, 0, T, s * lim, C_BOUND))
                g.add(band(ax_y, 0, T, s * lim, s * lim_y, C_BOUND))
            return g

        def readout(i):
            tag = "u0" if i == 0 else f"cart{i - 1}"
            qd = d[f"{tag}_p0_qdot"]
            head = "no bound on y (|τ| ≤ 100 N is inactive)" if i == 0 else f"|y| ≤ {limits[i - 1]:g} m"
            body = (
                f"{head}\npeak |dy/dt| = {np.abs(qd[TRANS]).max():.1f} m/s\npeak |dθ/dt| = {np.abs(qd[ROT]).max():.1f} rad/s\n"
                f"peak |τ| = {np.abs(d[f'{tag}_p0_tau'][TRANS]).max():.0f} N  ·  cost ∫ τ² dt = {float(d[f'{tag}_cost']):.1f}\n"
                + ipopt_line(int(d[f"{tag}_iterations"]), bool(d[f"{tag}_converged"]))
            )
            return place(Text(body, font_size=19, color=GRAY_A, line_spacing=0.9), CODE_X, -1.6)

        def comment(i):
            texts = [
                f"Free solution: y spans {d['u0_p0_q'][TRANS].min():.2f} to {d['u0_p0_q'][TRANS].max():.2f} m.",
                "|y| ≤ 0.9 m is barely active: almost the same motion.",
                "|y| ≤ 0.7 m: y rests on the bound (dy/dt ≈ 0), the swing between the bounds is faster.",
                "|y| ≤ 0.5 m: dy/dt ≈ 0 while y sits on a bound, the fast swing peaks higher and τ reaches its 100 N bound.",
            ]
            return place(say(texts[i], 18, chars=52), CODE_X, -3.0)

        info = readout(0)
        note = comment(0)
        self.play(FadeIn(panel2), FadeIn(info), FadeIn(note))
        self.wait(1)
        bmobs = y_bound_mobs(limits[0])
        self.play(*[Create(m) if isinstance(m, DashedLine) else FadeIn(m) for m in bmobs])
        self.wait(0.5)
        for i in range(1, len(limits) + 1):
            new_y, new_v, new_w = state_curves(f"cart{i - 1}")
            new_lines = code_lines(limits[i - 1])
            for old, nw in zip(lim_lines, new_lines):
                nw.move_to(old, aligned_edge=LEFT)
            self.play(
                Transform(y_c, new_y),
                Transform(v_c, new_v),
                Transform(w_c, new_w),
                Transform(bmobs, y_bound_mobs(limits[i - 1])),
                Transform(info, readout(i)),
                Transform(note, comment(i)),
                *[Transform(o, nw) for o, nw in zip(lim_lines, new_lines)],
                run_time=2.6,
            )
            self.wait(1.2)
        tight_exit = str(d["cart_tight_exit"]).replace("_", " ").lower()
        end = footer(
            f"Continuation: each solve starts from the previous one. |y| ≤ 0.4 m: IPOPT reports {tight_exit} "
            "(out of reach for |τ| ≤ 100 N, T = 1 s).",
            size=16,
        )
        self.play(FadeIn(end))
        self.wait(3)
        end2 = footer("Bounds act on decision variables; ConstraintFcn handles general path constraints.", size=16)
        self.play(Transform(end, end2))
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
                (0, "# one model, dynamics and objective per phase", GRAY_B),
                (0, "n_shooting = (12, 18)", WHITE),
                (0, "phase_time = (0.5, 1.0)", WHITE),
                (0, "phase_transitions = PhaseTransitionList()", C_PH1),
                (0, "phase_transitions.add(", C_PH1),
                (1, "PhaseTransitionFcn.CONTINUOUS,", C_PH1),
                (1, "phase_pre_idx=0)", C_PH1),
                (0, "# x, q̇ at the end of phase 0 == start of phase 1", GRAY_B),
            ],
        )
        self.play(FadeIn(panel))
        cont_q, cont_w = curves("continuous")
        self.play(Create(cont_q), Create(cont_w), run_time=3)
        info = Text(
            f"CONTINUOUS: phase 1 starts exactly where phase 0 ended\n"
            f"ω at the junction = {d['continuous_p0_qdot'][ROT, -1]:.2f} rad/s on both sides\n"
            f"cost = {float(d['continuous_cost']):.1f}\n"
            + ipopt_line(int(d["continuous_iterations"]), bool(d["continuous_converged"])),
            font_size=18,
            color=GRAY_A,
            line_spacing=0.9,
        )
        place(info, CODE_X, -1.95)
        self.play(FadeIn(info))
        self.wait(2)

        # ---- discontinuous
        new_lines = [
            code("PhaseTransitionFcn.DISCONTINUOUS,", 19, C_PH1),
            code("# no link: phase 1 restarts at rest (bounds)", 19, GRAY_B),
        ]
        old_a, old_b = panel[1][5], panel[1][7]
        new_lines[0].move_to(old_a, aligned_edge=LEFT)
        new_lines[1].scale_to_fit_width(min(new_lines[1].width, CODE_W)).move_to(old_b, aligned_edge=LEFT)
        disc_q, disc_w = curves("discontinuous")
        ang_v = "ω"
        info2 = Text(
            "DISCONTINUOUS: the states are not linked, both jump\n"
            f"y {d['discontinuous_p0_q'][TRANS, -1]:.2f} → {d['discontinuous_p1_q'][TRANS, 0]:.2f} m\n"
            f"{ang_v} {d['discontinuous_p0_qdot'][ROT, -1]:.2f} → {d['discontinuous_p1_qdot'][ROT, 0]:.2f} rad/s\n"
            f"cost = {float(d['discontinuous_cost']):.1f}\n"
            + ipopt_line(int(d["discontinuous_iterations"]), bool(d["discontinuous_converged"])),
            font_size=18,
            color=GRAY_A,
            line_spacing=0.9,
        )
        place(info2, CODE_X, -1.95)
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
        end = footer(
            "PhaseTransitionFcn.IMPACT also exists (rigid contact): the velocity jump is then given by the impact model."
        )
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
            bar = Line(ax_q.c2p(0, 4.5), ax_q.c2p(te, 4.5), color=C_TIME, stroke_width=8)
            ticks = VGroup(
                Line(ax_q.c2p(0, 4.2), ax_q.c2p(0, 4.8), color=C_TIME, stroke_width=4),
                Line(ax_q.c2p(te, 4.2), ax_q.c2p(te, 4.8), color=C_TIME, stroke_width=4),
            )
            return VGroup(bar, ticks)

        def bound_mobs(u):
            return VGroup(hline(ax_u, 0, Tmax, u, C_BOUND), hline(ax_u, 0, Tmax, -u, C_BOUND))

        def readout(i):
            mob = Text(
                f"|τ| ≤ {u_maxs[i]:g} N  →  optimal duration T* = {t_end(i):.3f} s\n"
                + ipopt_line(int(d[f"u{i}_iterations"]), bool(d[f"u{i}_converged"])),
                font_size=20,
                color=WHITE,
                line_spacing=0.9,
            )
            return place(mob, CODE_X, -1.55)

        curve_q, curve_u = curves(0)
        dur = duration_mob(0)
        bnd = bound_mobs(float(u_maxs[0]))
        info = readout(0)
        dur_lbl = M(dec(f"T* = {t_end(0):.3f} s"), 20, C_TIME).next_to(dur[0], RIGHT, buff=0.12)
        self.play(Create(curve_q), Create(curve_u), Create(dur), Create(bnd), FadeIn(info), FadeIn(dur_lbl), run_time=2)
        info_txt = say(
            "The torque sits on the bound most of the time (bang-bang like, with some chattering): "
            "the fastest swing-up uses all the force.",
            18,
        )
        place(info_txt, CODE_X, -2.95)
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
                    dur_lbl,
                    M(dec(f"T* = {t_end(i):.3f} s"), 20, C_TIME).next_to(ax_q.c2p(t_end(i), 4.5), RIGHT, buff=0.12),
                ),
                run_time=2.5,
            )
            self.wait(1)
        end = footer("Less force → longer T*: the free duration replaces a manual search on phase_time.")
        self.play(FadeIn(end))
        self.wait(3)


# ====================================================================================================================
# Scene 5 - parameters
# ====================================================================================================================
class Parameters(Scene):
    def construct(self):
        d = np.load(DATA_DIR / "features_parameters.npz")
        weights = d["weights"]
        n, T = int(d["n_shooting"]), float(d["final_time"])
        t_nodes = np.linspace(0, T, n + 1)
        nx, nu = int(d["layout_x_nodes"]), int(d["layout_u_nodes"])
        px, pu = int(d["layout_x_per_node"]), int(d["layout_u_per_node"])
        total = int(d["layout_total"])

        title = scene_title("Parameters", "a number optimized with the trajectory, but not a function of time")
        self.play(FadeIn(title))

        # ---------------------------------------------------------------- beat 1: where a parameter lives
        # Decision vector [dt | X | U | parameters]; all the sizes are read from the real OCP used for the solves.
        x0, ybox = -6.6, 1.5
        boxes = [
            ("dt", 0.6, GRAY_B, f"{int(d['layout_dt'])}"),
            ("X = (q, q̇)", 2.9, C_STATE, f"{nx} nodes × {px} = {nx * px}"),
            ("U = τ", 1.9, C_CTRL, f"{nu} nodes × {pu} = {nu * pu}"),
            ("p", 0.6, C_PAR, f"{int(d['layout_params'])}"),
        ]
        vec = VGroup()
        cx = x0
        for name, w, col, count in boxes:
            rect = Rectangle(width=w, height=0.55, stroke_color=col, stroke_width=3, fill_color=col, fill_opacity=0.22)
            rect.move_to([cx + w / 2, ybox, 0])
            lab = fit(Text(name, font_size=19, color=col), w - 0.1).move_to(rect)
            cnt = fit(Text(count, font_size=16, color=GRAY_B), w + 0.1).next_to(rect, DOWN, buff=0.1)
            vec.add(VGroup(rect, lab, cnt))
            cx += w + 0.05
        vec_cap = Text(f"decision vector of the problem: {total} variables", font_size=19, color=GRAY_B)
        vec_cap.move_to([x0, ybox + 0.65, 0], aligned_edge=LEFT)
        note_dt = Text("dt = duration of the phase (pinned by its bounds here)", font_size=16, color=GRAY_B)
        note_dt.move_to([x0, ybox - 0.75, 0], aligned_edge=LEFT)
        self.play(FadeIn(vec_cap))
        for grp in vec:
            self.play(FadeIn(grp), run_time=0.5)
        self.play(FadeIn(note_dt))

        # the same value p is used at all the nodes
        xs = np.linspace(x0 + 0.1, -0.5, nu)
        y_dots, y_box = -0.7, -2.35
        dots = VGroup(*[Dot([x, y_dots, 0], radius=0.06, color=C_CTRL) for x in xs])
        dots_cap = M(f"nodes k = 0 … {nu - 1}: one control τ<sub>k</sub> each", 16, C_CTRL)
        dots_cap.move_to([x0, y_dots + 0.32, 0], aligned_edge=LEFT)
        pbox = VGroup(
            Rectangle(width=1.5, height=0.5, stroke_color=C_PAR, stroke_width=3, fill_color=C_PAR, fill_opacity=0.25),
            Text("max_tau", font_size=19, color=C_PAR),
        ).move_to([-3.55, y_box, 0])
        fan = VGroup(
            *[Line([-3.55, y_box + 0.25, 0], [x, y_dots - 0.06, 0], stroke_width=1.5, color=C_PAR) for x in xs]
        )
        fan.set_opacity(0.55)
        fan_txt = M("the <b>same</b> value at every node:\n|τ<sub>k</sub>| ≤ max_tau", 16, C_PAR)
        fan_txt.move_to([-2.6, y_box - 0.05, 0], aligned_edge=LEFT)
        self.play(Indicate(vec[3], color=C_PAR, scale_factor=1.25), FadeIn(dots), FadeIn(dots_cap), FadeIn(pbox))
        self.play(Create(fan, lag_ratio=0.02), FadeIn(fan_txt), run_time=2)
        foot = footer("X and U have one value per node; a parameter is a single value for the whole trajectory.")
        self.play(FadeIn(foot))

        panel1 = code_panel(
            [
                (0, "# declare, bound and initialize the parameter", GRAY_B),
                (0, "parameters = ParameterList(use_sx=True)", C_PAR),
                (0, "parameters.add(", C_PAR),
                (1, '"max_tau", no_model_change, size=1,', C_PAR),
                (1, 'scaling=VariableScaling("max_tau", [1]))', C_PAR),
                (0, "parameter_bounds.add(", WHITE),
                (1, '"max_tau", min_bound=0, max_bound=100,', WHITE),
                (1, "interpolation=InterpolationType.CONSTANT)", WHITE),
                (0, 'parameter_init["max_tau"] = 50', WHITE),
                (0, "bio_model = TorqueBiorbdModel(", C_PAR),
                (1, "MODEL, parameters=parameters)", C_PAR),
            ],
        )
        self.play(FadeIn(panel1), run_time=1.5)
        self.wait(3.5)

        # ---------------------------------------------------------------- beat 2: four real solves
        self.play(*[FadeOut(m) for m in self.mobjects if m is not title])
        self.add(title)
        y_lim = 40
        ax_u = make_axes([-3.55, 0.85, 0], 5.6, 2.7, [0, T], [-y_lim, y_lim], 0.5, 20)
        ax_q = make_axes([-3.55, -2.25, 0], 5.6, 1.6, [0, T], [-0.3, 3.6], 0.5, 1)
        decos = VGroup(
            axis_label("τ(t)  actuated force (N)", ax_u, C_CTRL),
            axis_label("θ(t)  angle (rad)", ax_q, C_STATE),
            time_label(ax_q),
            x_ticks(ax_q, [0, 0.5, 1.0], "{:.1f}"),
            y_ticks(ax_u, [-40, -20, 0, 20, 40]),
            y_ticks(ax_q, [0, 3]),
        )
        self.play(Create(ax_u), Create(ax_q), FadeIn(decos))

        def p_of(i):
            return float(d[f"w{i}_max_tau"])

        def bound_mobs(p):
            g = VGroup()
            for sign in (1, -1):
                g.add(hline(ax_u, 0, T, sign * p, C_PAR))
                g.add(band(ax_u, 0, T, sign * p, sign * y_lim, C_PAR, 0.18))
                lab = Text("+max_tau" if sign > 0 else "−max_tau", font_size=16, color=C_PAR)
                lab.next_to(ax_u.c2p(0.0, sign * p), UR if sign > 0 else DR, buff=0.04)
                g.add(lab.align_to(ax_u.c2p(0.02, 0), LEFT))
            return g

        panel2 = code_panel(
            [
                (0, "# cost on the parameter, then |tau| <= max_tau", GRAY_B),
                (0, "parameter_objectives.add(", C_PAR),
                (1, "ObjectiveFcn.Parameter.MINIMIZE_PARAMETER,", C_PAR),
                (1, 'key="max_tau", weight=0.001, quadratic=True)', C_PAR),
                (0, "def max_tau_upper(controller):", WHITE),
                (1, 'return (controller.parameters["max_tau"].cx', WHITE),
                (3, '- controller.controls["tau"].cx[0])', WHITE),
                (0, "constraints.add(max_tau_upper,", WHITE),
                (1, "node=Node.ALL_SHOOTING, min_bound=0, max_bound=np.inf)", WHITE),
                (0, "# max_tau_lower: same with +  (max_tau + tau >= 0)", GRAY_B),
            ],
            top=2.35,
        )
        weight_line = panel2[1][3]

        def curves(i):
            return (
                steps(ax_u, t_nodes, d[f"w{i}_p0_tau"][TRANS], C_CTRL),
                poly(ax_q, t_nodes, d[f"w{i}_p0_q"][ROT], C_STATE),
            )

        def readout(i):
            p = p_of(i)
            tau = d[f"w{i}_p0_tau"][TRANS]
            integral = float((tau**2).sum() * T / n)
            body = (
                f"weight = {weights[i]:g}  →  max_tau* = {p:.2f} N\n"
                f"peak |τ| = {np.abs(tau).max():.2f} N\n"
                f"∫ τ² dt = {integral:.1f}  ·  weight · max_tau² = {weights[i] * p * p:.1f}\n"
                + ipopt_line(int(d[f"w{i}_iterations"]), bool(d[f"w{i}_converged"]))
            )
            return place(Text(body, font_size=18, color=GRAY_A, line_spacing=0.9), CODE_X, -1.75)

        comments = [
            "Almost free peak: max_tau sits on the peak of the minimum-effort torque, the dashed lines touch the curve.",
            "A higher price on the peak: the optimizer accepts a larger ∫ τ² dt to get a lower max_tau.",
            "Same idea, further: the torque is flattened against the two dashed lines.",
            "Still one number: the lines stay horizontal because max_tau does not depend on time.",
        ]

        def comment_mob(i):
            return place(say(comments[i], 18), CODE_X, -3.0)

        self.play(FadeIn(panel2))
        bnd = bound_mobs(p_of(0))
        curve_u, curve_q = curves(0)
        info = readout(0)
        comment = comment_mob(0)
        self.play(Create(curve_u), Create(curve_q), FadeIn(bnd), FadeIn(info), FadeIn(comment), run_time=1.8)
        self.wait(1.5)
        for i in range(1, len(weights)):
            new_line = code(f'key="max_tau", weight={weights[i]:g}, quadratic=True)', 19, C_PAR)
            # same scale as the (shrunk) code panel: width ratio of the original line
            new_line.scale(weight_line.width / code('key="max_tau", weight=0.001, quadratic=True)', 19).width)
            new_line.move_to(weight_line, aligned_edge=LEFT)
            new_u, new_q = curves(i)
            self.play(
                Transform(curve_u, new_u),
                Transform(curve_q, new_q),
                Transform(bnd, bound_mobs(p_of(i))),
                Transform(info, readout(i)),
                Transform(comment, comment_mob(i)),
                Transform(weight_line, new_line),
                run_time=2.4,
            )
            self.wait(1.0)
        end = footer(
            "Each solve starts from the previous solution (continuation): the problem is non-convex, see FEATURES.md."
        )
        self.play(FadeIn(end))
        self.wait(3)


# ====================================================================================================================
# Scene 6 - impact
# ====================================================================================================================
class Impact(Scene):
    def construct(self):
        d = np.load(DATA_DIR / "features_impact.npz")
        t_ph = d["phase_times"]
        T_total = float(t_ph.sum())
        t_imp = float(t_ph[0])
        n0 = len(d["impact_p0_t"])
        # global time and states of the IMPACT solve; at t_imp the last node of phase 0 (before) and the first node of
        # phase 1 (after) share the same time
        t_all = np.concatenate([d["impact_p0_t"], d["impact_p1_t"]])
        q_all = np.concatenate([d["impact_p0_q"], d["impact_p1_q"]], axis=1)
        v_all = np.concatenate([d["impact_p0_qdot"], d["impact_p1_qdot"]], axis=1)
        vz_pre, vz_post = float(d["impact_p0_qdot"][1, -1]), float(d["impact_p1_qdot"][1, 0])
        vx_pre, vx_post = float(d["impact_p0_qdot"][0, -1]), float(d["impact_p1_qdot"][0, 0])
        energy_lost = 0.5 * 1.0 * (vz_pre**2 - vz_post**2)  # the mass of point_floor.bioMod is 1 kg

        title = scene_title(
            "PhaseTransitionFcn.IMPACT", "a point mass hits the floor: the velocity jumps, the position does not"
        )
        self.play(FadeIn(title))

        # ------------------------------------------------ axes: a side view (x, z) and the velocities
        z_lo, z_hi = -0.6, 1.15
        v_lo, v_hi = -5.2, 4.3
        ax_s = make_axes([-3.55, 1.05, 0], 5.6, 2.2, [0, 3.1], [z_lo, z_hi], 1, 0.5)
        ax_v = make_axes([-3.55, -2.0, 0], 5.6, 2.1, [0, T_total], [v_lo, v_hi], 0.5, 1)
        floor = Line(ax_s.c2p(0, 0), ax_s.c2p(3.1, 0), color=GRAY_A, stroke_width=5)
        ground = band(ax_s, 0, 3.1, z_lo, 0, GRAY_D, 0.35)
        decos = VGroup(
            axis_label("side view (x, z): sketch, the two scales differ", ax_s, GRAY_B),
            axis_label("velocity (m/s)", ax_v, GRAY_B),
            time_label(ax_v),
            x_ticks(ax_v, [0, 0.5, 1.0, 1.5], "{:.1f}"),
            y_ticks(ax_v, [-4, -2, 0, 2]),
            Text("floor  z = 0", font_size=16, color=GRAY_A).move_to(ax_s.c2p(0.06, -0.16), aligned_edge=LEFT),
        )
        bands_v = VGroup(
            band(ax_v, 0, t_imp, v_lo, v_hi, C_PH0, 0.16), band(ax_v, t_imp, T_total, v_lo, v_hi, C_PH1, 0.16)
        )
        ph_lab = VGroup(
            Text("phase 0: flight", font_size=16, color=C_PH0).move_to(ax_v.c2p(t_imp / 2, 3.75)),
            Text("phase 1: on the floor", font_size=16, color=C_PH1).move_to(
                ax_v.c2p(t_imp + (T_total - t_imp) / 2, 3.75)
            ),
        )
        self.play(
            Create(ax_s), Create(ax_v), FadeIn(decos), FadeIn(ground), Create(floor), FadeIn(bands_v), FadeIn(ph_lab)
        )

        panel = code_panel(
            [
                (0, "# phase 0: flight, phase 1: contact", GRAY_B),
                (0, "# point_floor.bioMod: contact Mass_contact, axis z", GRAY_B),
                (0, "models = (", WHITE),
                (1, "TorqueBiorbdModel(MODEL_IMPACT),", WHITE),
                (1, "TorqueBiorbdModel(MODEL_IMPACT,", C_PH1),
                (2, "contact_types=[ContactType.RIGID_EXPLICIT]))", C_PH1),
                (0, "phase_transitions = PhaseTransitionList()", C_BOUND),
                (0, "phase_transitions.add(", C_BOUND),
                (1, "PhaseTransitionFcn.IMPACT, phase_pre_idx=0)", C_BOUND),
            ],
        )
        rule = M(
            "IMPACT constraint at the transition:\nq<sub>after</sub> = q<sub>before</sub>\n"
            "q̇<sub>after</sub> = biorbd ComputeConstraintImpulsesDirect(q, q̇<sub>before</sub>)",
            17,
            YELLOW_C,
        )
        place(rule, CODE_X, -1.65)
        self.play(FadeIn(panel), FadeIn(rule), run_time=1.2)

        # ------------------------------------------------ animate the real trajectory
        tr = ValueTracker(0.0)

        def state_at(t):
            xz = [float(np.interp(t, t_all, q_all[i])) for i in (0, 1)]
            v = [float(np.interp(t, t_all, v_all[i])) for i in (0, 1)]
            return xz, v

        def dot():
            xz, _ = state_at(tr.get_value())
            col = C_PH0 if tr.get_value() < t_imp else C_PH1
            return Dot(ax_s.c2p(*xz), radius=0.1, color=col)

        def vel_arrow():
            xz, v = state_at(tr.get_value())
            start = ax_s.c2p(*xz)
            vec = np.array([v[0] * 0.28, v[1] * 0.28, 0])
            if np.linalg.norm(vec) < 0.08:
                return VGroup()
            return Arrow(
                start, start + vec, buff=0.1, color=C_STATE, stroke_width=5, max_tip_length_to_length_ratio=0.3
            )

        def trail():
            k = max(int(np.searchsorted(t_all, tr.get_value(), side="right")), 2)
            pts = [ax_s.c2p(a, b) for a, b in zip(q_all[0, :k], q_all[1, :k])]
            return VMobject(color=GRAY_B, stroke_width=2).set_points_as_corners(pts)

        def curve_upto(idx, color):
            def make():
                out = VGroup()
                for sel in (np.arange(n0), np.arange(n0, len(t_all))):
                    keep = sel[t_all[sel] <= tr.get_value() + 1e-9]
                    if len(keep) >= 2:
                        out.add(poly(ax_v, t_all[keep], v_all[idx][keep], color, 5))
                return out

            return make

        d_dot = always_redraw(dot)
        d_arrow = always_redraw(vel_arrow)
        d_trail = always_redraw(trail)
        c_vz = always_redraw(curve_upto(1, C_STATE))
        c_vx = always_redraw(curve_upto(0, C_CTRL))
        lab_vz = M("v<sub>z</sub>", 19, C_STATE).move_to(ax_v.c2p(0.09, -3.4))
        lab_vx = M("v<sub>x</sub>", 19, C_CTRL).move_to(ax_v.c2p(0.16, 0.7))
        cursor = always_redraw(
            lambda: DashedLine(
                ax_v.c2p(tr.get_value(), v_lo), ax_v.c2p(tr.get_value(), v_hi), color=GRAY_B, stroke_width=2
            )
        )
        self.add(d_trail, c_vz, c_vx, cursor, d_arrow, d_dot)
        self.play(FadeIn(lab_vz), FadeIn(lab_vx))
        self.play(tr.animate.set_value(t_imp), run_time=3.2, rate_func=linear)

        # the impact: the jump of the vertical velocity
        jump = Arrow(
            ax_v.c2p(t_imp, vz_pre),
            ax_v.c2p(t_imp, vz_post),
            buff=0,
            color=C_BOUND,
            stroke_width=7,
            max_tip_length_to_length_ratio=0.15,
        )
        jump_lbl = M(f"jump  +{vz_post - vz_pre:.2f} m/s", 18, C_BOUND).next_to(
            ax_v.c2p(t_imp, (vz_pre + vz_post) / 2), RIGHT, buff=0.15
        )
        pre_lbl = M(f"v<sub>z</sub> before = {vz_pre:.2f}", 17, C_STATE).next_to(
            ax_v.c2p(t_imp, vz_pre), RIGHT, buff=0.15
        )
        pre_lbl.shift(UP * 0.1)
        post_lbl = M(f"v<sub>z</sub> after = {abs(vz_post):.2f}", 17, C_STATE).next_to(
            ax_v.c2p(t_imp, vz_post), RIGHT, buff=0.15
        )
        post_lbl.shift(UP * 0.3)
        self.play(GrowArrow(jump), FadeIn(jump_lbl), FadeIn(pre_lbl), FadeIn(post_lbl))
        info = M(
            ipopt_line(int(d["impact_iterations"]), bool(d["impact_converged"]))
            + f"\nbefore: v<sub>x</sub> = {vx_pre:.2f}, v<sub>z</sub> = {vz_pre:.2f} m/s\n"
            f"after: v<sub>x</sub> = {vx_post:.2f}, v<sub>z</sub> = {abs(vz_post):.2f} m/s\n"
            f"lost energy ½ m v<sub>z</sub>² = {energy_lost:.2f} J (m = 1 kg)",
            18,
            GRAY_A,
        )
        place(info, CODE_X, -2.75)
        self.play(FadeIn(info))
        self.wait(1.5)
        self.play(tr.animate.set_value(T_total), run_time=3.2, rate_func=linear)
        self.wait(1.5)

        # ------------------------------------------------ CONTINUOUS instead: infeasible
        cx_all = np.concatenate([d["continuous_p0_q"][0], d["continuous_p1_q"][0]])
        cz_all = np.concatenate([d["continuous_p0_q"][1], d["continuous_p1_q"][1]])
        bad = DashedVMobject(
            VMobject(color=C_BOUND, stroke_width=4).set_points_as_corners(
                [ax_s.c2p(a, b) for a, b in zip(cx_all, cz_all)]
            ),
            num_dashes=60,
        )
        new_line = code("PhaseTransitionFcn.CONTINUOUS, phase_pre_idx=0)", 19, C_BOUND)
        old_line = panel[1][8]
        # same scale as the (shrunk) code panel
        new_line.scale(old_line.width / code("PhaseTransitionFcn.IMPACT, phase_pre_idx=0)", 19).width)
        new_line.move_to(old_line, aligned_edge=LEFT)
        req = DashedLine(ax_v.c2p(t_imp, vz_pre), ax_v.c2p(T_total, vz_pre), color=C_BOUND, stroke_width=4)
        req_lbl = M(f"CONTINUOUS would keep v<sub>z</sub> = {vz_pre:.2f}", 17, C_BOUND).next_to(req, UP, buff=0.05)
        req_lbl.align_to(req, RIGHT)
        new_info = M(
            f"CONTINUOUS: IPOPT reports {str(d['continuous_exit']).replace('_', ' ').lower()}\n"
            f"after {int(d['continuous_iterations'])} iterations.\n"
            f"v<sub>z</sub> cannot drop to 0, the contact keeps z̈ = 0\n"
            f"and the mass would sink through the floor.",
            18,
            C_BOUND,
        )
        place(new_info, CODE_X, -2.75)
        note = fit(Text("red dashes: last IPOPT iterate, NOT a solution", font_size=16, color=C_BOUND), 4.2)
        note.move_to(ax_s.c2p(3.1, 1.0), aligned_edge=RIGHT)
        self.play(
            FadeOut(rule),
            FadeOut(jump),
            FadeOut(jump_lbl),
            FadeOut(post_lbl),
            FadeOut(pre_lbl),
            Transform(old_line, new_line),
            Transform(info, new_info),
            run_time=1.2,
        )
        self.play(Create(bad), Create(req), FadeIn(req_lbl), FadeIn(note), run_time=2.5)
        end = footer(
            "IMPACT: inelastic, frictionless impact on the contact axes of the model; the floor is z = 0 by construction."
        )
        self.play(FadeIn(end))
        self.wait(3.5)

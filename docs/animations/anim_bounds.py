"""
Manim CE scene: bounds, initial guess and InterpolationType, with REAL bioptim objects and solves
(data/bounds_guess.npz, produced by ``generate_bounds_data.py``).

Beat 1: the rotation q_rot of a pendulum swing-up is given to bioptim with four InterpolationType (real
``InitialGuess`` objects: array shape passed by the user and value used at each of the 21 nodes) next to the real bounds
(pinned at the first and last node). Beat 2: the same swing-up solved by IPOPT from three initial guesses (zeros,
straight line, coarse solve), with iterations, cost and status.

Scene: BoundsInitialGuess (about 20 s of content at native speed). Render (from the repository root):
    python docs/animations/render_series.py anim_bounds.py BoundsInitialGuess --lang both
"""

import numpy as np
from manim import *

from features_scenes import (  # noqa: E402  (importing it also sets the default fonts)
    C_BOUND,
    C_PAR,
    C_STATE,
    CODE_X,
    DATA_DIR,
    axis_label,
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

C_GHOST = GRAY_B
C_GUESS = ORANGE  # the initial guess (input), distinct from the solution (yellow, C_STATE)
TYPES = ["constant", "first_last", "linear", "each_frame"]
IT_NAMES = ["IT.CONSTANT", "IT.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT", "IT.LINEAR", "IT.EACH_FRAME"]
RUNS = ["zeros", "linear", "fromcoarse"]
RUN_IT = ["IT.CONSTANT", "IT.LINEAR", "IT.EACH_FRAME"]
RUN_GUESS = ["constant", "linear", "each_frame"]  # which beat-1 guess curve is the initial guess of the run


class BoundsInitialGuess(Scene):
    def construct(self):
        d = np.load(DATA_DIR / "bounds_guess.npz")
        n, T, q_end = int(d["n_shooting"]), float(d["final_time"]), float(d["q_end"])
        t = np.linspace(0, T, n + 1)
        shape = {k: [int(v) for v in d[f"{k}_shape"]] for k in TYPES}
        nodes = {k: d[f"{k}_nodes"] for k in TYPES}
        b_shape = [int(v) for v in d["bound_min_shape"]]
        b_hi = float(d["bound_max"][1])
        ncoarse = int(d["n_coarse"])

        title = scene_title("Bounds and initial guess", f"pendulum swing-up, N = {n} intervals, T = {T:g} s")
        self.play(FadeIn(title), run_time=0.4)

        # ================================================================ beat 1: one guess, four InterpolationType
        ax = make_axes([-3.55, 0.25, 0], 5.6, 3.9, [0, T], [-1.0, 4.0], 0.5, 1.0)
        decos = VGroup(
            axis_label("rotation angle q_rot (rad)", ax),
            time_label(ax),
            x_ticks(ax, [0, 0.5, 1.0], "{:.1f}"),
            y_ticks(ax, [0, q_end], "{:g}"),
        )
        self.play(Create(ax), FadeIn(decos), run_time=0.7)

        pins = VGroup(
            Circle(radius=0.13, color=C_BOUND, stroke_width=4).move_to(ax.c2p(0, float(d["bound_min"][0]))),
            Circle(radius=0.13, color=C_BOUND, stroke_width=4).move_to(ax.c2p(T, float(d["bound_min"][-1]))),
        )

        def guess_curve(k):
            return poly(ax, t, nodes[k], C_GUESS, 2).set_stroke(opacity=0.6).set_fill(opacity=0)

        def guess_dots(k):
            return VGroup(*[Dot(ax.c2p(t[i], nodes[k][i]), radius=0.055, color=C_GUESS) for i in range(n + 1)])

        def code_lines(i):
            return code_panel(
                [
                    (0, "# IT = InterpolationType", GRAY_B),
                    (0, 'x_bounds["q"][:, 0] = 0', C_BOUND),
                    (0, 'x_bounds["q"][1, -1] = 3.14', C_BOUND),
                    (0, f"interp = {IT_NAMES[i]}", C_GUESS),
                    (0, "x_init = InitialGuessList()", WHITE),
                    (0, 'x_init.add("q", guess, interpolation=interp)', WHITE),
                ],
                size=17,
            )

        def readout(k):
            r, c = shape[k]
            body = (
                f"guess: array of shape ({r}, {c}), used at {n + 1} nodes\n"
                f"bounds (red circles): array of shape ({b_shape[0]}, {b_shape[1]}), ±{b_hi:.2f} rad in between"
            )
            return place(Text(body, font_size=19, color=GRAY_A, line_spacing=0.9), CODE_X, -0.75)

        remarks = {
            "constant": "One column: the same value at every node.",
            "first_last": "Three columns: the first node, all the middle nodes, the last node.",
            "linear": "Two columns: the first and the last value, a straight line between them.",
            "each_frame": f"One column per node ({n + 1} columns): any curve you want.",
        }

        def remark(k):
            return place(say(remarks[k]), CODE_X, -1.95)

        panel = code_lines(0)
        code_line = panel[1][3]  # the changing "interp = ..." line
        curve, dots, info, rem = (
            guess_curve("constant"),
            guess_dots("constant"),
            readout("constant"),
            remark("constant"),
        )
        self.play(FadeIn(pins), FadeIn(panel), run_time=0.6)
        self.play(Create(curve), FadeIn(dots), FadeIn(info), FadeIn(rem), run_time=1.0)
        self.wait(0.5)
        for i in (1, 2, 3):
            k = TYPES[i]
            new_line = code_lines(i)[1][3]
            self.play(
                Transform(curve, guess_curve(k)),
                Transform(dots, guess_dots(k)),
                Transform(code_line, new_line),
                Transform(info, readout(k)),
                Transform(rem, remark(k)),
                run_time=1.1,
            )
            self.wait(0.5)
        foot1 = footer(
            f"Real InitialGuess and Bounds objects; the last guess is a coarse {ncoarse}-interval solve. "
            "SPLINE and CUSTOM also exist."
        )
        self.play(FadeIn(foot1), run_time=0.4)
        self.wait(0.5)

        # ================================================================ beat 2: three guesses, three real solves
        self.play(*[FadeOut(m) for m in self.mobjects if m is not title], run_time=0.6)
        self.add(title)
        ax_q = make_axes([-3.55, 0.85, 0], 5.6, 2.5, [0, T], [-1.0, 4.0], 0.5, 1.0)
        ax_d = make_axes([-3.55, -2.15, 0], 5.6, 1.6, [0, T], [-1.5, 2.2], 0.5, 1.0)
        decos2 = VGroup(
            axis_label("q_rot (rad): guess (dashed) and solution", ax_q, C_STATE),
            axis_label("Δq_rot = q_rot − q_rot from zeros (rad)", ax_d, C_PAR),
            time_label(ax_d),
            x_ticks(ax_d, [0, 0.5, 1.0], "{:.1f}"),
            y_ticks(ax_q, [0, q_end], "{:g}"),
            y_ticks(ax_d, [-1, 0, 1, 2]),
        )
        self.play(Create(ax_q), Create(ax_d), FadeIn(decos2), run_time=0.8)
        zero = hline(ax_d, 0, T, 0, GRAY_D, dashed=False)
        self.add(zero)

        sol = {r: d[f"{r}_q_rot"] for r in RUNS}
        iters = {r: int(d[f"{r}_iterations"]) for r in RUNS}
        cost = {r: float(d[f"{r}_cost"]) for r in RUNS}
        peak = {r: float(np.abs(d[f"{r}_tau"]).max()) for r in RUNS}
        conv = {r: int(d[f"{r}_status"]) == 0 for r in RUNS}

        def ghost(i):
            return DashedVMobject(poly(ax_q, t, nodes[RUN_GUESS[i]], C_GHOST, 3), num_dashes=40).set_opacity(0.8)

        def solution(i):
            return poly(ax_q, t, sol[RUNS[i]], C_STATE, 5)

        def diff(i):
            return poly(ax_d, t, sol[RUNS[i]] - sol["zeros"], C_PAR, 5)

        def code2(i):
            return code_panel(
                [
                    (0, f"interp = {RUN_IT[i]}", C_GUESS),
                    (0, "x_init = InitialGuessList()", WHITE),
                    (0, 'x_init.add("q", guess, interpolation=interp)', WHITE),
                    (0, '# same for "qdot" and "tau"', GRAY_B),
                    (0, "ocp = OptimalControlProgram(", WHITE),
                    (1, "bio_model, 20, 1.0, dynamics=dynamics,", WHITE),
                    (1, "x_init=x_init, u_init=u_init, ...)", WHITE),
                ],
                size=17,
            )

        def readout2(i):
            r = RUNS[i]
            body = ipopt_line(iters[r], conv[r]) + f"\ncost = {cost[r]:.2f}  ·  peak |τ| = {peak[r]:.1f} N"
            return place(Text(body, font_size=19, color=GRAY_A, line_spacing=0.9), CODE_X, -1.05)

        panel2 = code2(0)
        code_line2 = panel2[1][0]
        g, s, df, info2 = ghost(0), solution(0), diff(0), readout2(0)
        self.play(FadeIn(panel2), Create(g), run_time=0.8)
        self.play(Create(s), Create(df), FadeIn(info2), run_time=1.2)
        self.wait(0.4)
        for i in (1, 2):
            self.play(
                Transform(g, ghost(i)),
                Transform(s, solution(i)),
                Transform(df, diff(i)),
                Transform(code_line2, code2(i)[1][0]),
                Transform(info2, readout2(i)),
                run_time=1.3,
            )
            self.wait(0.4)
        c0, c1, c2 = (cost[r] for r in RUNS)
        rem2 = place(
            say(f"The costs {c0:.1f}, {c1:.1f} and {c2:.1f} are three different local minima: the guess picks one."),
            CODE_X,
            -2.4,
        )
        self.play(FadeIn(rem2))
        foot2 = footer(
            f"Warm start: the third guess comes from a coarse solve ({int(d['coarse_iterations'])} iterations, not "
            "counted above) interpolated on the nodes."
        )
        self.play(FadeIn(foot2))
        self.wait(2.5)

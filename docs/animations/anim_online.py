"""
Manim Community animation on the ONLINE PLOT of bioptim (``Solver.IPOPT(show_online_optim=True)``).

A live GUI window cannot be captured reliably, so scene 1 RE-DRAWS the online panel from REAL IPOPT iterates
(``data/online_iterates.npz``, see ``generate_online_data.py``: the k-th iterate = a real solve with
``set_maximum_iterations(k)``); it is labelled as such on screen. Scene 2 shows two REAL matplotlib figures produced by
bioptim (``sol.graphs(show_bounds=True, save_name=...)``, Agg backend) embedded as images.

Scenes:
    1. OnlineIterates   the 2x2 online panel (theta, tau, custom plot, IPOPT output) refreshed at every iteration
    2. OfflineGraphs    the offline twin ``sol.graphs()`` (real figures)

Helpers are imported read-only from ``features_scenes.py``. No LaTeX. Render: see notes/online.md.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from manim import *

from features_scenes import (  # noqa: E402  (also sets the default fonts)
    CODE_X,
    CODE_W,
    code_panel,
    fit,
    make_axes,
    poly,
    scene_title,
    steps,
    x_ticks,
    y_ticks,
)

DATA_DIR = Path(__file__).parent / "data"
C_STATE, C_CTRL, C_CUSTOM = YELLOW_C, GREEN_C, TEAL_C
C_PR, C_DU = RED_C, BLUE_C


class OnlineIterates(Scene):
    def construct(self):
        d = np.load(DATA_DIR / "online_iterates.npz")
        t, T, n = d["t"], float(d["final_time"]), int(d["n_iter"])
        q, tau, cost = d["q"], d["tau"], d["cost"]
        pr, du = d["hist_inf_pr"], d["hist_inf_du"]

        def lg(v):
            return np.log10(max(float(v), 1e-11))

        title = scene_title("Online plot", "watch the solve, iteration by iteration")

        # ---- the (re-drawn) window
        win = RoundedRectangle(corner_radius=0.08, width=6.9, height=6.35, stroke_color=GRAY_B, stroke_width=2)
        win.move_to([-3.55, -0.42, 0])
        head = Text("bioptim online plot", font_size=17, color=GRAY_B)
        head.move_to(win.get_top() + DOWN * 0.22).align_to(win, LEFT).shift(RIGHT * 0.2)
        disclaimer = Text("re-drawn from real IPOPT iterates, not a screen capture", font_size=16, color=ORANGE)
        disclaimer.move_to(win.get_bottom() + UP * 0.2)

        cx, rows = [-5.35, -1.95], [1.05, -1.75]
        aw, ah = 2.5, 1.65
        ax_q = make_axes([cx[0], rows[0], 0], aw, ah, [0, T], [-1, 3.6])
        ax_u = make_axes([cx[1], rows[0], 0], aw, ah, [0, T], [-28, 14])
        ax_c = make_axes([cx[0], rows[1], 0], aw, ah, [0, T], [-60, 210])
        ax_i = make_axes([cx[1], rows[1], 0], aw, ah, [0, n], [-11, 5.2])
        axes = [ax_q, ax_u, ax_c, ax_i]

        def lab(s, ax, col):
            return Text(s, font_size=14, color=col).next_to(ax, UP, buff=0.06).align_to(ax, LEFT)

        labels = VGroup(
            lab("q  (pendulum angle, rad)", ax_q, C_STATE),
            lab("tau  (force, N)", ax_u, C_CTRL),
            lab('custom: "angle (deg)"', ax_c, C_CUSTOM),
            lab("IPOPT output", ax_i, GRAY_B),
        )
        zero = VGroup(
            *[
                DashedLine(ax.c2p(0, 0), ax.c2p(ax.x_range[1], 0), color=GRAY_D, stroke_width=1.5)
                for ax in (ax_q, ax_u, ax_c)
            ]
        )
        ticks = VGroup(
            y_ticks(ax_q, [0, 3]),
            y_ticks(ax_u, [-25, 0]),
            y_ticks(ax_c, [0, 180]),
            y_ticks(ax_i, [-10, -5, 0, 5]),
            x_ticks(ax_q, [0, 1]),
            x_ticks(ax_u, [0, 1]),
            x_ticks(ax_c, [0, 1]),
            x_ticks(ax_i, [0, n]),
        )
        for grp in ticks:
            for m in grp:
                m.scale(0.8)
        legend = VGroup(Text("inf_pr", font_size=13, color=C_PR), Text("inf_du", font_size=13, color=C_DU))
        legend.arrange(RIGHT, buff=0.15).next_to(ax_i, UP, buff=0.06).align_to(ax_i, RIGHT)
        log_note = Text("log scale", font_size=13, color=GRAY_B).move_to(ax_i.c2p(n * 0.5, -8.6))

        self.play(
            FadeIn(title),
            Create(win),
            FadeIn(head),
            FadeIn(disclaimer),
            *[Create(a) for a in axes],
            FadeIn(labels),
            FadeIn(zero),
            FadeIn(ticks),
            FadeIn(legend),
            FadeIn(log_note),
            run_time=0.9,
        )

        # ---- code beside the window (verified: interface_utils.py, ipopt_options.py, optimal_control_program.py)
        pa = code_panel(
            [
                (0, "solver = Solver.IPOPT(show_online_optim=True)", YELLOW_C),
                (0, "# same as online_optim=OnlineOptim.DEFAULT; or e.g.", GRAY_B),
                (0, "Solver.IPOPT(online_optim=OnlineOptim.MULTIPROCESS_SERVER,", WHITE),
                (5, "show_options=dict(show_bounds=True))", WHITE),
                (0, "sol = ocp.solve(solver)   # window refreshed at each iteration", WHITE),
            ],
            size=17,
            top=2.45,
            caption="1. ask the solver for the live window",
        )
        pb = code_panel(
            [
                (0, 'ocp.add_plot("angle (deg)",', TEAL_C),
                (3, "lambda t0, phases_dt, node_idx, x, u, p, a, d:", WHITE),
                (3, "x[[1], :] * 180 / np.pi,", WHITE),
                (3, "plot_type=PlotType.PLOT)", WHITE),
                (0, "ocp.add_plot_ipopt_outputs()   # f, inf_pr, inf_du", C_DU),
            ],
            size=17,
            top=0.15,
            caption="2. optional: custom plots and the IPOPT-output panel",
        )
        self.play(FadeIn(pa), FadeIn(pb), run_time=0.6)

        # ---- the replay: everything is a function of the iteration counter
        kt = ValueTracker(0)

        def cur():
            return int(round(kt.get_value()))

        def curves():
            k = cur()
            g = VGroup(
                poly(ax_q, t, q[k], C_STATE, 3),
                steps(ax_u, t, tau[k], C_CTRL, 3),
                poly(ax_c, t, q[k] * 180 / np.pi, C_CUSTOM, 3),
            )
            if k > 0:
                g.add(
                    VMobject(color=C_PR, stroke_width=3).set_points_as_corners(
                        [ax_i.c2p(i, lg(pr[i])) for i in range(1, k + 1)] * (1 + (k == 1))
                    ),
                    VMobject(color=C_DU, stroke_width=3).set_points_as_corners(
                        [ax_i.c2p(i, lg(du[i])) for i in range(1, k + 1)] * (1 + (k == 1))
                    ),
                )
            g.add(
                Dot(ax_i.c2p(k, lg(pr[k])), radius=0.04, color=C_PR),
                Dot(ax_i.c2p(k, lg(du[k])), radius=0.04, color=C_DU),
            )
            return g

        def readout():
            k = cur()
            r = VGroup(
                Text(f"iteration {k:2d}", font_size=17, weight=BOLD, color=WHITE),
                Text(f"f = {cost[k]:.1f}", font_size=16, color=GRAY_B),
            ).arrange(RIGHT, buff=0.35)
            return r.move_to(win.get_top() + DOWN * 0.22).align_to(win, RIGHT).shift(LEFT * 0.2)

        self.add(always_redraw(curves), always_redraw(readout))
        self.play(kt.animate.set_value(n), run_time=8, rate_func=linear)
        done = Text(f"Solve_Succeeded: {n} iterations, cost {float(d['full_cost']):.2f}", font_size=20, color=GREEN_C)
        fit(done, CODE_W)
        done.move_to([CODE_X, -3.55, 0], aligned_edge=LEFT)
        self.play(FadeIn(done))
        self.wait(1.2)


class OfflineGraphs(Scene):
    def construct(self):
        title = scene_title("The offline twin", "the same plots, after the solve")
        img_q = ImageMobject(str(DATA_DIR / "online_graphs_q_states.png"))
        img_c = ImageMobject(str(DATA_DIR / "online_graphs_angle (deg).png"))
        for im in (img_q, img_c):
            im.set_height(2.4)
        pair = Group(img_q, img_c).arrange(RIGHT, buff=0.2).move_to([-3.5, 0.3, 0])
        note = Text("real matplotlib figures saved by bioptim (Agg backend)", font_size=17, color=ORANGE)
        cap_q = Text("q_states", font_size=16, color=GRAY_B).next_to(img_q, UP, buff=0.08)
        cap_c = Text('custom plot "angle (deg)"', font_size=16, color=GRAY_B).next_to(img_c, UP, buff=0.08)
        note.next_to(pair, DOWN, buff=0.25)
        fit(note, 6.6)
        self.play(FadeIn(title), run_time=0.5)
        self.play(FadeIn(img_q), FadeIn(img_c), FadeIn(cap_q), FadeIn(cap_c), FadeIn(note), run_time=0.8)
        panel = code_panel(
            [
                (0, "sol = ocp.solve(Solver.IPOPT())", WHITE),
                (0, "sol.graphs(show_bounds=True,", YELLOW_C),
                (2, "show_now=False,", GRAY_B),
                (2, 'save_name="online_graphs")', YELLOW_C),
            ],
            size=19,
            top=2.45,
            caption="the same custom plots, drawn once from the solution",
        )
        self.play(FadeIn(panel), run_time=0.5)
        info = VGroup(
            Text("saves one png per figure: <save_name>_<figure>.png", font_size=18, color=GRAY_B),
            Text("show_bounds=True: axes follow the bounds, not the data", font_size=18, color=GRAY_B),
            Text("the IPOPT-output panel is live only (not in sol.graphs)", font_size=18, color=GRAY_B),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
        fit(info, CODE_W)
        info.move_to([CODE_X, -0.3, 0], aligned_edge=UL)
        self.play(FadeIn(info), run_time=0.5)
        self.wait(2.5)

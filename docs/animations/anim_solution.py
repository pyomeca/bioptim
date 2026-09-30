"""
Manim CE scene: the ``Solution`` object and its accessors. REAL bioptim / IPOPT solve stored in
``data/solution_pendulum.npz`` (see ``generate_solution_data.py``): 2-phase pendulum, multiple shooting with RK4
(phase 0: 4 intervals x 3 RK steps, phase 1: 3 intervals x 2 RK steps).  Every marker below is one column of an array
really returned by the accessor, drawn at its real time.

Scene: SolutionTour (about 17 s).  Render (from docs/animations):  manim render -qh anim_solution.py SolutionTour
"""

import numpy as np
from manim import *

from features_scenes import CODE_W, DATA_DIR, code, fit, scene_title

CODE_X0 = 0.15
W = WHITE
C_PH0, C_PH1 = BLUE_C, ORANGE
C_DEC, C_STEP, C_INTERP, C_INTEG = YELLOW_C, GREEN_C, PURPLE_B, RED_C
T0X, T1X = -5.0, -0.7  # screen x of t = 0 and t = T
ROT = 1


def caption(text, size=17, color=GRAY_B):
    return Text(text, font_size=size, color=color)


class SolutionTour(Scene):
    def construct(self):
        d = np.load(DATA_DIR / "solution_pendulum.npz")
        ns, ts = d["ns"], d["ts"]
        T = float(ts.sum())
        t_ph = float(ts[0])

        def tx(t):
            return T0X + (T1X - T0X) * t / T

        # markers (times and q_rot) as returned by the accessors
        dec_t = np.concatenate([d["decision_time_p0"], d["decision_time_p1"]])
        dec_q = d["decision_states_ALL"][ROT]
        step_t = np.concatenate([d["stepwise_time_p0"], d["stepwise_time_p1"]])
        step_q = d["stepwise_states_ALL"][ROT]
        int_q = np.concatenate([d["integrate_p0"][ROT], d["integrate_p1"][ROT]])
        ip_t, ip_q = d["interpolate100_time"], d["interpolate100_q"][ROT]
        n_dec, n_step, n_ip = dec_q.size, step_q.size, ip_q.size
        assert d["decision_states_ALL"].shape == (4, n_dec) and d["stepwise_states_ALL"].shape == (4, n_step)
        assert d["interpolate100_q"].shape == (2, n_ip)
        assert d["integrate_p0"].shape == (4, 17) and d["integrate_p1"].shape == (4, 10)
        assert (n_dec, n_step, n_ip) == (int(sum(ns) + len(ns)), 27, 100)
        lo, hi = float(step_q.min()), float(step_q.max())
        y_c0, y_c1 = 1.0, 2.05

        def yq(v):
            return y_c0 + (y_c1 - y_c0) * (v - lo) / (hi - lo)

        title = scene_title("The Solution object", "one solve, seen through different accessors")
        self.play(FadeIn(title), run_time=0.4)

        # ------------------------------------------------------------ time axis, phases, shooting nodes, curve
        y_top, y_bot = 2.3, -2.55
        band0 = Rectangle(
            width=tx(t_ph) - tx(0), height=y_top - y_bot, stroke_width=0, fill_color=C_PH0, fill_opacity=0.12
        ).move_to([(tx(0) + tx(t_ph)) / 2, (y_top + y_bot) / 2, 0])
        band1 = Rectangle(
            width=tx(T) - tx(t_ph), height=y_top - y_bot, stroke_width=0, fill_color=C_PH1, fill_opacity=0.12
        ).move_to([(tx(t_ph) + tx(T)) / 2, (y_top + y_bot) / 2, 0])
        ph_lab = VGroup(
            caption("phase 0", 15, C_PH0).move_to([(tx(0) + tx(t_ph)) / 2, y_top + 0.15, 0]),
            caption("phase 1", 15, C_PH1).move_to([(tx(t_ph) + tx(T)) / 2, y_top + 0.15, 0]),
        )
        node_lines = VGroup(
            *[
                DashedLine([tx(t), y_bot, 0], [tx(t), y_top, 0], color=GRAY_D, stroke_width=1.5)
                for t in np.unique(np.round(dec_t, 9))
            ]
        )
        tlab = VGroup(
            *[caption(f"{t:g}", 16).move_to([tx(t), y_bot - 0.22, 0]) for t in (0, t_ph, T)],
            caption("t (s)", 15).move_to([T1X + 0.55, y_bot - 0.22, 0]),
        )
        curve = VMobject(color=GRAY_B, stroke_width=3).set_points_as_corners(
            [[tx(t), yq(v), 0] for t, v in zip(step_t, step_q)]
        )
        curve_lab = caption("q_rot(t)", 16).move_to([-6.85, (y_c0 + y_c1) / 2, 0], aligned_edge=LEFT)
        self.play(
            FadeIn(VGroup(band0, band1, ph_lab, node_lines, tlab, curve_lab)), Create(curve, run_time=0.8), run_time=0.8
        )

        # ------------------------------------------------------------ right panel: code
        cap = caption("Bioptim code")
        solve = code("sol = ocp.solve(Solver.IPOPT())", 17, W)
        legend = VGroup(
            caption("SolutionMerge (argument  to_merge=)", 15, GRAY_B),
            code("KEYS    q, qdot stacked: 4 = 2 + 2", 14, GRAY_A),
            code("NODES   nodes side by side", 14, GRAY_A),
            code("PHASES  phases side by side", 14, GRAY_A),
            code("ALL     the three together", 14, GRAY_A),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        head = VGroup(cap, solve, legend).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
        fit(head, CODE_W)
        head.move_to([CODE_X0, 2.6, 0], aligned_edge=UL)
        self.play(FadeIn(head), run_time=0.5)

        # ------------------------------------------------------------ the four rows
        ys = [0.3, -0.55, -1.4, -2.25]
        n0, n1 = len(d["decision_time_p0"]), len(d["decision_time_p1"])
        s0, s1 = len(d["stepwise_time_p0"]), len(d["stepwise_time_p1"])
        rows = [
            dict(
                name="decision",
                color=C_DEC,
                t=dec_t,
                q=dec_q,
                r=0.075,
                code="sol.decision_states(to_merge=SolutionMerge.ALL)",
                shape=f"-> (4, {n_dec}) = {n0} + {n1} nodes, t = 0.6 s twice",
            ),
            dict(
                name="stepwise",
                color=C_STEP,
                t=step_t,
                q=step_q,
                r=0.05,
                code="sol.stepwise_states(to_merge=SolutionMerge.ALL)",
                shape=f"-> (4, {n_step}) = {s0} + {s1}: RK4 substeps + interval ends",
            ),
            dict(
                name="interpolate",
                color=C_INTERP,
                t=ip_t,
                q=ip_q,
                r=0.02,
                code='sol.interpolate(100)["q"]',
                shape=f"-> (2, {n_ip}) evenly spaced in time",
            ),
            dict(
                name="integrate",
                color=C_INTEG,
                t=step_t,
                q=int_q,
                r=0.05,
                code="sol.integrate(to_merge=[SolutionMerge.KEYS, SolutionMerge.NODES])",
                shape=f"-> [(4, {s0}), (4, {s1})] one per phase; max |x - stepwise| = {float(d['integrate_gap']):.0e}",
            ),
        ]
        prev_on_curve = None
        for y, row in zip(ys, rows):
            lab = code(row["name"], 15, row["color"]).move_to([-6.85, y, 0], aligned_edge=LEFT)
            line = Line([tx(0), y, 0], [tx(T), y, 0], color=GRAY_D, stroke_width=2)
            dots = VGroup(*[Dot([tx(t), y, 0], radius=row["r"], color=row["color"]) for t in row["t"]])
            on_curve = VGroup(
                *[
                    Dot([tx(t), yq(v), 0], radius=row["r"] * 1.15, color=row["color"])
                    for t, v in zip(row["t"], row["q"])
                ]
            )
            txt = VGroup(code(row["code"], 16, W), caption(row["shape"], 14, row["color"]))
            txt.arrange(DOWN, aligned_edge=LEFT, buff=0.06)
            fit(txt, CODE_W)
            txt.move_to([CODE_X0, y, 0], aligned_edge=LEFT)
            n = len(row["t"])
            self.play(FadeIn(lab), Create(line), FadeIn(txt), run_time=0.4)
            if prev_on_curve is not None:
                self.play(FadeOut(prev_on_curve), run_time=0.15)
            self.play(
                LaggedStart(*[FadeIn(a, scale=1.6) for a in dots], lag_ratio=1.0 / max(n, 8)),
                FadeIn(on_curve, run_time=0.9),
                run_time=1.0,
            )
            self.wait(0.7)
            prev_on_curve = on_curve
        self.play(FadeOut(prev_on_curve), run_time=0.2)

        # ------------------------------------------------------------ ending: the numbers and the plots
        c = float(d["cost"])
        dc0, dc1 = (float(v) for v in d["detailed_cost"])
        assert abs(c - dc0 - dc1) < 1e-6
        end = VGroup(
            VGroup(
                code("sol.cost", 17, W),
                code(f"-> {c:.2f}", 17, GRAY_A),
                code("sol.detailed_cost", 17, W).shift(RIGHT * 0.5),
                code(f"-> {dc0:.2f} (phase 0), {dc1:.2f} (phase 1)", 17, GRAY_A).shift(RIGHT * 0.5),
            ).arrange(RIGHT, buff=0.25),
            VGroup(
                code("sol.print_cost()", 17, W),
                code("sol.graphs()", 17, W),
                code("sol.animate()", 17, W),
            ).arrange(RIGHT, buff=0.5),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
        fit(end, 13.2)
        end.move_to([-6.85, -3.42, 0], aligned_edge=LEFT)
        self.play(FadeIn(end), run_time=0.5)
        self.wait(2.5)

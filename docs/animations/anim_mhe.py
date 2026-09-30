"""
Manim Community animation: Moving Horizon Estimation (MovingHorizonEstimator), driven by REAL bioptim/IPOPT solves stored
in ``data/mhe_results.npz`` (see ``generate_mhe_data.py``): a cart-pendulum motion is simulated, the pendulum angle is
"measured" with synthetic Gaussian noise, and a 0.5 s window (10 intervals) slides over the measurements, one solve per
new measurement (40 windows).  The estimate plotted is the first node of each window.

Scene: MHEWindow (about 15 s).  Render (from docs/animations):  manim render -qh anim_mhe.py MHEWindow
"""

import numpy as np
from manim import *

from features_scenes import (
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

C_TRUTH = GRAY_B
C_MEAS = RED_C
C_WIN = ORANGE
C_EST = GREEN_C
CODE_X0 = 0.15


def code_block(lines, size=15):
    block = VGroup(*[code(text, size, color) for _, text, color in lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.07)
    for line, (level, _, _) in zip(block, lines):
        line.shift(RIGHT * 0.3 * level)
    return block


def Lines(*lines, font_size=18, color=WHITE, line_spacing=0.9):
    """Left-aligned stack of Text lines. Unlike Paragraph, every line is translated on its own (no truncation in FR)."""
    return VGroup(*[Text(s, font_size=font_size, color=color) for s in lines]).arrange(
        DOWN, aligned_edge=LEFT, buff=0.16 * font_size / 18 * line_spacing
    )


class MHEWindow(Scene):
    def construct(self):
        d = np.load(DATA_DIR / "mhe_results.npz")
        n, dt, n_win, sigma = int(d["n"]), float(d["dt"]), int(d["n_steps"]), float(d["noise_std"])
        t, truth, meas = d["t"], d["truth_theta"], d["meas_theta"]
        pred, est = d["pred_theta"], d["est_theta"]
        e_meas, e_est = meas - truth, est - truth[:n_win]
        t_end = t[-1]

        title = scene_title(
            "Moving Horizon Estimation",
            f"synthetic data: simulated motion + Gaussian noise (sigma = {sigma:g} rad) on the pendulum angle",
        )
        self.play(FadeIn(title), run_time=0.6)

        # ------------------------------------------------------------------------------------------ axes and code
        ax_q = make_axes([-3.55, 0.85, 0], 5.6, 2.7, [0, t_end], [-2.0, 2.0], 0.5, 1)
        ax_e = make_axes([-3.55, -2.3, 0], 5.6, 1.6, [0, t_end], [-0.25, 0.25], 0.5, 0.25)
        lab_q = Text("pendulum angle (rad)", font_size=20, color=GRAY_B).next_to(ax_q.get_y_axis(), UP, buff=0.08)
        lab_q.align_to(ax_q.get_y_axis(), LEFT)
        lab_e = Text("error vs truth (rad)", font_size=20, color=GRAY_B).next_to(ax_e.get_y_axis(), UP, buff=0.08)
        lab_e.align_to(ax_e.get_y_axis(), LEFT)
        decos = VGroup(
            lab_q,
            lab_e,
            time_label(ax_e),
            x_ticks(ax_e, [0, 0.5, 1, 1.5, 2, 2.5], "{:g}"),
            y_ticks(ax_q, [-1.5, 0, 1.5], "{:g}"),
            y_ticks(ax_e, [-0.2, 0, 0.2], "{:g}"),
        )
        t_fine = np.linspace(0, t_end, 300)
        truth_fine = np.interp(t_fine, t, truth)
        truth_line = DashedVMobject(poly(ax_q, t_fine, truth_fine, C_TRUTH, 3), num_dashes=70)

        W = WHITE
        obj = code_block(
            [
                (0, "objectives.add(ObjectiveFcn.Lagrange.TRACK_STATE,", W),
                (1, "key='q', index=1, node=Node.ALL,", W),
                (1, "weight=1000, target=np.zeros((1, N + 1)))", W),
            ]
        )
        ctor = code_block(
            [
                (0, "mhe = MovingHorizonEstimator(", W),
                (1, "model, window_len=N, window_duration=N * dt,", W),
                (1, "common_objective_functions=objectives,", W),
                (1, "x_bounds=x_bounds, u_bounds=u_bounds)", W),
            ]
        )
        upd = code_block(
            [
                (0, "def update_function(mhe, step, sol):", W),
                (1, "mhe.update_objectives_target(", W),
                (2, "target=meas[None, step:step + N + 1],", W),
                (2, "list_index=0)", W),
                (1, "return step < n_windows", W),
            ]
        )
        slv = code_block(
            [
                (0, "sol = mhe.solve(update_function,", W),
                (1, "solver=Solver.IPOPT())", W),
            ]
        )
        panel = VGroup(obj, ctor, upd, slv).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        cap = Text("Bioptim code", font_size=20, color=GRAY_B)
        panel = VGroup(cap, panel).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        fit(panel, CODE_W)
        panel.move_to([CODE_X0, 2.35, 0], aligned_edge=UL)

        def box(mob, color=YELLOW):
            return SurroundingRectangle(mob, color=color, buff=0.06, stroke_width=2.5)

        def legend_item(color, text, kind="line"):
            a, b = LEFT * 0.35, RIGHT * 0.35
            if kind == "dash":
                mark = DashedLine(a, b, color=color).set_stroke(width=3)
            elif kind == "dot":
                mark = VGroup(*[Dot(p, radius=0.045, color=color) for p in (a, ORIGIN, b)])
            elif kind == "faded":
                mark = Line(a, b, color=color).set_stroke(width=4, opacity=0.6)
            else:
                mark = Line(a, b, color=color).set_stroke(width=7)
            return VGroup(mark, Text(text, font_size=17, color=WHITE)).arrange(RIGHT, buff=0.15)

        legend = VGroup(
            legend_item(C_TRUTH, "truth (simulated)", "dash"),
            legend_item(C_MEAS, "noisy measurement", "dot"),
            legend_item(C_WIN, "window fit", "faded"),
            legend_item(C_EST, "estimate (first node)"),
        ).arrange_in_grid(rows=2, cols=2, buff=(0.5, 0.12), col_alignments="ll")
        legend.move_to([CODE_X0, -2.02, 0], aligned_edge=UL)

        self.play(FadeIn(VGroup(ax_q, ax_e, decos)), FadeIn(panel), run_time=0.8)
        zero_e = DashedLine(ax_e.c2p(0, 0), ax_e.c2p(t_end, 0), color=GRAY_D).set_stroke(width=2)
        self.play(Create(truth_line), FadeIn(zero_e), FadeIn(legend), run_time=0.8)

        # ------------------------------------------------------------------------------------------ the sliding window
        def status_text(k):
            it, st = int(d["iterations"][k]), int(d["status"][k])
            ok = "converged" if st == 0 else "FAILED"
            txt = f"window {k + 1}/{n_win}   IPOPT: {it} iterations, status {st} ({ok})"
            return Text(txt, font_size=18, color=GRAY_B).move_to([CODE_X0 + 0.2, -3.0, 0], aligned_edge=LEFT)

        def rms(x):
            return float(np.sqrt(np.mean(np.asarray(x) ** 2)))

        def readout(k):
            txt = Lines(
                "Root-mean-square (RMS) error so far, in rad:",
                f"measurement {rms(e_meas[: k + 1]):.3f}   ·   estimate {rms(e_est[: k + 1]):.3f}",
                font_size=18,
                color=WHITE,
                line_spacing=0.9,
            )
            return txt.move_to([CODE_X0 + 0.2, -3.45, 0], aligned_edge=LEFT)

        y_top, y_bot = ax_q.c2p(0, 2.0)[1], ax_e.c2p(0, -0.25)[1]

        def win_band(k):
            r = Rectangle(
                width=ax_q.c2p(n * dt, 0)[0] - ax_q.c2p(0, 0)[0],
                height=y_top - y_bot,
                stroke_width=0,
                fill_color=C_WIN,
                fill_opacity=0.1,
            )
            x = (ax_q.c2p(k * dt, 0)[0] + ax_q.c2p(k * dt + n * dt, 0)[0]) / 2
            return r.move_to([x, (y_top + y_bot) / 2, 0])

        def dots(ax, idx, y):
            return VGroup(*[Dot(ax.c2p(t[i], y[i]), radius=0.035, color=C_MEAS) for i in idx])

        band = win_band(0)
        status, ro = status_text(0), readout(0)
        m_dots = dots(ax_q, range(n + 1), meas)
        me_dots = dots(ax_e, range(n + 1), e_meas)
        self.play(FadeIn(m_dots), FadeIn(me_dots), FadeIn(band), run_time=0.8)
        box_cur = box(obj)
        self.play(Create(box_cur), run_time=0.3)

        ghosts = []
        for k in range(n_win):
            rt = 1.0 if k == 0 else 0.11
            tw = k * dt + dt * np.arange(n + 1)
            pw = poly(ax_q, tw, pred[k], C_WIN, 4).set_stroke(opacity=0.7)
            new_status, new_ro = status_text(k), readout(k)
            self.remove(status, ro)
            self.add(new_status, new_ro)
            status, ro = new_status, new_ro
            anims = [band.animate.become(win_band(k)), FadeIn(pw)]
            ghosts.append(pw)
            for j, g in enumerate(reversed(ghosts[:-1])):
                anims.append(g.animate.set_stroke(opacity=max(0.0, 0.3 - 0.1 * j)))
            if k > 0:
                new_m = dots(ax_q, [k + n], meas)
                new_me = dots(ax_e, [k + n], e_meas)
                anims += [FadeIn(new_m), FadeIn(new_me)]
            if k == 0:
                bu = box(upd)
                self.play(ReplacementTransform(box_cur, bu), run_time=0.4)
                box_cur = bu
                self.play(*anims, run_time=rt)
                self.wait(0.3)
                bs = box(slv)
                self.play(ReplacementTransform(box_cur, bs), run_time=0.3)
                box_cur = bs
            else:
                self.play(*anims, run_time=rt)
            if k > 0:
                ge = Line(ax_q.c2p((k - 1) * dt, est[k - 1]), ax_q.c2p(k * dt, est[k]), color=C_EST).set_stroke(width=8)
                ee = Line(ax_e.c2p((k - 1) * dt, e_est[k - 1]), ax_e.c2p(k * dt, e_est[k]), color=C_EST).set_stroke(
                    width=6
                )
                self.add(ge, ee)
            if len(ghosts) > 4:
                self.remove(ghosts.pop(0))
        self.remove(*ghosts)

        # ------------------------------------------------------------------------------------------ outro
        msg = Lines(
            f"{n_win} real IPOPT solves, all converged (t = 0 to {(n_win - 1) * dt:.2f} s).",
            f"Root-mean-square error of the noisy measurement: {rms(e_meas[:n_win]):.3f} rad.",
            f"Root-mean-square error of the estimate: {rms(e_est):.3f} rad.",
            font_size=19,
            color=WHITE,
            line_spacing=0.9,
        )
        fit(msg, 5.9)
        msg.move_to([CODE_X0 + 0.2, -3.05, 0], aligned_edge=LEFT)
        self.play(
            FadeOut(status), FadeOut(ro), FadeOut(box_cur), FadeOut(band), FadeOut(legend), FadeIn(msg), run_time=0.6
        )
        self.wait(2.0)

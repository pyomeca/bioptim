"""
Manim Community animation: receding-horizon control (NonlinearModelPredictiveControl), driven by REAL bioptim/IPOPT
solves stored in ``data/nmpc_results.npz`` (see ``generate_nmpc_data.py``): 30 consecutive windows of 1 s (10 nodes),
a cart following a sine reference, the window being advanced by one node at each solve.

Scene: NMPCWindow (about 15 s).  Render (from docs/animations):  manim render -qh anim_nmpc.py NMPCWindow
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
    steps,
    time_label,
    x_ticks,
    y_ticks,
)

C_PRED = ORANGE
C_APPLIED = GREEN_C
C_REF = GRAY_B
CODE_X0 = 0.15


def code_block(lines, size=18):
    block = VGroup(*[code(text, size, color) for _, text, color in lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
    for line, (level, _, _) in zip(block, lines):
        line.shift(RIGHT * 0.3 * level)
    return block


def Lines(*lines, font_size=18, color=WHITE, line_spacing=0.9):
    """Left-aligned stack of Text lines. Unlike Paragraph, every line is translated on its own (no truncation in FR)."""
    return VGroup(*[Text(s, font_size=font_size, color=color) for s in lines]).arrange(
        DOWN, aligned_edge=LEFT, buff=0.16 * font_size / 18 * line_spacing
    )


class NMPCWindow(Scene):
    def construct(self):
        d = np.load(DATA_DIR / "nmpc_results.npz")
        n, dt, n_steps = int(d["n"]), float(d["dt"]), int(d["n_steps"])
        amp, period = float(d["amp"]), float(d["period"])
        t_end = n_steps * dt + n * dt  # 3 s of motion + the last horizon

        title = scene_title(
            "Nonlinear model predictive control", "solve a 1 s window, apply the first control, slide by one node"
        )
        self.play(FadeIn(title), run_time=0.6)

        # ------------------------------------------------------------------------------------------ axes and code
        ax_q = make_axes([-3.55, 0.85, 0], 5.6, 2.7, [0, t_end], [-0.8, 0.8], 1, 0.4)
        ax_u = make_axes([-3.55, -2.3, 0], 5.6, 1.8, [0, t_end], [-70, 70], 1, 70)
        lab_q = Text("cart position (m)", font_size=20, color=GRAY_B).next_to(ax_q.get_y_axis(), UP, buff=0.08)
        lab_q.align_to(ax_q.get_y_axis(), LEFT)
        lab_u = Text("cart force (N)", font_size=20, color=GRAY_B).next_to(ax_u.get_y_axis(), UP, buff=0.08)
        lab_u.align_to(ax_u.get_y_axis(), LEFT)
        decos = VGroup(
            lab_q,
            lab_u,
            time_label(ax_u),
            x_ticks(ax_u, [0, 1, 2, 3, 4], "{:g}"),
            y_ticks(ax_q, [-0.5, 0, 0.5], "{:g}"),
            y_ticks(ax_u, [-60, 0, 60]),
        )
        t_fine = np.linspace(0, t_end, 200)
        ref = poly(ax_q, t_fine, amp * np.sin(2 * np.pi * t_fine / period), C_REF, 3)
        ref = DashedVMobject(ref, num_dashes=60)

        W = WHITE
        ctor = code_block(
            [
                (0, "nmpc = NonlinearModelPredictiveControl(", W),
                (1, "model, window_len=10, window_duration=1.0,", W),
                (1, "common_objective_functions=objectives,", W),
                (1, "x_bounds=x_bounds, u_bounds=u_bounds)", W),
            ]
        )
        upd = code_block(
            [
                (0, "def update_function(nmpc, step, sol):", W),
                (1, "nmpc.update_objectives_target(", W),
                (2, "target=ref(step), list_index=0)", W),
                (1, "return step < n_steps", W),
            ]
        )
        slv = code_block(
            [
                (0, "sol = nmpc.solve(update_function,", W),
                (1, "solver=Solver.IPOPT())", W),
            ]
        )
        panel = VGroup(ctor, upd, slv).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        cap_code = Text("Bioptim code", font_size=20, color=GRAY_B)
        panel = VGroup(cap_code, panel).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        fit(panel, CODE_W)
        panel.move_to([CODE_X0, 2.3, 0], aligned_edge=UL)

        def box(mob, color=YELLOW):
            return SurroundingRectangle(mob, color=color, buff=0.06, stroke_width=2.5)

        def legend_item(color, text, dashed=False, thick=False, faded=False):
            a, b = LEFT * 0.35, RIGHT * 0.35
            line = DashedLine(a, b, color=color) if dashed else Line(a, b, color=color)
            line.set_stroke(width=7 if thick else 3, opacity=0.45 if faded else 1)
            return VGroup(line, Text(text, font_size=18, color=WHITE)).arrange(RIGHT, buff=0.15)

        legend = VGroup(
            legend_item(C_REF, "cyclic reference", dashed=True),
            legend_item(C_PRED, "prediction over the window (faded)", faded=True),
            legend_item(C_APPLIED, "applied: first control / next state", thick=True),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        legend.move_to([CODE_X0, -2.2, 0], aligned_edge=LEFT)

        self.play(FadeIn(VGroup(ax_q, ax_u, decos)), FadeIn(panel), run_time=0.8)
        self.play(Create(ref), FadeIn(legend), run_time=0.8)
        box_ctor = box(ctor)
        self.play(Create(box_ctor), run_time=0.4)

        # ------------------------------------------------------------------------------------------ the sliding window
        def status_text(k):
            it, st = int(d["iterations"][k]), int(d["status"][k])
            ok = "converged" if st == 0 else "FAILED"
            txt = f"window {k + 1}/{n_steps}   IPOPT: {it} iterations, status {st} ({ok})"
            return Text(txt, font_size=18, color=GRAY_B).move_to([CODE_X0, -3.1, 0], aligned_edge=LEFT)

        y_top, y_bot = ax_q.c2p(0, 0.8)[1], ax_u.c2p(0, -70)[1]

        def win_band(k):
            r = Rectangle(
                width=ax_q.c2p(n * dt, 0)[0] - ax_q.c2p(0, 0)[0],
                height=y_top - y_bot,
                stroke_width=0,
                fill_color=C_PRED,
                fill_opacity=0.1,
            )
            x = (ax_q.c2p(k * dt, 0)[0] + ax_q.c2p(k * dt + n * dt, 0)[0]) / 2
            return r.move_to([x, (y_top + y_bot) / 2, 0])

        band = win_band(0)
        status = status_text(0)
        ghosts = []
        self.add(band)

        for k in range(n_steps):
            rt = 1.0 if k < 3 else 0.17
            tw = k * dt + dt * np.arange(n + 1)
            pq = poly(ax_q, tw, d["pred_q"][k], C_PRED, 4).set_stroke(opacity=0.55)
            pu = steps(ax_u, tw, d["pred_tau"][k], C_PRED, 3).set_stroke(opacity=0.55)
            aq = poly(ax_q, [k * dt, (k + 1) * dt], d["applied_q"][k : k + 2], C_APPLIED, 8)
            au = Line(
                ax_u.c2p(k * dt, d["applied_tau"][k]), ax_u.c2p((k + 1) * dt, d["applied_tau"][k]), color=C_APPLIED
            ).set_stroke(width=8)
            new_status = status_text(k)
            self.remove(status)
            self.add(new_status)
            status = new_status
            anims = [band.animate.become(win_band(k)), FadeIn(pq), FadeIn(pu)]
            ghosts.append((pq, pu))
            for j, (gq, gu) in enumerate(reversed(ghosts[:-1])):
                op = max(0.0, 0.3 - 0.08 * j)
                anims += [gq.animate.set_stroke(opacity=op), gu.animate.set_stroke(opacity=op)]
            if k == 0:
                bu = box(upd)
                self.play(ReplacementTransform(box_ctor, bu), run_time=0.4)
                self.play(*anims, run_time=rt)
                cap = Text("predicted horizon", font_size=18, color=C_PRED).next_to(
                    ax_q.c2p(t_end, 0.8), UP, buff=0.05, aligned_edge=RIGHT
                )
                self.play(FadeIn(cap), run_time=0.3)
                self.wait(0.4)
                self.play(FadeOut(cap), run_time=0.2)
                self.play(Create(aq), Create(au), run_time=0.6)
                box_cur = bu
            else:
                self.play(*anims, AnimationGroup(Create(aq), Create(au)), run_time=rt)
            if k == 1:
                bs = box(slv)
                self.play(ReplacementTransform(box_cur, bs), run_time=0.3)
                box_cur = bs
            if len(ghosts) > 5:
                old = ghosts.pop(0)
                self.remove(*old)
        self.remove(*[m for g in ghosts[:-1] for m in g])

        # ------------------------------------------------------------------------------------------ outro
        msg = Lines(
            f"{n_steps} real IPOPT solves, all converged.",
            "The applied trajectory (green) chains the first node of each window.",
            font_size=19,
            color=WHITE,
            line_spacing=0.9,
        )
        fit(msg, 5.9)
        msg.move_to([CODE_X0, -2.2, 0], aligned_edge=LEFT)
        self.play(FadeOut(band), FadeOut(status), FadeOut(box_cur), FadeOut(legend), FadeIn(msg), run_time=0.6)
        self.wait(2.0)

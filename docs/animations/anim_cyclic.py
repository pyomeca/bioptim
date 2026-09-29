"""
Manim Community animation: cyclic NMPC (CyclicNonlinearModelPredictiveControl), driven by REAL bioptim/IPOPT solves
stored in ``data/cyclic_results.npz`` (see ``generate_cyclic_data.py``): a cart-pendulum, 4 cycles of 4 s (20 nodes),
one solve per cycle; the window advances by a whole cycle and the last node is bounded around the initial state.

Scene: CyclicNMPC (about 17 s).  Render (from docs/animations):  manim render -qh anim_cyclic.py CyclicNMPC
"""

import numpy as np
from manim import *

from features_scenes import (
    CODE_W,
    DATA_DIR,
    M,
    code,
    fit,
    make_axes,
    poly,
    scene_title,
    time_label,
    x_ticks,
    y_ticks,
)

C_CYC = [BLUE_C, ORANGE, BLUE_C, ORANGE]
C_REF = GRAY_B
C_FIRST = GREEN_C
C_LAST = RED_C
CODE_X0 = 0.15


def code_block(lines, size=18):
    block = VGroup(*[code(text, size, color) for _, text, color in lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
    for line, (level, _, _) in zip(block, lines):
        line.shift(RIGHT * 0.3 * level)
    return block


def caption(text):
    return Text(text, font_size=17, color=GRAY_B)


class CyclicNMPC(Scene):
    def construct(self):
        d = np.load(DATA_DIR / "cyclic_results.npz")
        n, T, n_cycles = int(d["cycle_len"]), float(d["cycle_duration"]), int(d["n_cycles"])
        amps = d["amps"]
        t_end = n_cycles * T
        tw = np.linspace(0, T, n + 1)

        title = scene_title("Cyclic NMPC", "one solve per cycle; the window advances by a whole cycle")
        self.play(FadeIn(title), run_time=0.6)

        # ------------------------------------------------------------------------------------------ axes
        ax_q = make_axes([-3.55, 0.95, 0], 5.6, 2.3, [0, t_end], [-0.9, 0.9], 4, 0.9)
        ax_t = make_axes([-3.55, -1.75, 0], 5.6, 1.9, [0, t_end], [-1.0, 1.0], 4, 1.0)
        lab_q = Text("cart position (m)", font_size=20, color=GRAY_B).next_to(ax_q.get_y_axis(), UP, buff=0.08)
        lab_q.align_to(ax_q.get_y_axis(), LEFT)
        lab_t = Text("pendulum angle (rad)", font_size=20, color=GRAY_B).next_to(ax_t.get_y_axis(), UP, buff=0.08)
        lab_t.align_to(ax_t.get_y_axis(), LEFT)
        decos = VGroup(
            lab_q,
            lab_t,
            time_label(ax_t),
            x_ticks(ax_t, [0, 4, 8, 12, 16], "{:g}"),
            y_ticks(ax_q, [-0.7, 0, 0.7], "{:g}"),
            y_ticks(ax_t, [-1, 0, 1], "{:g}"),
        )
        # dashed reference of each cycle (the amplitude changes for cycle 3)
        refs = VGroup()
        for k in range(n_cycles):
            r = poly(ax_q, k * T + tw, amps[k] * np.sin(2 * np.pi * tw / T), C_REF, 3)
            refs.add(DashedVMobject(r, num_dashes=25))

        # ------------------------------------------------------------------------------------------ code
        W = WHITE
        ctor = code_block(
            [
                (0, "nmpc = CyclicNonlinearModelPredictiveControl(", W),
                (1, "model, cycle_len=20, cycle_duration=4.0,", W),
                (1, "common_objective_functions=objectives,", W),
                (1, "x_bounds=x_bounds, u_bounds=u_bounds)", W),
            ]
        )
        upd = code_block(
            [
                (0, "def update_function(nmpc, cycle, sol):", W),
                (1, "nmpc.update_objectives_target(", W),
                (2, "target=ref(amps[cycle]), list_index=0)", W),
                (1, "return cycle < n_cycles", W),
            ]
        )
        slv = code_block(
            [
                (0, "sol = nmpc.solve(update_function,", W),
                (1, "solver=Solver.IPOPT())", W),
            ]
        )
        cyc_cap = caption("after each solve, bioptim does (x_last = state at the last node):")
        cyc = code_block(
            [
                (0, "x_bounds[key][:, 0] = x_last", W),
                (0, "x_bounds[key].min[s, 2] = x_last - 0.01 * range", W),
                (0, "x_bounds[key].max[s, 2] = x_last + 0.01 * range", W),
            ]
        )
        panel = VGroup(ctor, upd, slv, VGroup(cyc_cap, cyc).arrange(DOWN, aligned_edge=LEFT, buff=0.1)).arrange(
            DOWN, aligned_edge=LEFT, buff=0.2
        )
        fit(panel, CODE_W)
        panel.move_to([CODE_X0, 2.3, 0], aligned_edge=UL)

        def box_of(mob):
            return SurroundingRectangle(mob, color=YELLOW, buff=0.06, stroke_width=2.5)

        def legend_item(mob, text):
            return VGroup(mob, Text(text, font_size=17, color=WHITE)).arrange(RIGHT, buff=0.12)

        legend = VGroup(
            legend_item(DashedLine(LEFT * 0.3, RIGHT * 0.3, color=C_REF).set_stroke(width=3), "reference"),
            legend_item(Dot(color=C_FIRST, radius=0.07), "first node (= end of previous cycle)"),
            legend_item(Dot(color=C_LAST, radius=0.07), "last node (bounded around the first)"),
        ).arrange(RIGHT, buff=0.3)
        fit(legend, 6.4)
        legend.move_to([-3.55, -3.2, 0])

        cur_box = box_of(ctor)
        self.play(FadeIn(VGroup(ax_q, ax_t, decos)), FadeIn(panel), run_time=0.8)
        self.play(Create(refs), FadeIn(legend), Create(cur_box), run_time=0.8)

        # ------------------------------------------------------------------------------------------ one solve per cycle
        y_top, y_bot = ax_q.c2p(0, 0.9)[1], ax_t.c2p(0, -1.0)[1]

        def win_band(k):
            x0, x1 = ax_q.c2p(k * T, 0)[0], ax_q.c2p((k + 1) * T, 0)[0]
            r = Rectangle(width=x1 - x0, height=y_top - y_bot, stroke_width=0, fill_color=C_CYC[k], fill_opacity=0.10)
            return r.move_to([(x0 + x1) / 2, (y_top + y_bot) / 2, 0])

        def status_text(k):
            it, st = int(d["iterations"][k]), int(d["status"][k])
            gap = np.abs(d["win_x_last"][k] - d["win_x_first"][k]).max()
            ok = "converged" if st == 0 else "FAILED"
            txt = f"cycle {k + 1}/{n_cycles}  IPOPT {it} it., status {st} ({ok})   max |x_last - x_first| = {gap:.3f}"
            return Text(txt, font_size=17, color=GRAY_B).move_to([-6.7, -3.65, 0], aligned_edge=LEFT)

        band = win_band(0)
        status = status_text(0)
        self.add(band, status)
        for k in range(n_cycles):
            tt = k * T + tw
            cq = poly(ax_q, tt, d["win_q"][k], C_CYC[k], 5)
            cth = poly(ax_t, tt, d["win_theta"][k], C_CYC[k], 5)
            first = VGroup(
                Dot(ax_q.c2p(tt[0], d["win_q"][k][0]), color=C_FIRST, radius=0.08),
                Dot(ax_t.c2p(tt[0], d["win_theta"][k][0]), color=C_FIRST, radius=0.08),
            )
            last = VGroup(
                Dot(ax_q.c2p(tt[-1], d["win_q"][k][-1]), color=C_LAST, radius=0.08),
                Dot(ax_t.c2p(tt[-1], d["win_theta"][k][-1]), color=C_LAST, radius=0.08),
            )
            if k == 0:
                nb = box_of(upd)
                self.play(ReplacementTransform(cur_box, nb), run_time=0.3)
                cur_box = nb
            new_status = status_text(k)
            self.remove(status)
            status = new_status
            self.add(status)
            self.play(band.animate.become(win_band(k)), Create(cq), Create(cth), run_time=1.8 if k == 0 else 1.5)
            self.play(FadeIn(first), FadeIn(last), run_time=0.3)
            if k == 0:
                nb = box_of(cyc)
                self.play(ReplacementTransform(cur_box, nb), run_time=0.3)
                cur_box = nb
            self.wait(0.15)

        msg = M(
            f"{n_cycles} real IPOPT solves, all converged. Each cycle starts where the previous one ended; the "
            f"amplitude change (cycle 3) pushes the gap to the edge of the 1 % slack.",
            18,
            WHITE,
        )
        fit(msg, 13.2)
        msg.to_edge(DOWN, buff=0.1)
        self.play(FadeOut(band), FadeOut(status), FadeOut(cur_box), FadeIn(msg), run_time=0.5)
        self.wait(2.0)

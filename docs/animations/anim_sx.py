"""
Manim CE scene: ``use_sx=False`` (MX, default) vs ``use_sx=True`` (SX). REAL timings stored in ``data/sx_timings.npz``
(see ``generate_sx_data.py``): same cart-pendulum OCP, RK4, N = 50, each configuration built and solved 3 times.
The graph drawing of beat 1 is a SCHEMATIC (labelled as such), not a dump of the real CasADi graph.

Scene: SxVsMx (about 20 s).  Render (from docs/animations):  manim render -qh anim_sx.py SxVsMx
"""

import numpy as np
from manim import *

from features_scenes import CODE_W, DATA_DIR, M, code, fit, scene_title

CODE_X0 = 0.15
C_MX = BLUE_C
C_SX = ORANGE
W = WHITE
N = 50


def caption(text, size=19, color=GRAY_B):
    return Text(text, font_size=size, color=color)


def code_block(lines, size=17):
    block = VGroup(*[code(text, size, color) for _, text, color in lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.11)
    for line, (level, _, _) in zip(block, lines):
        line.shift(RIGHT * 0.3 * level)
    return block


def box(text, w, h, color, pos, size=17):
    r = RoundedRectangle(corner_radius=0.08, width=w, height=h, color=color, stroke_width=3)
    r.set_fill(color, 0.15).move_to(pos)
    return VGroup(r, Text(text, font_size=size, color=W).move_to(pos))


def mx_graph(cx, cy):
    """A few big nodes: one node per CasADi operation (here one integration step)."""
    a = box("x_k, u_k", 1.2, 0.6, C_MX, [cx - 2.4, cy, 0])
    b = box("RK4 step  (calls dynamics)", 2.3, 0.6, C_MX, [cx, cy, 0], 15)
    c = box("x_k+1", 1.0, 0.6, C_MX, [cx + 2.1, cy, 0])
    arrows = VGroup(
        Arrow(a.get_right(), b.get_left(), buff=0.05, color=GRAY_B, stroke_width=3, max_tip_length_to_length_ratio=0.3),
        Arrow(b.get_right(), c.get_left(), buff=0.05, color=GRAY_B, stroke_width=3, max_tip_length_to_length_ratio=0.3),
    )
    return VGroup(a, b, c, arrows)


def sx_graph(cx, cy):
    """Many scalar nodes: layers of small circles with fixed pseudo-random edges (schematic)."""
    rng = np.random.default_rng(3)
    sizes = [4, 9, 9, 9, 4]
    xs = np.linspace(cx - 2.6, cx + 2.6, len(sizes))
    layers = [[np.array([x, cy + (i - (n - 1) / 2) * 0.24, 0]) for i in range(n)] for x, n in zip(xs, sizes)]
    edges = VGroup()
    for la, lb in zip(layers[:-1], layers[1:]):
        for pb in lb:
            for pa in rng.choice(len(la), size=2, replace=False):
                edges.add(Line(la[pa], pb, color=GRAY_D, stroke_width=1.5))
    dots = VGroup(*[Dot(p, radius=0.055, color=C_SX) for layer in layers for p in layer])
    return VGroup(edges, dots)


class SxVsMx(Scene):
    def construct(self):
        d = np.load(DATA_DIR / "sx_timings.npz")
        n = N

        def med(cfg, key):
            return float(np.median(d[f"{cfg}_n{n}_{key}"]))

        stat = {}
        for cfg in ("mx", "sx"):
            stat[cfg] = dict(
                build=med(cfg, "build"),
                setup=float(np.median(d[f"{cfg}_n{n}_wall"] - d[f"{cfg}_n{n}_solve"])),
                solve=med(cfg, "solve"),
                iters=int(d[f"{cfg}_n{n}_iters"][0]),
                cost=float(d[f"{cfg}_n{n}_cost"][0]),
            )
        reps = int(d["repeats"])

        title = scene_title(
            "use_sx: MX or SX graph?",
            f"same optimal control problem (N = {n}, RK4, IPOPT), median of {reps} measured runs",
        )
        self.play(FadeIn(title), run_time=0.4)

        # ---------------------------------------------------------------- code (right)
        cap1 = caption("Bioptim code")
        code1 = code_block(
            [
                (0, "ocp = OptimalControlProgram(", W),
                (1, "bio_model, n_shooting, final_time,", W),
                (1, "dynamics=DynamicsOptions(...),", W),
                (1, "use_sx=True)   # default: False (MX)", C_SX),
            ]
        )
        cap2 = caption("timings read after the solve")
        code2 = code_block(
            [
                (0, "sol = ocp.solve(Solver.IPOPT())", W),
                (0, "sol.real_time_to_optimize   # IPOPT time", W),
                (0, "sol.iterations", W),
            ]
        )
        panel = VGroup(cap1, code1, cap2, code2).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        panel[2:].shift(DOWN * 0.15)
        panel[3].shift(DOWN * 0.08)
        fit(panel, CODE_W)
        panel.move_to([CODE_X0, 2.3, 0], aligned_edge=UL)
        self.play(FadeIn(panel), run_time=0.5)

        # ---------------------------------------------------------------- beat 1: schematic of the two graphs
        mx = mx_graph(-3.7, 1.35)
        mx_lab = M("<b>MX</b>: few big nodes, each one a whole function call", 19, C_MX)
        sx = sx_graph(-3.7, -0.9)
        sx_lab = M("<b>SX</b>: every scalar operation is a node (one big flat graph)", 19, C_SX)
        fit(mx_lab, 6.2).move_to([-6.95, 2.2, 0], aligned_edge=LEFT)
        fit(sx_lab, 6.2).move_to([-6.95, 0.3, 0], aligned_edge=LEFT)
        sch = caption("schematic, not the real CasADi graph", 16).move_to([-6.95, -2.35, 0], aligned_edge=LEFT)
        self.play(FadeIn(mx_lab), FadeIn(mx), run_time=0.7)
        self.play(FadeIn(sx_lab), FadeIn(sx), FadeIn(sch), run_time=0.9)
        self.wait(2.0)
        self.play(FadeOut(VGroup(mx_lab, mx, sx_lab, sx, sch)), run_time=0.4)

        # ---------------------------------------------------------------- beat 2: measured bars
        base_y, height = -1.7, 3.4
        vmax = 1.3
        x0, gw = -6.0, 1.95
        groups = [
            ("build", "problem build", "constructor"),
            ("setup", "solver set-up", "in ocp.solve"),
            ("solve", "IPOPT solve", f"{stat['mx']['iters']} iterations"),
        ]

        def yv(v):
            return base_y + height * v / vmax

        axis = Line([-6.4, base_y, 0], [-0.4, base_y, 0], color=GRAY_B)
        yaxis = Line([-6.4, base_y, 0], [-6.4, base_y + height + 0.1, 0], color=GRAY_B)
        ticks = VGroup(
            *[Text(f"{v:g}", font_size=15, color=GRAY_B).move_to([-6.7, yv(v), 0]) for v in (0, 0.5, 1.0)],
            *[DashedLine([-6.4, yv(v), 0], [-0.4, yv(v), 0], color=GRAY_D, stroke_width=1.2) for v in (0.5, 1.0)],
        )
        ylab = caption("wall time (s), median", 16).move_to([-6.95, base_y + height + 0.35, 0], aligned_edge=LEFT)
        self.play(FadeIn(VGroup(axis, yaxis, ticks, ylab)), run_time=0.4)

        bars, labels = {}, {}
        for gi, (key, name, sub) in enumerate(groups):
            gx = x0 + gi * gw + 0.35
            for ci, (cfg, color) in enumerate((("mx", C_MX), ("sx", C_SX))):
                v = stat[cfg][key]
                bar = Rectangle(
                    width=0.6, height=max(yv(v) - base_y, 0.02), stroke_width=0, fill_color=color, fill_opacity=0.9
                )
                bar.move_to([gx + ci * 0.68, base_y, 0], aligned_edge=DOWN)
                lab = Text(f"{v:.2f}", font_size=17, color=W).next_to(bar, UP, buff=0.06)
                bars[(cfg, key)], labels[(cfg, key)] = bar, lab
            nm = Text(name, font_size=17, color=W).move_to([gx + 0.34, base_y - 0.25, 0])
            sb = code(sub, 12, GRAY_B)
            fit(sb, gw - 0.1).move_to([gx + 0.34, base_y - 0.55, 0])
            self.play(
                FadeIn(nm),
                FadeIn(sb),
                *[GrowFromEdge(bars[(c, key)], DOWN) for c in ("mx", "sx")],
                *[FadeIn(labels[(c, key)]) for c in ("mx", "sx")],
                run_time=0.9,
            )
        leg = (
            VGroup(code("use_sx=False (MX)", 15, C_MX), code("use_sx=True (SX)", 15, C_SX))
            .arrange(RIGHT, buff=0.5)
            .move_to([-3.5, base_y - 1.0, 0])
        )
        self.play(FadeIn(leg), run_time=0.3)

        # readout computed from the data
        mxs, sxs = stat["mx"], stat["sx"]
        per_mx, per_sx = 1000 * mxs["solve"] / mxs["iters"], 1000 * sxs["solve"] / sxs["iters"]
        tot_mx = mxs["build"] + mxs["setup"] + mxs["solve"]
        tot_sx = sxs["build"] + sxs["setup"] + sxs["solve"]
        breakeven = (sxs["setup"] - mxs["setup"]) / ((mxs["solve"] - sxs["solve"]) / mxs["iters"])
        self.play(FadeOut(VGroup(cap2, code2)), run_time=0.3)
        msg = VGroup(
            M(
                f"<b>per IPOPT iteration</b>   {per_mx:.0f} ms (MX)  →  {per_sx:.0f} ms (SX),  ÷ {per_mx / per_sx:.1f}",
                20,
                W,
            ),
            M(f"<b>same optimum</b>   cost {mxs['cost']:.4f} in {mxs['iters']} iterations, both", 20, W),
            M(f"<b>total, one solve</b>   {tot_mx:.2f} s (MX)  vs  {tot_sx:.2f} s (SX)", 20, W),
            M(f"SX pays its longer set-up back after about {breakeven:.0f} iterations", 19, GRAY_B),
            M("(this pendulum, this machine)", 17, GRAY_B),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.17)
        fit(msg, CODE_W)
        msg.move_to([CODE_X0, 0.35, 0], aligned_edge=UL)
        self.play(FadeIn(msg[:3], lag_ratio=0.3), run_time=1.2)
        self.play(FadeIn(msg[3:]), run_time=0.6)
        self.wait(2.0)

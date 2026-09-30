"""
Manim CE scene: ``PhaseDynamics.SHARED_DURING_THE_PHASE`` vs ``PhaseDynamics.ONE_PER_NODE``. REAL data from
``generate_phasedyn_data.py`` (``data/phasedyn_bench.npz``, ``phasedyn_hold.npz``, ``phasedyn_series.npz``): the
time-dependent pendulum of example_pendulum_time_dependent.py, RK4, SX, N = 30 / 60 / 120.

Scene: PhaseDynamicsScene (about 20 s).  Render (from docs/animations):  manim render -qh anim_phasedyn.py PhaseDynamicsScene
"""

import numpy as np
from manim import *

from features_scenes import (
    CODE_W,
    DATA_DIR,
    M,
    code,
    fit,
    scene_title,
    make_axes,
    poly,
    band,
    y_ticks,
    x_ticks,
    time_label,
)

CODE_X0 = 0.15
C_SH = BLUE_C
C_PN = ORANGE
W = WHITE


def code_block(lines, size=17):
    block = VGroup(*[code(text, size, color) for _, text, color in lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
    for line, (level, _, _) in zip(block, lines):
        line.shift(RIGHT * 0.3 * level)
    return block


def caption(text, size=19, color=GRAY_B):
    return Text(text, font_size=size, color=color)


def panel_of(*parts):
    panel = VGroup(*parts).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
    fit(panel, CODE_W)
    return panel.move_to([CODE_X0, 2.3, 0], aligned_edge=UL)


class PhaseDynamicsScene(Scene):
    def construct(self):
        bench = np.load(DATA_DIR / "phasedyn_bench.npz")
        hold = np.load(DATA_DIR / "phasedyn_hold.npz")
        series = np.load(DATA_DIR / "phasedyn_series.npz")
        n0 = 30

        title = scene_title(
            "PhaseDynamics: one function or one per node", "time-dependent pendulum, RK4 integrator, measured timings"
        )
        self.play(FadeIn(title), run_time=0.4)

        # ------------------------------------------------------------ beat 1: what the enum means
        xs = np.linspace(-6.4, -0.9, n0)
        cx = float(xs.mean())
        u_sh, u_pn = int(bench["shared_30_unique_integrators"]), int(bench["per_node_30_unique_integrators"])

        lab_sh = code("SHARED_DURING_THE_PHASE", 17, C_SH).move_to([-6.95, 2.4, 0], aligned_edge=LEFT)
        box = VGroup(
            RoundedRectangle(width=1.7, height=0.5, corner_radius=0.1, color=C_SH, stroke_width=3),
            code("f", 18, C_SH),
        ).move_to([cx, 1.75, 0])
        dots_a = VGroup(*[Dot([x, 1.0, 0], radius=0.06, color=GRAY_B) for x in xs])
        fan = VGroup(*[Line(box.get_bottom(), d.get_top(), color=C_SH, stroke_width=1.3) for d in dots_a])
        read_a = M(f"<b>{u_sh}</b> integrator Function object, reused at the {n0} nodes", 20, W).move_to(
            [-6.95, 0.55, 0], aligned_edge=LEFT
        )

        lab_pn = code("ONE_PER_NODE", 17, C_PN).move_to([-6.95, -0.35, 0], aligned_edge=LEFT)
        boxes = VGroup(
            *[Square(0.13, color=C_PN, stroke_width=2.5).set_fill(C_PN, 0.35).move_to([x, -0.95, 0]) for x in xs]
        )
        dots_b = VGroup(*[Dot([x, -1.5, 0], radius=0.06, color=GRAY_B) for x in xs])
        links = VGroup(
            *[Line(b.get_bottom(), d.get_top(), color=C_PN, stroke_width=1.3) for b, d in zip(boxes, dots_b)]
        )
        read_b = M(f"<b>{u_pn}</b> integrator Function objects, one per node", 20, W).move_to(
            [-6.95, -2.05, 0], aligned_edge=LEFT
        )
        foot = caption("counted with id() on nlp.dynamics after building the problem", 15).move_to(
            [-6.95, -2.85, 0], aligned_edge=LEFT
        )

        cap1 = caption("Bioptim code")
        code1 = code_block(
            [(0, "DynamicsOptions(phase_dynamics=", W), (1, "PhaseDynamics.SHARED_DURING_THE_PHASE)", C_SH)]
        )
        cap2 = caption("what nlp.dynamics holds (ode_solver_base.py)")
        code2 = code_block(
            [
                (0, "SHARED:", C_SH),
                (1, "dynamics = dynamics * nlp.ns", W),
                (0, "ONE_PER_NODE:", C_PN),
                (1, "for node_index in range(1, nlp.ns):", W),
                (2, "dynamics.append(initialize_integrator(", W),
                (3, "ocp, nlp, dynamics_index=0,", W),
                (3, "node_index=node_index))", W),
            ]
        )
        panel1 = panel_of(cap1, code1, cap2, code2)
        panel1[2:].shift(DOWN * 0.2)

        self.play(FadeIn(VGroup(lab_sh, box, dots_a, fan)), FadeIn(panel1), run_time=0.7)
        self.play(FadeIn(read_a), run_time=0.4)
        self.play(FadeIn(VGroup(lab_pn, boxes, dots_b, links)), run_time=0.7)
        self.play(FadeIn(VGroup(read_b, foot)), run_time=0.4)
        self.wait(2.6)

        # ------------------------------------------------------------ beat 2: cost of the option, same optimum
        self.play(
            FadeOut(VGroup(lab_sh, box, dots_a, fan, read_a, lab_pn, boxes, dots_b, links, read_b, foot)),
            FadeOut(VGroup(cap2, code2)),
            run_time=0.5,
        )
        ns = [30, 60, 120]

        def med(opt, n, key):
            return float(np.median(bench[f"{opt}_{n}_{key}"]))

        panels = []
        for j, (key, name) in enumerate((("build", "problem build time (s)"), ("solve", "IPOPT solve time (s)"))):
            x0 = -6.6 + j * 3.4
            base_y, height, width = -0.6, 2.6, 2.7
            vmax = max(med(o, n, key) for o in ("shared", "per_node") for n in ns) * 1.15
            g = VGroup(
                caption(name, 17).move_to([x0 + width / 2, base_y + height + 0.35, 0]),
                Line([x0 - 0.1, base_y, 0], [x0 + width, base_y, 0], color=GRAY_B),
            )
            for i, n in enumerate(ns):
                gx = x0 + 0.1 + i * 0.9
                g.add(Text(f"N={n}", font_size=14, color=GRAY_B).move_to([gx + 0.28, base_y - 0.22, 0]))
                for k, (opt, color) in enumerate((("shared", C_SH), ("per_node", C_PN))):
                    v = med(opt, n, key)
                    h = height * v / vmax
                    bar = Rectangle(width=0.36, height=h, stroke_width=0, fill_color=color, fill_opacity=0.9)
                    bar.move_to([gx + 0.18 + 0.4 * k, base_y + h / 2, 0])
                    g.add(bar, Text(f"{v:.1f}", font_size=13, color=W).next_to(bar, UP, buff=0.04))
            panels.append(g)
        legend = VGroup(
            VGroup(Square(0.16, stroke_width=0, fill_color=C_SH, fill_opacity=0.9), code("SHARED", 14, C_SH)).arrange(
                RIGHT, buff=0.1
            ),
            VGroup(
                Square(0.16, stroke_width=0, fill_color=C_PN, fill_opacity=0.9), code("ONE_PER_NODE", 14, C_PN)
            ).arrange(RIGHT, buff=0.1),
        ).arrange(RIGHT, buff=0.5)
        legend.move_to([-3.7, -1.3, 0])
        cs, cp = float(bench["shared_30_cost"]), float(bench["per_node_30_cost"])
        dq = float(bench["max_dq"])
        dq_txt = "0 (identical)" if dq == 0 else f"{dq:.0e} rad"
        same = M(
            f"<b>same optimum</b> (30 nodes):  cost {cs:.4f} vs {cp:.4f},  max |Δq| = {dq_txt}",
            19,
            W,
        ).move_to([-6.95, -2.2, 0], aligned_edge=LEFT)
        fit(same, 6.3)
        note = Paragraph(
            "N is the number of nodes.",
            "Median of 3 runs on one loaded machine (noisy), CasADi SX with RK4.",
            font_size=14,
            color=GRAY_B,
            line_spacing=0.9,
        ).move_to([-6.95, -2.85, 0], aligned_edge=LEFT)
        code_b = code_block(
            [
                (0, "DynamicsOptions(ode_solver=OdeSolver.RK4(),", W),
                (1, "phase_dynamics=PhaseDynamics.ONE_PER_NODE)", C_PN),
                (0, "sol = ocp.solve(Solver.IPOPT())", W),
            ]
        )
        panel2 = panel_of(caption("Bioptim code"), code_b)
        self.play(FadeOut(VGroup(cap1, code1)), FadeIn(panel2), FadeIn(VGroup(*panels)), FadeIn(legend), run_time=0.7)
        self.play(FadeIn(VGroup(same, note)), run_time=0.5)
        self.wait(2.8)

        # ------------------------------------------------------------ beat 3: where ONE_PER_NODE is required
        self.play(FadeOut(VGroup(*panels, legend, same, note)), FadeOut(panel2), run_time=0.5)
        t = np.linspace(0, float(hold["T"]), int(hold["N"]) + 1)
        ax = make_axes([-3.7, 0.0, 0], 5.6, 3.6, (0, 1), (-0.2, 3.4), 0.2, 1.0)
        yt, xt = y_ticks(ax, [0, 1, 2, 3]), x_ticks(ax, [0, 0.5, 1])
        ylab = (
            caption("pendulum rotation (rad)", 17)
            .next_to(ax.get_y_axis(), UP, buff=0.1)
            .align_to(ax.get_y_axis(), LEFT)
        )
        xlab = time_label(ax)
        nodes = [int(k) for k in hold["nodes"]]
        bd = band(ax, t[nodes[0]], t[nodes[-1]], -0.2, 3.4, C_PN, 0.18)
        bd_lab = Text(f"nodes {nodes[0]} to {nodes[-1]}", font_size=15, color=C_PN).move_to(ax.c2p(0.9, 0.3))
        ghost = poly(ax, t, hold["free_q"][1], GRAY_C, 3)
        curve = poly(ax, t, hold["hold_q"][1], C_PN, 5)
        gl = caption("without the constraint", 15, GRAY_C).move_to(ax.c2p(0.05, 2.5), aligned_edge=LEFT)

        code_c = code_block(
            [
                (0, "multinode_constraints.add(", W),
                (1, "MultinodeConstraintFcn.CUSTOM,", W),
                (1, "custom_function=hold_rotation,", W),
                (1, f"nodes_phase=(0,) * {len(nodes)},", W),
                (1, f"nodes=({nodes[0]} ... {nodes[-1]}))", W),
            ]
        )
        err = str(hold["shared_error"])
        cut = "more penalties than available in a multinode constraint"
        assert cut in err
        res_sh = VGroup(
            code("SHARED_DURING_THE_PHASE", 16, C_SH),
            code("ValueError: ... " + cut, 13, RED_C),
            code("... use phase_dynamics=PhaseDynamics.ONE_PER_NODE", 13, RED_C),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        res_pn = VGroup(
            code("ONE_PER_NODE", 16, C_PN),
            Text(
                f"IPOPT status {int(hold['hold_status'])}, {int(hold['hold_iterations'])} iterations, "
                f"cost {float(hold['hold_cost']):.1f} (free: {float(hold['free_cost']):.1f})",
                font_size=16,
                color=W,
            ),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        d = float(series["max_dq"])
        d_txt = "0" if d == 0 else f"{d:.0e}"
        cost_series = float(series["shared_cost"])
        assert abs(cost_series - float(series["per_node_cost"])) < 1e-6 * cost_series
        res_ts = VGroup(
            caption("numerical_data_timeseries (external forces):", 15),
            Text(f"both options solve, cost {cost_series:.2f}, max |Δq| = {d_txt} rad", font_size=16, color=W),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        panel3 = panel_of(caption("Bioptim code: 7 nodes of ONE phase"), code_c, res_sh, res_pn, res_ts)
        self.play(FadeIn(VGroup(ax, yt, xt, ylab, xlab, bd, bd_lab, ghost, gl)), FadeIn(panel3[:2]), run_time=0.6)
        self.play(FadeIn(panel3[2]), run_time=0.5)
        self.play(FadeIn(panel3[3]), Create(curve), run_time=1.2)
        self.play(FadeIn(panel3[4]), run_time=0.5)
        self.wait(2.5)

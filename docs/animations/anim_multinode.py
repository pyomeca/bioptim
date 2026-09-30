"""
Manim CE scene: multinode constraint / objective linking the first and last node of a movement (cyclic movement).
REAL bioptim / IPOPT solves stored in ``data/multinode_pendulum.npz`` (see ``generate_multinode_data.py``): cart-pendulum,
N = 30, T = 2 s, three solves (no link, MultinodeObjectiveFcn.STATES_EQUALITY, MultinodeConstraintFcn.STATES_EQUALITY).

Scene: MultinodeLink (about 18 s).  Render (from docs/animations):  manim render -qh anim_multinode.py MultinodeLink
"""

import numpy as np
from manim import *

from features_scenes import CODE_W, DATA_DIR, M, code, fit, make_axes, poly, scene_title, time_label, x_ticks, y_ticks

CODE_X0 = 0.15
C_FREE = GRAY_B
C_OBJ = ORANGE
C_CON = BLUE_C
C_LINK = YELLOW_C
C_GAP = RED_C
W = WHITE


def code_block(lines, size=16):
    block = VGroup(*[code(text, size, color) for _, text, color in lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.09)
    for line, (level, _, _) in zip(block, lines):
        line.shift(RIGHT * 0.3 * level)
    return block


def caption(text, size=18, color=GRAY_B):
    return Text(text, font_size=size, color=color)


def sci(v):
    return f"{v:.1e}" if v < 1e-3 else f"{v:.2f}" if v >= 1 else f"{v:.3f}"


class MultinodeLink(Scene):
    def construct(self):
        d = np.load(DATA_DIR / "multinode_pendulum.npz")
        n, T, w_obj = int(d["N"]), float(d["T"]), float(d["w_obj"])
        tk = np.linspace(0, T, n + 1)
        modes = [("free", C_FREE), ("obj", C_OBJ), ("cons", C_CON)]
        x = {m: d[f"{m}_x"] for m, _ in modes}

        title = scene_title(
            "Multinode penalties", "link two distant nodes: here the last node must return to the first"
        )
        self.play(FadeIn(title), run_time=0.4)

        # ---------------------------------------------------------------- axes and time grid (left)
        rows = [(0, "cart position (m)", 0.35), (1, "pendulum angle (rad)", -1.95)]
        axes, labs = [], []
        for idx, name, yc in rows:
            allv = np.concatenate([x[m][idx] for m, _ in modes])
            lo, hi = np.floor(allv.min()), np.ceil(allv.max())
            ax = make_axes([-3.55, yc, 0], 5.6, 1.7, [0, T], [lo, hi], 1, 1)
            axes.append(ax)
            lab = Text(name, font_size=17, color=GRAY_B).next_to(ax.get_y_axis(), UP, buff=0.06)
            lab.align_to(ax.get_y_axis(), LEFT)
            labs.append(lab)
        ticks = VGroup(*[y_ticks(ax, [ax.y_range[0], ax.y_range[1]], "{:g}") for ax in axes])
        xt = x_ticks(axes[1], [0, 1, 2], "{:g}")
        xlab = time_label(axes[1])
        y_grid = 2.05
        gx = [axes[0].c2p(t, 0)[0] for t in tk]
        dots = VGroup(*[Dot([px, y_grid, 0], radius=0.045, color=GRAY_C) for px in gx])
        p0, p1 = dots[0].get_center(), dots[-1].get_center()
        arc = ArcBetweenPoints(p0 + UP * 0.05, p1 + UP * 0.05, angle=-0.55, color=C_LINK, stroke_width=4)
        hl = VGroup(*[Dot(dots[k].get_center(), radius=0.1, color=C_LINK) for k in (0, n)])
        lab0 = code("Node.START", 15, C_LINK).next_to(dots[0], DOWN, buff=0.14).align_to(dots[0], LEFT)
        lab1 = code("Node.END", 15, C_LINK).next_to(dots[-1], DOWN, buff=0.14).align_to(dots[-1], RIGHT)
        lab_arc = M(f"x(END) = x(START)   ({n + 1} nodes, phase 0)", 18, C_LINK).move_to([-3.55, y_grid + 0.65, 0])
        guides = VGroup(
            *[
                DashedLine([px, y_grid - 0.05, 0], axes[1].c2p(t, axes[1].y_range[0]), color=GRAY_D, stroke_width=1.5)
                for px, t in ((gx[0], 0), (gx[-1], T))
            ]
        )

        # ---------------------------------------------------------------- code (right)
        cap_c = caption("Bioptim code: hard link (MultinodeConstraint)")
        code_c = code_block(
            [
                (0, "multinode_constraints = MultinodeConstraintList()", W),
                (0, "multinode_constraints.add(", W),
                (1, "MultinodeConstraintFcn.STATES_EQUALITY,", W),
                (1, "nodes_phase=(0, 0), nodes=(Node.START, Node.END),", W),
                (1, 'key="all")', W),
            ]
        )
        cap_o = caption(f"soft link: MultinodeObjective, weight {w_obj:g}")
        code_o = code_block(
            [
                (0, "multinode_objectives = MultinodeObjectiveList()", W),
                (0, "multinode_objectives.add(", W),
                (1, "MultinodeObjectiveFcn.STATES_EQUALITY,", W),
                (1, "nodes_phase=(0, 0), nodes=(Node.START, Node.END),", W),
                (1, f'weight={w_obj:g}, key="all")', W),
            ]
        )
        code_p = code_block(
            [(0, "OptimalControlProgram(..., multinode_constraints=..., multinode_objectives=...)", GRAY_A)]
        )
        panel = VGroup(cap_c, code_c, cap_o, code_o, code_p).arrange(DOWN, aligned_edge=LEFT, buff=0.13)
        panel[2].shift(DOWN * 0.1)
        panel[4].shift(DOWN * 0.1)
        fit(panel, CODE_W)
        panel.move_to([CODE_X0, 2.6, 0], aligned_edge=UL)

        # ---------------------------------------------------------------- beat 1: the link on the time grid
        self.play(FadeIn(dots), run_time=0.5)
        self.play(Create(arc), FadeIn(hl), FadeIn(lab0), FadeIn(lab1), FadeIn(lab_arc), run_time=1.0)
        self.play(FadeIn(cap_c), FadeIn(code_c), run_time=0.6)
        self.play(FadeIn(cap_o), FadeIn(code_o), FadeIn(code_p), run_time=0.6)
        self.wait(0.6)

        # ---------------------------------------------------------------- beat 2: trajectories, gap readout
        self.play(FadeIn(VGroup(*axes, *labs, ticks, xt, xlab, guides)), run_time=0.6)
        head = VGroup(caption("solution", 16), caption("‖x(END) − x(START)‖", 16), caption("cost", 16))
        col_x = [CODE_X0, CODE_X0 + 2.15, CODE_X0 + 5.2]
        y0 = -1.2
        for h, cx in zip(head, col_x):
            h.move_to([cx, y0, 0], aligned_edge=LEFT)
        self.play(FadeIn(head), run_time=0.3)
        names = {"free": "no link", "obj": "objective (soft)", "cons": "constraint (hard)"}
        prev = []
        for row, (m, color) in enumerate(modes):
            xm = x[m]
            objs = []
            for i, (idx, _, _) in enumerate(rows):
                ax = axes[i]
                curve = poly(ax, tk, xm[idx], color, width=5)
                s = Dot(ax.c2p(0, xm[idx][0]), radius=0.07, color=color)
                e = Dot(ax.c2p(T, xm[idx][-1]), radius=0.07, color=color)
                gap = Line(ax.c2p(T, xm[idx][0]), ax.c2p(T, xm[idx][-1]), color=C_GAP, stroke_width=5)
                base = DashedLine(ax.c2p(0, xm[idx][0]), ax.c2p(T, xm[idx][0]), color=GRAY_D, stroke_width=1.5)
                objs.append(VGroup(base, gap, curve, s, e))
            yy = y0 - 0.45 - row * 0.5
            gn, cost = float(d[f"{m}_gap_norm"]), float(d[f"{m}_cost"])
            line = VGroup(
                Text(names[m], font_size=18, color=color).move_to([col_x[0], yy, 0], aligned_edge=LEFT),
                Text(sci(gn), font_size=20, color=W).move_to([col_x[1] + 0.3, yy, 0], aligned_edge=LEFT),
                Text(f"{cost:.2f}", font_size=20, color=W).move_to([col_x[2], yy, 0], aligned_edge=LEFT),
            )
            anims = [Create(o[2]) for o in objs] + [FadeIn(o[3]) for o in objs] + [FadeIn(o[4]) for o in objs]
            if prev:
                anims += [g.animate.set_stroke(opacity=0.3) for g in prev]
            self.play(*anims, run_time=1.6)
            self.play(*[FadeIn(o[0]) for o in objs], *[Create(o[1]) for o in objs], FadeIn(line), run_time=0.6)
            prev = [o[2] for o in objs]
            self.wait(0.9 if m != "cons" else 0.6)
        note = Paragraph(
            "Red bar: gap between the two linked nodes (dashed: start value).",
            "The constraint closes the cycle exactly; the objective only pushes towards it.",
            "Both cost more control effort.",
            font_size=16,
            color=GRAY_B,
            line_spacing=0.9,
        )
        fit(note, 5.9)
        note.move_to([CODE_X0, y0 - 1.7, 0], aligned_edge=UL)
        self.play(FadeIn(note), run_time=0.5)
        self.wait(2.2)

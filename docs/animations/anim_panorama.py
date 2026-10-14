"""
Manim CE scene: panorama of the Bioptim penalty library. The names and the counts come from the REAL enums
``ObjectiveFcn.Lagrange``, ``ObjectiveFcn.Mayer`` and ``ConstraintFcn`` of this version (``data/panorama_library.npz``,
see ``generate_panorama_data.py``; no solver involved), grouped by what the penalty acts on. Second beat: how to pick
between Lagrange / Mayer and between objective / constraint.

Scene: PenaltyPanorama (about 16 s).  Render (from docs/animations):  manim render -qh anim_panorama.py PenaltyPanorama
"""

import numpy as np
from manim import *

from features_scenes import CODE_W, DATA_DIR, code, fit, scene_title

C_LAG = GREEN_C
C_MAY = ORANGE
C_CON = RED_C
W = WHITE
FAMS = [("Lagrange", "L", C_LAG), ("Mayer", "M", C_MAY), ("Constraint", "C", C_CON)]
LEGEND = {"Lagrange": "ObjectiveFcn.Lagrange", "Mayer": "ObjectiveFcn.Mayer", "Constraint": "ConstraintFcn"}

# (group, example names with their family, color): the names are checked against the enums in the scene
CARDS = [
    ("controls", [("MINIMIZE_CONTROL", "Lagrange"), ("TRACK_CONTROL", "Constraint")], BLUE_C),
    ("states", [("MINIMIZE_STATE", "Lagrange"), ("TRACK_STATE", "Constraint")], TEAL_C),
    ("time", [("MINIMIZE_TIME", "Mayer"), ("TIME_CONSTRAINT", "Constraint")], YELLOW_C),
    ("markers", [("SUPERIMPOSE_MARKERS", "Mayer"), ("TRACK_MARKERS", "Constraint")], PURPLE_B),
    ("segments", [("MINIMIZE_SEGMENT_VELOCITY", "Mayer"), ("TRACK_SEGMENT_ROTATION", "Constraint")], GREEN_C),
    ("center of mass", [("MINIMIZE_COM_POSITION", "Lagrange"), ("TRACK_COM_VELOCITY", "Constraint")], MAROON_B),
    ("contacts / forces", [("TRACK_SUM_REACTION_FORCES", "Mayer"), ("NON_SLIPPING", "Constraint")], ORANGE),
    ("power / energy", [("MINIMIZE_POWER", "Lagrange"), ("MINIMIZE_FATIGUE", "Mayer")], GOLD_C),
    ("continuity", [("STATE_CONTINUITY", "Mayer"), ("STATE_CONTINUITY", "Constraint")], GRAY_B),
    ("stochastic", [("SYMMETRIC_MATRIX", "Constraint"), ("SEMIDEFINITE_POSITIVE_MATRIX", "Constraint")], PINK),
    ("other", [("CUSTOM", "Lagrange"), ("TRACK_PARAMETER", "Constraint")], GRAY_C),
]


def code_block(lines, size=17):
    block = VGroup(*[code(text, size, color) for _, text, color in lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
    for line, (level, _, _) in zip(block, lines):
        line.shift(RIGHT * 0.3 * level)
    return block


def caption(text, size=19, color=GRAY_B):
    return Text(text, font_size=size, color=color)


def counts_row(values, size=20):
    """Colored 'L 8   M 9   C 6' row; a zero is dimmed."""
    return VGroup(
        *[
            Text(f"{short} {v}", font_size=size, weight=BOLD, color=color if v else GRAY_D)
            for (_, short, color), v in zip(FAMS, values)
        ]
    ).arrange(RIGHT, buff=0.3)


class PenaltyPanorama(Scene):
    def construct(self):
        d = np.load(DATA_DIR / "panorama_library.npz")
        names = {f: [str(n) for n in d[f"{f}_names"]] for f, _, _ in FAMS}
        groups = {f: [str(g) for g in d[f"{f}_groups"]] for f, _, _ in FAMS}
        canon = {f: [str(n) for n in d[f"{f}_canonical"]] for f, _, _ in FAMS}

        def count(fam, group):
            return sum(1 for g in groups[fam] if g == group)

        for _, examples, _ in CARDS:  # every displayed name must exist in its enum
            for name, fam in examples:
                assert name in names[fam], (name, fam)
        for fam, _, _ in FAMS:  # the cards cover every name exactly once
            assert sum(count(fam, c[0]) for c in CARDS) == len(names[fam])

        # ------------------------------------------------------------------ title with colored legend
        title = scene_title("The penalty library")
        legend = VGroup(
            *[
                VGroup(
                    Text(short, font_size=22, weight=BOLD, color=color),
                    Text(LEGEND[fam], font_size=20, color=GRAY_B),
                ).arrange(RIGHT, buff=0.12)
                for fam, short, color in FAMS
            ]
        ).arrange(RIGHT, buff=0.5)
        legend.next_to(title, DOWN, buff=0.14)
        head = VGroup(title, legend)
        self.play(FadeIn(head), run_time=0.4)

        # ------------------------------------------------------------------ beat 1: grouped cards
        card_w, card_h, gap = 3.3, 1.62, 0.15
        x0 = -(4 * card_w + 3 * gap) / 2 + card_w / 2
        y0 = 1.75

        def make_card(body_items, color, i):
            box = RoundedRectangle(corner_radius=0.12, width=card_w, height=card_h, stroke_color=color, stroke_width=3)
            box.set_fill(color, 0.08)
            body = VGroup(*body_items).arrange(DOWN, aligned_edge=LEFT, buff=0.07)
            fit(body, card_w - 0.35)
            body.move_to(box).align_to(box, LEFT).shift(RIGHT * 0.18)
            return VGroup(box, body).move_to([x0 + (i % 4) * (card_w + gap), y0 - (i // 4) * (card_h + gap), 0])

        cards = []
        for i, (group, examples, color) in enumerate(CARDS):
            items = [
                Text(group, font_size=21, weight=BOLD, color=color),
                counts_row([count(f, group) for f, _, _ in FAMS]),
                *[code(n, 15, GRAY_A) for n, _ in examples],
            ]
            cards.append(make_card(items, color, i))
        # last slot: totals computed from the enums
        items = [
            Text("total", font_size=21, weight=BOLD),
            counts_row([len(names[f]) for f, _, _ in FAMS]),
            Text("names in the enums", font_size=16, color=GRAY_B),
            counts_row([len(set(canon[f])) for f, _, _ in FAMS]),
            Text("distinct penalty functions", font_size=16, color=GRAY_B),
        ]
        cards.append(make_card(items, W, 11))
        foot = Text(
            "several names can point to one function (objectives: MINIMIZE_CONTROL and TRACK_CONTROL)",
            font_size=18,
            color=GRAY_B,
        ).to_edge(DOWN, buff=0.75)
        self.play(LaggedStart(*[FadeIn(c, scale=0.85) for c in cards], lag_ratio=0.3, run_time=4.2))
        self.play(FadeIn(foot), run_time=0.4)
        self.wait(2.2)

        # ------------------------------------------------------------------ beat 2: how to pick
        sub = scene_title("How to pick")
        self.play(FadeOut(VGroup(*cards, foot, legend, title)), FadeIn(sub), run_time=0.6)

        ax_x0, ax_x1 = -6.4, -0.9
        xs = np.linspace(ax_x0, ax_x1, 11)

        def lane(y, label, label_color):
            line = Line([ax_x0, y, 0], [ax_x1, y, 0], color=GRAY_D, stroke_width=3)
            dots = VGroup(*[Dot([x, y, 0], radius=0.07, color=GRAY_B) for x in xs])
            lab = Text(label, font_size=20, weight=BOLD, color=label_color).move_to(
                [ax_x0, y + 0.45, 0], aligned_edge=LEFT
            )
            return VGroup(line, dots, lab)

        yA, yB, yC = 1.8, 0.25, -1.3
        laneA = lane(yA, "Lagrange: integral over the interval", C_LAG)
        laneB = lane(yB, "Mayer: one node", C_MAY)
        laneC = lane(yC, "Constraint: node(s) where g must hold", C_CON)
        band = Rectangle(width=ax_x1 - ax_x0, height=0.3, stroke_width=0, fill_color=C_LAG, fill_opacity=0.45)
        band.move_to([(ax_x0 + ax_x1) / 2, yA, 0])
        fA = Text("∫ g(x, u) dt   on Node.ALL_SHOOTING (default)", font_size=19, color=C_LAG)
        fA.move_to([ax_x0, yA - 0.42, 0], aligned_edge=LEFT)
        markB = Dot([xs[-1], yB, 0], radius=0.16, color=C_MAY)
        fB = Text("g(x) at Node.END (default)", font_size=19, color=C_MAY).move_to(
            [ax_x0, yB - 0.42, 0], aligned_edge=LEFT
        )
        markC = Dot([xs[-1], yC, 0], radius=0.16, color=C_CON)
        fC = Text("min_bound ≤ g(x) ≤ max_bound (both 0: g = 0)", font_size=19, color=C_CON)
        fC.move_to([ax_x0, yC - 0.42, 0], aligned_edge=LEFT)
        warn = caption("A Lagrange objective on another node raises RuntimeError (objective_functions.py)", 16)
        warn.move_to([ax_x0, -2.75, 0], aligned_edge=LEFT)

        cap1 = caption("objectives are minimized")
        code1 = code_block(
            [
                (0, "objectives.add(", W),
                (1, "ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,", C_LAG),
                (1, 'key="tau", weight=1e-2)', W),
                (0, "objectives.add(", W),
                (1, "ObjectiveFcn.Mayer.MINIMIZE_STATE,", C_MAY),
                (1, 'key="qdot", node=Node.END, weight=100)', W),
            ]
        )
        cap2 = caption("constraints must be satisfied")
        code2 = code_block(
            [
                (0, "constraints.add(", W),
                (1, "ConstraintFcn.SUPERIMPOSE_MARKERS,", C_CON),
                (1, "node=Node.END,", W),
                (1, 'first_marker="hand", second_marker="target")', W),
            ]
        )
        panel = VGroup(cap1, code1, cap2, code2).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        VGroup(cap2, code2).shift(DOWN * 0.2)
        fit(panel, CODE_W)
        panel.move_to([0.15, 2.25, 0], aligned_edge=UL)

        self.play(FadeIn(laneA), FadeIn(cap1), FadeIn(code1), run_time=0.6)
        self.play(FadeIn(band), FadeIn(fA), run_time=0.6)
        self.play(FadeIn(laneB), run_time=0.4)
        self.play(FadeIn(markB, scale=2), FadeIn(fB), run_time=0.5)
        self.play(FadeIn(laneC), FadeIn(cap2), FadeIn(code2), run_time=0.4)
        self.play(FadeIn(markC, scale=2), FadeIn(fC), run_time=0.5)
        self.play(FadeIn(warn), run_time=0.4)
        self.wait(2.5)

"""
Manim CE scenes: reading the decision vector. REAL Bioptim objects stored in ``data/vector_layout.npz`` (see
``generate_vector_data.py``): pendulum swing, N = 3 shooting intervals, one parameter, layout read from
``ocp.vector_layout.index_map`` and values from ``sol.vector``.

Scenes (about 20 s in total; render from docs/animations):
    VectorOrdering     RK4: the strip dt | X | U | parameters, VARIABLE_MAJOR then TIME_MAJOR (the cells move)
    VectorCollocation  COLLOCATION(degree 3): every X node holds 4 columns; reading them with decision_states()
"""

import numpy as np
from manim import *

from features_scenes import CODE_W, DATA_DIR, code, fit, scene_title

W = WHITE
C_DT, C_X, C_U, C_P = TEAL_C, BLUE_C, GREEN_C, PURPLE_B
VAR_NAMES = {"states": ["q0", "q1", "dq0", "dq1"], "controls": ["tau0", "tau1"]}
COLORS = {"states": C_X, "controls": C_U, "time": C_DT, "parameters": C_P}


def caption(text, size=19, color=GRAY_B):
    return Text(text, font_size=size, color=color)


def code_block(lines, size=20):
    block = VGroup(*[code(text, size, color) for _, text, color in lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.11)
    for line, (level, _, _) in zip(block, lines):
        line.shift(RIGHT * 0.3 * level)
    return block


def layout(d, tag):
    """List of dicts (kind, node, start, stop, ncols) in vector order, from the real index_map (empty blocks dropped)."""
    out = []
    for k, a, b, c in zip(d[f"{tag}_keys"], d[f"{tag}_start"], d[f"{tag}_stop"], d[f"{tag}_ncols"]):
        kind, _, node = str(k).partition(":")
        if kind == "algebraic_states":
            continue  # size 0 with RK4 and this model
        out.append(dict(kind=kind, node=int(node) if node else None, start=int(a), stop=int(b), ncols=int(c)))
    return out


def block_name(b):
    return {"time": "dt", "parameters": "p"}.get(b["kind"]) or ("X" if b["kind"] == "states" else "U") + str(b["node"])


def cell_names(b):
    if b["kind"] == "time":
        return ["dt"]
    if b["kind"] == "parameters":
        return ["p"]
    names = VAR_NAMES[b["kind"]]
    return [names[j % len(names)] for j in range(b["stop"] - b["start"])]


def make_cell(name, value, color, w=0.5, h=0.5):
    rect = Rectangle(width=w - 0.04, height=h, stroke_width=1.5, stroke_color=color)
    rect.set_fill(color, 0.32)
    label = fit(Text(name, font_size=12, color=W).move_to(rect), w - 0.1)
    val = fit(
        Text(f"{value:.1f}" if abs(value) >= 10 else f"{value:.2f}", font_size=11, color=GRAY_B), w - 0.06
    ).next_to(rect, DOWN, buff=0.08)
    return VGroup(rect, label, val)


def bracket(x0, x1, y, label, color, size=14):
    line = Line([x0, y, 0], [x1, y, 0], color=color, stroke_width=3)
    tick0 = Line([x0, y, 0], [x0, y + 0.08, 0], color=color, stroke_width=3)
    tick1 = Line([x1, y, 0], [x1, y + 0.08, 0], color=color, stroke_width=3)
    text = Text(label, font_size=size, color=color).next_to(line, DOWN, buff=0.06)
    return VGroup(line, tick0, tick1, text)


def code_panel_at(cap_text, lines, width=12.2):
    panel = VGroup(caption(cap_text, 19), code_block(lines)).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
    fit(panel, width)
    return panel.move_to([-6.6, -0.5, 0], aligned_edge=UL)


class VectorOrdering(Scene):
    def construct(self):
        d = np.load(DATA_DIR / "vector_layout.npz")
        var, tim = layout(d, "rk4_var"), layout(d, "rk4_time")
        x = d["rk4_var_x"]
        n_tot = int(d["rk4_var_total"])
        assert np.allclose(np.sort(x), np.sort(d["rk4_time_x"]))

        self.play(
            FadeIn(scene_title("Reading the decision vector", "pendulum, RK4, N = 3 intervals, real values")),
            run_time=0.4,
        )

        slot = lambda i: np.array([-5.75 + 0.5 * i, 1.4, 0])
        idx = VGroup(*[Text(str(i), font_size=11, color=GRAY_C).move_to(slot(i) + UP * 0.45) for i in range(n_tot)])

        cells = {}  # (block name, j) -> cell, value taken at its VARIABLE_MAJOR position
        for b in var:
            for j, name in enumerate(cell_names(b)):
                cell = make_cell(name, x[b["start"] + j], COLORS[b["kind"]])
                cells[(block_name(b), j)] = cell.move_to(slot(b["start"] + j) + DOWN * 0.15)
        strip = VGroup(*cells.values())

        def brackets(blocks):
            g = VGroup()
            for b in blocks:
                a, e = slot(b["start"])[0] - 0.23, slot(b["stop"] - 1)[0] + 0.23
                g.add(bracket(a, e, 0.55, block_name(b), COLORS[b["kind"]]))
            return g

        b1 = next(b for b in var if block_name(b) == "X1")
        b1t = next(b for b in tim if block_name(b) == "X1")
        note_alg = caption("algebraic_states: empty block here (0 variables), between U and p", 15)
        note_alg.move_to([0, -0.05, 0])
        panel = code_panel_at(
            "layout of the flat vector: ocp.vector_layout.index_map",
            [
                (0, "OptimalControlProgram(..., ordering_strategy=", W),
                (1, "OrderingStrategy.VARIABLE_MAJOR)   # default", W),
                (0, 'ocp.vector_layout.index_map[(0, "states", 1)]', W),
                (1, f"# -> (slice({b1['start']}, {b1['stop']}), 1)   node 1 of X", C_X),
            ],
        )
        x1_cells = lambda: VGroup(*[c for (n, _), c in cells.items() if n == "X1"])

        self.play(FadeIn(idx), run_time=0.3)
        self.play(LaggedStart(*[FadeIn(c, shift=DOWN * 0.15) for c in strip], lag_ratio=0.03), run_time=1.5)
        br_var = brackets(var)
        self.play(FadeIn(br_var), FadeIn(note_alg), FadeIn(panel), run_time=0.5)
        hl = SurroundingRectangle(x1_cells(), color=YELLOW, buff=0.06, stroke_width=3)
        self.play(Create(hl), run_time=0.4)
        self.wait(1.3)

        # ------------------------------------------------ TIME_MAJOR: the same cells move
        self.play(FadeOut(hl), FadeOut(br_var), FadeOut(panel), FadeOut(note_alg), run_time=0.3)
        tim_pos = {(block_name(b), j): b["start"] + j for b in tim for j in range(b["stop"] - b["start"])}
        panel2 = code_panel_at(
            "same 24 numbers, other order",
            [
                (0, "OptimalControlProgram(..., ordering_strategy=", W),
                (1, "OrderingStrategy.TIME_MAJOR)", W),
                (0, 'ocp.vector_layout.index_map[(0, "states", 1)]', W),
                (1, f"# -> (slice({b1t['start']}, {b1t['stop']}), 1)   was slice({b1['start']}, {b1['stop']})", C_X),
            ],
        )
        moves = [c.animate.move_to(slot(tim_pos[key]) + DOWN * 0.15) for key, c in cells.items()]
        self.play(FadeIn(panel2), *moves, run_time=2.0)
        hl2 = SurroundingRectangle(x1_cells(), color=YELLOW, buff=0.06, stroke_width=3)
        self.play(FadeIn(brackets(tim)), Create(hl2), run_time=0.5)
        self.wait(1.0)

        # ------------------------------------------------ reading: never by hand
        q_var, q_time = float(d["rk4_var_q_nodes"][1, 1]), float(d["rk4_time_q_nodes"][1, 1])
        panel3 = code_panel_at(
            "read it with the helpers, whatever the ordering",
            [
                (0, "states = sol.decision_states()", W),
                (
                    0,
                    f'states["q"][1][1, 0]   # rotation at node 1: {q_var:.2f} (VARIABLE_MAJOR), {q_time:.2f} (TIME_MAJOR)',
                    C_X,
                ),
                (0, 'sol.decision_controls()["tau"][0]      # control at node 0', C_U),
                (0, 'sol.parameters["max_tau"]              # the parameter', C_P),
            ],
        )
        self.play(FadeOut(panel2), FadeOut(hl2), run_time=0.3)
        self.play(FadeIn(panel3), run_time=0.5)
        self.wait(2.0)


class VectorCollocation(Scene):
    def construct(self):
        d = np.load(DATA_DIR / "vector_layout.npz")
        blocks = layout(d, "colloc_var")
        n_tot = int(d["colloc_var_total"])
        n_cols = int(d["colloc_var_n_states_decision_steps_1"])
        self.play(
            FadeIn(
                scene_title("Collocation: extra columns per node", "pendulum, COLLOCATION(polynomial_degree=3), N = 3")
            ),
            run_time=0.4,
        )

        cw = 0.18
        xs, cursor = {}, -6.35
        for b in blocks:
            for j in range(b["stop"] - b["start"]):
                if b["kind"] == "states" and b["ncols"] > 1 and j % 4 == 0 and j > 0:
                    cursor += 0.05
                xs[b["start"] + j] = cursor + cw / 2
                cursor += cw
            cursor += 0.1
        groups, brs = {}, VGroup()
        for b in blocks:
            g = VGroup()
            for j in range(b["stop"] - b["start"]):
                col = j // 4 if (b["kind"] == "states" and b["ncols"] > 1) else 0
                r = Rectangle(width=cw - 0.02, height=0.5, stroke_width=1, stroke_color=COLORS[b["kind"]])
                g.add(r.set_fill(COLORS[b["kind"]], 0.85 if col == 0 else 0.4).move_to([xs[b["start"] + j], 1.4, 0]))
            groups[block_name(b)] = g
            a, e = xs[b["start"]] - cw / 2, xs[b["stop"] - 1] + cw / 2
            brs.add(bracket(a, e, 1.05, block_name(b), COLORS[b["kind"]], 13))
            brs.add(Text(str(b["start"]), font_size=11, color=GRAY_C).move_to([a + 0.1, 1.85, 0]))
        tot = caption(f"{n_tot} variables, index of the first cell above each block", 15).move_to(
            [-6.35, 2.35, 0], aligned_edge=LEFT
        )
        self.play(LaggedStart(*[FadeIn(g) for g in groups.values()], lag_ratio=0.1), run_time=1.4)
        self.play(FadeIn(brs), FadeIn(tot), run_time=0.4)

        # ---- zoom on X1: 16 cells = 4 variables x 4 columns
        x1 = groups["X1"]
        hl = SurroundingRectangle(x1, color=YELLOW, buff=0.06, stroke_width=3)
        self.play(Create(hl), run_time=0.4)
        blk = d["colloc_var_x1_block"]  # (4 states, 4 columns)
        gx0, gy0, s = -6.3, -0.2, 0.62
        heads = ["node", "pt 1", "pt 2", "pt 3"]
        grid, grid_cells = VGroup(), {}
        for c in range(4):
            grid.add(Text(heads[c], font_size=13, color=GRAY_B).move_to([gx0 + 0.55 + c * s, gy0 + 0.45, 0]))
        for r in range(4):
            grid.add(Text(VAR_NAMES["states"][r], font_size=13, color=GRAY_B).move_to([gx0, gy0 - r * 0.42, 0]))
            for c in range(4):
                cell = Rectangle(width=s - 0.04, height=0.38, stroke_width=1.2, stroke_color=C_X)
                cell.set_fill(C_X, 0.8 if c == 0 else 0.35).move_to([gx0 + 0.55 + c * s, gy0 - r * 0.42, 0])
                grid_cells[(r, c)] = cell
                grid.add(
                    Text(
                        f"{blk[r, c]:.1f}" if abs(blk[r, c]) >= 10 else f"{blk[r, c]:.2f}", font_size=11, color=W
                    ).move_to(cell),
                )
        grid_rects = VGroup(*grid_cells.values())
        note = caption("16 consecutive numbers, filled column after column", 15)
        note.move_to([-6.35, -2.3, 0], aligned_edge=LEFT)
        self.play(
            FadeIn(grid),
            FadeIn(note),
            *[TransformFromCopy(x1[c * 4 + r], grid_cells[(r, c)]) for c in range(4) for r in range(4)],
            run_time=1.0,
        )

        q1_shape = tuple(d["colloc_var_q1"].shape)
        panel = VGroup(
            caption("why 4 columns, and how to read them", 19),
            code_block(
                [
                    (0, "OdeSolver.COLLOCATION(polynomial_degree=3)", W),
                    (0, f"nlp.n_states_decision_steps(1)   # {n_cols} = 1 + degree", C_X),
                    (0, 'q1 = sol.decision_states()["q"][1]', W),
                    (1, f"# shape {q1_shape}: rows q0, q1 / columns node, pt 1..3", GRAY_A),
                ]
            ),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        fit(panel, CODE_W)
        panel.move_to([0.15, 0.35, 0], aligned_edge=UL)
        self.play(FadeIn(panel), run_time=0.5)
        self.wait(2.5)

"""
Manim CE scene: architecture of Bioptim, from the user's inputs to the Solution. Every class / file / method shown was
checked against the source of this version (see notes/arch.md). Beat 2 reads REAL sizes of a tiny pendulum OCP
(``data/arch_sizes.npz``, see ``generate_arch_data.py``).

Scene: ArchitecturePath (about 16 s).  Render (from docs/animations):  manim render -qh anim_arch.py ArchitecturePath
"""

import numpy as np
from manim import *

from features_scenes import DATA_DIR, MONO, code, fit, scene_title

ROWS = [
    # (box label, module path (relative to bioptim/), key method, color)
    (
        "User inputs",
        "bio_model, dynamics, objective_functions, constraints,",
        "x_bounds, u_bounds, x_init, ode_solver",
        GRAY_B,
    ),
    (
        "OptimalControlProgram",
        "optimization/optimal_control_program.py",
        "__init__ → _check_arguments_and_build_nlp",
        BLUE_C,
    ),
    ("NonLinearProgram", "optimization/non_linear_program.py", "one per phase · declare_shooting_points", BLUE_C),
    ("ConfigureProblem", "dynamics/configure_problem.py", "initialize(ocp, nlp) → dynamics_func", TEAL_C),
    ("Penalties", "limits/penalty_option.py · penalty.py", "_finalize_penalties → _declare_continuity", ORANGE),
    (
        "OptimizationVectorHelper",
        "optimization/optimization_vector.py",
        "vector · bounds_vectors · init_vector",
        YELLOW_C,
    ),
    (
        "SolverInterface",
        "interfaces/ipopt_interface.py (fatrop, sqp, acados)",
        "solve → interface_utils.generic_solve",
        RED_C,
    ),
    (
        "Solution",
        "optimization/solution/solution.py",
        "from_dict · decision_states · integrate · graphs",
        GREEN_C,
    ),
]
# code lines of the right panel, revealed together with a row: (row, text)
CODE = [
    (0, "OptimalControlProgram("),
    (0, "  bio_model, N, T,"),
    (0, "  dynamics=DynamicsOptions(...),"),
    (0, "  x_bounds=, u_bounds=, x_init=,"),
    (0, "  objective_functions=, constraints=)"),
    (6, "sol = ocp.solve(Solver.IPOPT())"),
    (7, "sol.decision_states()"),
    (7, "sol.graphs()"),
]
ROW_Y0, ROW_DY = 2.15, 0.72
BOX_W, BOX_H = 3.0, 0.52
BOX_X = -5.45
TXT_X = -3.7
TXT_W = 5.9
PANEL_X = 2.55
PANEL_W = 4.4


class ArchitecturePath(Scene):
    def construct(self):
        d = np.load(DATA_DIR / "arch_sizes.npz")
        n_ph, N = int(d["n_phases"]), int(d["n_shooting"])
        nx, nu = int(d["nx"]), int(d["nu"])
        nv, n_g = int(d["n_variables"]), int(d["n_g"])
        n_t, n_x, n_u = int(d["n_time"]), int(d["n_x"]), int(d["n_u"])
        assert nv == n_t + n_x + n_u and n_g == N * nx

        title = scene_title("Architecture of Bioptim", "from your inputs to the Solution")
        self.play(FadeIn(title), run_time=0.4)

        cap = Text("Bioptim code", font_size=19, color=GRAY_B).move_to([PANEL_X, 2.5, 0], aligned_edge=LEFT)
        code_lines = VGroup(*[code(t, 15) for _, t in CODE])
        code_lines.arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        code_lines[5:].shift(DOWN * 0.2)
        fit(code_lines, PANEL_W)
        code_lines.move_to([PANEL_X, 2.05, 0], aligned_edge=UL)
        self.add(cap)

        boxes, texts, arrows = [], [], []
        for i, (name, path, method, color) in enumerate(ROWS):
            y = ROW_Y0 - i * ROW_DY
            box = RoundedRectangle(
                corner_radius=0.1, width=BOX_W, height=BOX_H, stroke_color=color, stroke_width=3, fill_opacity=0
            ).move_to([BOX_X, y, 0])
            box.set_fill(color, 0.14)
            label = Text(name, font_size=19, weight=BOLD, color=WHITE).move_to(box)
            fit(label, BOX_W - 0.2)
            t1 = code(path, 13, GRAY_A)
            t2 = code(method, 13, color)
            info = VGroup(t1, t2).arrange(DOWN, aligned_edge=LEFT, buff=0.05)
            fit(info, TXT_W)
            info.move_to([TXT_X, y, 0], aligned_edge=LEFT)
            boxes.append(VGroup(box, label))
            texts.append(info)
            if i:
                arrows.append(
                    Arrow(
                        [BOX_X, y + ROW_DY - BOX_H / 2, 0],
                        [BOX_X, y + BOX_H / 2, 0],
                        buff=0.03,
                        stroke_width=3,
                        color=GRAY_B,
                        max_tip_length_to_length_ratio=0.5,
                    )
                )

        # ---------------------------------------------------------------- beat 1: the path, box after box
        for i in range(len(ROWS)):
            anims = [FadeIn(boxes[i], shift=DOWN * 0.1), FadeIn(texts[i])]
            if i:
                anims.append(GrowArrow(arrows[i - 1]))
            anims += [FadeIn(code_lines[j]) for j, (r, _) in enumerate(CODE) if r == i]
            self.play(*anims, run_time=0.85)
        self.wait(1.2)

        # ---------------------------------------------------------------- beat 2: same path, real numbers
        tags = [
            f"pendulum, N = {N}, T = 1 s",
            f"n_phases = {n_ph}",
            f"len(ocp.nlp) = {n_ph} · nx = {nx}, nu = {nu}",
            "states q, qdot · controls tau",
            f"1 Lagrange objective, {n_g} continuity",
            f"vector {nv} = {n_t} + {n_x} + {n_u}  (t + X + U)",
            f"IPOPT status {int(d['status'])} · {n_g} constraints",
            f"cost = {float(d['cost']):.2f}",
        ]
        cap2 = Text("the same path, real pendulum OCP", font_size=19, color=GRAY_B).move_to(cap, aligned_edge=LEFT)
        self.play(FadeOut(code_lines), FadeOut(cap), FadeIn(cap2), run_time=0.4)
        for i, tag in enumerate(tags):
            t = Text(tag, font_size=17, color=WHITE)
            fit(t, PANEL_W)
            t.move_to([PANEL_X, ROW_Y0 - i * ROW_DY, 0], aligned_edge=LEFT)
            self.play(
                FadeIn(t, shift=RIGHT * 0.15),
                boxes[i][0].animate.set_stroke(WHITE, 5),
                run_time=0.3,
            )
            self.play(boxes[i][0].animate.set_stroke(ROWS[i][3], 3), run_time=0.1)
        self.wait(3.0)

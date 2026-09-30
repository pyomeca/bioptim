"""
Manim CE scene: which dynamics a Bioptim model can have, and how to choose (DynamicsOverview, about 19 s of content).

Beat 1: a map of the model classes of ``bioptim/models/biorbd/model_dynamics.py`` (torque-based, muscle-based, model
structure) with what is a state and what is a control, and the video of the series that shows each one in detail. The
names of eight rows are read from REAL optimal control problems (``data/dynamics_names.npz`` and the four solves below); the
holonomic, variational and stochastic rows are read from the classes (see the notes).
Beat 2: the SAME task solved with TorqueBiorbdModel, TorqueActivationBiorbdModel, TorqueDerivativeBiorbdModel and
JointAccelerationBiorbdModel (REAL bioptim / IPOPT solves, ``data/dynamics_compare.npz``, generator
``generate_dynamics_data.py``): control curve (its own unit), decision-vector size, own cost, IPOPT status, and the motion
of the first joint with the torque solution as a dashed ghost and the difference below.

Render (from the repo root):
    python docs/animations/render_series.py anim_dynamics.py DynamicsOverview --lang en
"""

import numpy as np
from manim import *

from features_scenes import (
    C_CTRL,
    C_PAR,
    C_STATE,
    CODE_X,
    DATA_DIR,
    axis_label,
    code,
    code_panel,
    dec,
    footer,
    hline,
    ipopt_line,
    make_axes,
    place,
    poly,
    say,
    scene_title,
    steps,
    time_label,
    x_ticks,
    y_ticks,
)

C_GHOST = GRAY_B
TAGS = ("tau", "act", "der", "acc")
CLASSES = {
    "tau": "TorqueBiorbdModel",
    "act": "TorqueActivationBiorbdModel",
    "der": "TorqueDerivativeBiorbdModel",
    "acc": "JointAccelerationBiorbdModel",
}
# column left edges of the map
X_CLS, X_STATE, X_CTRL, X_VIDEO = -6.9, -2.65, 0.45, 2.7


def names(arr):
    """['q:2', 'qdot:2'] (or ['q', 'qdot']) -> 'q, qdot'."""
    return ", ".join(str(a).split(":")[0] for a in arr)


def nice_ceil(x):
    mag = 10 ** np.floor(np.log10(x))
    for m in (1, 2, 2.5, 5, 10):
        if x <= m * mag:
            return float(m * mag)


class DynamicsOverview(Scene):
    def construct(self):
        d = np.load(DATA_DIR / "dynamics_compare.npz")
        nm = np.load(DATA_DIR / "dynamics_names.npz")
        n, T, q_end, tmax = int(d["N"]), float(d["T"]), float(d["q_end"]), float(d["tau_max"])
        t = d["tau_t"]

        # ============================================================ beat 1: the map
        sub1 = "what is a state and what is a control in each dynamics of this version"
        title = scene_title("Choosing the dynamics of a model", sub1)
        self.play(FadeIn(title), run_time=0.4)

        this = "this video"
        rows = [
            (
                "Torque-based",
                [
                    (
                        "TorqueBiorbdModel",
                        names(d["tau_state_names"]),
                        names(d["tau_control_names"]),
                        "Lagrange, Mayer and nodes",
                    ),
                    ("TorqueActivationBiorbdModel", names(d["act_state_names"]), names(d["act_control_names"]), this),
                    ("TorqueDerivativeBiorbdModel", names(d["der_state_names"]), names(d["der_control_names"]), this),
                    ("JointAccelerationBiorbdModel", names(d["acc_state_names"]), names(d["acc_control_names"]), this),
                ],
            ),
            (
                "Muscle-based",
                [
                    (
                        "MusclesBiorbdModel",
                        names(nm["muscles_states"]),
                        names(nm["muscles_controls"]),
                        "Muscle-driven reaching",
                    ),
                    (
                        "MusclesWithExcitationsBiorbdModel",
                        names(nm["excit_states"]),
                        names(nm["excit_controls"]),
                        "Excitation to activation",
                    ),
                ],
            ),
            (
                "Model structure",
                [
                    (
                        "TorqueFreeFloatingBaseBiorbdModel",
                        names(nm["floating_states"]),
                        names(nm["floating_controls"]),
                        "Free-floating base reorientation",
                    ),
                    ("HolonomicTorqueBiorbdModel", "q_u, qdot_u", "tau", "Holonomic constraint: double pendulum"),
                    ("VariationalTorqueBiorbdModel", "q, lambdas", "tau", "Discrete mechanics (variational)"),
                    (
                        "MultiTorqueBiorbdModel",
                        names(nm["multi_states"]),
                        names(nm["multi_controls"]),
                        "Two bodies in one OCP",
                    ),
                    ("StochasticTorqueBiorbdModel", "q, qdot", "tau, k, ref, cov", "Robust path constraint (SOCP)"),
                ],
            ),
        ]

        def header_cell(text, x, color=GRAY_B):
            return Text(text, font_size=17, color=color).move_to([x, 0, 0], aligned_edge=LEFT)

        head = VGroup(
            header_cell("Model class", X_CLS),
            header_cell("states x", X_STATE, C_STATE),
            header_cell("controls u", X_CTRL, C_CTRL),
            header_cell("Shown in detail in", X_VIDEO),
        )
        y = 2.3
        head.shift(UP * y)
        self.play(FadeIn(head), run_time=0.4)

        y -= 0.36
        groups = []
        for group_name, items in rows:
            cells = VGroup(
                Text(group_name, font_size=18, color=WHITE, weight=BOLD).move_to([X_CLS, y, 0], aligned_edge=LEFT)
            )
            y -= 0.34
            for cls, states, controls, video in items:
                split = states.count(",") >= 3  # the free-floating base names are long: two lines
                st_text = states.replace(", qdot_roots", ",\nqdot_roots") if split else states
                row_h = 0.52 if split else 0.33
                yc = y - (row_h - 0.33) / 2
                st = code(st_text, 16, C_STATE).move_to([X_STATE, yc, 0], aligned_edge=LEFT)
                if split:
                    st.move_to([X_STATE, yc, 0], aligned_edge=LEFT)
                cells.add(
                    code(cls, 16, WHITE).move_to([X_CLS, yc, 0], aligned_edge=LEFT),
                    st,
                    code(controls, 16, C_CTRL).move_to([X_CTRL, yc, 0], aligned_edge=LEFT),
                    Text(video, font_size=17, color=GRAY_A if video == this else GRAY_B).move_to(
                        [X_VIDEO, yc, 0], aligned_edge=LEFT
                    ),
                )
                y -= row_h
            y -= 0.06
            groups.append(cells)

        foot1 = footer(
            "Names come from real problems built with each class; the holonomic, variational and stochastic rows are read from the class source."
        )
        for i, cells in enumerate(groups):
            self.play(FadeIn(cells), *([FadeIn(foot1)] if i == 0 else []), run_time=0.9)
        self.wait(1.2)

        # ============================================================ beat 2: same task, four dynamics
        sub2 = f"same task, four dynamics: reach {q_end:g} rad at rest with the elbow only, N = {n}, T = {T:g} s"
        sub2_mob = Text(sub2, font_size=22, color=GRAY_B).move_to(title[1])
        self.play(*[FadeOut(m) for m in self.mobjects if m is not title], Transform(title[1], sub2_mob), run_time=0.7)
        self.add(title)

        # data of the four solves
        q0 = {tag: d[f"{tag}_q"][0] for tag in TAGS}
        u_el = {
            tag: d[f"{tag}_u"][-1] for tag in TAGS
        }  # the elbow control (the last row: tau[1], taudot[1], qddot_joints[0])
        lim = {tag: nice_ceil(1.15 * float(np.abs(u_el[tag]).max())) for tag in TAGS}
        unit_label = {
            "tau": "τ₂  elbow torque (N·m)",
            "act": f"a₂  torque activation (1 = {tmax:g} N·m)",
            "der": "dτ₂/dt  torque rate (N·m/s)",
            "acc": "d²q₂/dt²  elbow acceleration (rad/s²)",
        }
        diff = {tag: q0[tag] - q0["tau"] for tag in TAGS}

        ax_u = make_axes([-3.55, 1.15, 0], 5.6, 1.3, [0, T], [-1, 1], 1, 1)
        ax_q = make_axes([-3.55, -0.7, 0], 5.6, 1.3, [0, T], [-0.8, 1.2], 1, 1)
        ax_d = make_axes([-3.55, -2.4, 0], 5.6, 1.0, [0, T], [-0.8, 0.8], 1, 1)
        lab_q = axis_label("θ₁(t)  first-joint angle (rad)", ax_q, C_STATE)
        lab_d = axis_label("Δθ₁ = θ₁ − θ₁(torque model) (rad)", ax_d, C_PAR)
        lab_u = axis_label(unit_label["tau"], ax_u, C_CTRL)
        static = VGroup(
            lab_q,
            lab_d,
            time_label(ax_d),
            x_ticks(ax_d, [0, 2, 4]),
            y_ticks(ax_q, [-0.5, 0, 0.5, 1]),
            y_ticks(ax_d, [-0.5, 0, 0.5]),
        )
        self.play(Create(ax_u), Create(ax_q), Create(ax_d), FadeIn(static), FadeIn(lab_u), run_time=0.8)

        def u_ticks(tag):
            """Tick labels of the control axis (normalised to [-1, 1]): -limit, 0, +limit in the unit of the control."""
            return VGroup(
                *[
                    Text(dec(f"{v * lim[tag]:g}"), font_size=16, color=GRAY_B).next_to(
                        ax_u.c2p(ax_u.x_range[0], v), LEFT, buff=0.08
                    )
                    for v in (-1, 0, 1)
                ]
            )

        def code_lines(tag):
            return [
                (0, f"bio_model = {CLASSES[tag]}(MODEL)", C_CTRL),
                (0, "objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,", WHITE),
                (1, f'key="{d[f"{tag}_u_key"]}", weight=1)', C_CTRL),
                (0, "ocp = OptimalControlProgram(bio_model, N, T, ...)", WHITE),
            ]

        def readout(tag):
            body = (
                f"states per node: {int(d[f'{tag}_n_states'])}, controls per node: {int(d[f'{tag}_n_controls'])}\n"
                f"decision vector: {int(d[f'{tag}_n_vector'])} variables\n"
                f"minimised ∫ u² dt = {float(d[f'{tag}_cost']):.3g} (in the unit of u²)\n"
                f"elbow torque: peak {float(d[f'{tag}_tau2_peak']):.1f} N·m, ∫ τ₂² dt = {float(d[f'{tag}_tau2_effort']):.1f}\n"
                + ipopt_line(int(d[f"{tag}_iterations"]), bool(d[f"{tag}_converged"]))
            )
            return place(Text(body, font_size=19, color=GRAY_A, line_spacing=0.9), CODE_X, -0.45)

        def curves(tag):
            return (
                steps(ax_u, t, u_el[tag] / lim[tag], C_CTRL, 4),
                poly(ax_q, t, q0[tag], C_STATE, 5),
                poly(ax_d, t, diff[tag], C_PAR, 5),
            )

        zero = hline(ax_d, 0, T, 0, GRAY_D, dashed=False)
        panel = code_panel(code_lines("tau"), size=18)
        cu, cq, cd = curves("tau")
        ticks_u = u_ticks("tau")
        info = readout("tau")
        self.play(FadeIn(panel), FadeIn(ticks_u), run_time=0.5)
        self.play(Create(cu), Create(cq), Create(zero), Create(cd), FadeIn(info), run_time=1.3)
        self.wait(0.6)

        ghost = DashedVMobject(poly(ax_q, t, q0["tau"], C_GHOST, 3), num_dashes=40).set_opacity(0.8)
        remark = None
        for tag in TAGS[1:]:
            new_u, new_q, new_d = curves(tag)
            new_ticks = u_ticks(tag)
            new_lab = axis_label(unit_label[tag], ax_u, C_CTRL)
            new_panel = code_panel(code_lines(tag), size=18)
            anims = [
                Transform(cu, new_u),
                Transform(cq, new_q),
                Transform(cd, new_d),
                Transform(lab_u, new_lab),
                FadeOut(ticks_u),
                FadeIn(new_ticks),
                Transform(info, readout(tag)),
                Transform(panel[1][0], new_panel[1][0]),
                Transform(panel[1][2], new_panel[1][2]),
            ]
            if tag == "act":
                anims.append(FadeIn(ghost))
                remark = place(
                    say("Same motion as the torque model: the activation is the torque divided by the maximal torque."),
                    CODE_X,
                    -2.3,
                )
                anims.append(FadeIn(remark))
            if tag == "der":
                new_remark = place(
                    say("A different control, bounded and penalised in its own unit, gives a different motion."),
                    CODE_X,
                    -2.3,
                )
                anims.append(Transform(remark, new_remark))
            self.play(*anims, run_time=1.3)
            ticks_u = new_ticks
            self.wait(0.6)

        foot2 = footer(
            "How to choose: take the control you must bound, penalise or measure (torque, activation, torque rate, acceleration, muscles)."
        )
        self.play(FadeIn(foot2))
        self.wait(2.5)

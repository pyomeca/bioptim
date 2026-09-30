"""
Manim CE scene: rigid contact and the unilateral contact force, driven by REAL solves (data/contact_leg.npz, see
``generate_contact_data.py``). A leg (light foot + heavy body, models/contact_leg.bioMod) stands on the floor with
``ContactType.RIGID_EXPLICIT`` and extends from 0.4 m to 0.9 m in 0.4 s. Solved twice:
    without a constraint on the contact force (the floor ends up PULLING the foot: F < 0),
    with ConstraintFcn.TRACK_EXPLICIT_RIGID_CONTACT_FORCES, min_bound=0 (unilateral contact).

Render (from docs/animations):  manim render -qh anim_contact.py UnilateralContact
"""

import sys
from pathlib import Path

import numpy as np
from manim import *

sys.path.insert(0, str(Path(__file__).parent))
from features_scenes import (  # noqa: E402  (also sets the default fonts)
    CODE_X,
    MONO,
    code_panel,
    make_axes,
    place,
    scene_title,
    axis_label,
    time_label,
    x_ticks,
    y_ticks,
)

DATA = Path(__file__).parent / "data" / "contact_leg.npz"
C_FREE, C_UNI, C_BODY = RED_C, GREEN_C, YELLOW_C
RUN = 3.4  # seconds of animation per replay of the 0.4 s motion
FLOOR_Y, X_FIG, SCALE = -2.75, -6.1, 3.6  # screen y of the floor, screen x of the leg, screen units per metre
ARROW_SCALE = 0.0009  # screen units per newton


class UnilateralContact(Scene):
    def construct(self):
        d = np.load(DATA)
        n, T = int(d["n_shooting"]), float(d["final_time"])
        t = np.linspace(0, T, n + 1)
        weight = (float(d["body_mass"]) + float(d["foot_mass"])) * 9.81
        F = {"free": d["free_F"], "uni": d["unilateral_F"]}
        Q = {"free": d["free_q"][1], "uni": d["unilateral_q"][1]}
        neg = int((F["free"] < 0).sum())
        t_neg = t[int(np.argmax(F["free"] < 0))]
        n_zero = int((F["uni"] < 1e-3).sum())

        self.play(FadeIn(scene_title("Rigid contact", "the floor can push, it cannot pull")), run_time=0.5)

        # ---- plots -----------------------------------------------------------------------------------------------
        ax_f = make_axes([-2.35, 0.85, 0], 4.3, 3.3, (0, T), (-700, 2500), y_step=1000)
        ax_q = make_axes([-2.35, -2.55, 0], 4.3, 1.3, (0, T), (0.3, 1.0))
        decos = VGroup(
            axis_label("contact force F (N)", ax_f),
            axis_label("leg length (m)", ax_q),
            time_label(ax_q),
            x_ticks(ax_q, [0, 0.2, 0.4], "{:.1f}"),
            y_ticks(ax_f, [0, 1000, 2000]),
            y_ticks(ax_q, [0.4, 0.9], "{:.1f}"),
        )
        neg_band = Rectangle(
            width=ax_f.c2p(T, 0)[0] - ax_f.c2p(0, 0)[0],
            height=ax_f.c2p(0, 0)[1] - ax_f.c2p(0, -700)[1],
            stroke_width=0,
            fill_color=RED_E,
            fill_opacity=0.35,
        ).move_to((ax_f.c2p(0, 0) + ax_f.c2p(T, -700)) / 2)
        neg_txt = Text("F < 0: floor pulls", font_size=15, color=RED_B).move_to(ax_f.c2p(0.14, -420))
        weight_line = DashedLine(ax_f.c2p(0, weight), ax_f.c2p(T, weight), color=GRAY_C, stroke_width=2)
        weight_txt = Text(f"body weight {weight:.0f} N", font_size=15, color=GRAY_B).move_to(
            ax_f.c2p(0.32, weight + 210)
        )

        # ---- the leg, drawn from the current time ----------------------------------------------------------------
        clock = ValueTracker(0.0)
        cur = {"key": "free"}

        def leg_at(tc):
            return float(np.interp(tc, t, Q[cur["key"]]))

        def force_at(tc):
            return float(F[cur["key"]][min(int(tc / (T / n) + 1e-9), n - 1)])

        floor = Line([X_FIG - 0.9, FLOOR_Y, 0], [X_FIG + 1.5, FLOOR_Y, 0], color=GRAY_B, stroke_width=4)
        hatch = VGroup(
            *[
                Line([x, FLOOR_Y, 0], [x - 0.15, FLOOR_Y - 0.15, 0], color=GRAY_D, stroke_width=2)
                for x in np.arange(X_FIG - 0.9, X_FIG + 1.5, 0.2)
            ]
        )

        def figure():
            tc = clock.get_value()
            top = FLOOR_Y + SCALE * leg_at(tc)
            leg = Line([X_FIG, FLOOR_Y + 0.08, 0], [X_FIG, top, 0], color=GRAY_A, stroke_width=6)
            foot = Rectangle(width=0.5, height=0.16, stroke_width=0, fill_color=GRAY_A, fill_opacity=1).move_to(
                [X_FIG, FLOOR_Y + 0.08, 0]
            )
            body = Circle(radius=0.32, stroke_width=0, fill_color=C_BODY, fill_opacity=1).move_to(
                [X_FIG, top + 0.32, 0]
            )
            f = force_at(tc)
            colour = C_FREE if f < -1 else C_UNI
            length = ARROW_SCALE * f
            if abs(length) > 0.05:
                arrow = Arrow(
                    [X_FIG + 0.9, FLOOR_Y, 0],
                    [X_FIG + 0.9, FLOOR_Y + length, 0],
                    buff=0,
                    color=colour,
                    stroke_width=8,
                    max_tip_length_to_length_ratio=0.35,
                )
            else:
                arrow = Dot([X_FIG + 0.9, FLOOR_Y, 0], radius=0.05, color=colour)
            return VGroup(leg, foot, body, arrow)

        fig = always_redraw(figure)

        def f_readout():
            f = force_at(clock.get_value())
            col = C_FREE if f < -1 else C_UNI
            return Text(f"F = {f:.0f} N", font_size=18, color=col, font=MONO).move_to([X_FIG + 0.2, -3.5, 0])

        f_read = always_redraw(f_readout)

        # ---- force curve drawn up to the clock -------------------------------------------------------------------
        def curve(key, color, upto):
            pts, u = [], F[key]
            for k in range(n):
                t0, t1 = t[k], min(t[k + 1], upto)
                if t0 >= upto:
                    break
                pts += [ax_f.c2p(t0, u[k]), ax_f.c2p(t1, u[k])]
            m = VMobject(color=color, stroke_width=4)
            if len(pts) >= 2:
                m.set_points_as_corners(pts)
            return m

        def q_curve(key, color, upto):
            ts = np.linspace(0, upto, max(2, int(60 * upto / T) + 2))
            m = VMobject(color=color, stroke_width=4)
            m.set_points_as_corners([ax_q.c2p(x, np.interp(x, t, Q[key])) for x in ts])
            return m

        live_f = always_redraw(lambda: curve(cur["key"], C_FREE if cur["key"] == "free" else C_UNI, clock.get_value()))
        live_q = always_redraw(lambda: q_curve(cur["key"], C_BODY, clock.get_value()))

        # ---- code ------------------------------------------------------------------------------------------------
        lines = [
            (0, "bio_model = TorqueBiorbdModel(", WHITE),
            (1, '"contact_leg.bioMod",', WHITE),
            (1, "contact_types=[ContactType.RIGID_EXPLICIT],", C_UNI),
            (0, ")", WHITE),
            (0, "constraints = ConstraintList()", WHITE),
            (0, "constraints.add(", WHITE),
            (1, "ConstraintFcn.TRACK_EXPLICIT_RIGID_CONTACT_FORCES,", C_UNI),
            (1, "node=Node.ALL_SHOOTING, contact_index=0,", WHITE),
            (1, "min_bound=0, max_bound=np.inf,", C_UNI),
            (0, ")", WHITE),
        ]
        panel = code_panel(lines, size=19, top=2.55, caption="Bioptim code")
        block = panel[1]
        add_lines = VGroup(*[block[i] for i in range(5, 10)])
        add_lines.set_opacity(0)
        base_lines = VGroup(*[block[i] for i in range(0, 5)])

        self.play(
            Create(ax_f),
            Create(ax_q),
            FadeIn(decos),
            FadeIn(neg_band),
            FadeIn(neg_txt),
            Create(weight_line),
            FadeIn(weight_txt),
            FadeIn(floor),
            FadeIn(hatch),
            FadeIn(panel[0]),
            FadeIn(base_lines),
            run_time=1.0,
        )
        self.add(fig, f_read, live_f, live_q)

        info1 = place(
            Text(
                "Rigid contact, no constraint on F:\nthe optimizer is free to use a negative force.",
                font_size=19,
                color=GRAY_A,
                line_spacing=0.9,
            ),
            CODE_X,
            -1.65,
        )
        self.play(FadeIn(info1), run_time=0.4)
        self.play(clock.animate.set_value(T), run_time=RUN, rate_func=linear)
        res1 = place(
            Text(
                f"{neg} nodes with F < 0 (from t = {t_neg:.2f} s), min {F['free'].min():.0f} N:\n"
                "the foot is being pulled down, as if glued to the floor.",
                font_size=19,
                color=C_FREE,
                line_spacing=0.9,
            ),
            CODE_X,
            -2.6,
        )
        self.play(FadeIn(res1), run_time=0.5)
        self.wait(1.0)

        # ---- switch on the unilateral constraint -----------------------------------------------------------------
        ghost_f = curve("free", C_FREE, T).set_stroke(opacity=0.35, width=3)
        ghost_q = q_curve("free", GRAY_C, T)
        ghost_q.set_stroke(opacity=0.5, width=2)
        self.play(FadeIn(ghost_f), FadeIn(ghost_q), FadeOut(res1), FadeOut(info1), run_time=0.4)
        cur["key"] = "uni"
        clock.set_value(0.0)
        self.play(add_lines.animate.set_opacity(1), run_time=0.6)
        info2 = place(
            Text(
                "Same problem + min_bound=0 on the vertical\ncontact force (unilateral contact).",
                font_size=19,
                color=GRAY_A,
            ),
            CODE_X,
            -1.65,
        )
        self.play(FadeIn(info2), run_time=0.3)
        self.play(clock.animate.set_value(T), run_time=RUN, rate_func=linear)
        res2 = place(
            Text(
                f"F ≥ 0 everywhere. The last {n_zero} nodes ({n_zero * T / n:.2f} s) sit at F = 0:\n"
                f"the body brakes in free fall, the foot barely touches.\n"
                f"Cost {float(d['free_cost']):.1f} → {float(d['unilateral_cost']):.1f} (a bit more effort).",
                font_size=19,
                color=C_UNI,
                line_spacing=0.9,
            ),
            CODE_X,
            -2.55,
        )
        self.play(FadeIn(res2), run_time=0.5)
        self.wait(2.5)

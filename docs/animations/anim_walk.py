"""
Manim CE scene: a hop cycle with contact phases and an IMPACT transition, driven by a REAL solve
(data/walk_hopper.npz, see ``generate_walk_data.py``). Vertical leg (light foot + heavy body, models/walk_hopper.bioMod):
    phase 0  flight (no contact)   -> PhaseTransitionFcn.IMPACT (touch-down)
    phase 1  stance (RIGID_EXPLICIT, F >= 0, F = 0 at take-off) -> CONTINUOUS
    phase 2  flight back to the apex.

Scene: Hopper (about 15 s).  Render (from docs/animations):  manim render -qh anim_walk.py Hopper
"""

import sys
from pathlib import Path

import numpy as np
from manim import *

sys.path.insert(0, str(Path(__file__).parent))
from features_scenes import (  # noqa: E402  (also sets the default fonts)
    CODE_W,
    CODE_X,
    MONO,
    band,
    code,
    fit,
    make_axes,
    place,
    poly,
    scene_title,
    steps,
    axis_label,
    x_ticks,
    y_ticks,
)

DATA = Path(__file__).parent / "data" / "walk_hopper.npz"
C_AIR, C_STANCE, C_IMPACT = BLUE_C, GREEN_C, RED_C
C_FOOT, C_BODY = ORANGE, YELLOW_C
FLOOR_Y, X_FIG, SCALE = -2.7, -6.05, 2.9  # screen y of the floor, screen x of the leg, screen units per metre
ARROW_SCALE = 0.0007  # screen units per newton


class Hopper(Scene):
    def construct(self):
        d = np.load(DATA)
        ns, ts = d["ns"], d["phase_times"]
        t_imp, t_off, T = float(ts[0]), float(ts[0] + ts[1]), float(ts.sum())
        mf, mb = float(d["foot_mass"]), float(d["body_mass"])
        # global time / states (bioptim times are cumulative; the phase boundaries appear twice: last node of a phase
        # and first node of the next one)
        t_all = np.concatenate([d[f"p{p}_t"] for p in range(3)])
        q_all = np.concatenate([d[f"p{p}_q"] for p in range(3)], axis=1)
        v_all = np.concatenate([d[f"p{p}_qdot"] for p in range(3)], axis=1)
        foot_v, body_v = v_all[0], v_all[0] + v_all[1]  # absolute vertical velocities
        idx = [slice(0, ns[0] + 1), slice(ns[0] + 1, ns[0] + ns[1] + 2), slice(ns[0] + ns[1] + 2, None)]
        F = d["p1_F"]
        t1 = d["p1_t"]
        # numbers computed from the data
        vf_pre, vf_post = float(d["p0_qdot"][0, -1]), float(d["p1_qdot"][0, 0])
        vb_pre = vf_pre + float(d["p0_qdot"][1, -1])
        vb_post = vf_post + float(d["p1_qdot"][1, 0])
        impulse = mf * (vf_post - vf_pre) + mb * (vb_post - vb_pre)  # change of total momentum
        e_lost = 0.5 * mf * (vf_pre**2 - vf_post**2) + 0.5 * mb * (vb_pre**2 - vb_post**2)
        f_max = float(F.max())
        weight = (mf + mb) * 9.81

        self.play(FadeIn(scene_title("Hopping: contact phases", "flight, impact, stance, take-off")), run_time=0.4)

        # ---- plots -----------------------------------------------------------------------------------------------
        v_lo, v_hi, f_lo, f_hi = -2.3, 2.9, -650, 3000
        ax_v = make_axes([-2.15, 0.95, 0], 3.8, 2.1, (0, T), (v_lo, v_hi))
        ax_f = make_axes([-2.15, -1.75, 0], 3.8, 1.85, (0, T), (f_lo, f_hi))
        decos = VGroup(
            axis_label("vertical velocity (m/s)", ax_v),
            axis_label("floor force F (N)", ax_f),
            Text("t (s)", font_size=16, color=GRAY_B).next_to(ax_f.c2p(T / 2, f_lo), DOWN, buff=0.4),
            x_ticks(ax_f, [0, 0.2, 0.4, 0.6], "{:.1f}"),
            y_ticks(ax_v, [-2, 0, 2]),
            y_ticks(ax_f, [0, 1000, 2000]),
        )
        bands = VGroup(
            *[
                band(ax, a, b, lo, hi, col, 0.16)
                for ax, lo, hi in ((ax_v, v_lo, v_hi), (ax_f, f_lo, f_hi))
                for a, b, col in ((0, t_imp, C_AIR), (t_imp, t_off, C_STANCE), (t_off, T, C_AIR))
            ]
        )
        ph_lab = VGroup(
            Text("flight", font_size=15, color=C_AIR).move_to(ax_v.c2p(t_imp / 2, 2.5)),
            Text("stance", font_size=15, color=C_STANCE).move_to(ax_v.c2p((t_imp + t_off) / 2, 2.5)),
            Text("flight", font_size=15, color=C_AIR).move_to(ax_v.c2p((t_off + T) / 2, 2.5)),
        )
        neg = band(ax_f, 0, T, f_lo, 0, C_IMPACT, 0.3)
        neg_txt = Text("F < 0 is forbidden", font_size=14, color=RED_B).move_to(ax_f.c2p(T * 0.5, -330))
        w_line = DashedLine(ax_f.c2p(0, weight), ax_f.c2p(T, weight), color=GRAY_C, stroke_width=2)
        w_txt = Text(f"weight {weight:.0f} N", font_size=14, color=GRAY_B)
        w_txt.move_to(ax_f.c2p(T - 0.01, weight + 250), aligned_edge=RIGHT)

        # ---- code (verified against generate_walk_data.py) -------------------------------------------------------
        W = WHITE
        lines = [
            (0, "models = (TorqueBiorbdModel(M),", C_AIR),
            (1, "TorqueBiorbdModel(M, contact_types=", C_STANCE),
            (2, "[ContactType.RIGID_EXPLICIT]),", C_STANCE),
            (1, "TorqueBiorbdModel(M))", C_AIR),
            (0, "trans = PhaseTransitionList()", W),
            (0, "trans.add(PhaseTransitionFcn.IMPACT, phase_pre_idx=0)", C_IMPACT),
            (0, "trans.add(PhaseTransitionFcn.CONTINUOUS, phase_pre_idx=1)", W),
            (0, "cons.add(ConstraintFcn.TRACK_EXPLICIT_RIGID_CONTACT_FORCES,", C_STANCE),
            (1, "node=Node.ALL_SHOOTING, contact_index=0,", C_STANCE),
            (1, "min_bound=0, max_bound=np.inf, phase=1)", C_STANCE),
            (0, "cons.add(ConstraintFcn.TRACK_EXPLICIT_RIGID_CONTACT_FORCES,", ORANGE),
            (1, "node=Node.PENULTIMATE, contact_index=0,", ORANGE),
            (1, "min_bound=0, max_bound=0, phase=1)", ORANGE),
        ]
        # compact panel (same convention as code_panel: caption above the code) so that the messages below never touch it
        block = VGroup(*[code(text, 15, color) for _, text, color in lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.07)
        for line, (level, _, _) in zip(block, lines):
            line.shift(RIGHT * 0.3 * level)
        cap = Text("Bioptim code (M = walk_hopper.bioMod)", font_size=19, color=GRAY_B)
        panel = VGroup(cap, block).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        fit(panel, CODE_W)
        panel.move_to([CODE_X, 2.55, 0], aligned_edge=UL)
        groups = [VGroup(*[block[i] for i in r]) for r in (range(0, 4), range(4, 7), range(7, 10), range(10, 13))]
        for g in groups:
            g.set_opacity(0)

        # ---- figure ---------------------------------------------------------------------------------------------
        clock = ValueTracker(0.0)

        def state(tc):
            return float(np.interp(tc, t_all, q_all[0])), float(np.interp(tc, t_all, q_all[1]))

        def force_at(tc):
            if not (t_imp <= tc < t_off):
                return 0.0
            return float(F[min(int((tc - t_imp) / (ts[1] / ns[1]) + 1e-9), ns[1] - 1)])

        def colour(tc):
            return C_STANCE if t_imp <= tc < t_off else C_AIR

        floor = Line([X_FIG - 0.9, FLOOR_Y, 0], [X_FIG + 1.3, FLOOR_Y, 0], color=GRAY_B, stroke_width=4)
        hatch = VGroup(
            *[
                Line([x, FLOOR_Y, 0], [x - 0.15, FLOOR_Y - 0.15, 0], color=GRAY_D, stroke_width=2)
                for x in np.arange(X_FIG - 0.9, X_FIG + 1.3, 0.2)
            ]
        )

        def figure():
            tc = clock.get_value()
            zf, leg = state(tc)
            yf, yb = FLOOR_Y + SCALE * zf, FLOOR_Y + SCALE * (zf + leg)
            rod = Line([X_FIG, yf, 0], [X_FIG, yb, 0], color=GRAY_A, stroke_width=6)
            foot = Rectangle(width=0.45, height=0.13, stroke_width=0, fill_color=C_FOOT, fill_opacity=1).move_to(
                [X_FIG, yf - 0.02, 0]
            )
            body = Circle(radius=0.28, stroke_width=0, fill_color=C_BODY, fill_opacity=1).move_to([X_FIG, yb + 0.28, 0])
            f = force_at(tc)
            out = VGroup(rod, foot, body)
            if f > 1:
                out.add(
                    Arrow(
                        [X_FIG + 0.85, FLOOR_Y, 0],
                        [X_FIG + 0.85, FLOOR_Y + ARROW_SCALE * f, 0],
                        buff=0,
                        color=C_STANCE,
                        stroke_width=8,
                        max_tip_length_to_length_ratio=0.3,
                    )
                )
            return out

        def readout():
            tc = clock.get_value()
            return Text(f"F = {force_at(tc):4.0f} N", font_size=18, color=colour(tc), font=MONO).move_to(
                [X_FIG + 0.2, -3.5, 0]
            )

        fig = always_redraw(figure)
        f_read = always_redraw(readout)

        # ---- curves drawn up to the clock ------------------------------------------------------------------------
        def v_curves():
            out = VGroup()
            for arr, col in ((foot_v, C_FOOT), (body_v, C_BODY)):
                for sl in idx:
                    tt, yy = t_all[sl], arr[sl]
                    keep = tt <= clock.get_value() + 1e-9
                    if keep.sum() >= 2:
                        out.add(poly(ax_v, tt[keep], yy[keep], col, 5))
            return out

        def f_curve():
            tc = clock.get_value()
            if tc <= t_imp + 1e-6:
                return VGroup()
            k = int(min(np.ceil((tc - t_imp) / (ts[1] / ns[1]) - 1e-9), ns[1]))
            tt = np.append(t1[:k], min(tc, t_off))
            return steps(ax_f, tt, F[:k], C_STANCE, 5)

        cursors = always_redraw(
            lambda: VGroup(
                DashedLine(
                    ax_v.c2p(clock.get_value(), v_lo), ax_v.c2p(clock.get_value(), v_hi), color=GRAY_B, stroke_width=2
                ),
                DashedLine(
                    ax_f.c2p(clock.get_value(), f_lo), ax_f.c2p(clock.get_value(), f_hi), color=GRAY_B, stroke_width=2
                ),
            )
        )
        lab_foot = Text("foot", font_size=15, color=C_FOOT).move_to(ax_v.c2p(0.56, -1.2))
        lab_body = Text("body", font_size=15, color=C_BODY).move_to(ax_v.c2p(0.56, -1.7))

        self.play(
            Create(ax_v),
            Create(ax_f),
            FadeIn(decos),
            FadeIn(bands),
            FadeIn(ph_lab),
            FadeIn(neg),
            FadeIn(neg_txt),
            Create(w_line),
            FadeIn(w_txt),
            FadeIn(floor),
            FadeIn(hatch),
            FadeIn(panel[0]),
            run_time=0.9,
        )
        self.add(fig, f_read, cursors, always_redraw(v_curves), always_redraw(f_curve))
        self.add(lab_foot, lab_body)
        info = place(
            Text(
                "Phase 0, flight: free fall of the foot\nand of the body, no contact in the model.",
                font_size=19,
                color=GRAY_A,
            ),
            CODE_X,
            -2.0,
        )
        self.play(groups[0].animate.set_opacity(1), FadeIn(info), run_time=0.5)

        # ---- phase 0 -> impact --------------------------------------------------------------------------------------
        self.play(clock.animate.set_value(t_imp), run_time=2.0, rate_func=linear)
        jump = Arrow(
            ax_v.c2p(t_imp, vf_pre),
            ax_v.c2p(t_imp, vf_post),
            buff=0,
            color=C_IMPACT,
            stroke_width=7,
            max_tip_length_to_length_ratio=0.15,
        )
        jump_txt = Text(f"foot: {vf_pre:.2f} → {vf_post:.0f} m/s", font_size=15, color=C_FOOT)
        jump_txt.move_to(ax_v.c2p(t_imp + 0.01, 1.35), aligned_edge=LEFT)
        body_txt = Text(f"body: {vb_post:.2f} m/s, no jump", font_size=15, color=C_BODY)
        body_txt.move_to(ax_v.c2p(t_imp + 0.01, 0.75), aligned_edge=LEFT)
        info2 = place(
            Text(
                f"IMPACT: q continuous, foot velocity {vf_pre:.2f} → 0 m/s.\n"
                f"Impulse {impulse:.1f} N·s on the foot only, {e_lost:.1f} J lost.\n"
                "Inelastic, frictionless, point contact.",
                font_size=19,
                color=C_IMPACT,
                line_spacing=0.9,
            ),
            CODE_X,
            -2.2,
        )
        self.play(
            groups[1].animate.set_opacity(1),
            GrowArrow(jump),
            FadeIn(jump_txt),
            FadeIn(body_txt),
            FadeOut(info),
            FadeIn(info2),
            run_time=0.8,
        )
        self.wait(1.7)

        # ---- stance ---------------------------------------------------------------------------------------------
        info3 = place(
            Text(
                "Phase 1, stance: rigid contact, foot fixed at z = 0.\n"
                "F ≥ 0 (a floor cannot pull); F = 0 at the last node (take-off).",
                font_size=19,
                color=C_STANCE,
                line_spacing=0.9,
            ),
            CODE_X,
            -2.1,
        )
        self.play(
            groups[2].animate.set_opacity(1),
            FadeOut(info2),
            FadeOut(jump),
            FadeOut(jump_txt),
            FadeOut(body_txt),
            FadeIn(info3),
            run_time=0.6,
        )
        self.play(clock.animate.set_value(t_off), run_time=3.0, rate_func=linear)
        self.play(groups[3].animate.set_opacity(1), run_time=0.6)
        self.wait(0.6)

        # ---- take-off, flight back to the apex ----------------------------------------------------------------
        info4 = place(
            Text(
                f"Peak force {f_max:.0f} N ({f_max / weight:.1f} times the weight).\n"
                "F ≥ 0 is not binding here: F > 0 until take-off.\n"
                "Then phase 2: flight back to the apex (periodic).\n"
                f"IPOPT status {int(d['status'])}, {int(d['iterations'])} iterations.",
                font_size=18,
                color=GRAY_A,
                line_spacing=0.9,
            ),
            CODE_X,
            -2.75,
        )
        self.play(FadeOut(info3), FadeIn(info4), run_time=0.4)
        self.play(clock.animate.set_value(T), run_time=1.8, rate_func=linear)
        self.wait(2.0)

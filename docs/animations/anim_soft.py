"""
Manim CE scene: soft (compliant) contact, ContactType.SOFT_EXPLICIT.  A 1 kg ball is pushed onto a spring-damper ground.
REAL bioptim / IPOPT solves stored in ``data/soft_ball.npz`` (see ``generate_soft_data.py``): the same push (tracked
half-cosine reference) for three stiffnesses of the bioMod, warm started; the contact force is the one computed by
``bio_model.soft_contact_forces()``.

Scene: SoftContact (about 20 s).  Render (from docs/animations):  manim render -qh anim_soft.py SoftContact
"""

import numpy as np
from manim import *

from features_scenes import CODE_W, DATA_DIR, M, code, fit, make_axes, poly, scene_title

CODE_X0 = 0.15
W = WHITE
C_K = {1e4: BLUE_C, 1e5: ORANGE, 1e6: RED_C}
C_REF = GRAY_B
C_F = YELLOW_C
SCALE = 20.0  # sketch: 1 m -> 20 scene units (exaggerated so that 3 cm is visible)


def caption(text, size=19, color=GRAY_B):
    return Text(text, font_size=size, color=color)


def code_block(lines, size=17):
    block = VGroup(*[code(text, size, color) for _, text, color in lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.11)
    for line, (level, _, _) in zip(block, lines):
        line.shift(RIGHT * 0.3 * level)
    return block


def panel_of(*items, top=2.3):
    panel = VGroup(*items).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
    fit(panel, CODE_W)
    return panel.move_to([CODE_X0, top, 0], aligned_edge=UL)


class SoftContact(Scene):
    def construct(self):
        d = np.load(DATA_DIR / "soft_ball.npz")
        radius = float(d["radius"])
        ks = [1e4, 1e5, 1e6]

        def val(k, key):
            return d[f"k{k:g}_{key}"]

        t = val(1e4, "t")
        n = len(t)

        title = scene_title(
            "Soft contact: a spring-damper ground", "ContactType.SOFT_EXPLICIT, real solves of a pushed ball"
        )
        self.play(FadeIn(title), run_time=0.4)

        # ------------------------------------------------------------ code, beat 1 (caption above its code)
        cap_a = caption("Bioptim code: soft contact declared in the bioMod")
        code_a = code_block(
            [
                (0, "softcontact Contact1", W),
                (1, "parent ball", W),
                (1, "type sphere", W),
                (1, "radius 0.05", W),
                (1, "stiffness 1e4", YELLOW_C),
                (1, "damping 2", YELLOW_C),
                (0, "endsoftcontact", W),
            ]
        )
        cap_b = caption("Bioptim code: model with the soft contact dynamics")
        code_b = code_block(
            [
                (0, "bio_model = TorqueBiorbdModel(path,", W),
                (1, "contact_types=[ContactType.SOFT_EXPLICIT])", W),
            ]
        )
        cap_c = caption("force from biorbd; measured c = 0.298 (depth > 1 cm)")
        code_c = code_block([(0, "F = c * k * depth^1.5 * (1 + 1.5 * damping * speed)", GRAY_A)])
        panel = panel_of(cap_a, code_a, cap_b, code_b, cap_c, code_c)

        # ------------------------------------------------------------ sketch (left) driven by the k = 1e4 solution
        gx, gy = -5.0, -0.6  # ground surface (center x, y)
        ground = Rectangle(width=3.4, height=1.0, stroke_width=0, fill_color=GRAY_D, fill_opacity=0.55).move_to(
            [gx, gy - 0.5, 0]
        )
        surface = Line([gx - 1.7, gy, 0], [gx + 1.7, gy, 0], color=GRAY_B, stroke_width=3)
        gname = caption("soft ground (z = 0)", 16).move_to([gx, gy - 0.9, 0])
        z0 = val(1e4, "q")[0]
        idx = ValueTracker(0)

        def zc(i):
            return gy + float(z0[i]) * SCALE

        def ball():
            i = int(idx.get_value())
            return Circle(
                radius=radius * SCALE, color=BLUE_C, stroke_width=4, fill_color=BLUE_E, fill_opacity=0.45
            ).move_to([gx, zc(i), 0])

        def arrow():
            i = int(idx.get_value())
            f = float(val(1e4, "force")[i])
            if f < 0.3:
                return Dot([gx, gy, 0], radius=0.001, color=C_F)
            return Arrow(
                [gx, gy, 0],
                [gx, gy + f * 0.06, 0],
                buff=0,
                color=C_F,
                stroke_width=7,
                max_tip_length_to_length_ratio=0.35,
            )

        def readout():
            i = int(idx.get_value())
            dep = float(val(1e4, "depth")[i]) * 100
            f = float(val(1e4, "force")[i])
            return (
                VGroup(
                    Text(f"depth = {max(dep, 0):.1f} cm", font_size=18, color=W),
                    Text(f"contact force = {f:.1f} N", font_size=18, color=C_F),
                )
                .arrange(DOWN, aligned_edge=LEFT, buff=0.08)
                .move_to([-6.9, -2.9, 0], aligned_edge=LEFT)
            )

        ball_m = always_redraw(ball)
        arrow_m = always_redraw(arrow)
        read_m = always_redraw(readout)
        depth_lab = caption("depth = radius − z", 16, GRAY_B).move_to([gx + 0.3, 2.35, 0])
        # plots (right of the sketch)
        ax_d = make_axes([-1.75, 1.2, 0], 2.7, 1.5, (0, t[-1]), (-1, 3.5))
        ax_f = make_axes([-1.75, -1.0, 0], 2.7, 1.5, (0, t[-1]), (0, 17))
        pl_d = caption("depth (cm)", 15).next_to(ax_d, UP, buff=0.05).align_to(ax_d, LEFT)
        pl_f = caption("contact force (N)", 15).next_to(ax_f, UP, buff=0.05).align_to(ax_f, LEFT)
        zero_d = DashedLine(ax_d.c2p(0, 0), ax_d.c2p(t[-1], 0), color=GRAY_D, stroke_width=2)
        depth_pts = np.array(val(1e4, "depth")) * 100
        force_pts = np.array(val(1e4, "force"))
        tr_d = always_redraw(
            lambda: poly(ax_d, t[: int(idx.get_value()) + 1], depth_pts[: int(idx.get_value()) + 1], BLUE_C, 4)
        )
        tr_f = always_redraw(
            lambda: poly(ax_f, t[: int(idx.get_value()) + 1], force_pts[: int(idx.get_value()) + 1], C_F, 4)
        )
        self.play(
            FadeIn(VGroup(ground, surface, gname, ax_d, ax_f, pl_d, pl_f, zero_d, depth_lab)),
            FadeIn(panel),
            run_time=0.6,
        )
        self.add(ball_m, arrow_m, read_m, tr_d, tr_f)
        self.play(idx.animate.set_value(n - 1), run_time=4.5, rate_func=linear)
        self.wait(1.2)

        # ------------------------------------------------------------ beat 2: three stiffnesses
        beat1 = VGroup(
            ground, surface, gname, ax_d, ax_f, pl_d, pl_f, zero_d, depth_lab, ball_m, arrow_m, read_m, tr_d, tr_f
        )
        for m in (ball_m, arrow_m, read_m, tr_d, tr_f):
            m.clear_updaters()
        self.play(FadeOut(beat1), FadeOut(panel), run_time=0.5)

        ax1 = make_axes([-3.5, 1.05, 0], 5.6, 1.9, (0, t[-1]), (-1, 3.5))
        ax2 = make_axes([-3.5, -1.55, 0], 5.6, 1.9, (0, t[-1]), (0, 400))
        lab1 = caption("depth (cm), dashed: reference", 16).next_to(ax1, UP, buff=0.05).align_to(ax1, LEFT)
        lab2 = caption("contact force (N)", 16).next_to(ax2, UP, buff=0.05).align_to(ax2, LEFT)
        z_ref = radius * 100 - 100 * (0.06 + (0.02 - 0.06) * 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, n))))
        ref = DashedVMobject(poly(ax1, t, z_ref, WHITE, 3), num_dashes=60).set_z_index(5)
        zero1 = DashedLine(ax1.c2p(0, 0), ax1.c2p(t[-1], 0), color=GRAY_D, stroke_width=2)
        t_lab = caption("t (s)", 14).next_to(ax2, DOWN, buff=0.05).align_to(ax2, RIGHT)

        cap_d = caption("Bioptim code: same push, only the number in the bioMod changes")
        code_d = code_block(
            [
                (0, "stiffness 1e4   ->   1e5   ->   1e6", YELLOW_C),
                (0, 'ObjectiveFcn.Lagrange.TRACK_STATE, key="q",', W),
                (1, "node=Node.ALL, target=reference", W),
                (0, "OdeSolver.RK4(n_integration_steps=5)", W),
            ]
        )
        panel2 = panel_of(cap_d, code_d)
        self.play(FadeIn(VGroup(ax1, ax2, lab1, lab2, zero1, t_lab, ref)), FadeIn(panel2), run_time=0.5)

        rows = []
        for k in ks:
            c = C_K[k]
            c1 = poly(ax1, t, np.array(val(k, "depth")) * 100, c, 5)
            c2 = poly(ax2, t, np.array(val(k, "force")), c, 5)
            row = VGroup(
                code(f"{k:.0e}".replace("e+0", "e"), 17, c),
                Text(f"max depth = {val(k, 'depth').max() * 100:.1f} cm", font_size=18, color=W),
                Text(f"peak force = {val(k, 'force').max():.0f} N", font_size=18, color=W),
                Text(f"IPOPT: {int(val(k, 'iterations'))} iterations", font_size=18, color=GRAY_A),
            ).arrange(RIGHT, buff=0.3)
            rows.append(row)
            self.play(Create(c1), Create(c2), run_time=1.1)
        table = VGroup(caption("stiffness and result (IPOPT status 0 for all three)", 17), *rows).arrange(
            DOWN, aligned_edge=LEFT, buff=0.14
        )
        fit(table, CODE_W)
        table.move_to([CODE_X0, 0.45, 0], aligned_edge=UL)
        self.play(FadeIn(table), run_time=0.7)
        self.wait(0.6)

        s1 = int(d["steps1_status"])
        f5, f20 = float(d["steps5_force"].max()), float(d["steps20_force"].max())
        msg = VGroup(
            M("<b>the stiffer the ground, the harder the problem</b>", 20, W),
            M(f"stiffness 1e6, RK4 with 1 step per interval: IPOPT status {s1} (failed)", 18, GRAY_B),
            M(f"5 steps or 20 steps: same peak force ({f5:.0f} N vs {f20:.0f} N)", 18, GRAY_B),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
        fit(msg, CODE_W)
        msg.move_to([CODE_X0, -1.7, 0], aligned_edge=UL)
        self.play(FadeIn(msg[0]), run_time=0.5)
        self.play(FadeIn(msg[1:]), run_time=0.6)
        self.wait(2.0)

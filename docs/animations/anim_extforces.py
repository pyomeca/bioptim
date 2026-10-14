"""
Manim CE scene: external forces (ExternalForceSetTimeSeries + numerical_data_timeseries). REAL bioptim / IPOPT solves
stored in ``data/extforces_arm.npz`` (see ``generate_extforces_data.py``): a planar two-link arm tracks a smooth reach
(N = 30, T = 1.5 s); a wind-like push acts on the hand during the movement. Ghost = same OCP without the force.

Scene: ExternalForces (about 15 s).  Render (from docs/animations):  manim render -qh anim_extforces.py ExternalForces
"""

import numpy as np
from manim import *

from features_scenes import CODE_W, DATA_DIR, M, code, fit, make_axes, scene_title, time_label, x_ticks, y_ticks

CODE_X0 = 0.15
C_SH = BLUE_C
C_EL = ORANGE
C_F = RED_C
W = WHITE


def code_block(lines, size=15.5):
    block = VGroup(*[code(text, size, color) for _, text, color in lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
    for line, (level, _, _) in zip(block, lines):
        line.shift(RIGHT * 0.3 * level)
    return block


def caption(text, size=18, color=GRAY_B):
    return Text(text, font_size=size, color=color)


def prefix_steps(ax, t, u, upto, color, width=4, opacity=1.0):
    """Piecewise-constant control u[k] on [t[k], t[k+1]], drawn up to time ``upto``."""
    pts = []
    for k in range(len(u)):
        if t[k] >= upto:
            break
        t1 = min(t[k + 1], upto)
        pts += [ax.c2p(t[k], u[k]), ax.c2p(t1, u[k])]
    if len(pts) < 2:
        return VMobject()
    return VMobject(color=color, stroke_width=width).set_points_as_corners(pts).set_stroke(opacity=opacity)


class ExternalForces(Scene):
    def construct(self):
        d = np.load(DATA_DIR / "extforces_arm.npz")
        n, T = int(d["N"]), float(d["T"])
        t = np.linspace(0, T, n + 1)
        force = -d["force"][1]  # magnitude of the push along -y (N)
        tau_f, tau_p = d["free_tau"], d["push_tau"]
        dtau = tau_p - tau_f
        peak = float(d["peak"])

        title = scene_title("External forces", "a wind-like push on the hand: same reach, one extra input")
        self.play(FadeIn(title), run_time=0.4)

        # ---------------------------------------------------------------- axes (left)
        ax_f = make_axes([-3.55, 1.95, 0], 5.6, 0.95, [0, T], [0, 16], 0.5, 8)
        ax_t = make_axes([-3.55, 0.05, 0], 5.6, 1.9, [0, T], [-3.5, 9.5], 0.5, 1)
        ax_d = make_axes([-3.55, -2.35, 0], 5.6, 1.5, [0, T], [-3, 5], 0.5, 1)

        def lab(text, ax):
            return caption(text, 17).next_to(ax.get_y_axis(), UP, buff=0.06).align_to(ax.get_y_axis(), LEFT)

        decos = VGroup(
            lab("push force on the hand, along -y (N)", ax_f),
            lab("torques (N·m)", ax_t),
            lab("torque with the force minus without (N·m)", ax_d),
            time_label(ax_d),
            x_ticks(ax_d, [0, 0.5, 1, 1.5]),
            y_ticks(ax_f, [0, 15]),
            y_ticks(ax_t, [0, 4, 8]),
            y_ticks(ax_d, [-2, 0, 2, 4]),
        )
        zero_d = DashedLine(ax_d.c2p(0, 0), ax_d.c2p(T, 0), color=GRAY_D).set_stroke(width=2)

        # ---------------------------------------------------------------- code (right)
        cap1 = caption("Bioptim code", 19)
        code1 = code_block(
            [
                (0, "# the force set (external_forces.py); force is a (3, N) array", GRAY_B),
                (0, "hand = np.tile([[0], [0], [-0.3]], (1, N))", W),
                (0, "fset = ExternalForceSetTimeSeries(nb_frames=N)", W),
                (0, 'fset.add_translational_force("push", "Forearm",', W),
                (1, "force, point_of_application_in_local=hand)", W),
            ]
        )
        code2 = code_block(
            [
                (0, "# given to the model and to the dynamics", GRAY_B),
                (0, "bio_model = TorqueBiorbdModel(path, external_force_set=fset)", W),
                (0, "DynamicsOptions(numerical_data_timeseries={", W),
                (1, '"external_forces": fset.to_numerical_time_series()})', W),
            ]
        )
        panel = VGroup(cap1, code1, code2).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
        panel[2].shift(DOWN * 0.1)
        fit(panel, CODE_W)
        panel.move_to([CODE_X0, 2.55, 0], aligned_edge=UL)

        # ---------------------------------------------------------------- arm sketch (right, below the code)
        sh = np.array([1.9, -1.75, 0.0])
        scale = 3.2

        def to_screen(yz):
            return sh + np.array([yz[0] * scale, yz[1] * scale, 0])

        def arm_at(prefix, tt, color, width, opacity=1.0):
            e = np.array([np.interp(tt, t, d[f"{prefix}_elbow"][:, c]) for c in (1, 2)])
            h = np.array([np.interp(tt, t, d[f"{prefix}_hand"][:, c]) for c in (1, 2)])
            pts = [to_screen([0, 0]), to_screen(e), to_screen(h)]
            g = VGroup(
                Line(pts[0], pts[1], color=color, stroke_width=width),
                Line(pts[1], pts[2], color=color, stroke_width=width),
                *[Dot(p, radius=0.06, color=color) for p in pts],
            )
            return g.set_stroke(opacity=opacity).set_fill(opacity=opacity)

        tr = ValueTracker(0.0)
        base = Dot(sh, radius=0.12, color=GRAY_B)
        arm_lab = caption("arm in the (y, z) plane, shoulder at the grey dot", 15).move_to(
            [CODE_X0 + 0.3, -1.32, 0], aligned_edge=LEFT
        )

        def force_now(tt):
            return float(np.interp(tt, t[:-1], force))

        def push_arrow():
            tt = tr.get_value()
            h = np.array([np.interp(tt, t, d["push_hand"][:, c]) for c in (1, 2)])
            tip = to_screen(h)
            f = force_now(tt)
            if f < 0.3:
                return VMobject()
            length = 1.4 * f / peak
            return Arrow(
                tip + RIGHT * length, tip, buff=0, color=C_F, stroke_width=7, max_tip_length_to_length_ratio=0.35
            )

        def readout():
            tt = tr.get_value()
            return caption(f"t = {tt:.2f} s    F = {force_now(tt):4.1f} N", 18, W).move_to(
                [CODE_X0 + 0.3, -1.0, 0], aligned_edge=LEFT
            )

        ghost_arm = always_redraw(lambda: arm_at("free", tr.get_value(), GRAY_B, 4, 0.55))
        arm = always_redraw(lambda: arm_at("push", tr.get_value(), WHITE, 7))
        arrow = always_redraw(push_arrow)
        ro = always_redraw(readout)

        # ---------------------------------------------------------------- curves
        def curves(tt):
            g = VGroup()
            g.add(prefix_steps(ax_t, t, tau_p[0], tt, C_SH, 5), prefix_steps(ax_t, t, tau_p[1], tt, C_EL, 5))
            g.add(prefix_steps(ax_d, t, dtau[0], tt, C_SH, 5), prefix_steps(ax_d, t, dtau[1], tt, C_EL, 5))
            g.add(prefix_steps(ax_f, t, force, tt, C_F, 5))
            return g

        live = always_redraw(lambda: curves(tr.get_value()))
        ghosts = VGroup(
            *[
                DashedVMobject(prefix_steps(ax_t, t, tau_f[j], T, c, 4, 0.7), num_dashes=60)
                for j, c in ((0, C_SH), (1, C_EL))
            ]
        )

        def leg(color, text, dashed):
            a, b = LEFT * 0.3, RIGHT * 0.3
            mark = (
                DashedLine(a, b, color=color).set_stroke(width=4) if dashed else Line(a, b, color=color, stroke_width=6)
            )
            return VGroup(mark, caption(text, 15, W)).arrange(RIGHT, buff=0.1)

        legend = VGroup(
            leg(C_SH, "shoulder", False),
            leg(C_EL, "elbow", False),
            leg(GRAY_B, "dashed: no force", True),
        ).arrange(RIGHT, buff=0.35)
        legend.move_to([-3.75, -3.5, 0])

        self.play(FadeIn(VGroup(ax_f, ax_t, ax_d, decos, zero_d, legend)), FadeIn(panel), run_time=0.7)
        self.play(Create(ghosts), FadeIn(base), FadeIn(ghost_arm), FadeIn(arm_lab), run_time=1.0)
        box = SurroundingRectangle(VGroup(code1), color=YELLOW, buff=0.07, stroke_width=2.5)
        self.play(Create(box), run_time=0.3)
        self.add(live, arm, arrow, ro)
        self.play(tr.animate.set_value(0.35 * T), run_time=2.2, rate_func=linear)
        self.play(
            ReplacementTransform(box, SurroundingRectangle(code2, color=YELLOW, buff=0.07, stroke_width=2.5)),
            tr.animate.set_value(0.5 * T),
            run_time=0.5,
            rate_func=linear,
        )
        box = [m for m in self.mobjects if isinstance(m, SurroundingRectangle)][0]
        self.play(tr.animate.set_value(0.9 * T), run_time=2.3, rate_func=linear)
        self.play(FadeOut(box), tr.animate.set_value(T), run_time=1.2, rate_func=linear)
        self.wait(0.3)

        # ---------------------------------------------------------------- outro
        pk = np.abs(dtau).max(axis=1)
        msg = M(
            f"2 real IPOPT solves (status {int(d['free_status'])} and {int(d['push_status'])}),\n"
            f"peak push {peak:g} N (hand 0.6 m from the shoulder).\n"
            f"Peak torque change: shoulder {pk[0]:.1f} N·m, elbow {pk[1]:.1f} N·m.",
            18,
            W,
        )
        fit(msg, 5.2)
        msg.move_to([CODE_X0 + 0.15, -3.1, 0], aligned_edge=LEFT)
        self.play(FadeIn(msg), run_time=0.5)
        self.wait(2.5)

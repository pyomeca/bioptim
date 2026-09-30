"""
Manim CE scene: free-floating base.  A planar body in zero gravity (trunk with a free root that has NO torque + two arms)
reorients its trunk by moving the arms only, and the angular momentum stays zero.  REAL bioptim / IPOPT solve stored in
``data/floating_reorient.npz`` (see ``generate_floating_data.py``, model ``models/floating_trunk_2arms.bioMod``).

Scenes: FloatingReorient (about 18 s).  Render (from docs/animations):  manim render -qh anim_floating.py FloatingReorient
"""

import numpy as np
from manim import *

from features_scenes import CODE_W, DATA_DIR, M, code, fit, make_axes, poly, scene_title, time_label, x_ticks, y_ticks

CODE_X0 = 0.15
C_ROOT = YELLOW_C
C_ARML = BLUE_C
C_ARMR = TEAL_C
C_LROOT = ORANGE
C_LJ = BLUE_C
W = WHITE
SC = 1.7  # screen units per metre
CENTER = np.array([-3.7, 1.25, 0.0])
TRUNK_HALF = (0.2, 0.3)  # (y, z) half sizes, from the bioMod mesh
SHOULDER = 0.2, 0.3
ARM_LEN = 0.7


def code_block(lines, size=17):
    block = VGroup(*[code(text, size, color) for _, text, color in lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.11)
    for line, (level, _, _) in zip(block, lines):
        line.shift(RIGHT * 0.3 * level)
    return block


def caption(text, size=19, color=GRAY_B):
    return Text(text, font_size=size, color=color)


def rot(a, v):
    return np.array([np.cos(a) * v[0] - np.sin(a) * v[1], np.sin(a) * v[0] + np.cos(a) * v[1]])


def screen(p):
    return CENTER + np.array([p[0] * SC, p[1] * SC, 0.0])


def body_points(q):
    """q = [y, z, theta, phiL, phiR] -> corners of the trunk and the two arm segments (y right, z up)."""
    root, th = np.array([q[0], q[1]]), q[2]
    hy, hz = TRUNK_HALF
    corners = [root + rot(th, np.array(c)) for c in [(-hy, -hz), (hy, -hz), (hy, hz), (-hy, hz)]]
    arms = []
    for side, phi in ((1, q[3]), (-1, q[4])):
        sh = root + rot(th, np.array([side * SHOULDER[0], SHOULDER[1]]))
        arms.append((sh, sh + rot(th + phi, np.array([0.0, -ARM_LEN]))))
    nose = (root + rot(th, np.array([0.0, hz])), root + rot(th, np.array([0.0, hz + 0.16])))
    return corners, arms, nose


def draw_body(q, ghost=False):
    corners, arms, nose = body_points(q)
    trunk = Polygon(*[screen(c) for c in corners], color=GRAY_C if ghost else C_ROOT, stroke_width=2 if ghost else 4)
    trunk.set_fill(C_ROOT, 0 if ghost else 0.25)
    grp = VGroup(trunk)
    for (a, b), col in zip(arms, (C_ARML, C_ARMR)):
        grp.add(Line(screen(a), screen(b), color=GRAY_C if ghost else col, stroke_width=3 if ghost else 7))
    grp.add(Line(screen(nose[0]), screen(nose[1]), color=GRAY_C if ghost else W, stroke_width=2 if ghost else 5))
    return grp.set_opacity(0.4) if ghost else grp


def node_state(x, tr):
    """Linear interpolation between nodes of a (n, N+1) array at fractional node index tr."""
    k = int(np.clip(np.floor(tr), 0, x.shape[1] - 2))
    f = tr - k
    return (1 - f) * x[:, k] + f * x[:, k + 1]


class FloatingReorient(Scene):
    def construct(self):
        d = np.load(DATA_DIR / "floating_reorient.npz")
        q = np.vstack([d["q_roots"], d["q_joints"]])
        N, T = int(d["N"]), float(d["T"])
        t = np.linspace(0, T, N + 1)
        theta_end = float(q[2, -1])

        title = scene_title(
            "Free floating base: reorientation",
            f"no root torque, zero gravity, actuated arms only, N = {N}, T = {T:g} s",
        )
        self.play(FadeIn(title), run_time=0.4)

        # ------------------------------------------------------------ code (right), part 1
        cap1 = caption("Bioptim code")
        code1 = code_block(
            [
                (0, "bio_model = TorqueFreeFloatingBaseBiorbdModel(path)", W),
                (0, "# states: q_roots, q_joints, qdot_roots, qdot_joints", GRAY_B),
                (0, "# controls: tau_joints only (no torque on the root)", GRAY_B),
            ]
        )
        code2 = code_block(
            [
                (0, "# boundary conditions", GRAY_B),
                (0, 'x_bounds["q_joints"][:, [0, -1]] = 0', W),
                (0, 'x_bounds["q_roots"][2, -1] = 0.8', W),
            ]
        )
        panel = VGroup(cap1, code1, code2).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        panel[2].shift(DOWN * 0.15)
        fit(panel, CODE_W)
        panel.move_to([CODE_X0, 2.3, 0], aligned_edge=UL)

        # ------------------------------------------------------------ plot: angles vs time
        ax = make_axes([-3.7, -2.05, 0], 5.4, 1.9, [0, T], [-1.6, 1.6], y_step=1.6)
        ylab = caption("angles (rad)", 16).next_to(ax, UP, buff=0.05).align_to(ax, LEFT)
        yt = y_ticks(ax, [-1.5, 0, 1.5])
        xt = x_ticks(ax, [0, T])
        xl = time_label(ax)
        zero = DashedLine(ax.c2p(0, 0), ax.c2p(T, 0), color=GRAY_D, stroke_width=1.5)
        c_root = poly(ax, t, q[2], C_ROOT, 5)
        c_l = poly(ax, t, q[3], C_ARML, 3)
        c_r = poly(ax, t, q[4], C_ARMR, 3)
        leg = VGroup(caption("root", 15, C_ROOT), caption("arm L", 15, C_ARML), caption("arm R", 15, C_ARMR)).arrange(
            RIGHT, buff=0.3
        )
        leg.next_to(ax, UP, buff=0.05).align_to(ax, RIGHT)

        ghost = draw_body(q[:, 0], ghost=True)
        tracker = ValueTracker(0)
        body = always_redraw(lambda: draw_body(node_state(q, tracker.get_value())))
        readout = always_redraw(
            lambda: M(f"root angle  {node_state(q, tracker.get_value())[2]:+.2f} rad", 20, C_ROOT).move_to(
                [-3.7, -0.75, 0]
            )
        )

        self.play(FadeIn(panel), FadeIn(VGroup(ax, ylab, yt, xt, xl, zero, leg, ghost, body, readout)), run_time=0.6)
        self.play(
            tracker.animate.set_value(N),
            Create(c_root),
            Create(c_l),
            Create(c_r),
            run_time=6.0,
            rate_func=linear,
        )
        end_note = VGroup(
            M(f"arms back to 0, trunk rotated by <b>{theta_end:.2f} rad</b> ({np.degrees(theta_end):.0f}°)", 21, W),
            M("with zero torque on the root", 19, GRAY_B),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        fit(end_note, CODE_W)
        end_note.move_to([CODE_X0, -0.4, 0], aligned_edge=UL)
        self.play(FadeIn(end_note), run_time=0.5)
        self.wait(1.3)

        # ------------------------------------------------------------ part 2: angular momentum
        L, Lr, Lj = d["L"], d["L_root"], d["L_joints"]
        body.clear_updaters()
        readout.clear_updaters()
        self.play(
            FadeOut(VGroup(ax, ylab, yt, xt, xl, zero, leg, c_root, c_l, c_r, readout, end_note, panel, body, ghost)),
            run_time=0.5,
        )
        axm = make_axes([-3.7, -0.15, 0], 5.4, 2.2, [0, T], [-12, 12], y_step=12)
        yl = (
            caption("angular momentum about the centre of mass (kg m²/s)", 16)
            .next_to(axm, UP, buff=0.05)
            .align_to(axm, LEFT)
        )
        ytm = y_ticks(axm, [-10, 0, 10])
        xtm = x_ticks(axm, [0, T])
        xlm = time_label(axm)
        cLr = poly(axm, t, Lr, C_LROOT, 4)
        cLj = poly(axm, t, Lj, C_LJ, 4)
        cL = poly(axm, t, L, W, 5)
        lab_r = code("L_root", 15, C_LROOT).next_to(axm.c2p(0.05, 10), RIGHT, buff=0.05)
        lab_j = code("L_joints", 15, C_LJ).next_to(lab_r, RIGHT, buff=0.3)
        lab_t = code("L", 15, W).next_to(lab_j, RIGHT, buff=0.3)
        lim = float(np.ceil(np.abs(L).max() * 1e5 * 1.3)) * 1e-5
        axt = make_axes([-3.7, -3.0, 0], 5.4, 1.0, [0, T], [-lim, lim], y_step=lim)
        ytt = y_ticks(axt, [-lim, 0, lim], "{:.0e}")
        lab_z = caption("same L, zoomed (tight axis)", 15).next_to(axt, UP, buff=0.04).align_to(axt, LEFT)
        cLz = poly(axt, t, L, W, 4)

        cap3 = caption("Bioptim code")
        code3 = code_block(
            [
                (0, "# biorbd, about the centre of mass", GRAY_B),
                (0, "am = bio_model.angular_momentum()", W),
                (0, "L = am(q, qdot, [])[0]", W),
                (0, "L_root = am(q, [qdot_roots; 0], [])[0]", W),
                (0, "L_joints = am(q, [0; qdot_joints], [])[0]", W),
            ]
        )
        panel3 = VGroup(cap3, code3).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        fit(panel3, CODE_W)
        panel3.move_to([CODE_X0, 2.3, 0], aligned_edge=UL)
        self.play(FadeIn(VGroup(axm, yl, ytm, xtm, xlm, lab_r, lab_j, lab_t, panel3)), run_time=0.5)
        self.play(Create(cLr), Create(cLj), run_time=1.6, rate_func=linear)
        self.play(Create(cL), run_time=1.0, rate_func=linear)
        self.play(FadeIn(VGroup(axt, ytt, lab_z)), Create(cLz), run_time=0.9)
        msg = VGroup(
            M("<b>L = L_root + L_joints</b>", 22, W),
            M(f"max |L_root| = {np.abs(Lr).max():.1f}    max |L_joints| = {np.abs(Lj).max():.1f}", 20, GRAY_A),
            M(f"max |L| = {np.abs(L).max():.1e}", 22, W),
            M("the trunk counter-rotates exactly against the arms:\nangular momentum is conserved", 19, GRAY_B),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        fit(msg, CODE_W)
        msg.move_to([CODE_X0, 0.0, 0], aligned_edge=UL)
        self.play(FadeIn(msg), run_time=0.6)
        self.wait(2.5)

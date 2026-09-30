"""
Manim Community animation: stochastic optimal control with sensory noise and a feedback gain K
(StochasticOptimalControlProgram, SocpType.COLLOCATION, StochasticTorqueBiorbdModel), driven by REAL bioptim/IPOPT solves
stored in ``data/socpk_results.npz`` (see ``generate_socpk_data.py``).

Problem (bioptim/examples/toy_examples/stochastic_optimal_control/arm_reaching_torque_driven_collocations.py): a two-link
arm reaches a target in 0.8 s with motor noise and sensory noise on the hand position and velocity.  The optimiser
returns the mean motion, the feedback gain K (a control of the problem) and the state covariance P at every node.
Beat 1: the mean hand path with the covariance ellipses and the hand standard deviation over time.
Beat 2: K as a heat map for low and high sensory noise (two real solves) and their difference.

Scene: StochasticArmFeedback (about 17 s at native speed).  Render: see notes/socpk.md.
"""

import numpy as np
from manim import *

from features_scenes import (
    C_BOUND,
    C_CTRL,
    C_PAR,
    C_STATE,
    CODE_X,
    DATA_DIR,
    axis_label,
    code_panel,
    dec,
    footer,
    hline,
    make_axes,
    place,
    poly,
    say,
    scene_title,
    time_label,
    x_ticks,
    y_ticks,
)

C_GAIN_NEG = BLUE_C  # negative entries of K (positive entries use C_CTRL: K is a control)
C_ZERO = "#1E1E1E"  # colour of a zero entry of a heat map
SIGMAS = 2  # the ellipses show this number of standard deviations


def gain_matrix(k):
    """(8, N + 1) -> rows ordered shoulder torque (x, y, vx, vy inputs), then elbow torque (K is stored column-major)."""
    return np.array([k[ref * 2 + tor] for tor in range(2) for ref in range(4)])


def heat_image(mat, vmax, width, height, center):
    """Heat map of a matrix (rows top to bottom), green > 0, blue < 0, dark = 0, saturating at +-vmax."""
    pos, neg, zero = color_to_rgb(C_CTRL), color_to_rgb(C_GAIN_NEG), color_to_rgb(C_ZERO)
    img = np.zeros(mat.shape + (3,))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = float(np.clip(mat[i, j] / vmax, -1, 1))
            img[i, j] = zero + abs(v) * ((pos if v > 0 else neg) - zero)
    mob = ImageMobject((255 * img).astype(np.uint8))
    mob.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    mob.stretch_to_fit_width(width).stretch_to_fit_height(height).move_to(center)
    frame = Rectangle(width=width, height=height, stroke_color=GRAY_B, stroke_width=2).move_to(center)
    return mob, frame


class StochasticArmFeedback(Scene):
    def construct(self):
        # ---------------------------------------------------------------- data: everything shown comes from the npz
        d = np.load(DATA_DIR / "socpk_results.npz")
        n, T, bound = int(d["n"]), float(d["final_time"]), float(d["end_bound_m"])
        t = np.linspace(0, T, n + 1)
        hand, hand_all, det_hand = d["low_hand"], d["low_hand_all"], d["det_hand_all"]
        cov = d["low_hand_cov"]  # (n + 1, 2, 2): hand covariance [x, y] at every node
        std_major = np.array([np.sqrt(np.linalg.eigvalsh(c).max()) for c in cov]) * 1e3  # mm
        peak_std, end_std = float(std_major.max()), float(std_major[-1])
        gap = float(np.abs(d["det_hand"] - hand).max()) * 1e3  # mm, deterministic mean path vs stochastic mean path
        k_low, k_high = gain_matrix(d["low_k"]), gain_matrix(d["high_k"])
        s_low, s_high = float(d["low_sens"]), float(d["high_sens"])
        std_high = np.array([np.sqrt(np.linalg.eigvalsh(c).max()) for c in d["high_hand_cov"]]) * 1e3
        tried = [float(s) for s in d["tried_sens"]]
        assert len(set(int(x) for x in d["tried_status"])) == 1  # same IPOPT status for every failed run

        title = scene_title(
            "Stochastic control: the feedback gain K",
            f"SOCP (stochastic optimal control problem): arm reaching, N = {n}, T = {T:g} s",
        )
        self.play(FadeIn(title), run_time=0.4)

        # ================================================================ beat 1: hand path and covariance ellipses
        x_lo, x_hi, y_lo, y_hi = 0.26, 0.55, -0.03, 0.05  # forward hand position y (horizontal), lateral x (vertical)
        w_p = 5.6
        unit = w_p / (x_hi - x_lo)  # scene units per metre, the same in both directions (the ellipses are round)
        ax_p = make_axes([-3.55, 1.25, 0], w_p, unit * (y_hi - y_lo), [x_lo, x_hi], [y_lo, y_hi], 0.1, 0.02)
        ax_s = make_axes([-3.55, -1.4, 0], w_p, 1.7, [0, T], [0, 12], 0.4, 4)
        decos = VGroup(
            axis_label("hand path (m): x lateral, y forward", ax_p, C_STATE),
            axis_label("hand position, largest standard deviation (mm)", ax_s, C_CTRL),
            time_label(ax_s),
            x_ticks(ax_p, [0.3, 0.4, 0.5], "{:.1f}"),
            y_ticks(ax_p, [0, 0.04]),
            x_ticks(ax_s, [0, 0.4, 0.8], "{:.1f}"),
            y_ticks(ax_s, [0, 5, 10]),
        )
        self.play(Create(ax_p), Create(ax_s), FadeIn(decos), run_time=0.8)

        # ghost = the deterministic solve (no noise, no feedback); its mean path is the same to within a fraction of a mm
        ghost = DashedVMobject(poly(ax_p, det_hand[1], det_hand[0], GRAY_B, 3), num_dashes=40).set_opacity(0.8)
        target = Circle(radius=0.07, color=C_BOUND, stroke_width=3).move_to(ax_p.c2p(hand[1, -1], hand[0, -1]))
        target_lab = Text("target", font_size=14, color=C_BOUND).next_to(target, UP, buff=0.05)
        self.play(Create(ghost), FadeIn(target), FadeIn(target_lab), run_time=0.8)

        panel = code_panel(
            [
                (0, "problem_type = SocpType.COLLOCATION(polynomial_degree=3, ...)", WHITE),
                (0, "bio_model = StochasticTorqueBiorbdModel(", WHITE),
                (1, "biorbd_model_path, problem_type=problem_type,", WHITE),
                (1, "motor_noise_magnitude=motor_noise_magnitude,", C_PAR),
                (1, "sensory_noise_magnitude=sensory_noise_magnitude,", C_PAR),
                (1, "sensory_reference=sensory_reference, ...)", WHITE),
                (0, "socp = StochasticOptimalControlProgram(bio_model, ...)", WHITE),
            ],
            size=17,
        )
        self.play(FadeIn(panel), run_time=0.6)

        path = poly(ax_p, hand_all[1], hand_all[0], C_STATE, 5)
        curve_s = poly(ax_s, t, std_major, C_CTRL, 5)
        ellipses = VGroup()
        for i in range(0, n + 1, 4):
            c = cov[i]
            c_plot = np.array([[c[1, 1], c[1, 0]], [c[0, 1], c[0, 0]]])  # axes of the plot: (forward y, lateral x)
            w, v = np.linalg.eigh(c_plot)
            el = Ellipse(
                width=2 * SIGMAS * np.sqrt(w[1]) * unit, height=2 * SIGMAS * np.sqrt(w[0]) * unit, stroke_width=2
            )
            el.rotate(np.arctan2(v[1, 1], v[0, 1])).move_to(ax_p.c2p(hand[1, i], hand[0, i]))
            ellipses.add(el.set_stroke(C_CTRL, 2).set_fill(C_CTRL, 0.22))
        self.play(Create(path), run_time=0.8)
        self.play(LaggedStart(*[FadeIn(e) for e in ellipses], lag_ratio=0.1), Create(curve_s), run_time=1.6)
        bound_line = hline(ax_s, 0.55 * T, T, bound * 1e3, C_BOUND)
        info = place(
            Text(
                f"hand std: peak {peak_std:.1f} mm, {end_std:.1f} mm at the target\n"
                f"bound: {bound * 1e3:.0f} mm on x and on y (dashed red)\n"
                f"mean path vs deterministic solve: {gap:.2f} mm apart at most\n"
                f"IPOPT: {int(d['low_iterations'])} iterations, "
                + ("converged" if int(d["low_status"]) == 0 else "not converged"),
                font_size=19,
                color=GRAY_A,
                line_spacing=0.9,
            ),
            CODE_X,
            -1.0,
        )
        self.play(Create(bound_line), FadeIn(info), run_time=0.6)
        remark = place(
            say("The mean path is the same as the deterministic one (dashed): the uncertainty lives in the ellipses."),
            CODE_X,
            -2.5,
        )
        foot = footer(
            f"Ellipses: {SIGMAS} standard deviations (std) of the hand position from the optimised covariance P, fixed "
            "to initial_cov at the first node."
        )
        self.play(FadeIn(remark), FadeIn(foot))
        self.wait(1.2)

        # ================================================================ beat 2: the feedback gain K, low vs high noise
        self.play(*[FadeOut(m) for m in self.mobjects if m is not title])
        self.add(title)
        h_map, w_map, x_map = 1.1, 5.3, -3.25
        vmax = float(np.percentile(np.abs(np.concatenate([k_low, k_high])), 97))  # both maps share this scale
        dk = k_high - k_low
        dmax = float(np.abs(dk).max())
        tops = [2.15, 0.55, -1.05]
        centers = [[x_map, top - h_map / 2, 0] for top in tops]
        img0, fr0 = heat_image(k_low, vmax, w_map, h_map, centers[0])
        img1, fr1 = heat_image(k_high, vmax, w_map, h_map, centers[1])
        img2, fr2 = heat_image(dk, dmax, w_map, h_map, centers[2])

        def map_label(text, top, color=GRAY_B):
            return Text(text, font_size=20, color=color).move_to([x_map - w_map / 2, top + 0.2, 0], aligned_edge=LEFT)

        rows = VGroup()
        for c in centers:
            for name, dy in (("shoulder", 0.28), ("elbow", -0.28)):
                rows.add(Text(name, font_size=14, color=GRAY_B).move_to([x_map - w_map / 2 - 0.4, c[1] + dy, 0]))
        lab_axes = VGroup(
            Text("t (s)", font_size=16, color=GRAY_B).move_to([x_map + w_map / 2 + 0.45, tops[2] - h_map - 0.22, 0]),
            *[
                Text(dec(f"{v:.1f}"), font_size=16, color=GRAY_B).move_to(
                    [x_map - w_map / 2 + w_map * v / T, tops[2] - h_map - 0.22, 0]
                )
                for v in (0, 0.4, 0.8)
            ],
        )
        lab0 = map_label("feedback gain K, low sensory noise", tops[0], C_CTRL)
        lab1 = map_label("K, high sensory noise", tops[1], C_CTRL)
        lab2 = map_label("difference K(high) − K(low)", tops[2], C_PAR)
        legend = Text(
            "K in N·m per m or m/s · green > 0 · blue < 0\n"
            f"colour saturates at ±{vmax:.0f} (difference map: ±{dmax:.0f})",
            font_size=16,
            color=GRAY_B,
            line_spacing=0.9,
        ).move_to([x_map - w_map / 2, -2.95, 0], aligned_edge=LEFT)
        self.play(FadeIn(Group(lab0, img0, fr0, rows[:2], legend, lab_axes)), run_time=0.8)

        panel = code_panel(
            [
                (0, f"sens = {s_low:.1f}", C_STATE),
                (0, "pos = cas.DM(np.array([(sens * WPQ_STD) ** 2 / DT] * 2))", WHITE),
                (0, "vel = cas.DM(np.array([(sens * WPQDOT_STD) ** 2 / DT] * 2))", WHITE),
                (0, "sensory = cas.vertcat(pos, vel)", C_PAR),
                (0, "tau_fb = k_matrix @ ((sensory_input - ref) + sensory_noise)", WHITE),
            ],
            size=17,
        )
        panel_high = code_panel(
            [
                (0, f"sens = {s_high:.1f}", C_STATE),
                (0, "pos = cas.DM(np.array([(sens * WPQ_STD) ** 2 / DT] * 2))", WHITE),
                (0, "vel = cas.DM(np.array([(sens * WPQDOT_STD) ** 2 / DT] * 2))", WHITE),
                (0, "sensory = cas.vertcat(pos, vel)", C_PAR),
                (0, "tau_fb = k_matrix @ ((sensory_input - ref) + sensory_noise)", WHITE),
            ],
            size=17,
        )
        self.play(FadeIn(panel), run_time=0.6)
        self.play(
            Transform(panel[1][0], panel_high[1][0]),
            FadeIn(Group(lab1, img1, fr1, rows[2:4])),
            run_time=0.8,
        )
        self.play(FadeIn(Group(lab2, img2, fr2, rows[4:6])), run_time=0.8)

        info2 = place(
            Text(
                f"sensory noise ×{s_high:g}: mean |K| = {np.abs(k_low).mean():.1f} → {np.abs(k_high).mean():.1f}\n"
                f"cost = {float(d['low_cost']):.1f} → {float(d['high_cost']):.1f}  ·  peak hand std = {peak_std:.1f} → "
                f"{float(std_high.max()):.1f} mm\n"
                f"IPOPT: {int(d['low_iterations'])} then {int(d['high_iterations'])} iterations, "
                + ("both converged" if int(d["low_status"]) == 0 and int(d["high_status"]) == 0 else "not converged"),
                font_size=19,
                color=GRAY_A,
                line_spacing=0.9,
            ),
            CODE_X,
            -0.85,
        )
        remark2 = place(say("Noisier sensors call for larger feedback gains overall, at a higher cost."), CODE_X, -2.15)
        foot2 = footer(
            f"Two real solves, no warm start. Noise ×{tried[0]:g} and ×{tried[1]:g} ended with IPOPT status "
            f"{int(d['tried_status'][0])} (not shown); ×{s_high:g} is the largest that converged."
        )
        self.play(FadeIn(info2), FadeIn(remark2), FadeIn(foot2), run_time=0.8)
        self.wait(2.5)

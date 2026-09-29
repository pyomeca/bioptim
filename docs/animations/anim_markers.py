"""
Manim CE scene: tracking measured markers with ObjectiveFcn.Lagrange.TRACK_MARKERS, driven by a REAL bioptim solve
(data/markers_tracking.npz, see ``generate_markers_data.py``). The "measured" markers are SYNTHETIC (known motion of the
double pendulum + 1 cm Gaussian noise). The model trajectory shown is the k-th IPOPT iterate, then the final solution
is played back in time.

Render (from docs/animations):  manim render -qh anim_markers.py TrackMarkers
"""

import sys
from pathlib import Path

import numpy as np
from manim import *

sys.path.insert(0, str(Path(__file__).parent))
from features_scenes import (  # noqa: E402  (also sets the default fonts)
    CODE_X,
    axis_label,
    code_panel,
    make_axes,
    place,
    poly,
    scene_title,
    time_label,
    x_ticks,
)

DATA = Path(__file__).parent / "data" / "markers_tracking.npz"
C_MODEL, C_MEAS, C_ERR = GREEN_C, GRAY_B, RED_C
PIVOT = np.array([-3.2, 2.05, 0.0])
SCALE = 1.25  # scene units per metre
NODE_SHOW = 10  # node drawn as a stick figure while the iterates converge
YR = (-0.5, 2.5)  # log10(error in cm) range of the error axis


class TrackMarkers(Scene):
    def construct(self):
        d = np.load(DATA)
        n, T = int(d["n_shooting"]), float(d["final_time"])
        t = np.linspace(0, T, n + 1)
        tr = list(d["tracked"])  # marker indices tracked (elbow, tip)
        meas, clean = d["markers_meas"], d["markers_clean"]
        its = [int(k) for k in d["iter_list"]]
        n_iter = int(d["iterations"])

        def pt(mk, i, node):
            """Scene point of marker i (y-z plane) at a possibly fractional node."""
            a = min(int(np.floor(node)), n - 1)
            w = node - a
            p = (1 - w) * mk[:, i, a] + w * mk[:, i, a + 1]
            return PIVOT + SCALE * np.array([p[1], p[2], 0.0])

        def dist(mk, ref):
            return np.linalg.norm(mk[1:, tr, :] - ref[1:, tr, :], axis=0)  # (2 markers, N+1), metres

        def rms(mk, ref=meas):
            return 100 * np.sqrt((dist(mk, ref) ** 2).mean())

        self.play(
            FadeIn(scene_title("Tracking measured markers", "synthetic data: known motion + 1 cm noise")), run_time=0.5
        )

        # ---- measured markers (static): all nodes, both tracked markers
        meas_dots = VGroup(*[Dot(pt(meas, i, k), radius=0.035, color=C_MEAS) for i in tr for k in range(n + 1)])
        base = Dot(PIVOT, radius=0.07, color=WHITE)

        def trace(mk):
            return VGroup(
                *[
                    VMobject(color=C_MODEL, stroke_width=3, stroke_opacity=0.7).set_points_as_corners(
                        [pt(mk, i, k) for k in range(n + 1)]
                    )
                    for i in tr
                ]
            )

        def stick(mk, node):
            p0, p1, p2 = pt(mk, 0, node), pt(mk, tr[0], node), pt(mk, tr[1], node)
            g = VGroup(
                Line(p0, p1, color=C_MODEL, stroke_width=6),
                Line(p1, p2, color=C_MODEL, stroke_width=6),
                Dot(p1, radius=0.08, color=C_MODEL),
                Dot(p2, radius=0.08, color=C_MODEL),
            )
            for i, p in zip(tr, (p1, p2)):
                m = pt(meas, i, node)
                g.add(Line(p, m, color=C_ERR, stroke_width=4))
                g.add(Circle(radius=0.09, color=WHITE, stroke_width=2).move_to(m))
            return g

        # ---- error axis (log scale, cm): RMS over the two markers at each node
        ax = make_axes([-3.6, -2.3, 0], 5.6, 1.7, (0, T), YR)
        tf = lambda v: np.log10(np.maximum(v, 1e-3))
        decos = VGroup(
            axis_label("marker error (cm, log scale)", ax),
            time_label(ax),
            x_ticks(ax, [0, 0.5, 1.0, 1.5], "{:.1f}"),
        )
        for v in (1, 10, 100):
            decos.add(Text(f"{v}", font_size=16, color=GRAY_B).next_to(ax.c2p(0, np.log10(v)), LEFT, buff=0.08))
        noise_rms = rms(clean, meas)  # the noise itself: measured vs noise-free markers
        noise_line = DashedLine(ax.c2p(0, tf(noise_rms)), ax.c2p(T, tf(noise_rms)), color=YELLOW_C, stroke_width=2)
        noise_txt = Text(f"noise level ({noise_rms:.1f} cm)", font_size=15, color=YELLOW_C).next_to(
            ax.c2p(T, tf(noise_rms)), UP, buff=0.06, aligned_edge=RIGHT
        )

        def err_plot(mk):
            e = 100 * np.sqrt((dist(mk, meas) ** 2).mean(axis=0))
            return poly(ax, t, tf(e), C_ERR, 4)

        panel = code_panel(
            [
                (0, "objectives = ObjectiveList()", WHITE),
                (0, "objectives.add(", WHITE),
                (1, "ObjectiveFcn.Lagrange.TRACK_MARKERS,", C_MODEL),
                (1, "weight=1000, node=Node.ALL,", WHITE),
                (1, "marker_index=[1, 3], axes=[Axis.Y, Axis.Z],", WHITE),
                (1, "target=markers_measured,  # (2, 2, N+1)", YELLOW_C),
                (0, ")", WHITE),
                (0, "objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,", GRAY_B),
                (1, 'key="tau", weight=1e-3)  # small regulariser', GRAY_B),
            ],
            size=19,
            top=2.35,
            caption="Bioptim code",
        )

        def readout(label, mk):
            return place(
                Text(f"IPOPT iteration {label}   ·   RMS error {rms(mk):.1f} cm", font_size=21, color=WHITE),
                CODE_X,
                -1.3,
            )

        def mk_it(k):
            return d[f"markers_it{k}"]

        mk_final = d["markers_opt"]
        legend = place(
            Text(
                "grey dots: measured (synthetic)   ·   green: model markers\nred: gap between model and measurement",
                font_size=16,
                color=GRAY_B,
                line_spacing=0.9,
            ),
            CODE_X,
            -2.2,
        )

        # ---- iteration 0
        tr_mob, st_mob, ep = trace(mk_it(its[0])), stick(mk_it(its[0]), NODE_SHOW), err_plot(mk_it(its[0]))
        ro = readout(its[0], mk_it(its[0]))
        self.play(
            FadeIn(meas_dots),
            FadeIn(base),
            Create(ax),
            FadeIn(decos),
            FadeIn(panel),
            FadeIn(noise_line),
            FadeIn(noise_txt),
            run_time=1.0,
        )
        self.play(Create(tr_mob), FadeIn(st_mob), Create(ep), FadeIn(ro), FadeIn(legend), run_time=0.8)
        self.wait(0.4)
        for k in its[1:] + ["final"]:
            mk = mk_final if k == "final" else mk_it(k)
            lab = f"{n_iter} (converged)" if k == "final" else k
            self.play(
                Transform(tr_mob, trace(mk)),
                Transform(st_mob, stick(mk, NODE_SHOW)),
                Transform(ep, err_plot(mk)),
                Transform(ro, readout(lab, mk)),
                run_time=0.55 if k != "final" else 0.8,
                rate_func=smooth,
            )
        self.wait(0.3)

        # ---- playback of the final solution in time
        tracker = ValueTracker(NODE_SHOW)
        self.remove(st_mob)
        live = always_redraw(lambda: stick(mk_final, tracker.get_value()))
        cursor = always_redraw(
            lambda: Line(
                ax.c2p(tracker.get_value() * T / n, YR[0]),
                ax.c2p(tracker.get_value() * T / n, YR[1]),
                color=WHITE,
                stroke_width=2,
            )
        )
        self.add(live, cursor)
        self.play(tracker.animate.set_value(0), run_time=0.4, rate_func=smooth)
        self.play(tracker.animate.set_value(n), run_time=3.6, rate_func=linear)
        self.remove(live, cursor)
        self.add(stick(mk_final, n))
        summary = place(
            Text(
                f"model vs measured: {rms(mk_final):.2f} cm RMS (about the noise)\n"
                f"model vs noise-free truth: {rms(mk_final, clean):.2f} cm RMS",
                font_size=20,
                color=YELLOW_C,
                line_spacing=0.9,
            ),
            CODE_X,
            -3.3,
        )
        self.play(FadeIn(summary), run_time=0.5)
        self.wait(2.0)

"""
PLACEHOLDER data generator for ``scene_template.py`` (see ../STANDARD.md, section 5).

This script does NOT use bioptim: the numbers below are computed by hand with numpy so that the template scene runs
anywhere. In a real video this file is called ``generate_<topic>_data.py`` (in docs/animations/), it builds the OCP,
runs IPOPT, and stores REAL solver output. Replace ``fake_solve`` by ``prepare_ocp`` + ``ocp.solve(solver)`` (see
generate_deriv_data.py for a compact model) and keep the same keys. Never ship fabricated curves in a real video.

Output: docs/animations/data/template_demo.npz (git-ignored, regenerate it with the command below)
    t            (N + 1,) node times (s)
    y_ref        (N + 1,) reference curve (the ghost: e.g. the solution before the change)
    y_new        (N + 1,) curve after the change
    weight       the parameter that changes between the two runs
    cost_ref, cost_new, iterations, status (0 = Solve_Succeeded), converged

Usage, from the repository root:  python docs/animations/templates/generate_template_data.py
"""

from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "data"
N, T = 30, 1.0
WEIGHT = 10.0


def fake_solve(weight: float):
    """Stand-in for ``ocp.solve``: returns (t, curve, cost, iterations, status). REPLACE by a real bioptim solve."""
    t = np.linspace(0.0, T, N + 1)
    y = np.sin(np.pi * t) / (1.0 + 0.05 * weight)  # the larger the weight, the smaller the peak
    cost = float(np.sum(y[:-1] ** 2) * T / N + weight * 1e-3)
    return t, y, cost, 12 + int(weight), 0


def main():
    t, y_ref, cost_ref, _, _ = fake_solve(0.0)
    _, y_new, cost_new, iterations, status = fake_solve(WEIGHT)
    OUT.mkdir(exist_ok=True)
    np.savez(
        OUT / "template_demo.npz",
        t=t,
        y_ref=y_ref,
        y_new=y_new,
        weight=WEIGHT,
        cost_ref=cost_ref,
        cost_new=cost_new,
        iterations=iterations,
        status=status,
        converged=status == 0,
        n_shooting=N,
        final_time=T,
    )
    print(f"template_demo.npz written: cost {cost_ref:.4f} -> {cost_new:.4f}, status {status}, {iterations} iterations")


if __name__ == "__main__":
    main()

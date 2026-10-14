"""Validate that the installed FATROP backend solves a nonlinear OCP."""

from pathlib import Path
import sys

import casadi
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bioptim import OdeSolver, OrderingStrategy, Solver
from bioptim.examples.getting_started.basic_ocp import prepare_ocp


def main() -> None:
    ocp = prepare_ocp(
        str(ROOT / "bioptim/examples/models/pendulum.bioMod"),
        final_time=1,
        n_shooting=30,
        use_sx=False,
        ode_solver=OdeSolver.RK4(),
        ordering_strategy=OrderingStrategy.TIME_MAJOR,
    )
    solver = Solver.FATROP()
    solver.set_print_level(0)
    ocp.solve(solver)
    interface = ocp.ocp_solver
    stats = interface.shaked_ocp_solver.stats()
    x = np.array(interface.out["sol"]["x"])
    evaluate_g = casadi.Function("validate_fatrop_constraints", [interface.nlp["x"]], [interface.nlp["g"]])
    g = np.array(evaluate_g(x))
    violations = np.maximum(
        0.0,
        np.concatenate(
            [
                np.array(interface.limits["lbx"]) - x,
                x - np.array(interface.limits["ubx"]),
                np.array(interface.limits["lbg"]) - g,
                g - np.array(interface.limits["ubg"]),
            ]
        ),
    )
    max_violation = float(np.max(violations))
    if (
        not stats["success"]
        or not np.isfinite(x).all()
        or not np.isfinite(g).all()
        or not np.isfinite(violations).all()
        or max_violation > 1e-7
    ):
        raise RuntimeError(f"FATROP smoke failed: {stats['return_status']}, max violation={max_violation}")
    print(f"FATROP backend converged; maximum constraint/bound violation: {max_violation:.3g}")


if __name__ == "__main__":
    main()

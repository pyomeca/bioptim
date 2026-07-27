import numpy as np


def scaled_control_bounds(nlp) -> tuple[np.ndarray, np.ndarray]:
    """Assemble Acados control bounds in solver order and scaled coordinates."""

    lower = np.empty(nlp.controls.shape)
    upper = np.empty(nlp.controls.shape)
    for key in nlp.controls.keys():
        bounds = nlp.u_bounds[key].scale(nlp.u_scaling[key].scaling)
        index = nlp.controls[key].index
        lower[index] = np.asarray(bounds.min[:, 0], dtype=float)
        upper[index] = np.asarray(bounds.max[:, 0], dtype=float)

    if np.any(lower > upper):
        raise ValueError(f"Scaled Acados control bounds are inconsistent: lower={lower}, upper={upper}")
    return lower, upper

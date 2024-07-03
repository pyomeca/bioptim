import biorbd_casadi as biorbd
import numpy as np
from typing import Any

from .biorbd_model import BiorbdModel


def animate_with_pyorerun(
    solution: "SolutionData",
    show_now: bool = True,
    tracked_markers: list[np.ndarray] = None,
    models: BiorbdModel | list[BiorbdModel] = None,
    **kwargs: Any,
) -> None:
    try:
        import pyorerun
    except ModuleNotFoundError:
        raise RuntimeError("pyorerun must be install to animate the model")

    if not isinstance(solution, (list, tuple)):
        solution = [solution]

    if tracked_markers is None:
        tracked_markers = [None] * len(solution)
    prerun = pyorerun.MultiPhaseRerun()

    for idx_phase, (data, model, tm) in enumerate(zip(solution, models, tracked_markers)):

        if "q_roots" in data and "q_joints" in data:
            try:
                data["q"] = np.vstack((data["q_roots"], data["q_joints"]))
            except:
                raise NotImplementedError(
                    "Found q_roots and q_joints in the solution. This is not supported yet with animation in pyorerun"
                )

        prerun.add_phase(t_span=data["time"], phase=idx_phase)

        if not isinstance(model, biorbd.Model):
            raise NotImplementedError(
                f"Animation is only implemented for biorbd models. Got {model.__class__.__name__}"
            )

        biorbd_model = pyorerun.BiorbdModel.from_biorbd_object(model)

        prerun.add_animated_model(
            biorbd_model,
            data["q"],
            tracked_markers=tm if tm is not None else None,
            phase=idx_phase,
        )

    prerun.rerun(notebook=not show_now)

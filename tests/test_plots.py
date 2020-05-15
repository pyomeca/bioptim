"""
Test for file IO
"""
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from biorbd_optim import Data, OdeSolver, ShowResult

# Load align_markers
PROJECT_FOLDER = Path(__file__).parent / ".."
spec = importlib.util.spec_from_file_location(
    "align_markers", str(PROJECT_FOLDER) + "/examples/torque_driven_ocp/align_markers.py"
)
align_markers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(align_markers)


def test_plot_graphs_one_phase():
    ocp = align_markers.prepare_ocp(biorbd_model_path=str(PROJECT_FOLDER) + "/examples/torque_driven_ocp/cube.bioMod")
    sol = ocp.solve()

    plt = ShowResult(ocp, sol)
    plt.graphs()

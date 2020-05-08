import importlib.util
from pathlib import Path

from biorbd_optim import OptimalControlProgram, ShowResult

# Load multiphase_align_markers
PROJECT_FOLDER = Path(__file__).parent
spec = importlib.util.spec_from_file_location(
    "multiphase_align_markers", str(PROJECT_FOLDER) + "/multiphase_align_markers.py"
)
eocar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eocar)


def run_and_save_ocp():
    ocp = eocar.prepare_ocp(biorbd_model_path=str(PROJECT_FOLDER) + "/cube.bioMod",)
    sol = ocp.solve()
    OptimalControlProgram.save(ocp, sol, "cube_ocp_sol")


if __name__ == "__main__":
    run_and_save_ocp()

    ocp, sol = OptimalControlProgram.load(biorbd_model_path="cube.bioMod", name="cube_ocp_sol.bo")
    result = ShowResult(ocp, sol)
    # result.graphs()
    result.animate(nb_frames=40)
)


import importlib.util
from pathlib import Path

from biorbd_optim import OptimalControlProgram, ShowResult

# Load eocar
PROJECT_FOLDER = Path(__file__).parent
spec = importlib.util.spec_from_file_location(
    "eocar", str(PROJECT_FOLDER) + "/eocar.py"
)
eocar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eocar)


def run_and_save_ocp():
    ocp = eocar.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/eocar.bioMod",
    )
    sol = ocp.solve()
    OptimalControlProgram.save(ocp, sol, "eocar_ocp_sol")


if __name__ == "__main__":
    run_and_save_ocp()

    ocp, sol = OptimalControlProgram.load(biorbd_model_path="eocar.bioMod", name="eocar_ocp_sol")
    result = ShowResult(ocp, sol)
    result.graphs()
    result.animate(40)

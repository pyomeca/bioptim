from .optimal_control_program import OptimalControlProgram, Data


def get_data_from_bo(bo_path, nb_frames=-1):
    ocp, sol = OptimalControlProgram.load(bo_path)
    return Data.get_data(ocp, sol, interpolate_nb_frames=nb_frames, concatenate=False)


def from_bo_to_bob(bo_path, bob_path):
    ocp, sol = OptimalControlProgram.load(bo_path)
    ocp.save_get_data(sol, bob_path)

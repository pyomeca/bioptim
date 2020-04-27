from biorbd_optim import OptimalControlProgram, ShowResult

ocp, sol = OptimalControlProgram.load(biorbd_model_path="eocar.bioMod", name="eocar_ocp_sol")
result = ShowResult(ocp, sol)
result.graphs()
result.animate(40)

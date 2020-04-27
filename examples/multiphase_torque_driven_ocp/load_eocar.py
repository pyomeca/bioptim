from biorbd_optim import OptimalControlProgram, ShowResult

ocp, sol = OptimalControlProgram.load(biorbd_model_path="eocar.bioMod", name="eocar_tata")
result = ShowResult(ocp, sol)
result.graphs()
result.animate(40)

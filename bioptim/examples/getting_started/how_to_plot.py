"""
This example shows how to extract the data from the Solution object and plot it using matplotlib.
"""

import matplotlib.pyplot as plt
from bioptim.examples.getting_started.basic_ocp import prepare_ocp
from bioptim import Solver, SolutionMerge, TimeAlignment
from bioptim.examples.utils import ExampleUtils

"""
If pendulum is run as a script, it will perform the optimization and animates it
"""

# --- Prepare the ocp --- #
biorbd_model_path = ExampleUtils.folder + "/models/pendulum.bioMod"
ocp = prepare_ocp(biorbd_model_path=biorbd_model_path, final_time=1, n_shooting=400, n_threads=2)

# --- Solve the ocp --- #
sol = ocp.solve(Solver.IPOPT(show_online_optim=False))


# --- Create a custom figure of the results --- #
fig, axs = plt.subplots(2, 2, figsize=(10, 15))

# Plotting the solution for decision
decision_time = sol.decision_time(to_merge=SolutionMerge.NODES, time_alignment=TimeAlignment.STATES)
decision_states = sol.decision_states(to_merge=SolutionMerge.NODES)
for i in range(2):
    axs[0, i].step(decision_time, decision_states["q"][i, :], label="Decision q", where="post")

# Retrieve stepwise states from the solution object.
stepwise_time = sol.stepwise_time(to_merge=SolutionMerge.NODES, time_alignment=TimeAlignment.STATES)
stepwise_states = sol.stepwise_states(to_merge=SolutionMerge.NODES)
for i in range(2):
    axs[1, i].plot(stepwise_time, stepwise_states["q"][i, :], label="Stepwise q")

# Plotting the solution for decision
decision_time = sol.decision_time(to_merge=SolutionMerge.NODES, time_alignment=TimeAlignment.CONTROLS)
decision_controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
for i in range(2):
    axs[0, i].step(decision_time, decision_controls["tau"][i, :], label="Decision tau", where="post")
    axs[0, i].set_xlabel("Time [s]")
    axs[0, i].grid(True)
    axs[0, i].legend()

# Retrieve stepwise states from the solution object.
stepwise_time = sol.stepwise_time(to_merge=SolutionMerge.NODES, time_alignment=TimeAlignment.CONTROLS)
stepwise_controls = sol.stepwise_controls(to_merge=SolutionMerge.NODES)
for i in range(2):
    axs[1, i].step(stepwise_time, stepwise_controls["tau"][i, :], label="Stepwise tau", where="post")
    axs[1, i].set_xlabel("Time [s]")
    axs[1, i].grid(True)
    axs[1, i].legend()

axs[0, 0].set_title("DoF 1")
axs[0, 1].set_title("DoF 2")
axs[0, 0].set_ylabel("Decision")
axs[1, 0].set_ylabel("Stepwise")
plt.show()

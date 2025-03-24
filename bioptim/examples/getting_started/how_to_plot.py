"""
This example shows how to extract the data from the Solution object and plot it using matplotlib.
"""

import matplotlib.pyplot as plt
from bioptim.examples.getting_started.pendulum import prepare_ocp
from bioptim import Solver, SolutionMerge, TimeAlignment, Shooting, SolutionIntegrator, OdeSolver, DefectType

"""
If pendulum is run as a script, it will perform the optimization and animates it
"""

# --- Prepare the ocp --- #
ocp = prepare_ocp(biorbd_model_path="models/pendulum.bioMod", final_time=1, n_shooting=400, n_threads=2, ode_solver=OdeSolver.COLLOCATION(polynomial_degree=3, defects_type=DefectType.TAU_EQUALS_INVERSE_DYNAMICS))

# --- Solve the ocp --- #
sol = ocp.solve(Solver.IPOPT(show_online_optim=False))


# --- Plot the reintegration to confirm dynamics consistency --- #
decision_time = sol.decision_time(to_merge=SolutionMerge.NODES, time_alignment=TimeAlignment.STATES)
decision_states = sol.decision_states(to_merge=SolutionMerge.NODES)
q, qdot = decision_states["q"], decision_states["qdot"]
# time_integrated, sol_integrated = sol.integrate(shooting_type=Shooting.SINGLE,
#                                 integrator=SolutionIntegrator.SCIPY_DOP853,
#                                 to_merge=SolutionMerge.NODES,
#                                 return_time=True,
#                                )
import numpy as np
sol_integrated = sol.integrate(shooting_type=Shooting.SINGLE,
                                integrator=SolutionIntegrator.SCIPY_DOP853,
                                to_merge=SolutionMerge.NODES,
                                return_time=False,
                               )
time_integrated = np.linspace(0, 1, sol_integrated["q"].shape[1])
q_integrated, qdot_integrated = sol_integrated["q"], sol_integrated["qdot"]

fig, axs = plt.subplots(4, 1, figsize=(10, 10))
for i_dof in range(2):
    axs[i_dof].plot(decision_time, q[i_dof, :], marker="o", linestyle='none', fillstyle='none', color="tab:red",
                    label="Optimal solution - q")
    axs[i_dof].plot(time_integrated, q_integrated[i_dof, :], ".", linestyle='none', color="tab:red",
                    label="Reintegration - q")
    axs[i_dof+2].plot(decision_time, qdot[i_dof, :], marker="o", linestyle='none', fillstyle='none', color="tab:blue",
                    label="Optimal solution - qdot")
    axs[i_dof+2].plot(time_integrated, qdot_integrated[i_dof, :], ".", linestyle='none', color="tab:blue",
                    label="Reintegration - qdot")
    axs[i_dof].set_title(f"{ocp.nlp[0].model.name_dof[i_dof]}")
axs[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("reintegration_.png")
plt.show()


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


# --- Plot the reintegration to confirm dynamics consistency --- #
decision_time = sol.decision_time(to_merge=SolutionMerge.NODES, time_alignment=TimeAlignment.STATES)
decision_states = sol.decision_states(to_merge=SolutionMerge.NODES)
q, qdot = decision_states["q"], decision_states["qdot"]
# time_integrated, sol_integrated = sol.integrate(shooting_type=Shooting.SINGLE,
#                                 integrator=SolutionIntegrator.SCIPY_DOP853,
#                                 to_merge=SolutionMerge.NODES,
#                                 return_time=True,
#                                )
import numpy as np
sol_integrated = sol.integrate(shooting_type=Shooting.SINGLE,
                                integrator=SolutionIntegrator.SCIPY_DOP853,
                                to_merge=SolutionMerge.NODES,
                                return_time=False,
                               )
time_integrated = np.linspace(0, 1, sol_integrated["q"].shape[1])
q_integrated, qdot_integrated = sol_integrated["q"], sol_integrated["qdot"]

fig, axs = plt.subplots(4, 1, figsize=(10, 10))
for i_dof in range(2):
    axs[i_dof].plot(decision_time, q[i_dof, :-1], marker="o", linestyle='none', fillstyle='none', color="tab:red",
                    label="Optimal solution - q")
    axs[i_dof].plot(time_integrated, q_integrated[i_dof, :], ".", linestyle='none', color="tab:red",
                    label="Reintegration - q")
    axs[i_dof+2].plot(decision_time, qdot[i_dof, :-1], marker="o", linestyle='none', fillstyle='none', color="tab:blue",
                    label="Optimal solution - qdot")
    axs[i_dof+2].plot(time_integrated, qdot_integrated[i_dof, :], ".", linestyle='none', color="tab:blue",
                    label="Reintegration - qdot")
    axs[i_dof].set_title(f"{ocp.nlp[0].model.name_dof[i_dof]}")
axs[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("reintegration_.png")
plt.show()

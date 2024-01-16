
# This code is designed to read data from a pickle file and plot various solutions for a pendulum system.
# It's likely part of a verification process for "Testing solution outputs to manually display graphs of pendulum #813".
# The script handles the extraction and processing of the pendulum system's state (q, qdot) and control (tau) variables.
# These variables are visualized for two degrees of freedom across a specified time interval.
# Matplotlib is used for plotting, with adjustments made for layout to prevent overlapping of the subplots.


import pickle
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Set the backend for Matplotlib to 'Qt5Agg'
matplotlib.use('Qt5Agg') # Use 'Qt5Agg' for PyQt5 compatibility, 'Qt6Agg' if using PyQt6

# Load data from the pickle file
with open("pendulum.pkl", "rb") as file:
    data = pickle.load(file)

# Extract and process q, qdot, and tau solutions from the data
# Extracting the first and last elements of each solution for two degrees of freedom (DOF)
q_sol_1 = [point[0][0] for point in data["q_sol"]]
q_sol_2 = [point[-1][-1] for point in data["q_sol"]]
qdot_sol_1 = [point[0][0] for point in data["qdot_sol"]]
qdot_sol_2 = [point[-1][-1] for point in data["qdot_sol"]]
tau_sol_1 = [point[0][0] for point in data["tau_sol"]]
tau_sol_2 = [point[-1][-1] for point in data["tau_sol"]]

# Append the last value to tau solutions to ensure equal length with the time array
tau_sol_1.append(tau_sol_1[-1])
tau_sol_2.append(tau_sol_2[-1])

# Extract time data and convert each DM (Differential Matrix) to a NumPy array
time = data["time"]
numpy_array = np.concatenate([np.array(t.full()).flatten()[:-1] if i < len(time) - 1 else np.array(t.full()).flatten() for i, t in enumerate(time)])

# Create a time array between 0 and 1 with 400 intervals (total 401 points)
Time = np.linspace(0, 1, 401)

# Create a figure and a set of subplots
# 3 rows for q, qdot, tau and 2 columns for each DOF
fig, axs = plt.subplots(3, 2, figsize=(10, 15))

# Plotting q solutions for both DOFs
axs[0, 0].plot(Time, q_sol_1)
axs[0, 0].set_title("q Solution for first DOF")
axs[0, 0].set_ylabel("q")
axs[0, 0].set_xlabel("Time")
axs[0, 0].grid(True)

axs[0, 1].plot(Time, q_sol_2)
axs[0, 1].set_title("q Solution for second DOF")
axs[0, 1].set_ylabel("q")
axs[0, 1].set_xlabel("Time")
axs[0, 1].grid(True)

axs[1, 0].plot(Time, qdot_sol_1)
axs[1, 0].set_title("qdot Solution for first DOF")
axs[1, 0].set_ylabel("qdot")
axs[1, 0].set_xlabel("Time")
axs[1, 0].grid(True)

axs[1, 1].plot(Time, qdot_sol_2)
axs[1, 1].set_title("qdot Solution for second DOF")
axs[1, 1].set_ylabel("qdot")
axs[1, 1].set_xlabel("Time")
axs[1, 1].grid(True)

axs[2, 0].plot(Time, tau_sol_1)
axs[2, 0].set_title("Tau Solution for first DOF")
axs[2, 0].set_ylabel("Tau")
axs[2, 0].set_xlabel("Time")
axs[2, 0].grid(True)

axs[2, 1].plot(Time, tau_sol_2)
axs[2, 1].set_title("Tau Solution for second DOF")
axs[2, 1].set_ylabel("Tau")
axs[2, 1].set_xlabel("Time")
axs[2, 1].grid(True)

plt.tight_layout()
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
plt.savefig("pendulum_graph.jpg")

# Display the plot
plt.show()
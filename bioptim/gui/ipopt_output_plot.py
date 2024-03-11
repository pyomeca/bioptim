import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from casadi import jacobian
from ..interfaces.ipopt_interface import IpoptInterface


def create_ipopt_output_plot(ocp):
    """
    This function creates the plots for the ipopt output: x, f, g, inf_pr, inf_du.
    """
    fig, axs = plt.subplots(5, 1, num="IPOPT output")
    axs[0].set_ylabel("x", fontweight="bold")
    axs[1].set_ylabel("f", fontweight="bold")
    axs[2].set_ylabel("g", fontweight="bold")
    axs[3].set_ylabel("inf_pr", fontweight="bold")
    axs[4].set_ylabel("inf_du", fontweight="bold")

    colors = get_cmap("viridis")
    for i in range(5):
        axs[i].plot([0], [0], linestyle="-", markersize=3, color=colors(i / 5))
        axs[i].get_xaxis().set_visible(False)
        axs[i].grid(True)

    fig.tight_layout()

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()

    ocp.ipopt_plots = {
        "x": [],
        "f": [],
        "g": [],
        "inf_pr": [],
        "inf_du": [],
        "plots": axs,
    }


def update_ipopt_output_plot(args, ocp):
    """
    This function updated the plots for the ipopt output: x, f, g, inf_pr, inf_du.
    We currently do not have access to the iteration number, weather we are currently in restoration, the lg(mu), the length of the current step, the alpha_du, or the alpha_pr.
    inf_pr is obtained from the maximum absolute value of the constraints.
    inf_du is obtained from the maximum absolute value of the equation 4a in the ipopt original paper.
    """

    from ..interfaces.ipopt_interface import IpoptInterface

    x = args["x"]
    print("x : ", x)
    f = args["f"]
    print("f : ", f)
    g = args["g"]
    print("g : ", g)
    print(args)

    if f != 0 and g.shape[0] != 0 and (len(ocp.ipopt_plots["f"]) == 0 or args["f"] != ocp.ipopt_plots["f"][-1]):
        print("max g : ", np.max(np.abs(g)))
        inf_pr = np.max(np.abs(args["g"]))

        lam_x = args["lam_x"]
        lam_g = args["lam_g"]
        lam_p = args["lam_p"]

        interface = IpoptInterface(ocp)

        v = interface.ocp.variables_vector

        all_objectives = interface.dispatch_obj_func()
        all_g, all_g_bounds = interface.dispatch_bounds(
            include_g=True, include_g_internal=True, include_g_implicit=True
        )

        grad_f = jacobian(all_objectives, v)
        grad_g = jacobian(all_g, v)

        eq_4a = grad_f + grad_g @ lam_g - lam_x
        inf_du = np.max(np.abs(eq_4a))

        ocp.ipopt_plots["x"].append(x)
        ocp.ipopt_plots["f"].append(f)
        ocp.ipopt_plots["g"].append(g)
        ocp.ipopt_plots["inf_pr"].append(inf_pr)
        ocp.ipopt_plots["inf_du"].append(inf_du)

        ocp.ipopt_plots.plots[0].set_ydata(ocp.ipopt_plots["x"])
        ocp.ipopt_plots.plots[1].set_ydata(ocp.ipopt_plots["f"])
        ocp.ipopt_plots.plots[2].set_ydata(ocp.ipopt_plots["g"])
        ocp.ipopt_plots.plots[3].set_ydata(ocp.ipopt_plots["inf_pr"])
        ocp.ipopt_plots.plots[4].set_ydata(ocp.ipopt_plots["inf_du"])

        for i in range(5):
            ocp.ipopt_plots.plots[i].set_xdata(range(len(ocp.ipopt_plots["x"])))

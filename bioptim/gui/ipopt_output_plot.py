import pickle

import numpy as np
from casadi import jacobian, gradient, sum1, Function
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap

from ..misc.parameters_types import Str, Int


def create_ipopt_output_plot(ocp, interface):
    """
    This function creates the plots for the ipopt output: f, g, inf_pr, inf_du.
    """
    ipopt_fig, axs = plt.subplots(3, 1, num="IPOPT output")
    axs[0].set_ylabel("f", fontweight="bold")
    axs[1].set_ylabel("inf_pr", fontweight="bold")
    axs[2].set_ylabel("inf_du", fontweight="bold")

    plots = []
    colors = get_cmap("viridis")
    for i in range(3):
        plot = axs[i].plot([0], [1], linestyle="-", marker=".", color="k")
        plots.append(plot[0])
        axs[i].grid(True)
        axs[i].set_yscale("log")

    plot = axs[2].plot([0], [1], linestyle="-", marker=".", color=colors(0.1), label="grad_f")
    plots.append(plot[0])
    plot = axs[2].plot([0], [1], linestyle="-", marker=".", color=colors(0.5), label="grad_g")
    plots.append(plot[0])
    plot = axs[2].plot([0], [1], linestyle="-", marker=".", color=colors(0.9), label="lam_x")
    plots.append(plot[0])
    axs[2].legend()

    ipopt_fig.tight_layout()

    try:
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
    except:
        pass

    v_sym = interface.ocp.variables_vector

    all_objectives = interface.dispatch_obj_func()
    all_g, all_g_bounds = interface.dispatch_bounds(include_g=True, include_g_internal=True)

    grad_f_func = Function("grad_f", [v_sym], [gradient(sum1(all_objectives), v_sym)])
    grad_g_func = Function("grad_g", [v_sym], [jacobian(all_g, v_sym).T])

    ocp.ipopt_plots = {
        "grad_f_func": grad_f_func,
        "grad_g_func": grad_g_func,
        "f": [],
        "inf_pr": [],
        "inf_du": [],
        "grad_f": [],
        "grad_g": [],
        "lam_x": [],
        "axs": axs,
        "plots": plots,
        "ipopt_fig": ipopt_fig,
    }


def update_ipopt_output_plot(args, ocp):
    """
    This function updated the plots for the ipopt output: x, f, g, inf_pr, inf_du.
    We currently do not have access to the iteration number, weather we are currently in restoration, the lg(mu), the length of the current step, the alpha_du, or the alpha_pr.
    inf_pr is obtained from the maximum absolute value of the constraints.
    inf_du is obtained from the maximum absolute value of the equation 4a in the ipopt original paper.
    """

    x = args["x"]
    f = args["f"]
    lam_x = args["lam_x"]
    lam_g = args["lam_g"]
    lam_p = args["lam_p"]

    inf_pr = np.max(np.abs(args["g"]))

    grad_f_func = ocp.ipopt_plots["grad_f_func"]
    grad_g_func = ocp.ipopt_plots["grad_g_func"]

    grad_f = grad_f_func(x)
    grad_g_lam = grad_g_func(x) @ lam_g
    eq_4a = np.max(np.abs(grad_f + grad_g_lam - lam_x))
    inf_du = np.max(np.abs(eq_4a))

    ocp.ipopt_plots["f"].append(float(f))
    ocp.ipopt_plots["inf_pr"].append(float(inf_pr))
    ocp.ipopt_plots["inf_du"].append(float(inf_du))
    ocp.ipopt_plots["grad_f"].append(float(np.max(np.abs(grad_f))))
    ocp.ipopt_plots["grad_g"].append(float(np.max(np.abs(grad_g_lam))))
    ocp.ipopt_plots["lam_x"].append(float(np.max(np.abs(lam_x))))

    ocp.ipopt_plots["plots"][0].set_ydata(ocp.ipopt_plots["f"])
    ocp.ipopt_plots["plots"][1].set_ydata(ocp.ipopt_plots["inf_pr"])
    ocp.ipopt_plots["plots"][2].set_ydata(ocp.ipopt_plots["inf_du"])
    ocp.ipopt_plots["plots"][3].set_ydata(ocp.ipopt_plots["grad_f"])
    ocp.ipopt_plots["plots"][4].set_ydata(ocp.ipopt_plots["grad_g"])
    ocp.ipopt_plots["plots"][5].set_ydata(ocp.ipopt_plots["lam_x"])

    ocp.ipopt_plots["axs"][0].set_ylim(np.min(ocp.ipopt_plots["f"]), np.max(ocp.ipopt_plots["f"]))
    ocp.ipopt_plots["axs"][1].set_ylim(np.min(ocp.ipopt_plots["inf_pr"]), np.max(ocp.ipopt_plots["inf_pr"]))
    ocp.ipopt_plots["axs"][2].set_ylim(
        np.min(
            np.array(
                [
                    1e8,
                    np.min(np.abs(ocp.ipopt_plots["inf_du"])),
                    np.min(np.abs(ocp.ipopt_plots["grad_f"])),
                    np.min(np.abs(ocp.ipopt_plots["grad_g"])),
                    np.min(np.abs(ocp.ipopt_plots["lam_x"])),
                ]
            )
        ),
        np.max(
            np.array(
                [
                    1e-8,
                    np.max(np.abs(ocp.ipopt_plots["inf_du"])),
                    np.max(np.abs(ocp.ipopt_plots["grad_f"])),
                    np.max(np.abs(ocp.ipopt_plots["grad_g"])),
                    np.max(np.abs(ocp.ipopt_plots["lam_x"])),
                ]
            )
        ),
    )

    for i in range(6):
        ocp.ipopt_plots["plots"][i].set_xdata(range(len(ocp.ipopt_plots["f"])))
    for i in range(3):
        ocp.ipopt_plots["axs"][i].set_xlim(0, len(ocp.ipopt_plots["f"]))

    ocp.ipopt_plots["ipopt_fig"].canvas.draw()


def save_ipopt_output(args, save_ipopt_iterations_info):
    """
    This function saves the ipopt outputs: x, f, g, lam_x, lam_g, lam_p every nb_iter_save iterations.
    """
    f = args["f"]

    if len(save_ipopt_iterations_info.f_list) != 0 and save_ipopt_iterations_info.f_list[-1] == f:
        return

    save_ipopt_iterations_info.f_list += [f]
    save_ipopt_iterations_info.current_iter += 1

    if save_ipopt_iterations_info.current_iter % save_ipopt_iterations_info.nb_iter_save != 0:
        return
    else:
        x = args["x"]
        g = args["g"]
        lam_x = args["lam_x"]
        lam_g = args["lam_g"]
        lam_p = args["lam_p"]

        save_path = (
            save_ipopt_iterations_info.path_to_results
            + save_ipopt_iterations_info.result_file_name
            + "_"
            + str(save_ipopt_iterations_info.current_iter)
            + ".pkl"
        )
        with open(save_path, "wb") as file:
            pickle.dump({"x": x, "f": f, "g": g, "lam_x": lam_x, "lam_g": lam_g, "lam_p": lam_p}, file)


class SaveIterationsInfo:
    """
    This class is used to store the ipopt outputs save info.
    """

    def __init__(self, path_to_results: Str, result_file_name: Str, nb_iter_save: Int):

        if not isinstance(path_to_results, str) or len(path_to_results) == 0:
            raise ValueError("path_to_results should be a non-empty string")
        if path_to_results[-1] != "/":
            path_to_results += "/"

        if not isinstance(result_file_name, str) or len(result_file_name) == 0:
            raise ValueError("result_file_name should be a non-empty string")
        if result_file_name[-4:] == ".pkl":
            result_file_name = result_file_name[:-4]
        result_file_name.replace(".", "-")

        if not isinstance(nb_iter_save, int) or nb_iter_save <= 0:
            raise ValueError("nb_iter_save should be a positive integer")

        self.path_to_results = path_to_results
        self.result_file_name = result_file_name
        self.nb_iter_save = nb_iter_save
        self.current_iter = 0
        self.f_list = []

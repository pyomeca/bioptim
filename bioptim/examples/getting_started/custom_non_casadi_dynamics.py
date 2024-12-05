"""
TODO: Explain what is this example about

This example is similar to the getting_started/pendulum.py example, but the dynamics are computed using a non-casadi 
based model. This is useful when the dynamics are computed using a different library (e.g. TensorFlow, PyTorch, etc.)
"""

import biorbd
from bioptim import (
    OptimalControlProgram,
    DynamicsFcn,
    Objective,
    ObjectiveFcn,
    BoundsList,
    OdeSolver,
    OdeSolverBase,
    PhaseDynamics,
    ControlType,
    InitialGuessList,
    Dynamics,
    CasadiFunctionInterface,
    BiorbdModel,
    PenaltyController,
    Node,
)
from casadi import Function, jacobian, MX, DM

import numpy as np


class CasadiFunctionInterfaceTest(CasadiFunctionInterface):
    """
    This example implements a somewhat simple 5x1 function, with x and y inputs (x => 3x1; y => 4x1) of the form
        f(x, y) = np.array(
            [
                x[0] * y[1] + y[0] * y[0],
                x[1] * x[1] + 2 * y[1],
                x[0] * x[1] * x[2],
                x[2] * x[1] + 2 * y[3] * y[2],
                y[0] * y[1] * y[2] * y[3],
            ]
        )

    It implements the equation (5x1) and the jacobians for the inputs x (5x3) and y (5x4).
    """

    def __init__(self, model, opts={}):
        super(CasadiFunctionInterfaceTest, self).__init__("CasadiFunctionInterfaceTest", opts)

    def inputs_len(self) -> list[int]:
        return [2, 2]

    def outputs_len(self) -> list[int]:
        return [2]

    def function(self, *args):
        x, y = args
        x = np.array(x)[:, 0]
        y = np.array(y)[:, 0]
        return np.array([x[0] * y[1] + x[0] * y[0] * y[0], x[1] * x[1] + 2 * y[1]])

    def jacobians(self, *args):
        x, y = args
        x = np.array(x)[:, 0]
        y = np.array(y)[:, 0]
        jacobian_x = np.array([[y[1] + y[0] * y[0], 0, 0], [0, 2 * x[1], 0]])
        jacobian_y = np.array([[x[0] * 2 * y[0], x[0], 0, 0], [0, 2, 0, 0]])
        return [jacobian_x, jacobian_y]


class ForwardDynamicsInterface(CasadiFunctionInterface):
    def __init__(self, model: BiorbdModel, opts={}):
        self.non_casadi_model = biorbd.Model(model.path)
        super(ForwardDynamicsInterface, self).__init__("ForwardDynamicsInterface", opts)

    def inputs_len(self) -> list[int]:
        return [1]

    def outputs_len(self) -> list[int]:
        return [1]

    def function(self, *args):
        return [args[0]]  # self.non_casadi_model.ForwardDynamics(*self.mx_in()[:3])

    def jacobians(self, *args):
        return [0, 0, DM(1), 0, 0]


def custom_func_track_markers(controller: PenaltyController) -> MX:
    return controller.model.custom_interface(controller.states["q"].cx, controller.controls["tau"].cx)


def prepare_ocp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    use_sx: bool = False,
    n_threads: int = 1,
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
    expand_dynamics: bool = True,
    control_type: ControlType = ControlType.CONSTANT,
) -> OptimalControlProgram:
    """
    The initialization of an ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the biorbd model
    final_time: float
        The time in second required to perform the task
    n_shooting: int
        The number of shooting points to define int the direct multiple shooting program
    ode_solver: OdeSolverBase = OdeSolver.RK4()
        Which type of OdeSolver to use
    use_sx: bool
        If the SX variable should be used instead of MX (can be extensive on RAM)
    n_threads: int
        The number of threads to use in the paralleling (1 = no parallel computing)
    phase_dynamics: PhaseDynamics
        If the dynamics equation within a phase is unique or changes at each node.
        PhaseDynamics.SHARED_DURING_THE_PHASE is much faster, but lacks the capability to have changing dynamics within
        a phase. A good example of when PhaseDynamics.ONE_PER_NODE should be used is when different external forces
        are applied at each node
    expand_dynamics: bool
        If the dynamics function should be expanded. Please note, this will solve the problem faster, but will slow down
        the declaration of the OCP, so it is a trade-off. Also depending on the solver, it may or may not work
        (for instance IRK is not compatible with expanded dynamics)
    control_type: ControlType
        The type of the controls

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    bio_model = BiorbdModel(biorbd_model_path)
    bio_model.custom_interface = CasadiFunctionInterfaceTest(bio_model)

    # Add objective functions
    objective_functions = Objective(custom_func_track_markers, custom_type=ObjectiveFcn.Mayer, node=Node.START)

    # Dynamics
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN, expand_dynamics=expand_dynamics, phase_dynamics=phase_dynamics)

    # Path bounds
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["q"][:, [0, -1]] = 0  # Start and end at 0...
    x_bounds["q"][1, -1] = 3.14  # ...but end with pendulum 180 degrees rotated
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:, [0, -1]] = 0  # Start and end without any velocity

    # Initial guess (optional since it is 0, we show how to initialize anyway)
    x_init = InitialGuessList()
    x_init["q"] = [0] * bio_model.nb_q
    x_init["qdot"] = [0] * bio_model.nb_qdot

    # Define control path bounds
    n_tau = bio_model.nb_tau
    u_bounds = BoundsList()
    u_bounds["tau"] = [-100] * n_tau, [100] * n_tau  # Limit the strength of the pendulum to (-100 to 100)...
    u_bounds["tau"][1, :] = 0  # ...but remove the capability to actively rotate

    # Initial guess (optional since it is 0, we show how to initialize anyway)
    u_init = InitialGuessList()
    u_init["tau"] = [0] * n_tau

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        ode_solver=ode_solver,
        use_sx=use_sx,
        n_threads=n_threads,
        control_type=control_type,
    )


def main():
    """
    If pendulum is run as a script, it will perform the optimization and animates it
    """

    # --- Prepare the ocp --- #
    ocp = prepare_ocp(biorbd_model_path="models/pendulum.bioMod", final_time=1, n_shooting=400, n_threads=2)

    # --- Solve the ocp --- #
    sol = ocp.solve()

    # --- Show the results graph --- #
    # sol.print_cost()
    sol.graphs(show_bounds=True, save_name="results.png")


if __name__ == "__main__":
    main()


######## OCP FAST ########
# from casadi import *

# T = 10.0  # Time horizon
# N = 20  # number of control intervals

# # Declare model variables
# x1 = MX.sym("x1")
# x2 = MX.sym("x2")
# x = vertcat(x1, x2)
# u = MX.sym("u")

# # Model equations
# xdot = vertcat((1 - x2**2) * x1 - x2 + u, x1)


# # Formulate discrete time dynamics
# if False:
#     # CVODES from the SUNDIALS suite
#     dae = {"x": x, "p": u, "ode": xdot}
#     opts = {"tf": T / N}
#     F = integrator("F", "cvodes", dae, opts)
# else:
#     # Fixed step Runge-Kutta 4 integrator
#     M = 4  # RK4 steps per interval
#     DT = T / N / M
#     f = Function("f", [x, u], [xdot])
#     X0 = MX.sym("X0", 2)
#     U = MX.sym("U")
#     X = X0
#     Q = 0
#     for j in range(M):
#         k1 = f(X, U)
#         k2 = f(X + DT / 2 * k1, U)
#         k3 = f(X + DT / 2 * k2, U)
#         k4 = f(X + DT * k3, U)
#         X = X + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
#     F = Function("F", [X0, U], [X], ["x0", "p"], ["xf"])

# # Start with an empty NLP
# w = []
# w0 = []
# lbw = []
# ubw = []
# g = []
# lbg = []
# ubg = []

# # "Lift" initial conditions
# Xk = MX.sym("X0", 2)
# w += [Xk]
# lbw += [0, 1]
# ubw += [0, 1]
# w0 += [0, 1]

# # Formulate the NLP
# for k in range(N):
#     # New NLP variable for the control
#     Uk = MX.sym("U_" + str(k))
#     w += [Uk]
#     lbw += [-1]
#     ubw += [1]
#     w0 += [0]

#     # Integrate till the end of the interval
#     Fk = F(x0=Xk, p=Uk)
#     Xk_end = Fk["xf"]

#     # New NLP variable for state at end of interval
#     Xk = MX.sym("X_" + str(k + 1), 2)
#     w += [Xk]
#     lbw += [-0.25, -inf]
#     ubw += [inf, inf]
#     w0 += [0, 0]

#     # Add equality constraint
#     g += [Xk_end - Xk]
#     lbg += [0, 0]
#     ubg += [0, 0]

# nd = N + 1

# import gpflow
# import time

# from tensorflow_casadi import TensorFlowEvaluator


# class GPR(TensorFlowEvaluator):
#     def __init__(self, session, opts={}):
#         X = tf.compat.v1.placeholder(shape=(1, nd), dtype=np.float64)
#         mean = tf.reshape(tf.reduce_mean(X), (1, 1))
#         TensorFlowEvaluator.__init__(self, [X], [mean], session, opts)
#         self.counter = 0
#         self.time = 0

#     def eval(self, arg):
#         self.counter += 1
#         t0 = time.time()
#         ret = TensorFlowEvaluator.eval(self, arg)
#         self.time += time.time() - t0
#         return [ret]


# import tensorflow as tf

# with tf.compat.v1.Session() as session:
#     GPR = GPR(session)

#     w = vertcat(*w)

#     # Create an NLP solver
#     prob = {"f": sum1(GPR(w[0::3])), "x": w, "g": vertcat(*g)}
#     options = {"ipopt": {"hessian_approximation": "limited-memory"}}
#     solver = nlpsol("solver", "ipopt", prob, options)

#     # Solve the NLP
#     sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

# print("Ncalls", GPR.counter)
# print("Total time [s]", GPR.time)
# w_opt = sol["x"].full().flatten()

# # Plot the solution
# x1_opt = w_opt[0::3]
# x2_opt = w_opt[1::3]
# u_opt = w_opt[2::3]

# tgrid = [T / N * k for k in range(N + 1)]
# import matplotlib.pyplot as plt

# plt.figure(1)
# plt.clf()
# plt.plot(tgrid, x1_opt, "--")
# plt.plot(tgrid, x2_opt, "-")
# plt.step(tgrid, vertcat(DM.nan(1), u_opt), "-.")
# plt.xlabel("t")
# plt.legend(["x1", "x2", "u"])
# plt.grid()
# plt.show()


#
#
#
######### TENSORFLOW CASADI #########
# import casadi
# import tensorflow as tf


# class TensorFlowEvaluator(casadi.Callback):
#     def __init__(self, t_in, t_out, session, opts={}):
#         """
#         t_in: list of inputs (tensorflow placeholders)
#         t_out: list of outputs (tensors dependeant on those placeholders)
#         session: a tensorflow session
#         """
#         casadi.Callback.__init__(self)
#         assert isinstance(t_in, list)
#         self.t_in = t_in
#         assert isinstance(t_out, list)
#         self.t_out = t_out
#         self.construct("TensorFlowEvaluator", opts)
#         self.session = session
#         self.refs = []

#     def get_n_in(self):
#         return len(self.t_in)

#     def get_n_out(self):
#         return len(self.t_out)

#     def get_sparsity_in(self, i):
#         return casadi.Sparsity.dense(*self.t_in[i].get_shape().as_list())

#     def get_sparsity_out(self, i):
#         return casadi.Sparsity.dense(*self.t_out[i].get_shape().as_list())

#     def eval(self, arg):
#         # Associate each tensorflow input with the numerical argument passed by CasADi
#         d = dict((v, arg[i].toarray()) for i, v in enumerate(self.t_in))
#         # Evaluate the tensorflow expressions
#         ret = self.session.run(self.t_out, feed_dict=d)
#         return ret

#     # Vanilla tensorflow offers just the reverse mode AD
#     def has_reverse(self, nadj):
#         return nadj == 1

#     def get_reverse(self, nadj, name, inames, onames, opts):
#         # Construct tensorflow placeholders for the reverse seeds
#         adj_seed = [
#             tf.compat.v1.placeholder(shape=self.sparsity_out(i).shape, dtype=tf.float64) for i in range(self.n_out())
#         ]
#         # Construct the reverse tensorflow graph through 'gradients'
#         grad = tf.gradients(self.t_out, self.t_in, grad_ys=adj_seed)
#         # Create another TensorFlowEvaluator object
#         callback = TensorFlowEvaluator(self.t_in + adj_seed, grad, self.session)
#         # Make sure you keep a reference to it
#         self.refs.append(callback)

#         # Package it in the nominal_in+nominal_out+adj_seed form that CasADi expects
#         nominal_in = self.mx_in()
#         nominal_out = self.mx_out()
#         adj_seed = self.mx_out()
#         return casadi.Function(
#             name, nominal_in + nominal_out + adj_seed, callback.call(nominal_in + adj_seed), inames, onames
#         )


# if __name__ == "__main__":
#     from casadi import *

#     a = tf.compat.v1.placeholder(shape=(2, 2), dtype=tf.float64)
#     b = tf.compat.v1.placeholder(shape=(2, 1), dtype=tf.float64)

#     y = tf.matmul(tf.sin(a), b)

#     with tf.compat.v1.Session() as session:
#         f_tf = TensorFlowEvaluator([a, b], [y], session)

#         a = MX.sym("a", 2, 2)
#         b = MX.sym("a", 2, 1)
#         y = f_tf(a, b)
#         yref = mtimes(sin(a), b)

#         f = Function("f", [a, b], [y])
#         fref = Function("f", [a, b], [yref])

#         print(f(DM([[1, 2], [3, 4]]), DM([[1], [3]])))
#         print(fref(DM([[1, 2], [3, 4]]), DM([[1], [3]])))

#         f = Function("f", [a, b], [jacobian(y, a)])
#         fref = Function("f", [a, b], [jacobian(yref, a)])
#         print(f(DM([[1, 2], [3, 4]]), DM([[1], [3]])))
#         print(fref(DM([[1, 2], [3, 4]]), DM([[1], [3]])))

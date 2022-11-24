"""
This script doesn't use biorbd
This an example of how to use bioptim to solve a simple pendulum problem
"""
import numpy as np
from casadi import sin, MX, vertcat

from bioptim import (
    OptimalControlProgram,
    DynamicsFcn,
    Dynamics,
    Bounds,
    InterpolationType,
    InitialGuess,
    ObjectiveFcn,
    Objective,
    OdeSolver,
    CostType,
    Solver,
    BiorbdModel,
    Model,
    NonLinearProgram,
    ConfigureProblem,
    DynamicsList,
    DynamicsEvaluation,
)


class MyModel(Model):
    """This is a custom model that inherits from bioptim.Model"""

    def __init__(self):
        self.com = MX(np.array([-0.0005, 0.0688, -0.9542]))

    def DeepCopy(self, *args):
        raise NotImplementedError("DeepCopy is not implemented")

    def AddSegment(self, *args):
        raise NotImplementedError("AddSegment is not implemented")

    def getGravity(self):
        raise -9.81

    def setGravity(self, newGravity):
        raise NotImplementedError("setGravity is not implemented")

    def getBodyBiorbdId(self, segmentName):
        raise NotImplementedError("getBodyBiorbdId is not implemented")

    def getBodyRbdlId(self, segmentName):
        raise NotImplementedError("getBodyRbdlId is not implemented")

    def getBodyRbdlIdToBiorbdId(self, idx):
        raise NotImplementedError("getBodyRbdlIdToBiorbdId is not implemented")

    def getBodyBiorbdIdToRbdlId(self, idx):
        raise NotImplementedError("getBodyBiorbdIdToRbdlId is not implemented")

    def getDofSubTrees(self):
        raise NotImplementedError("getDofSubTrees is not implemented")

    def getDofIndex(self, SegmentName, dofName):
        raise NotImplementedError("getDofIndex is not implemented")

    def nbGeneralizedTorque(self):
        return 1

    def nbSegment(self):
        return 1

    def nbQuat(self):
        return 0

    def nbQ(self):
        return 1

    def nbQdot(self):
        return 1

    def nbQddot(self):
        return 1

    def nbRoot(self):
        return 0

    def updateSegmentCharacteristics(self, idx, characteristics):
        raise NotImplementedError("updateSegmentCharacteristics is not implemented")

    def segment(self, *args):
        raise NotImplementedError("segment is not implemented")

    def segments(self, i):
        raise NotImplementedError("updateSegmentCharacteristics is not implemented")

    def dispatchedForce(self, *args):
        raise NotImplementedError("updateSegmentCharacteristics is not implemented")

    def UpdateKinematicsCustom(self, Q=None, Qdot=None, Qddot=None):
        raise NotImplementedError("updateSegmentCharacteristics is not implemented")

    def allGlobalJCS(self, *args):
        raise NotImplementedError("updateSegmentCharacteristics is not implemented")

    def globalJCS(self, *args):
        raise NotImplementedError("updateSegmentCharacteristics is not implemented")

    def localJCS(self, *args):
        raise NotImplementedError("localJCS is not implemented")

    def projectPoint(self, *args):
        raise NotImplementedError("projectPoint is not implemented")

    def projectPointJacobian(self, *args):
        raise NotImplementedError("projectPointJacobian is not implemented")

    def mass(self):
        return 1

    def CoM(self, Q, updateKin=True):
        raise NotImplementedError("CoM is not implemented")

    def CoMbySegmentInMatrix(self, Q, updateKin=True):
        raise NotImplementedError("CoMbySegmentInMatrix is not implemented")

    def CoMbySegment(self, *args):
        raise NotImplementedError("CoMbySegment is not implemented")

    def CoMdot(self, Q, Qdot, updateKin=True):
        raise NotImplementedError("CoMdot is not implemented")

    def CoMddot(self, Q, Qdot, Qddot, updateKin=True):
        raise NotImplementedError("CoMddot is not implemented")

    def CoMdotBySegment(self, *args):
        raise NotImplementedError("CoMdotBySegment is not implemented")

    def CoMddotBySegment(self, *args):
        raise NotImplementedError("CoMddotBySegment is not implemented")

    def CoMJacobian(self, Q, updateKin=True):
        raise NotImplementedError("CoMJacobian is not implemented")

    def meshPoints(self, *args):
        raise NotImplementedError("meshPoints is not implemented")

    def meshPointsInMatrix(self, Q, updateKin=True):
        raise NotImplementedError("meshPointsInMatrix is not implemented")

    def meshFaces(self, *args):
        raise NotImplementedError("meshFaces is not implemented")

    def mesh(self, *args):
        raise NotImplementedError("mesh is not implemented")

    def angularMomentum(self, Q, Qdot, updateKin=True):
        raise NotImplementedError("angularMomentum is not implemented")

    def massMatrix(self, Q, updateKin=True):
        return self.mass() * self.com[2] ** 2

    def massMatrixInverse(self, Q, updateKin=True):
        raise NotImplementedError("massMatrixInverse is not implemented")

    def CalcAngularMomentum(self, *args):
        raise NotImplementedError("CalcAngularMomentum is not implemented")

    def CalcSegmentsAngularMomentum(self, *args):
        raise NotImplementedError("CalcSegmentsAngularMomentum is not implemented")

    def bodyAngularVelocity(self, Q, Qdot, updateKin=True):
        raise NotImplementedError("bodyAngularVelocity is not implemented")

    def CalcMatRotJacobian(self, Q, segmentIdx, rotation, G, updateKin):
        raise NotImplementedError("CalcMatRotJacobian is not implemented")

    def JacobianSegmentRotMat(self, Q, segmentIdx, updateKin):
        raise NotImplementedError("JacobianSegmentRotMat is not implemented")

    def computeQdot(self, Q, QDot, k_stab=1):
        return QDot

    def segmentAngularVelocity(self, Q, Qdot, idx, updateKin=True):
        raise NotImplementedError("segmentAngularVelocity is not implemented")

    def CalcKineticEnergy(self, Q, QDot, updateKin=True):
        raise NotImplementedError("CalcKineticEnergy is not implemented")

    def CalcPotentialEnergy(self, Q, updateKin=True):
        raise NotImplementedError("CalcPotentialEnergy is not implemented")

    def nameDof(self):
        return ["rotx"]

    def contactNames(self):
        raise NotImplementedError("contactNames is not implemented")

    def nbSoftContacts(self):
        return 0

    def softContactNames(self):
        raise NotImplementedError("softContactNames is not implemented")

    def muscleNames(self):
        raise NotImplementedError("muscleNames is not implemented")

    def torque(self, tau_activations, q, qdot):
        raise NotImplementedError("torque is not implemented")

    def ForwardDynamicsFreeFloatingBase(self, q, qdot, qddot_joints):
        raise NotImplementedError("ForwardDynamicsFreeFloatingBase is not implemented")

    def ForwardDynamics(self, q, qdot, tau, fext=None, f_contacts=None):
        return (tau - self.mass() * -9.81 * self.com[2] * sin(q)) / (self.mass() * self.com[2] ** 2)

    def ForwardDynamicsConstraintsDirect(self, *args):
        raise NotImplementedError("ForwardDynamicsConstraintsDirect is not implemented")

    def InverseDynamics(self, q, qdot, qddot, f_ext=None, f_contacts=None):
        return self.mass() * self.com[2] ** 2 * qddot[0] + self.mass() * -9.81 * self.com[2] * sin(q[0])

    def NonLinearEffect(self, Q, QDot, f_ext=None, f_contacts=None):
        raise NotImplementedError("NonLinearEffect is not implemented")

    def ContactForcesFromForwardDynamicsConstraintsDirect(self, Q, QDot, Tau, f_ext=None):
        raise NotImplementedError("ContactForcesFromForwardDynamicsConstraintsDirect is not implemented")

    def bodyInertia(self, Q, updateKin=True):
        raise NotImplementedError("bodyInertia is not implemented")

    def ComputeConstraintImpulsesDirect(self, Q, QDotPre):
        raise NotImplementedError("ComputeConstraintImpulsesDirect is not implemented")

    def checkGeneralizedDimensions(self, Q=None, Qdot=None, Qddot=None, torque=None):
        raise NotImplementedError("checkGeneralizedDimensions is not implemented")

    def stateSet(self):
        raise NotImplementedError("stateSet is not implemented")

    def activationDot(self, muscle_states):
        raise NotImplementedError("activationDot is not implemented")

    def muscularJointTorque(self, muscle_states, q, qdot):
        raise NotImplementedError("muscularJointTorque is not implemented")

    def getConstraints(self):
        raise NotImplementedError("markers is not implemented")

    def markers(self, Q, updateKin=True):
        raise NotImplementedError("markers is not implemented")

    def nbRigidContacts(self):
        return 0

    def path(self):
        return self.model.path()


def custom_dynamics(
    states: MX,
    controls: MX,
    parameters: MX,
    nlp: NonLinearProgram,
) -> DynamicsEvaluation:
    """
    Parameters
    ----------
    states: Union[MX, SX]
        The state of the system
    controls: Union[MX, SX]
        The controls of the system
    parameters: Union[MX, SX]
        The parameters acting on the system
    nlp: NonLinearProgram
        A reference to the phase
    Returns
    -------
    The derivative of the states in the tuple[Union[MX, SX]] format
    """

    return DynamicsEvaluation(
        dxdt=vertcat(states[0], nlp.model.ForwardDynamics(states[0], states[1], controls[0])), defects=None
    )


def custom_configure_my_dynamics(ocp: OptimalControlProgram, nlp: NonLinearProgram):
    """
    Tell the program which variables are states and controls.
    The user is expected to use the ConfigureProblem.configure_xxx functions.
    Parameters
    ----------
    ocp: OptimalControlProgram
        A reference to the ocp
    nlp: NonLinearProgram
        A reference to the phase
    """

    ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True)

    ConfigureProblem.configure_dynamics_function(ocp, nlp, custom_dynamics, expand=True)


def prepare_ocp(
    model: Model,
    final_time: float,
    n_shooting: int,
    ode_solver: OdeSolver = OdeSolver.RK4(n_integration_steps=1),
    use_sx: bool = True,
    n_threads: int = 1,
) -> OptimalControlProgram:
    """
    The initialization of an ocp

    Parameters
    ----------
    model: Model
        The path to the biorbd model
    final_time: float
        The time in second required to perform the task
    n_shooting: int
        The number of shooting points to define int the direct multiple shooting program
    ode_solver: OdeSolver = OdeSolver.RK4()
        Which type of OdeSolver to use
    use_sx: bool
        If the SX variable should be used instead of MX (can be extensive on RAM)
    n_threads: int
        The number of threads to use in the paralleling (1 = no parallel computing)

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    # Add objective functions
    objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")

    # Dynamics
    # dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN)
    # Declare the dynamics
    dynamics = DynamicsList()
    dynamics.add(custom_configure_my_dynamics, dynamic_function=custom_dynamics)

    # Path constraint
    x_bounds = Bounds(
        min_bound=np.array([[0, -6.28, 3.14], [0, -20, 0]]),
        max_bound=np.array([[0, 6.28, 3.14], [0, 20, 0]]),
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
    )
    # x_bounds[:, [0, -1]] = 0
    # x_bounds[0, -1] = 3.14
    # x_bounds[0, 0] = 1
    # x_bounds[0, -1] = 1

    # Initial guess
    n_q = model.nbQ()
    n_qdot = model.nbQdot()
    x_init = InitialGuess([20] * (n_q + n_qdot))

    # Define control path constraint
    n_tau = model.nbGeneralizedTorque()
    tau_min, tau_max, tau_init = -100, 100, 20
    u_bounds = Bounds([tau_min] * n_tau, [tau_max] * n_tau)

    u_init = InitialGuess([tau_init] * n_tau)

    return OptimalControlProgram(
        model,
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
    )


def main():
    """
    If pendulum is run as a script, it will perform the optimization and animates it
    """

    # --- Prepare the ocp --- #
    ocp = prepare_ocp(model=MyModel(), final_time=1, n_shooting=30)

    # Custom plots
    ocp.add_plot_penalty(CostType.ALL)

    # --- Print ocp structure --- #
    ocp.print(to_console=True, to_graph=False)

    # --- Solve the ocp --- #
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_maximum_iterations(100)

    sol = ocp.solve(solver=solver)
    sol.graphs(show_bounds=True)

    # --- Show the results in a bioviz animation --- #
    sol.detailed_cost_values()
    sol.print_cost()
    # sol.animate(n_frames=100)


if __name__ == "__main__":
    main()

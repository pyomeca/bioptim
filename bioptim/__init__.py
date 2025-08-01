"""
Bioptim is a direct multiple shooting optimal control program (ocp) framework for biomechanics based on biorbd.
It provides all the necessary tools to declare, modify, augment and solve an ocp.

Depending on the solver requested, normal and multiphase program, moving horizon estimator (MHE),
nonlinear model predictive control (NMPC), free time and parameter optimization, tracking, minimizing and maximizing
programs, from different joint torques and muscle driven dynamics can be solved.

Thanks to the CasADi backend of biorbd that allows to algorithmically differentiate the code, the descending gradient
algorithms are solving these highly nonlinear optimization program super efficiently. The Ipopt and ACADOS solvers
provide fast and robust solution for the optimal control program.

The examples provided cover a large spectrum of what bioptim is capable of, but is definitely not limited to. You will
find trivial example such balancing a pendulum upward up to arm movements controlled by the muscle EMG. The framework
has also been used gait tracking, hand prosthetic simulations, jumping simulation and violin movement optimization.
The latter being especially challenging because of the closed-loop involved in the kinematics.

So don't wait any further challenge biorbd to find you the best movement!
-------------------------------------------------------------------------


# --- The main interface --- #
OptimalControlProgram
    The main class to define an ocp. This class prepares the full program and gives all the needed interface to
    modify and solve the program
NonLinearProgram
    A nonlinear program that describes a phase in the ocp
Solution
    Data manipulation, showing and storage


# --- Some useful options --- #
Axis
    Selection of valid axis (X, Y or Z)
Node
    Selection of valid node
InterpolationType
    Selection of valid type of interpolation
OdeSolver
    Selection of valid integrator
PlotType
    Selection of valid plots
Solver
    Selection of valid nonlinear solvers
ControlType
    Selection of valid controls
CostType
    Selection of valid penalty type
IPOPT
    Selection of IPOPT options
ACADOS
    Selection of ACADOS options

# --- Managing the dynamics --- #
ConfigureProblem
    Dynamics configuration for the most common ocp
DynamicsOptions
    A placeholder for the chosen dynamics by the user
DynamicsOptionsList
    A list of DynamicsOptions if more than one is required, typically when more than one phases are declared
DynamicsEvaluation
    A placeholder for the dynamics evaluation in explicit dxdt or in implicit defects


# --- Managing the constraints --- #
Constraint
    A placeholder for a constraint
ConstraintList
    A list of Constraint if more than one is required
ConstraintFcn
    Selection of valid constraint functions


# --- Managing the objective functions --- #
Objective
    A placeholder for an objective function
ObjectiveFcn
    Selection of valid objective functions
ObjectiveList
    A list of Constraint if more than one is required


# --- Managing the parameters --- #
ParameterList
    A list of Parameter
ParameterConstrainer
    A placeholder for the ParameterList


# --- Managing the boundaries of the variables --- #
Bounds
    A placeholder for bounds constraints
BoundsList
    A list of Bounds if more than one is required


# --- Managing the initial guesses of the variables --- #
InitialGuess
    A placeholder for the initial guess
InitialGuessList
    A list of InitialGuess if more than one is required


# --- Managing the transitions between phases for multiphase programs --- #
PhaseTransitionList
    A list of PhaseTransition
PhaseTransitionFcn
    Selection of valid phase transition functions

# --- Managing the multinode constraint and objective for multiphase programs at specified nodes--- #
MultinodeConstraintList
    A list of MultinodeConstraint
MultinodeConstraintListFcn
    Selection of valid phase MultinodeConstraint functions
MultinodeObjectiveList
    A list of MultinodeObjective
MultinodeObjectiveListFcn
    Selection of valid phase MultinodeObjective functions

# --- Mapping indices between vector --- #
Mapping
    Mapping of index set to a different index set
BiMapping
    Mapping of two index sets between each other
BiMappingList
    A list of BiMapping


# --- Version of bioptim --- #
__version__
    The current version of bioptim
----------------------------------


Requirements and Installation
-----------------------------
bioptim requires minimally CasADi, [Ipopt, ACADOS], biorbd and bioviz. To install ACADOS, one is invited to have a look
at the installation script at 'external/acados_install.sh'. All the other requirements can be installed from conda
on the conda-forge channel using the following command:
`conda install -c conda-forge biorbd=*=*casadi* bioviz=*=*casadi*`

If one is interested in the conda-forge version of bioptim, they can install every requirements and bioptim using the
following command
`conda install -c conda-forge bioptim`


Examples
--------
Examples of all sort can be found in the 'examples' folder.
The first example one should have a look at is the example/getting_started/pendulum.py. This is a trivial yet
challenging optimal control problem. Once one is familiar with the bioptim nomenclature, it is suggested to have a look
at all the examples in 'examples/getting_started' which should tackle most of the questioning one could have when using
the bioptim API. For time optimization, one should have a look at the 'example/optimal_time_ocp' and some examples of
multiphase can be found in 'examples/torque_driven_ocp'. For ACADOS specific examples, you can have a look at
'example/acados'. Please note that ACADOS needs to be installed.

"""

from .dynamics.configure_problem import ConfigureProblem, DynamicsOptionsList, DynamicsOptions
from .dynamics.configure_variables import AutoConfigure, States, Controls, AlgebraicStates, ConfigureVariables
from .dynamics.dynamics_evaluation import DynamicsEvaluation
from .dynamics.dynamics_functions import DynamicsFunctions
from .dynamics.fatigue.effort_perception import EffortPerception, TauEffortPerception
from .dynamics.fatigue.fatigue_dynamics import FatigueList
from .dynamics.fatigue.michaud_fatigue import MichaudFatigue, MichaudTauFatigue
from .dynamics.fatigue.xia_fatigue import XiaFatigue, XiaTauFatigue, XiaFatigueStabilized
from .dynamics.ode_solvers import OdeSolver, OdeSolverBase
from .gui.online_callback_server import PlottingServer
from .gui.plot import CustomPlot
from .interfaces import Solver
from .limits.constraints import ConstraintFcn, ConstraintList, Constraint, ParameterConstraintList
from .limits.fatigue_path_conditions import FatigueBounds, FatigueInitialGuess
from .limits.multinode_constraint import MultinodeConstraintFcn, MultinodeConstraintList, MultinodeConstraint
from .limits.multinode_objective import MultinodeObjectiveFcn, MultinodeObjectiveList, MultinodeObjective
from .limits.objective_functions import ObjectiveFcn, ObjectiveList, Objective, ParameterObjectiveList
from .limits.path_conditions import BoundsList, InitialGuessList, Bounds, InitialGuess
from .limits.penalty_controller import PenaltyController
from .limits.penalty_helpers import PenaltyHelpers
from .limits.phase_transition import PhaseTransitionFcn, PhaseTransitionList, PhaseTransition
from .misc.__version__ import __version__
from .misc.enums import (
    Axis,
    Node,
    InterpolationType,
    PlotType,
    ControlType,
    CostType,
    Shooting,
    VariableType,
    SolutionIntegrator,
    QuadratureRule,
    DefectType,
    MagnitudeType,
    MultiCyclicCycleSolutions,
    PhaseDynamics,
    OnlineOptim,
    ContactType,
)
from .misc.mapping import BiMappingList, BiMapping, Mapping, SelectionMapping, Dependency
from .models.biorbd.biorbd_model import BiorbdModel
from .models.biorbd.external_forces import ExternalForceSetTimeSeries, ExternalForceSetVariables
from .models.biorbd.holonomic_biorbd_model import HolonomicBiorbdModel
from .models.biorbd.multi_biorbd_model import MultiBiorbdModel
from .models.biorbd.stochastic_biorbd_model import StochasticBiorbdModel
from .models.biorbd.variational_biorbd_model import VariationalBiorbdModel
from .models.biorbd.model_dynamics import (
    TorqueBiorbdModel,
    StochasticTorqueBiorbdModel,
    HolonomicTorqueBiorbdModel,
    VariationalTorqueBiorbdModel,
    TorqueFreeFloatingBaseBiorbdModel,
    StochasticTorqueFreeFloatingBaseBiorbdModel,
    TorqueActivationBiorbdModel,
    TorqueDerivativeBiorbdModel,
    MusclesBiorbdModel,
    JointAccelerationBiorbdModel,
    MultiTorqueBiorbdModel,
)
from .models.protocols.biomodel import BioModel
from .models.protocols.holonomic_constraints import HolonomicConstraintsFcn, HolonomicConstraintsList
from .models.protocols.stochastic_biomodel import StochasticBioModel
from .models.protocols.abstract_model_dynamics import (
    TorqueDynamics,
    StochasticTorqueDynamics,
    HolonomicTorqueDynamics,
    VariationalTorqueDynamics,
    TorqueFreeFloatingBaseDynamics,
    StochasticTorqueFreeFloatingBaseDynamics,
    TorqueActivationDynamics,
    TorqueDerivativeDynamics,
    MusclesDynamics,
    JointAccelerationDynamics,
)
from .models.protocols.abstract_model import AbstractModel
from .optimization.multi_start import MultiStart
from .optimization.non_linear_program import NonLinearProgram
from .optimization.optimal_control_program import OptimalControlProgram
from .optimization.optimization_variable import OptimizationVariableList
from .optimization.parameters import ParameterList, ParameterContainer
from .optimization.problem_type import SocpType
from .optimization.receding_horizon_optimization import (
    CyclicNonlinearModelPredictiveControl,
    CyclicMovingHorizonEstimator,
    MultiCyclicNonlinearModelPredictiveControl,
)
from .optimization.receding_horizon_optimization import MovingHorizonEstimator, NonlinearModelPredictiveControl
from .optimization.solution.solution import Solution
from .optimization.solution.solution_data import SolutionMerge, TimeAlignment
from .optimization.stochastic_optimal_control_program import StochasticOptimalControlProgram
from .optimization.variable_scaling import VariableScalingList, VariableScaling
from .optimization.variational_optimal_control_program import VariationalOptimalControlProgram

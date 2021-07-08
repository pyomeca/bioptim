from typing import Callable, Any, Union
from enum import Enum

from casadi import MX, vertcat, Function
import numpy as np

from .dynamics_functions import DynamicsFunctions
from ..misc.enums import PlotType, ControlType
from ..misc.mapping import BiMapping, Mapping
from ..misc.options import UniquePerPhaseOptionList, OptionGeneric
from ..gui.plot import CustomPlot


class ConfigureProblem:
    """
    Dynamics configuration for the most common ocp

    Methods
    -------

    """

    @staticmethod
    def initialize(ocp, nlp):
        """
        Call the dynamics a first time

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        """

        nlp.dynamics_type.type.value[0](ocp, nlp, **nlp.dynamics_type.params)

    @staticmethod
    def custom(ocp, nlp, **extra_params):
        """
        Call the user-defined dynamics configuration function

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        """

        nlp.dynamics_type.configure(ocp, nlp, **extra_params)

    @staticmethod
    def torque_driven(ocp, nlp, with_contact=False):
        """
        Configure the dynamics for a torque driven program (states are q and qdot, controls are tau)

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        with_contact: bool
            If the dynamic with contact should be used
        """

        ConfigureProblem.configure_q(nlp, True, False)
        ConfigureProblem.configure_qdot(nlp, True, False)
        ConfigureProblem.configure_tau(nlp, False, True)

        if nlp.dynamics_type.dynamic_function:
            ConfigureProblem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.custom)
        else:
            ConfigureProblem.configure_dynamics_function(
                ocp, nlp, DynamicsFunctions.torque_driven, with_contact=with_contact
            )

        if with_contact:
            ConfigureProblem.configure_contact_function(ocp, nlp, DynamicsFunctions.forces_from_torque_driven)

    @staticmethod
    def torque_derivative_driven(ocp, nlp, with_contact=False):
        """
        Configure the dynamics for a torque driven program (states are q and qdot, controls are tau)

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        with_contact: bool
            If the dynamic with contact should be used
        """

        ConfigureProblem.configure_q(nlp, True, False)
        ConfigureProblem.configure_qdot(nlp, True, False)
        ConfigureProblem.configure_tau(nlp, True, False)
        ConfigureProblem.configure_taudot(nlp, False, True)

        if nlp.dynamics_type.dynamic_function:
            ConfigureProblem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.custom)
        else:
            ConfigureProblem.configure_dynamics_function(
                ocp, nlp, DynamicsFunctions.torque_derivative_driven, with_contact=with_contact
            )

        if with_contact:
            ConfigureProblem.configure_contact_function(ocp, nlp, DynamicsFunctions.forces_from_torque_driven)

    @staticmethod
    def torque_activations_driven(ocp, nlp, with_contact=False):
        """
        Configure the dynamics for a torque driven program (states are q and qdot, controls are tau activations).
        The tau activations are bounded between -1 and 1 and actual tau is computed from torque-position-velocity
        relationship

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        with_contact: bool
            If the dynamic with contact should be used
        """

        ConfigureProblem.configure_q(nlp, True, False)
        ConfigureProblem.configure_qdot(nlp, True, False)
        ConfigureProblem.configure_tau(nlp, False, True)

        if nlp.dynamics_type.dynamic_function:
            ConfigureProblem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.custom)
        else:
            ConfigureProblem.configure_dynamics_function(
                ocp, nlp, DynamicsFunctions.torque_activations_driven, with_contact=with_contact
            )

        if with_contact:
            ConfigureProblem.configure_contact_function(
                ocp, nlp, DynamicsFunctions.forces_from_torque_activation_driven
            )

    @staticmethod
    def muscle_driven(
        ocp, nlp, with_excitations: bool = False, with_residual_torque: bool = False, with_contact: bool = False
    ):
        """
        Configure the dynamics for a muscle driven program.
        If with_excitations is set to True, then the muscle muscle activations are computed from the muscle dynamics.
        The tau from muscle is computed using the muscle activations.
        If with_residual_torque is set to True, then tau are used as supplementary force in the
        case muscles are too weak.

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        with_excitations: bool
            If the dynamic should include the muscle dynamics
        with_residual_torque: bool
            If the dynamic should be added with residual torques
        with_contact: bool
            If the dynamic with contact should be used
        """

        ConfigureProblem.configure_q(nlp, True, False)
        ConfigureProblem.configure_qdot(nlp, True, False)
        if with_residual_torque:
            ConfigureProblem.configure_tau(nlp, False, True)
        ConfigureProblem.configure_muscles(nlp, with_excitations, True)

        if nlp.dynamics_type.dynamic_function:
            ConfigureProblem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.custom)
        else:
            ConfigureProblem.configure_dynamics_function(
                ocp, nlp, DynamicsFunctions.muscles_driven, with_contact=with_contact
            )

        if with_contact:
            ConfigureProblem.configure_contact_function(ocp, nlp, DynamicsFunctions.forces_from_muscle_driven)

    @staticmethod
    def configure_dynamics_function(ocp, nlp, dyn_func, **extra_params):
        """
        Configure the forward dynamics

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        dyn_func: Callable[states, controls, param]
            The function to get the derivative of the states
        """

        nlp.parameters = ocp.v.parameters_in_list

        dynamics = dyn_func(nlp.states.mx_reduced, nlp.controls.mx_reduced, nlp.parameters.mx, nlp, **extra_params)
        if isinstance(dynamics, (list, tuple)):
            dynamics = vertcat(*dynamics)
        nlp.dynamics_func = Function(
            "ForwardDyn",
            [nlp.states.mx_reduced, nlp.controls.mx_reduced, nlp.parameters.mx],
            [dynamics],
            ["x", "u", "p"],
            ["xdot"],
        ).expand()

    @staticmethod
    def configure_contact_function(ocp, nlp, dyn_func: Callable, **extra_params):
        """
        Configure the contact points

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        dyn_func: Callable[states, controls, param]
            The function to get the values of contact forces from the dynamics
        """

        nlp.contact_forces_func = Function(
            "contact_forces_func",
            [nlp.states.mx_reduced, nlp.controls.mx_reduced, nlp.parameters.mx],
            [dyn_func(nlp.states.mx_reduced, nlp.controls.mx_reduced, nlp.parameters.mx, nlp, **extra_params)],
            ["x", "u", "p"],
            ["contact_forces"],
        ).expand()

        all_contact_names = []
        for elt in ocp.nlp:
            all_contact_names.extend(
                [name.to_string() for name in elt.model.contactNames() if name.to_string() not in all_contact_names]
            )

        if "contact_forces" in nlp.plot_mapping:
            phase_mappings = nlp.plot_mapping["contact_forces"]
        else:
            contact_names_in_phase = [name.to_string() for name in nlp.model.contactNames()]
            phase_mappings = Mapping([i for i, c in enumerate(all_contact_names) if c in contact_names_in_phase])

        nlp.plot["contact_forces"] = CustomPlot(
            nlp.contact_forces_func, plot_type=PlotType.INTEGRATED, axes_idx=phase_mappings, legend=all_contact_names
        )

    @staticmethod
    def configure_new_variable(
        name: str, name_elements: list, nlp, as_states: bool, as_controls: bool, combine_plot: bool = False
    ):
        def define_cx(n_col: int) -> list:
            cx = [nlp.cx() for _ in range(n_col)]
            for idx in nlp.variable_mappings[name].to_first.map_idx:
                if idx is None:
                    continue
                for j in range(len(cx)):
                    sign = "-" if np.sign(idx) < 0 else ""
                    cx[j] = vertcat(cx[j], nlp.cx.sym(f"{sign}{name}_{name_elements[abs(idx)]}_{j}", 1, 1))
            return cx

        if name not in nlp.variable_mappings:
            nlp.variable_mappings[name] = BiMapping(range(len(name_elements)), range(len(name_elements)))
        legend = [f"{name}_{name_elements[idx]}" for idx in nlp.variable_mappings[name].to_first.map_idx]

        mx_states = MX()
        mx_controls = MX()
        for i in nlp.variable_mappings[name].to_second.map_idx:
            if i is None:
                continue
            sign = "-" if np.sign(i) < 0 else ""
            mx_states = vertcat(mx_states, MX.sym(f"{sign}{name}_{name_elements[abs(i)]}_MX", 1, 1))
            mx_controls = vertcat(mx_controls, MX.sym(f"{sign}{name}_{name_elements[abs(i)]}_MX", 1, 1))

        if as_states:
            cx = define_cx(n_col=2)

            nlp.states.append(name, cx, mx_states, nlp.variable_mappings[name])
            nlp.plot[f"{name}_states"] = CustomPlot(
                lambda x, u, p: x[nlp.states[name].index, :],
                plot_type=PlotType.INTEGRATED,
                legend=legend,
                # bounds=nlp.x_bounds[nlp.states[name].index],  # TODO This is empty (this is a bug)
            )

        if as_controls:
            cx = define_cx(n_col=2)

            nlp.controls.append(name, cx, mx_controls, nlp.variable_mappings[name])
            plot_type = PlotType.PLOT if nlp.control_type == ControlType.LINEAR_CONTINUOUS else PlotType.STEP
            nlp.plot[f"{name}_controls"] = CustomPlot(
                lambda x, u, p: u[nlp.controls[name].index, :],
                plot_type=plot_type,
                legend=legend,
                # bounds=nlp.u_bounds[nlp.controls[name].index],  # TODO This is empty (this is a bug)
                combine_to=f"{name}_states" if as_states and combine_plot else None,
            )

    @staticmethod
    def configure_q(nlp, as_states: bool, as_controls: bool):
        """
        Configure the generalized coordinates

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        as_states: bool
            If the generalized coordinates should be a state
        as_controls: bool
            If the generalized coordinates should be a control
        """

        name_q = [name.to_string() for name in nlp.model.nameDof()]
        ConfigureProblem.configure_new_variable("q", name_q, nlp, as_states, as_controls)

    @staticmethod
    def configure_qdot(nlp, as_states: bool, as_controls: bool):
        """
        Configure the generalized velocities

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        as_states: bool
            If the generalized velocities should be a state
        as_controls: bool
            If the generalized velocities should be a control
        """

        name_qdot = [str(i) for i in range(nlp.model.nbQdot())]
        ConfigureProblem._adjust_mapping_against_q("qdot", ["qdot", "taudot"], nlp)
        ConfigureProblem.configure_new_variable("qdot", name_qdot, nlp, as_states, as_controls)

    @staticmethod
    def configure_tau(nlp, as_states: bool, as_controls: bool):
        """
        Configure the generalized forces

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        as_states: bool
            If the generalized forces should be a state
        as_controls: bool
            If the generalized forces should be a control
        """

        name_tau = [str(i) for i in range(nlp.model.nbGeneralizedTorque())]
        ConfigureProblem._adjust_mapping_against_q("tau", ["qdot", "taudot"], nlp)
        ConfigureProblem.configure_new_variable("tau", name_tau, nlp, as_states, as_controls)

    @staticmethod
    def configure_taudot(nlp, as_states: bool, as_controls: bool):
        """
        Configure the generalized forces

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        as_states: bool
            If the generalized force derivatives should be a state
        as_controls: bool
            If the generalized force derivatives should be a control
        """

        name_tau = [str(i) for i in range(nlp.model.nbGeneralizedTorque())]
        ConfigureProblem._adjust_mapping_against_q("taudot", ["qdot", "tau"], nlp)
        ConfigureProblem.configure_new_variable("taudot", name_tau, nlp, as_states, as_controls, False)

    @staticmethod
    def configure_muscles(nlp, as_states: bool, as_controls: bool):
        """
        Configure the muscles

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        as_states: bool
            If the muscles should be a state
        as_controls: bool
            If the muscles should be a control
        """

        muscle_names = [names.to_string() for names in nlp.model.muscleNames()]
        ConfigureProblem.configure_new_variable("muscles", muscle_names, nlp, as_states, as_controls, True)

    @staticmethod
    def _adjust_mapping_against_q(name_to_adjust, names_to_compare, nlp):
        if "q" in nlp.variable_mappings and name_to_adjust not in nlp.variable_mappings:
            if nlp.model.nbQuat() > 0:
                for n in names_to_compare:
                    if n in nlp.variable_mappings:
                        nlp.variable_mappings[name_to_adjust] = nlp.variable_mappings[n]
                        break
                else:
                    q_map = list(nlp.variable_mappings["q"].to_first.map_idx)
                    target = list(range(nlp.model.nbQ()))
                    if q_map != target or q_map != target:
                        raise RuntimeError(
                            "It is not possible to define a q mapping without a qdot or tau mapping"
                            "while the model has quaternions"
                        )
                    nlp.variable_mappings[name_to_adjust] = BiMapping(
                        range(nlp.model.nbGeneralizedTorque()), range(nlp.model.nbGeneralizedTorque())
                    )
            else:
                nlp.variable_mappings[name_to_adjust] = nlp.variable_mappings["q"]


class DynamicsFcn(Enum):
    """
    Selection of valid dynamics functions
    """

    TORQUE_DRIVEN = (ConfigureProblem.torque_driven,)
    TORQUE_DERIVATIVE_DRIVEN = (ConfigureProblem.torque_derivative_driven,)
    TORQUE_ACTIVATIONS_DRIVEN = (ConfigureProblem.torque_activations_driven,)
    MUSCLE_DRIVEN = (ConfigureProblem.muscle_driven,)
    CUSTOM = (ConfigureProblem.custom,)


class Dynamics(OptionGeneric):
    """
    A placeholder for the chosen dynamics by the user

    Attributes
    ----------
    dynamic_function: Callable
        The custom dynamic function provided by the user
    configure: Callable
        The configuration function provided by the user that declares the NLP (states and controls),
        usually only necessary when defining custom functions

    """

    def __init__(
        self,
        dynamics_type: Union[Callable, DynamicsFcn],
        expand: bool = True,
        **params: Any,
    ):
        """
        configure: Callable
            The configuration function provided by the user that declares the NLP (states and controls),
            usually only necessary when defining custom functions
        expand: bool
            If the continuity constraint should be expand. This can be extensive on RAM
        dynamic_function: Callable
            The custom dynamic function provided by the user

        Parameters
        ----------
        dynamics_type: Union[Callable, DynamicsFcn]
            The chosen dynamic functions
        params: Any
            Any parameters to pass to the dynamic and configure functions
        """

        configure = None
        if not isinstance(dynamics_type, DynamicsFcn):
            configure = dynamics_type
            dynamics_type = DynamicsFcn.CUSTOM
        else:
            if "configure" in params:
                configure = params["configure"]
                del params["configure"]

        dynamic_function = None
        if "dynamic_function" in params:
            dynamic_function = params["dynamic_function"]
            del params["dynamic_function"]

        super(Dynamics, self).__init__(type=dynamics_type, **params)
        self.dynamic_function = dynamic_function
        self.configure = configure
        self.expand = expand


class DynamicsList(UniquePerPhaseOptionList):
    """
    A list of Dynamics if more than one is required, typically when more than one phases are declared

    Methods
    -------
    add(dynamics: DynamicsFcn, **extra_parameters)
        Add a new Dynamics to the list
    print(self)
        Print the DynamicsList to the console
    """

    def add(self, dynamics_type: Union[Callable, Dynamics, DynamicsFcn], **extra_parameters: Any):
        """
        Add a new Dynamics to the list

        Parameters
        ----------
        dynamics_type: Union[Callable, Dynamics, DynamicsFcn]
            The chosen dynamic functions
        extra_parameters: dict
            Any parameters to pass to Dynamics
        """

        if isinstance(dynamics_type, Dynamics):
            self.copy(dynamics_type)

        else:
            super(DynamicsList, self)._add(dynamics_type=dynamics_type, option_type=Dynamics, **extra_parameters)

    def print(self):
        """
        Print the DynamicsList to the console
        """
        raise NotImplementedError("Printing of DynamicsList is not ready yet")

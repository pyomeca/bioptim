import biorbd_casadi as biorbd
from casadi import MX, vertcat, Function, horzcat
from typing import Callable

from .biorbd_model import BiorbdModel
from ..utils import _var_mapping
from ..utils import bounds_from_ranges
from ...limits.path_conditions import Bounds
from ...misc.mapping import BiMapping, BiMappingList


class MultiBiorbdModel:
    """
    This class allows to define multiple biorbd models for the same phase.


    Attributes
    ----------
    models : list[BiorbdModel]
        The list of biorbd models to be handled in the optimal control program.
    extra_models : list[BiorbdModel]
        A list of extra biorbd models stored in the class for further use.

    Methods
    -------
    variable_index()
        Get the index of the variables in the global vector for a given model index.
    nb_models()
        Get the number of models.
    nb_extra_models()
        Get the number of extra models.

    """

    def __init__(
        self,
        bio_model: tuple[str | biorbd.Model | BiorbdModel, ...],
        extra_bio_models: tuple[str | biorbd.Model | BiorbdModel, ...] = (),
    ):
        """
        MultiBiorbdModel does not handle external_forces and parameters yet.
        """
        self.models = []
        if not isinstance(bio_model, tuple):
            raise ValueError("The models must be a 'str', 'biorbd.Model', 'bioptim.BiorbdModel'" " or a tuple of those")

        for model in bio_model:
            if isinstance(model, str):
                self.models.append(BiorbdModel(model))
            elif isinstance(model, biorbd.Model):
                self.models.append(BiorbdModel(model))
            elif isinstance(model, BiorbdModel):
                self.models.append(model)
                if model.parameters is not None:
                    raise NotImplementedError(
                        "MultiBiorbdModel does not handle parameters yet. Please use BiorbdModel instead."
                    )
            else:
                raise ValueError("The models should be of type 'str', 'biorbd.Model' or 'bioptim.BiorbdModel'")

        if not isinstance(extra_bio_models, tuple):
            raise ValueError("The models must be a 'str', 'biorbd.Model', 'bioptim.BiorbdModel'" " or a tuple of those")

        self.extra_models = []
        for model in extra_bio_models:
            if isinstance(model, str):
                self.extra_models.append(BiorbdModel(model))
            elif isinstance(model, biorbd.Model):
                self.extra_models.append(BiorbdModel(model))
            elif isinstance(model, BiorbdModel):
                self.extra_models.append(model)
            else:
                raise ValueError("The models should be of type 'str', 'biorbd.Model' or 'bioptim.BiorbdModel'")

        # Declaration of MX variables of the right shape for the creation of CasADi Functions
        self.q = MX.sym("q_mx", self.nb_q, 1)
        self.qdot = MX.sym("qdot_mx", self.nb_qdot, 1)
        self.qddot = MX.sym("qddot_mx", self.nb_qddot, 1)
        self.qddot_roots = MX.sym("qddot_roots_mx", self.nb_root, 1)
        self.qddot_joints = MX.sym("qddot_joints_mx", self.nb_qddot - self.nb_root, 1)
        self.tau = MX.sym("tau_mx", self.nb_tau, 1)
        self.muscle = MX.sym("muscle_mx", self.nb_muscles, 1)
        self.activations = MX.sym("activations_mx", self.nb_muscles, 1)
        self.parameters = MX.sym("parameters_to_be_implemented", 0, 1)

    def __getitem__(self, index):
        return self.models[index]

    def deep_copy(self, *args):
        raise NotImplementedError("Deep copy is not implemented yet for MultiBiorbdModel class")

    @property
    def path(self) -> tuple[list[str], list[str]]:
        return [model.path for model in self.models], [model.path for model in self.extra_models]

    def copy(self):
        all_paths = self.path
        return MultiBiorbdModel(tuple(all_paths[0]), tuple(all_paths[1]))

    def serialize(self) -> tuple[Callable, dict]:
        return MultiBiorbdModel, dict(bio_model=tuple(self.path[0]), extra_bio_models=tuple(self.path[1]))

    def variable_index(self, variable: str, model_index: int) -> range:
        """
        Get the index of the variables in the global vector for a given model index

        Parameters
        ----------
        variable: str
            The variable to get the index from such as 'q', 'qdot', 'qddot', 'tau', 'contact', 'markers'
        model_index: int
            The index of the model to get the index from

        Returns
        -------
        range
            The index of the variable in the global vector
        """
        current_idx = 0

        if variable == "q":
            for model in self.models[:model_index]:
                current_idx += model.nb_q
            return range(current_idx, current_idx + self.models[model_index].nb_q)

        elif variable == "qdot":
            for model in self.models[:model_index]:
                current_idx += model.nb_qdot
            return range(current_idx, current_idx + self.models[model_index].nb_qdot)

        elif variable == "qddot":
            for model in self.models[:model_index]:
                current_idx += model.nb_qddot
            return range(current_idx, current_idx + self.models[model_index].nb_qddot)

        elif variable == "qddot_joints":
            for model in self.models[:model_index]:
                current_idx += model.nb_qddot - model.nb_root
            return range(
                current_idx, current_idx + self.models[model_index].nb_qddot - self.models[model_index].nb_root
            )

        elif variable == "qddot_root":
            for model in self.models[:model_index]:
                current_idx += model.nb_root
            return range(current_idx, current_idx + self.models[model_index].nb_root)

        elif variable == "tau":
            for model in self.models[:model_index]:
                current_idx += model.nb_tau
            return range(current_idx, current_idx + self.models[model_index].nb_tau)

        elif variable == "contact":
            for model in self.models[:model_index]:
                current_idx += model.nb_rigid_contacts
            return range(current_idx, current_idx + self.models[model_index].nb_rigid_contacts)

        elif variable == "markers":
            for model in self.models[:model_index]:
                current_idx += model.nb_markers
            return range(current_idx, current_idx + self.models[model_index].nb_markers)

        elif variable == "segment":
            for model in self.models[:model_index]:
                current_idx += model.nb_segments
            return range(current_idx, current_idx + self.models[model_index].nb_segments)

        else:
            raise ValueError(
                "The variable must be 'q', 'qdot', 'qddot', 'tau', 'contact' or 'markers'" f" and {variable} was sent."
            )

    def global_variable_id(self, variable: str, model_index: int, model_variable_id: int) -> int:
        """
        Get the id of the variable in the global vector for a given model index

        Parameters
        ----------
        variable: str
            The variable to get the index from such as 'q', 'qdot', 'qddot', 'tau', 'contact', 'markers'
        model_index: int
            The index of the model to get the index from
        model_variable_id: int
            The id of the variable in the model vector

        Returns
        -------
        int
            The id of the variable in the global vector
        """
        return self.variable_index(variable, model_index)[model_variable_id]

    def local_variable_id(self, variable: str, global_index: int) -> tuple[int, int]:
        """
        Get the id of the variable in the local vector and the model index for a given index of the global vector

        Parameters
        ----------
        variable: str
            The variable to get the index from such as 'q', 'qdot', 'qddot', 'tau', 'contact', 'markers'
        global_index: int
            The index of the variable in the global vector

        Returns
        -------
        tuple(int, int)
            The id of the variable in the local vector and the model index
        """

        for model_id, model in enumerate(self.models):
            if global_index in self.variable_index(variable, model_id):
                return global_index - self.variable_index(variable, model_id)[0], model_id

    @property
    def nb_models(self) -> int:
        """
        Get the number of models

        Returns
        -------
        int
            The number of models
        """
        return len(self.models)

    @property
    def nb_extra_models(self) -> int:
        """
        Get the number of extra models

        Returns
        -------
        int
            The number of extra models
        """
        return len(self.extra_models)

    @property
    def gravity(self) -> Function:
        for i, model in enumerate(self.models):
            if i == 0:
                if self.parameters.shape[0] == 0:
                    biorbd_return = model.gravity()["gravity"]
                else:
                    biorbd_return = model.gravity()(self.parameters)["gravity"]
            else:
                if self.parameters.shape[0] == 0:
                    biorbd_return = vertcat(biorbd_return, model.gravity()["gravity"])
                else:
                    biorbd_return = vertcat(biorbd_return, model.gravity()(self.parameters)["gravity"])
        casadi_fun = Function(
            "gravity",
            [self.parameters],
            [biorbd_return],
            ["parameters"],
            ["gravity"],
        )
        return casadi_fun

    def set_gravity(self, new_gravity) -> None:
        # All models have the same gravity, but it could be changed if needed
        for model in self.models:
            model.set_gravity(new_gravity)
        return

    @property
    def nb_tau(self) -> int:
        return sum(model.nb_tau for model in self.models)

    @property
    def nb_segments(self) -> int:
        return sum(model.nb_segments for model in self.models)

    def segment_index(self, name) -> int:
        raise NotImplementedError("segment_index is not implemented for MultiBiorbdModel")

    @property
    def nb_quaternions(self) -> int:
        return sum(model.nb_quaternions for model in self.models)

    @property
    def nb_q(self) -> int:
        return sum(model.nb_q for model in self.models)

    @property
    def nb_qdot(self) -> int:
        return sum(model.nb_qdot for model in self.models)

    @property
    def nb_qddot(self) -> int:
        return sum(model.nb_qddot for model in self.models)

    @property
    def nb_root(self) -> int:
        return sum(model.nb_root for model in self.models)

    @property
    def segments(self) -> tuple[biorbd.Segment, ...]:
        out = ()
        for model in self.models:
            for seg in model.segments:
                out += (seg,)
        return out

    def homogeneous_matrices_in_global(self, segment_index, inverse=False) -> Function:
        local_segment_id, model_id = self.local_variable_id("segment", segment_index)
        q_model = self.models[model_id].q
        biorbd_return = self.models[model_id].homogeneous_matrices_in_global(local_segment_id, inverse)(
            q_model, self.parameters
        )
        casadi_fun = Function(
            "homogeneous_matrices_in_global",
            [self.models[model_id].q, self.parameters],
            [biorbd_return],
            ["q", "parameters"],
            ["homogeneous_matrices_in_global"],
        )
        return casadi_fun

    def homogeneous_matrices_in_child(self, segment_id) -> Function:
        local_id, model_id = self.local_variable_id("segment", segment_id)
        casadi_fun = self.models[model_id].homogeneous_matrices_in_child(local_id)(self.parameters)
        return casadi_fun

    @property
    def mass(self) -> Function:
        for i, model in enumerate(self.models):
            if i == 0:
                if self.parameters.shape[0] == 0:
                    biorbd_return = model.mass()["mass"]
                else:
                    biorbd_return = model.mass()(self.parameters)["mass"]
            else:
                if self.parameters.shape[0] == 0:
                    biorbd_return = vertcat(biorbd_return, model.mass()["mass"])
                else:
                    biorbd_return = vertcat(biorbd_return, model.mass()(self.parameters)["mass"])
        casadi_fun = Function(
            "mass",
            [self.parameters],
            [biorbd_return],
            ["parameters"],
            ["mass"],
        )
        return casadi_fun

    def center_of_mass(self) -> Function:
        biorbd_return = MX()
        for i, model in enumerate(self.models):
            q_model = self.q[self.variable_index("q", i)]
            biorbd_return = vertcat(biorbd_return, model.center_of_mass()(q_model, self.parameters))
        casadi_fun = Function(
            "center_of_mass",
            [self.q, self.parameters],
            [biorbd_return],
            ["q", "parameters"],
            ["center_of_mass"],
        )
        return casadi_fun

    def center_of_mass_velocity(self) -> Function:
        biorbd_return = MX()
        for i, model in enumerate(self.models):
            q_model = self.q[self.variable_index("q", i)]
            qdot_model = self.qdot[self.variable_index("qdot", i)]
            biorbd_return = vertcat(
                biorbd_return,
                model.center_of_mass_velocity()(q_model, qdot_model, self.parameters),
            )
        casadi_fun = Function(
            "center_of_mass_velocity",
            [self.q, self.qdot, self.parameters],
            [biorbd_return],
            ["q", "qdot", "parameters"],
            ["center_of_mass_velocity"],
        )
        return casadi_fun

    def center_of_mass_acceleration(self) -> Function:
        biorbd_return = MX()
        for i, model in enumerate(self.models):
            q_model = self.q[self.variable_index("q", i)]
            qdot_model = self.qdot[self.variable_index("qdot", i)]
            qddot_model = self.qddot[self.variable_index("qddot", i)]
            biorbd_return = vertcat(
                biorbd_return,
                model.center_of_mass_acceleration()(q_model, qdot_model, qddot_model, self.parameters),
            )
        casadi_fun = Function(
            "center_of_mass_acceleration",
            [self.q, self.qdot, self.qddot, self.parameters],
            [biorbd_return],
            ["q", "qdot", "qddot", "parameters"],
            ["center_of_mass_acceleration"],
        )
        return casadi_fun

    def mass_matrix(self) -> Function:
        biorbd_return = []
        for i, model in enumerate(self.models):
            q_model = self.q[self.variable_index("q", i)]
            biorbd_return += [model.mass_matrix()(q_model, self.parameters)]
        casadi_fun = Function(
            "mass_matrix",
            [self.q, self.parameters],
            biorbd_return,
        )
        return casadi_fun

    def non_linear_effects(self) -> Function:
        biorbd_return = []
        for i, model in enumerate(self.models):
            q_model = self.q[self.variable_index("q", i)]
            qdot_model = self.qdot[self.variable_index("qdot", i)]
            biorbd_return += [model.non_linear_effects()(q_model, qdot_model, self.parameters)]
        casadi_fun = Function(
            "non_linear_effects",
            [self.q, self.qdot, self.parameters],
            biorbd_return,
        )
        return casadi_fun

    def angular_momentum(self) -> Function:
        biorbd_return = MX()
        for i, model in enumerate(self.models):
            q_model = self.q[self.variable_index("q", i)]
            qdot_model = self.qdot[self.variable_index("qdot", i)]
            biorbd_return = vertcat(
                biorbd_return,
                model.angular_momentum()(q_model, qdot_model, self.parameters),
            )
        casadi_fun = Function(
            "angular_momentum",
            [self.q, self.qdot, self.parameters],
            [biorbd_return],
            ["q", "qdot", "parameters"],
            ["angular_momentum"],
        )
        return casadi_fun

    def reshape_qdot(self, k_stab=1) -> Function:
        biorbd_return = MX()
        for i, model in enumerate(self.models):
            q_model = self.q[self.variable_index("q", i)]
            qdot_model = self.qdot[self.variable_index("qdot", i)]
            biorbd_return = vertcat(
                biorbd_return,
                model.reshape_qdot(k_stab)(q_model, qdot_model, self.parameters),
            )
        casadi_fun = Function(
            "reshape_qdot",
            [self.q, self.qdot, self.parameters],
            [biorbd_return],
            ["q", "qdot", "parameters"],
            ["qdot_reshaped"],
        )
        return casadi_fun

    def segment_angular_velocity(self, idx) -> Function:
        biorbd_return = MX()
        for i, model in enumerate(self.models):
            q_model = self.q[self.variable_index("q", i)]
            qdot_model = self.qdot[self.variable_index("qdot", i)]
            biorbd_return = vertcat(
                biorbd_return,
                model.segment_angular_velocity(idx)(q_model, qdot_model, self.parameters),
            )
        casadi_fun = Function(
            "segment_angular_velocity",
            [self.q, self.qdot, self.parameters],
            [biorbd_return],
            ["q", "qdot", "parameters"],
            ["segment_angular_velocity"],
        )
        return casadi_fun

    @property
    def name_dof(self) -> tuple[str, ...]:
        return tuple([dof for model in self.models for dof in model.name_dof])

    @property
    def contact_names(self) -> tuple[str, ...]:
        return tuple([contact for model in self.models for contact in model.contact_names])

    @property
    def nb_soft_contacts(self) -> int:
        return sum(model.nb_soft_contacts for model in self.models)

    @property
    def soft_contact_names(self) -> tuple[str, ...]:
        return tuple([contact for model in self.models for contact in model.soft_contact_names])

    def soft_contact(self, soft_contact_index, *args):
        # What does that function return?
        current_number_of_soft_contacts = 0
        out = []
        for model in self.models:
            if soft_contact_index < current_number_of_soft_contacts + model.nb_soft_contacts:
                out = model.soft_contact(soft_contact_index - current_number_of_soft_contacts, *args)
                break
            current_number_of_soft_contacts += model.nb_soft_contacts
        return out

    @property
    def muscle_names(self) -> tuple[str, ...]:
        return tuple([muscle for model in self.models for muscle in model.muscle_names])

    @property
    def nb_muscles(self) -> int:
        return sum(model.nb_muscles for model in self.models)

    def torque(self) -> Function:
        biorbd_return = MX()
        for i, model in enumerate(self.models):
            model.model.closeActuator()
            tau_activations_model = self.tau[self.variable_index("tau", i)]
            q_model = self.q[self.variable_index("q", i)]
            qdot_model = self.qdot[self.variable_index("qdot", i)]
            biorbd_return = vertcat(
                biorbd_return,
                model.torque()(
                    tau_activations_model,
                    q_model,
                    qdot_model,
                    self.parameters,
                ),
            )
        casadi_fun = Function(
            "torque",
            [self.tau, self.q, self.qdot, self.parameters],
            [biorbd_return],
            ["tau", "q", "qdot", "parameters"],
            ["torque"],
        )
        return casadi_fun

    def forward_dynamics_free_floating_base(self) -> Function:
        biorbd_return = MX()
        for i, model in enumerate(self.models):
            q_model = self.q[self.variable_index("q", i)]
            qdot_model = self.qdot[self.variable_index("qdot", i)]
            qddot_joints_model = self.qddot_joints[self.variable_index("qddot_joints", i)]
            biorbd_return = vertcat(
                biorbd_return,
                model.forward_dynamics_free_floating_base()(
                    q_model,
                    qdot_model,
                    qddot_joints_model,
                    self.parameters,
                ),
            )
        casadi_fun = Function(
            "forward_dynamics_free_floating_base",
            [self.q, self.qdot, self.qddot_joints, self.parameters],
            [biorbd_return],
            ["q", "qdot", "qddot_joints", "parameters"],
            ["qddot"],
        )
        return casadi_fun

    def reorder_qddot_root_joints(self):
        biorbd_return = MX()
        for i, model in enumerate(self.models):
            qddot_root_model = self.qddot_roots[self.variable_index("qddot_root", i)]
            qddot_joints_model = self.qddot_joints[self.variable_index("qddot_joints", i)]
            biorbd_return = vertcat(
                biorbd_return,
                model.reorder_qddot_root_joints(qddot_root_model, qddot_joints_model),
            )

        casadi_fun = Function(
            "reorder_qddot_root_joints",
            [self.qddot_roots, self.qddot_joints],
            [biorbd_return],
            ["qddot_roots", "qddot_joints"],
            ["qddot"],
        )
        return casadi_fun

    def forward_dynamics(self, with_contact) -> Function:
        """External forces and contact forces are not implemented yet for MultiBiorbdModel."""

        biorbd_return = MX()
        for i, model in enumerate(self.models):
            q_model = self.q[self.variable_index("q", i)]
            qdot_model = self.qdot[self.variable_index("qdot", i)]
            tau_model = self.tau[self.variable_index("tau", i)]
            biorbd_return = vertcat(
                biorbd_return,
                model.forward_dynamics(with_contact=with_contact)(
                    q_model,
                    qdot_model,
                    tau_model,
                    [],
                    self.parameters,
                ),
            )
        casadi_fun = Function(
            "forward_dynamics",
            [self.q, self.qdot, self.tau, [], self.parameters],
            [biorbd_return],
            ["q", "qdot", "tau", "external_forces([])", "parameters"],
            ["qddot"],
        )
        return casadi_fun

    def inverse_dynamics(self) -> Function:
        """External forces and contact forces are not implemented yet for MultiBiorbdModel."""

        biorbd_return = MX()
        for i, model in enumerate(self.models):
            q_model = self.q[self.variable_index("q", i)]
            qdot_model = self.qdot[self.variable_index("qdot", i)]
            qddot_model = self.qddot[self.variable_index("qddot", i)]
            biorbd_return = vertcat(
                biorbd_return,
                model.inverse_dynamics()(q_model, qdot_model, qddot_model, [], self.parameters),
            )
        casadi_fun = Function(
            "inverse_dynamics",
            [self.q, self.qdot, self.qddot, [], self.parameters],
            [biorbd_return],
            ["q", "qdot", "qddot", "external_forces([])", "parameters"],
            ["tau"],
        )
        return casadi_fun

    def contact_forces_from_constrained_forward_dynamics(self) -> Function:
        """External forces are not implemented yet for MultiBiorbdModel."""
        biorbd_return = MX()
        for i, model in enumerate(self.models):
            q_model = self.q[self.variable_index("q", i)]
            qdot_model = self.qdot[self.variable_index("qdot", i)]
            tau_model = self.tau[self.variable_index("qddot", i)]  # Due to a bug in biorbd
            biorbd_return = vertcat(
                biorbd_return,
                model.contact_forces_from_constrained_forward_dynamics()(
                    q_model,
                    qdot_model,
                    tau_model,
                    [],
                    self.parameters,
                ),
            )
        casadi_fun = Function(
            "contact_forces_from_constrained_forward_dynamics",
            [self.q, self.qdot, self.tau, [], self.parameters],
            [biorbd_return],
            ["q", "qdot", "tau", "external_forces([])", "parameters"],
            ["contact_forces"],
        )
        return casadi_fun

    def qdot_from_impact(self) -> Function:
        biorbd_return = MX()
        for i, model in enumerate(self.models):
            q_model = self.q[self.variable_index("q", i)]
            qdot_pre_impact_model = self.qdot[self.variable_index("qdot", i)]
            biorbd_return = vertcat(
                biorbd_return,
                model.qdot_from_impact()(
                    q_model,
                    qdot_pre_impact_model,
                    self.parameters,
                ),
            )
        casadi_fun = Function(
            "qdot_from_impact",
            [self.q, self.qdot, self.parameters],
            [biorbd_return],
            ["q", "qdot_pre", "parameters"],
            ["qdot_post"],
        )
        return casadi_fun

    def muscle_activation_dot(self) -> Function:
        biorbd_return = MX()
        n_muscles = 0
        for model in self.models:
            muscle_states = model.model.stateSet()
            for k in range(model.nb_muscles):
                muscle_states[k].setActivation(self.activations[k + n_muscles])
                muscle_states[k].setExcitation(self.muscle[k + n_muscles])
            biorbd_return = vertcat(biorbd_return, model.model.activationDot(muscle_states).to_mx())
            n_muscles += model.nb_muscles
        casadi_fun = Function(
            "muscle_activation_dot",
            [self.muscle, self.activations, self.parameters],
            [biorbd_return],
            ["muscles", "activations", "parameters"],
            ["muscle_activations_dot"],
        )
        return casadi_fun

    def muscle_joint_torque(self) -> Function:
        biorbd_return = MX()
        for i, model in enumerate(self.models):
            muscles_states = model.model.stateSet()  # still call from Biorbd
            for k in range(model.nb_muscles):
                muscles_states[k].setActivation(self.muscle[k])
            q_model = self.q[self.variable_index("q", i)]
            qdot_model = self.qdot[self.variable_index("qdot", i)]
            biorbd_return = vertcat(
                biorbd_return, model.model.muscularJointTorque(muscles_states, q_model, qdot_model).to_mx()
            )
        casadi_fun = Function(
            "muscle_joint_torque",
            [self.muscle, self.q, self.qdot, self.parameters],
            [biorbd_return],
            ["muscles", "q", "qdot", "parameters"],
            ["muscle_joint_torque"],
        )
        return casadi_fun

    def markers(self) -> Function:
        biorbd_return = []
        for i, model in enumerate(self.models):
            q_model = self.q[self.variable_index("q", i)]
            biorbd_return += [model.markers()(q_model, self.parameters)]
        casadi_fun = Function(
            "markers",
            [self.q, self.parameters],
            [horzcat(*biorbd_return)],
            ["q", "parameters"],
            ["markers"],
        )
        return casadi_fun

    @property
    def nb_markers(self) -> int:
        return sum(model.nb_markers for model in self.models)

    def marker_index(self, name):
        for i, model in enumerate(self.models):
            if name in model.marker_names:
                marker_id = biorbd.marker_index(model.model, name)
                return self.variable_index("markers", model_index=i)[marker_id]

        raise ValueError(f"{name} is not in the MultiBiorbdModel")

    def marker(self, index, reference_segment_index=None) -> Function:
        local_marker_id, model_id = self.local_variable_id("markers", index)
        q_model = self.q[self.variable_index("q", model_id)]
        biorbd_return = self.models[model_id].marker(local_marker_id, reference_segment_index)(q_model, self.parameters)
        casadi_fun = Function(
            "marker",
            [self.q, self.parameters],
            [biorbd_return],
            ["q", "parameters"],
            ["marker"],
        )
        return casadi_fun

    @property
    def nb_rigid_contacts(self) -> int:
        """
        Returns the number of rigid contacts.
        Example:
            First contact with axis YZ
            Second contact with axis Z
            nb_rigid_contacts = 2
        """
        return sum(model.nb_rigid_contacts for model in self.models)

    @property
    def nb_contacts(self) -> int:
        """
        Returns the number of contact index.
        Example:
            First contact with axis YZ
            Second contact with axis Z
            nb_contacts = 3
        """
        return sum(model.nb_contacts for model in self.models)

    def rigid_contact_index(self, contact_index) -> tuple:
        """
        Returns the axis index of this specific rigid contact.
        Example:
            First contact with axis YZ
            Second contact with axis Z
            rigid_contact_index(0) = (1, 2)
        """

        model_selected = None
        for i, model in enumerate(self.models):
            if contact_index in self.variable_index("contact", i):
                model_selected = model
            # Note: may not work if the contact_index is not in the first model
        return model_selected.rigid_contact_index(contact_index)

    def markers_velocities(self, reference_index=None) -> Function:
        if reference_index is not None:
            raise RuntimeError("markers_velocities is not implemented yet with reference_index for MultiBiorbdModel")

        biorbd_return = []
        for i, model in enumerate(self.models):
            q_model = self.q[self.variable_index("q", i)]
            qdot_model = self.qdot[self.variable_index("qdot", i)]
            biorbd_return += [model.markers_velocities(reference_index)(q_model, qdot_model, self.parameters)]
        casadi_fun = Function(
            "markers_velocities",
            [self.q, self.qdot, self.parameters],
            [horzcat(*biorbd_return)],
            ["q", "qdot", "parameters"],
            ["markers_velocities"],
        )
        return casadi_fun

    def marker_velocity(self, marker_index: int) -> Function:
        biorbd_return = []
        for i, model in enumerate(self.models):
            q_model = self.q[self.variable_index("q", i)]
            qdot_model = self.qdot[self.variable_index("qdot", i)]
            biorbd_return += [model.marker_velocity(marker_index)(q_model, qdot_model, self.parameters)]
        casadi_fun = Function(
            "marker_velocity",
            [self.q, self.qdot, self.parameters],
            [horzcat(*biorbd_return)],
            ["q", "qdot", "parameters"],
            ["marker_velocity"],
        )
        return casadi_fun

    def tau_max(self) -> Function:
        out_max = MX()
        out_min = MX()
        for i, model in enumerate(self.models):
            q_model = self.q[self.variable_index("q", i)]
            qdot_model = self.qdot[self.variable_index("qdot", i)]
            torque_max, torque_min = model.tau_max()(q_model, qdot_model, self.parameters)
            out_max = vertcat(out_max, torque_max)
            out_min = vertcat(out_min, torque_min)
        casadi_fun = Function(
            "tau_max",
            [self.q, self.qdot, self.parameters],
            [out_max, out_min],
            ["q", "qdot", "parameters"],
            ["tau_max", "tau_min"],
        )
        return casadi_fun

    def rigid_contact_acceleration(self, contact_index, contact_axis) -> Function:
        model_selected = None
        model_idx = -1
        for i, model in enumerate(self.models):
            if contact_index in self.variable_index("contact", i):
                model_selected = model
                model_idx = i
        q_model = self.q[self.variable_index("q", model_idx)]
        qdot_model = self.qdot[self.variable_index("qdot", model_idx)]
        qddot_model = self.qddot[self.variable_index("qddot", model_idx)]
        biorbd_return = model_selected.rigid_contact_acceleration(contact_index, contact_axis)(
            q_model, qdot_model, qddot_model, self.parameters
        )
        casadi_fun = Function(
            "rigid_contact_acceleration",
            [self.q, self.qdot, self.qddot, self.parameters],
            [biorbd_return],
            ["q", "qdot", "qddot", "parameters"],
            ["rigid_contact_acceleration"],
        )
        return casadi_fun

    @property
    def nb_dof(self) -> int:
        return sum(model.nb_dof for model in self.models)

    @property
    def marker_names(self) -> tuple[str, ...]:
        return tuple([name for model in self.models for name in model.marker_names])

    def soft_contact_forces(self) -> Function:
        biorbd_return = MX()
        for i, model in enumerate(self.models):
            q_model = self.q[self.variable_index("q", i)]
            qdot_model = self.qdot[self.variable_index("qdot", i)]
            soft_contact_forces = model.soft_contact_forces()(q_model, qdot_model, self.parameters)
            biorbd_return = vertcat(biorbd_return, soft_contact_forces)
        casadi_fun = Function(
            "soft_contact_forces",
            [self.q, self.qdot, self.parameters],
            [biorbd_return],
            ["q", "qdot", "parameters"],
            ["soft_contact_forces"],
        )
        return casadi_fun

    def reshape_fext_to_fcontact(self):
        raise NotImplementedError("reshape_fext_to_fcontact is not implemented yet for MultiBiorbdModel")

    def normalize_state_quaternions(self) -> Function:
        biorbd_return = MX()
        for i, model in enumerate(self.models):
            q_model = self.q[self.variable_index("q", i)]
            q_normalized = model.normalize_state_quaternions()(q_model)
            biorbd_return = vertcat(biorbd_return, q_normalized)
        casadi_fun = Function(
            "normalize_state_quaternions",
            [self.q],
            [biorbd_return],
            ["q"],
            ["q_normalized"],
        )
        return casadi_fun

    def contact_forces(self) -> Function:
        """external_forces is not implemented yet for MultiBiorbdModel"""
        biorbd_return = MX()
        for i, model in enumerate(self.models):
            q_model = self.q[self.variable_index("q", i)]
            qdot_model = self.qdot[self.variable_index("qdot", i)]
            tau_model = self.tau[self.variable_index("tau", i)]

            contact_forces = model.contact_forces()(q_model, qdot_model, tau_model, [], self.parameters)
            biorbd_return = vertcat(biorbd_return, contact_forces)
        casadi_fun = Function(
            "contact_forces",
            [self.q, self.qdot, self.tau, [], self.parameters],
            [biorbd_return],
            ["q", "qdot", "tau", "external_forces", "parameters"],
            ["contact_forces"],
        )
        return casadi_fun

    def passive_joint_torque(self) -> Function:
        biorbd_return = MX()
        for i, model in enumerate(self.models):
            q_model = self.q[self.variable_index("q", i)]
            qdot_model = self.qdot[self.variable_index("qdot", i)]
            biorbd_return = vertcat(biorbd_return, model.passive_joint_torque()(q_model, qdot_model, self.parameters))
        casadi_fun = Function(
            "passive_joint_torque",
            [self.q, self.qdot, self.parameters],
            [biorbd_return],
            ["q", "qdot", "parameters"],
            ["passive_joint_torque"],
        )
        return casadi_fun

    def ligament_joint_torque(self) -> Function:
        biorbd_return = MX()
        for i, model in enumerate(self.models):
            q_model = self.q[self.variable_index("q", i)]
            qdot_model = self.qdot[self.variable_index("qdot", i)]
            biorbd_return = vertcat(biorbd_return, model.ligament_joint_torque()(q_model, qdot_model, self.parameters))
        casadi_fun = Function(
            "ligament_joint_torque",
            [self.q, self.qdot, self.parameters],
            [biorbd_return],
            ["q", "qdot", "parameters"],
            ["ligament_joint_torque"],
        )
        return casadi_fun

    def ranges_from_model(self, variable: str):
        return [the_range for model in self.models for the_range in model.ranges_from_model(variable)]

    def bounds_from_ranges(self, variables: str | list[str], mapping: BiMapping | BiMappingList = None) -> Bounds:
        return bounds_from_ranges(self, variables, mapping)

    def _var_mapping(self, key: str, range_for_mapping: int | list | tuple | range, mapping: BiMapping = None) -> dict:
        return _var_mapping(key, range_for_mapping, mapping)

    def lagrangian(self):
        raise NotImplementedError("lagrangian is not implemented yet for MultiBiorbdModel")

    def partitioned_forward_dynamics(self, q_u, qdot_u, q_v_init, tau):
        raise NotImplementedError("partitioned_forward_dynamics is not implemented yet for MultiBiorbdModel")

    @staticmethod
    def animate(
        ocp,
        solution,
        show_now: bool = True,
        show_tracked_markers: bool = False,
        viewer: str = "pyorerun",
        n_frames: int = 0,
        **kwargs,
    ):
        from .viewer_bioviz import animate_with_bioviz_for_loop
        from .viewer_pyorerun import animate_with_pyorerun

        if viewer == "bioviz":
            return animate_with_bioviz_for_loop(ocp, solution, show_now, show_tracked_markers, n_frames, **kwargs)
        if viewer == "pyorerun":
            return animate_with_pyorerun(ocp, solution, show_now, show_tracked_markers, **kwargs)



import numpy as np
from casadi import vertcat

from bioptim import (
    MusclesBiorbdModel,
    DynamicsFunctions,
    DynamicsEvaluation,
    ConfigureVariables,
    ExternalForceSetTimeSeries,
)


class WithResidualExternalForces(MusclesBiorbdModel):
    def __init__(self, biorbd_model_path: str, external_force_set: ExternalForceSetTimeSeries=None, with_residual_torque=True):
        """
        Custom muscle-driven model to handle the residual external forces.
        """
        super().__init__(
            biorbd_model_path, external_force_set=external_force_set, with_residual_torque=with_residual_torque
        )
        # TODO: add mesh_file_folder to all BiorbdModel ?
        self.control_configuration += [
            lambda ocp, nlp, as_states, as_controls,
                   as_algebraic_states: ConfigureVariables.configure_translational_forces(
                ocp, nlp, as_states=False, as_controls=True, as_algebraic_states=False, n_contacts=2
            )
        ]

    def dynamics(
            self,
            time,
            states,
            controls,
            parameters,
            algebraic_states,
            numerical_timeseries,
            nlp,
    ):


        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)

        # Get torques
        tau_residual = DynamicsFunctions.get(nlp.controls["tau"], controls)
        mus_activations = DynamicsFunctions.get(nlp.controls["muscles"], controls)
        tau = tau_residual + DynamicsFunctions.compute_tau_from_muscle(nlp, q, qdot, mus_activations, None)

        # Get external forces
        f_ext_residual_value = DynamicsFunctions.get(nlp.controls["contact_forces"], controls)
        f_ext_residual_position = DynamicsFunctions.get(nlp.controls["contact_positions"], controls)
        external_forces = nlp.get_external_forces(
            "external_forces", states, controls, algebraic_states, numerical_timeseries
        )
        # Left
        external_forces[:3] += f_ext_residual_position[:3]
        external_forces[6:9] += f_ext_residual_value[:3]
        # Right
        external_forces[9:12] += f_ext_residual_position[3:6]
        external_forces[15:18] += f_ext_residual_value[3:6]

        ddq = nlp.model.forward_dynamics()(q, qdot, tau, external_forces, nlp.parameters.cx)

        return DynamicsEvaluation(dxdt=vertcat(qdot, ddq), defects=None)


def animate_solution(biorbd_model_path: str,
                     phase_time: float,
                     n_shooting: int,
                     markers_exp: np.ndarray[float],
                     f_ext_exp: np.ndarray[float],
                     q_opt: np.ndarray[float],
                     muscles_opt: np.ndarray[float],
                     f_ext_position_opt: np.ndarray[float],
                     f_ext_value_opt: np.ndarray[float],
                     ):
    """
    Animate the solution with Pyorerun if it is installed.
    """
    try:
        from pyorerun import BiorbdModel, PhaseRerun, PyoMarkers, PyoMuscles
    except:
        raise RuntimeError("To animate the optimal solution, you must install Pyorerun.")

    # Add the model
    model = BiorbdModel(biorbd_model_path)
    model.options.transparent_mesh = False
    model.options.show_gravity = True
    viz = PhaseRerun(np.linspace(0, phase_time, n_shooting + 1))

    # Add experimental markers
    markers = PyoMarkers(data=markers_exp, marker_names=list(model.marker_names), show_labels=False)
    nb_muscles = len(model.muscle_names)
    emgs = PyoMuscles(
        data=np.hstack((muscles_opt, np.zeros((nb_muscles, 1)))),
        muscle_names=list(model.muscle_names),
        mvc=np.ones((nb_muscles, 1)),
    )

    # Add force plates to the animation
    platform_1_corners = np.array([[ 1.61578003e+00,  1.62178003e+00, -1.51845993e-01, -1.47781998e-01],
                                   [ 1.01549994e+00,  5.24955017e-01,  5.22567993e-01, 1.01329993e+00],
                                   [ 9.72140980e-03,  1.04408998e-02,  1.54146001e-04, 5.82982004e-04]])
    platform_2_corners = np.array([[ 1.62208997e+00,  1.61820996e+00, -1.52233002e-01, -1.52447006e-01],
                                   [ 5.01015015e-01,  7.29261017e-03,  8.11328983e-03, 5.00369019e-01],
                                   [ 9.21893978e-03,  6.36424017e-03,  2.45192990e-04, 5.27547002e-04]])
    viz.add_force_plate(num=1, corners=platform_1_corners)
    viz.add_force_plate(num=2, corners=platform_2_corners)
    viz.add_force_plate(num=3, corners=platform_1_corners)
    viz.add_force_plate(num=4, corners=platform_2_corners)
    viz.add_force_data(
        num=1,
        force_origin=f_ext_exp["left_leg"][:3, :],
        force_vector=f_ext_exp["left_leg"][6:9, :],
    )
    viz.add_force_data(
        num=2,
        force_origin=f_ext_exp["right_leg"][:3, :],
        force_vector=f_ext_exp["right_leg"][6:9, :],
    )
    viz.add_force_data(
        num=3,
        force_origin=np.hstack((f_ext_position_opt[:3, :], np.zeros((3, 1)))),
        force_vector=np.hstack((f_ext_value_opt[:3, :], np.zeros((3, 1)))),
    )
    viz.add_force_data(
        num=3,
        force_origin=np.hstack((f_ext_position_opt[3:6, :], np.zeros((3, 1)))),
        force_vector=np.hstack((f_ext_value_opt[3:6, :], np.zeros((3, 1)))),
    )

    # Add the kinematics
    viz.add_animated_model(model, q_opt, tracked_markers=markers, muscle_activations_intensity=emgs)

    # Play
    viz.rerun("OCP optimal solution")

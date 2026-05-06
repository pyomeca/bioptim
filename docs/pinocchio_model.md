# Pinocchio model backend

`PinocchioModel` is an optional backend that mirrors the rigid-body subset of `BiorbdModel`.
It loads URDF models with Pinocchio and wraps Pinocchio CasADi algorithms in the `Function` signatures expected by
bioptim.

## Current equivalent functions

| bioptim / biorbd surface | Pinocchio equivalent used |
| --- | --- |
| `forward_dynamics()` | `pinocchio.casadi.aba` |
| `inverse_dynamics()` | `pinocchio.casadi.rnea` |
| `mass_matrix()` | `pinocchio.casadi.crba` |
| `non_linear_effects()` | `pinocchio.casadi.nonLinearEffects` |
| `center_of_mass()` | `pinocchio.casadi.centerOfMass` |
| `center_of_mass_velocity()` | `pinocchio.casadi.jacobianCenterOfMass(q) @ qdot` |
| `center_of_mass_acceleration()` | `pinocchio.casadi.centerOfMass(q, qdot, qddot)` and `data.acom[0]` |
| `homogeneous_matrices_in_global()` | `pinocchio.casadi.framesForwardKinematics` and `data.oMf[frame_id]` |
| `marker()` / `markers()` | Pinocchio frames selected through `marker_names` |
| `marker_velocity()` | `pinocchio.casadi.forwardKinematics` and `getFrameVelocity` |
| `marker_acceleration()` | second-order `pinocchio.casadi.forwardKinematics` and `getFrameAcceleration` |
| `markers_jacobian()` | `pinocchio.casadi.computeFrameJacobian` |

The unavailable biorbd-specific surfaces currently raise `NotImplementedError`: muscles, ligaments, passive torques,
soft contacts, rigid contact dynamics, biorbd external forces, animation and model parameter callbacks.

## Analytical derivatives

Pinocchio exposes analytical derivatives for the rigid-body algorithms that dominate torque-driven dynamics:

| Quantity | Pinocchio function | Data returned by `PinocchioModel` |
| --- | --- | --- |
| `d aba / d(q, qdot, tau)` | `pinocchio.casadi.computeABADerivatives` | `ddq_dq`, `ddq_dqdot`, `ddq_dtau` |
| `d rnea / d(q, qdot, qddot)` | `pinocchio.casadi.computeRNEADerivatives` | `dtau_dq`, `dtau_dqdot`, `dtau_dqddot` |

These are exposed as CasADi `Function`s through `forward_dynamics_derivatives()` and
`inverse_dynamics_derivatives()`. They are not yet wired into bioptim's NLP graph construction, which still relies on
CasADi differentiation of the assembled symbolic graph. The next optimization step would be to add an optional derivative
provider interface in the dynamics layer so solvers can use these functions when the selected backend offers them.

On the biorbd side, the current bioptim integration already benefits from CasADi algorithmic differentiation through
`biorbd_casadi`, but this codebase does not expose a comparable first-class analytical ABA/RNEA derivative API to call
from `BiorbdModel`.

References used while mapping the API:

- Pinocchio automatic differentiation documentation: https://docs.ros.org/en/rolling/p/pinocchio/doc/a-features/k-automatic-differentiation.html
- Pinocchio frame Jacobian documentation: https://docs.ros.org/en/melodic/api/pinocchio/html/namespacepinocchio.html
- Pinocchio ABA derivatives API: https://docs.ros.org/en/ros2_packages/rolling/api/pinocchio/generated/file_include_pinocchio_algorithm_aba-derivatives.hxx.html
- Pinocchio RNEA derivatives API: https://docs.ros.org/en/rolling/p/pinocchio/generated/file_include_pinocchio_algorithm_rnea-derivatives.hxx.html


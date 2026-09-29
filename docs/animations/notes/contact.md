# Rigid contact and unilateral contact force (scene `UnilateralContact`, ~14 s)

## What it teaches
A vertical "leg" (`models/contact_leg.bioMod`: a 5 kg foot carrying a rigid contact along z, and a 70 kg body sliding
on it; `tau` on the body is an internal force) extends from 0.4 m to 0.9 m in 0.4 s, at rest at both ends, minimizing
`MINIMIZE_CONTROL` on `tau` (weight 1e-4, RK4, 40 nodes, |tau| <= 5000 N). The model is declared with
`TorqueBiorbdModel(path, contact_types=[ContactType.RIGID_EXPLICIT])`: the foot cannot move vertically and the floor
reacts with a normal force `F` read from the model (`model.rigid_contact_forces()`, the same function used by the penalty).

* No constraint on `F`: IPOPT status 0 (8 iterations), cost 41.8. F falls linearly from 2016 N to -545 N (weight is
  736 N); it is negative from t = 0.31 s (last 9 nodes): the floor would have to pull the foot down.
* Add `ConstraintFcn.TRACK_EXPLICIT_RIGID_CONTACT_FORCES, node=Node.ALL_SHOOTING, contact_index=0, min_bound=0,
  max_bound=np.inf`: IPOPT status 0 (17 iterations), cost 44.7. F >= 0 everywhere and the last 15 nodes (0.15 s) sit
  exactly at F = 0 (the body brakes in free fall at -g*M/m_body = -10.5 m/s2, which is the hardest braking a pushing-only
  floor allows). The leg trajectory changes only slightly (bottom plot, grey = unconstrained), the force profile changes a lot.

All numbers on screen are read from `data/contact_leg.npz` (real solves).

## Commands
From the repo root (conda env `captury_biobuddy` on the PATH): `PYTHONPATH=. python docs/animations/generate_contact_data.py`
(about 5 s, writes `data/contact_leg.npz`), then from `docs/animations`: `manim render -qh anim_contact.py UnilateralContact`.

## Honest caveats
* API of THIS version of the code: `contact_types` is an argument of the model (`TorqueBiorbdModel(..., contact_types=...)`),
  NOT of `DynamicsOptions`, and the force constraint is `ConstraintFcn.TRACK_EXPLICIT_RIGID_CONTACT_FORCES`
  (`TRACK_CONTACT_FORCES` does not exist any more). `bioptim/examples/toy_examples/torque_driven_ocp/example_rigid_contact.py`
  still uses the old names, so it does not run as is; I also tried its 3-segment leg model with RIGID_EXPLICIT and IPOPT
  stopped at the initial point (NaN: the free-flight dynamics of that model are extremely stiff), hence the simpler model.
* The unilateral solve started cold (linear interpolation guess) ends in "local infeasibility" (IPOPT status 1); the
  reported solution is warm-started from the unconstrained one and converges (status 0). The scene shows only the converged one.
* The contact is rigid during the whole motion (the foot never leaves the floor): with F = 0 the model is in the
  take-off limit but there is no flight phase (that would need two phases and `PhaseTransitionFcn`, see the impact scene).
* F is constrained at the shooting nodes (`Node.ALL_SHOOTING`), not continuously; the contact force is constant over each interval
  (piecewise-constant tau), so the staircase is exact for this discretization. Playback is slowed down ~8.5x (0.4 s in 3.4 s).
* One friction-free contact along z only; no tangential force, no slipping.

## Exercises
1. Shorten the duration to 0.3 s (use continuation from 0.5 s, cold starts fail): how does the F = 0 segment grow, and at what duration does the problem become infeasible?
2. Replace the constraint by `ConstraintFcn.TRACK_EXPLICIT_RIGID_CONTACT_FORCES` with `min_bound=100`: what does the body do at the end of the motion?
3. Use `ContactType.RIGID_IMPLICIT` (with `TRACK_ALGEBRAIC_STATE, key="rigid_contact_forces"`) and compare with the explicit result.

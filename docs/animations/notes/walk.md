# Hop cycle: contact phases and IMPACT (scene `Hopper`, ~15 s)

## What it teaches
A vertical "leg" (`models/walk_hopper.bioMod`: a 5 kg foot carrying a rigid contact along z, and a 70 kg body sliding
on it; `tau` on the body is an INTERNAL force, i.e. the leg actuator) performs one periodic hop in three phases:

| phase | model | what happens |
|-------|-------|--------------|
| 0, 10 nodes, 0.175 s | `TorqueBiorbdModel(M)` (no contact) | free fall from the apex (foot 0.15 m high, leg 0.9 m, at rest); `tau` is bounded to 0, so it is a pure ballistic flight |
| transition | `PhaseTransitionFcn.IMPACT`, `phase_pre_idx=0` | touch-down: q continuous, foot velocity -1.72 -> 0 m/s |
| 1, 20 nodes, 0.3 s | `TorqueBiorbdModel(M, contact_types=[ContactType.RIGID_EXPLICIT])` | stance, foot fixed at z = 0; `TRACK_EXPLICIT_RIGID_CONTACT_FORCES` `min_bound=0` on all shooting nodes (a floor cannot pull), and `min_bound=max_bound=0` on `Node.PENULTIMATE` (take-off) |
| transition | `PhaseTransitionFcn.CONTINUOUS`, `phase_pre_idx=1` | take-off |
| 2, 12 nodes, 0.2 s | `TorqueBiorbdModel(M)` | flight back to the apex configuration at rest (periodic hop), `|tau| <= 400 N` (weak leg spring in the air) |

Objective: `MINIMIZE_CONTROL` on `tau` in every phase (weight 1e-4), RK4 (5 steps per interval), IPOPT.

What the video shows (all numbers computed from `data/walk_hopper.npz`, a real solve, IPOPT status 0, 16 iterations, cost 85.5):
* vertical velocities of the foot and of the body (absolute) with the phase colours: the FOOT velocity jumps from
  -1.72 m/s to 0 at the impact, the BODY velocity does not jump (the impulse acts through the contact, on the foot
  only: 8.6 N.s = 5 kg x 1.72 m/s, 7.4 J of kinetic energy lost);
* the ground reaction force during the stance (piecewise constant per interval): it grows from 927 N to 2497 N (peak 3.4 x the
  weight, 736 N) and is 0 in the last interval (take-off constraint); the red band F < 0 is never entered;
* the exact Bioptim lines (as in `generate_walk_data.py`), revealed with the phase they set up.

## Commands
From the repo root (conda env `captury_biobuddy` on the PATH): `PYTHONPATH=. python docs/animations/generate_walk_data.py`
(about 2 s, writes `data/walk_hopper.npz`), then from `docs/animations`: `manim render -qh anim_walk.py Hopper`.

## Honest caveats
* IMPACT is inelastic, frictionless, with a point contact (biorbd `ComputeConstraintImpulsesDirect` on the model contacts).
  There is no horizontal motion, no foot slip, no leg segments/knee: this is a 1-D hopper, not a walking model.
* The stance force is NOT the human-like half sine. Minimum-effort `tau` with piecewise-constant controls gives a
  ramp that drops to 0 in the last interval: the take-off condition `F = 0` is imposed at the last shooting node only
  (`Node.PENULTIMATE` is the last shooting node with the explicit-contact constraint), so the drop lasts one interval (15 ms).
* The constraint `F >= 0` is NOT binding here: I re-solved without the two force constraints (warm start) and F stays
  > 0 (min 1149 N), cost 77.3 instead of 85.5; the extra cost comes from the take-off condition. The scene says so.
* Phase 0 is pure free fall (`tau` fixed to 0, its duration sqrt(2 x 0.15 / 9.81) is imposed). The periodic condition
  (foot at 0.15 m, leg 0.9 m, at rest at the end of phase 2, with fixed phase durations) leaves almost no freedom: IPOPT warns
  "NLP is overconstrained" (more equalities than variables, still status 0), and the foot rises and falls in the second flight
  because the leg spring (`tau` between -153 and +55 N) has to bring it back. Free phase durations would be more natural.
* The floor is z = 0 by construction (foot z is fixed to 0 in the stance bounds). Playback is slowed down about 12x (0.675 s in ~8 s).
* Plain API of THIS version: `contact_types` is an argument of the model, not of `DynamicsOptions`; the per-phase models are given as a tuple.

## Exercises
1. Set `contact_min=False` in `build` (no force constraints): check that F stays positive, then raise `Z0` (drop height)
   or lower `TAU_MAX` until the `F >= 0` constraint becomes active.
2. Replace `PhaseTransitionFcn.IMPACT` by `PhaseTransitionFcn.CONTINUOUS`: what does IPOPT report, and why (see the `Impact` scene)?
3. Make the phase durations free (`ObjectiveFcn.Mayer.MINIMIZE_TIME` with `phase_dynamics`/`bounds` on the phase times) and compare the flight time with the symmetric ballistic value.

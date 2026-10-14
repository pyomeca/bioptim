# Soft (compliant) contact: SoftContact

Scene `SoftContact` in `anim_soft.py` (about 15.6 s). Data: `data/soft_ball.npz`, produced by `generate_soft_data.py`
(real bioptim / IPOPT solves). Model: `models/soft_ball.bioMod` (1 kg ball, translation z, one soft-contact sphere of
radius 0.05 m, ground at z = 0).

## What it teaches
- A soft contact is declared in the bioMod (`softcontact ... type sphere ... stiffness ... damping ...`) and switched on
  in Python with `TorqueBiorbdModel(path, contact_types=[ContactType.SOFT_EXPLICIT])` (names checked in
  `bioptim/misc/enums.py`; `SOFT_IMPLICIT` also exists and adds the contact forces as decision variables).
- Beat 1: ball pushed 1 cm above the ground to 3 cm of penetration (tracked half-cosine, stiffness 1e4). Depth and force
  come from the solution; force = `bio_model.soft_contact_forces()` (index 5, vertical).
- Beat 2: same push with stiffness 1e4, 1e5, 1e6 (warm started). Stiffer ground: same depth needs 100x more force, so
  the ball cannot reach the reference at 1e6 (control cost), and IPOPT needs more iterations (10, 14, 60).
- Integration step: at 1e6, RK4 with 1 step per interval fails (status 1, 500 iterations); 5 and 20 steps give the same
  peak force (370 N).

## Honest caveats
- Force law measured on this model (not from documentation): F = c * k * depth^1.5 * (1 + 1.5 * damping * speed) with
  c = 0.298 for depth > 1 cm (smaller c at very small depth); log-log slope 1.514; damping factor 1.300 at 0.1 m/s.
- Reference tracking with control weight 1e-6: at 1e6 the depth reaches only 1.1 cm. Lowering the weight to 1e-7 or 1e-8
  made the 1e6 solve fail (status 1), so 1e-6 is kept for all three.
- Force decays over the last 2 to 3 nodes (no final velocity constraint, last interval control effect); this is in the data.
- No rigid-contact comparison (different formulation, not cheap to make honest).
- Render needs `--disable_caching` (manim hashing fails with the always_redraw closures).

## Commands
    PYTHONPATH=. python docs/animations/generate_soft_data.py        # about 25 s
    manim render -qh --disable_caching anim_soft.py SoftContact       # from docs/animations

## Exercises
1. Change `damping` in the bioMod (0, 2, 20) and compare the force during the descent.
2. Switch to `ContactType.SOFT_IMPLICIT` and compare iterations at stiffness 1e6.
3. Change `WEIGHT_TAU` and find when the stiffest solve stops converging.

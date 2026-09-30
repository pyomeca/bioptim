# Learning Bioptim with short animations

*Version française : [README.fr.md](README.fr.md).*

A series of **48 short videos** (between 14 and 106 seconds each, median 31 s, about 27 minutes per language; English and French) that teach
[Bioptim](https://github.com/pyomeca/bioptim), the optimal control library for biomechanics, one idea at a time. The
videos are made with [Manim Community](https://www.manim.community/) and every curve shown comes from a **real Bioptim
solve** (IPOPT), not from a sketch.

It is written for students and researchers in biomechanics and optimal control who know some Python and mechanics, but
not necessarily what a direct multiple shooting is. Each video points to the matching Bioptim example and to the exact
lines of the library code, so the animation is a door into the code rather than a replacement for it.

## How to use the series

1. Watch a video, then open its **example** and **code** links (table below) and run the example.
2. Use the level column: **1** = introductory, **2** = intermediate, **3** = advanced (needs the previous ideas).
3. Follow the sections in this order, or jump to what you need:
   *Fundamentals* -> *Discretization* -> *Objectives and constraints* -> *Phases and time* -> *Solvers and numerics* ->
   *Models and biomechanics* -> *Advanced control*. *Library overview* can be watched at any time.
4. Each video has a notes file (what was computed, numbers, honest limits) and a data generator you can edit and
   re-run.

### Start here

<!-- BEGIN START -->

| Scene (mp4) | Title | Why start here |
| --- | --- | --- |
| `OCPStatement` | **What is an OCP?** | The vocabulary: state, control, dynamics, cost, constraints, bounds. |
| `MultipleShooting` | **Direct multiple shooting** | How Bioptim turns an OCP into an NLP (nodes, defects). |
| `ArchitecturePath` | **From inputs to Solution** | The map of the library: from your inputs to the Solution. |
| `ObjectivesNodes` | **Lagrange, Mayer and nodes** | Lagrange versus Mayer terms and the Node enum. |
| `SolutionTour` | **Reading the Solution object** | How to read and post-process a solved problem. |
| `TrackState` | **Tracking a reference** | A first realistic objective: tracking a reference. |

<!-- END START -->

Then: `ControlTypes` (how controls are interpolated), `ConstraintsBounds`, `IpoptIterates`, and one model of your field
(`MuscleReaching`, `ExternalForces`, `Hopper`...).

## The 48 videos

Scene names are also the mp4 names (`<Scene>_en.mp4`, `<Scene>_fr.mp4`). "Example and code" links open the Bioptim
example and the relevant library lines; "Sources" gives the Manim scene, the script that produced the data and the notes.
This table is generated: see [Regenerating the tables](#regenerating-the-tables).

<!-- BEGIN TABLE -->

#### Fundamentals

| Scene (mp4) | Title and content | Level | Example and code | Sources |
| --- | --- | --- | --- | --- |
| `OCPStatement` | **What is an OCP?**<br>Writes the optimal control problem as a table mapping each mathematical piece (Lagrange and Mayer cost, dynamics, boundary values, bounds, path constraints) to its Bioptim call, then shows the pendulum swing-up… | 1 - introductory | [Example: pendulum swing-up](../../bioptim/examples/toy_examples/sqp_method/pendulum.py#L22-L97)<br>[Objective class](../../bioptim/limits/objective_functions.py#L21-L181)<br>[Constraint class](../../bioptim/limits/constraints.py#L25-L162)<br>[OptimalControlProgram constructor](../../bioptim/optimization/optimal_control_program.py#L160-L338) | [scene](dms_vs_dc.py) · [data generator](generate_pendulum_data.py) · [notes](README.md) |

#### Discretization

| Scene (mp4) | Title and content | Level | Example and code | Sources |
| --- | --- | --- | --- | --- |
| `TimeGrid` | **Discretizing time into nodes**<br>The horizon T is cut into N shooting intervals of length T/N. States live at the nodes and controls are piecewise constant on each interval. | 1 - introductory | [Example: pendulum swing-up (n_shooting)](../../bioptim/examples/toy_examples/sqp_method/pendulum.py#L22-L97)<br>[Node enum](../../bioptim/misc/enums.py#L33-L47) | [scene](dms_vs_dc.py) · [data generator](generate_pendulum_data.py) · [notes](README.md) |
| `MultipleShooting` | **Direct multiple shooting**<br>On the pendulum swing-up (20 shooting intervals), shows a window of nodes with OdeSolver.RK4(n_integration_steps=5): each interval is integrated from x_k, and the continuity constraint F(x_k, u_k) = x_(k+1) is a defect… | 1 - introductory | [Example: pendulum swing-up](../../bioptim/examples/toy_examples/sqp_method/pendulum.py#L22-L97)<br>[OdeSolver.RK4](../../bioptim/dynamics/ode_solvers.py#L35-L42)<br>[RK4 integration step](../../bioptim/dynamics/integrator.py#L366-L379)<br>[State continuity constraint](../../bioptim/limits/penalty.py#L1224-L1276) | [scene](dms_vs_dc.py) · [data generator](generate_pendulum_data.py) · [notes](README.md) |
| `Comparison` | **Multiple shooting vs collocation**<br>A table compares direct multiple shooting (OdeSolver.RK4) and direct collocation (OdeSolver.COLLOCATION) on the same pendulum: extra unknowns, constraints per interval, dynamics evaluations, approximation order, and… | 1 - introductory | [Example: pendulum with RK4 or collocation](../../bioptim/examples/toy_examples/sqp_method/pendulum.py#L22-L97)<br>[OdeSolver classes (RK4, COLLOCATION)](../../bioptim/dynamics/ode_solvers.py#L12-L357)<br>[COLLOCATION integrator](../../bioptim/dynamics/integrator.py#L569-L695) | [scene](dms_vs_dc.py) · [data generator](generate_pendulum_data.py) · [notes](README.md) |
| `DirectCollocation` | **Direct collocation**<br>Shows OdeSolver.COLLOCATION(polynomial_degree=3, method='legendre'): collocation points, the Lagrange polynomial per interval, and the defects (polynomial slope minus dynamics) that IPOPT drives to zero, over real… | 2 - intermediate | [Example: pendulum swing-up](../../bioptim/examples/toy_examples/sqp_method/pendulum.py#L22-L97)<br>[OdeSolver.COLLOCATION](../../bioptim/dynamics/ode_solvers.py#L111-L214)<br>[Lagrange interpolation](../../bioptim/dynamics/lagrange_interpolation.py#L15-L202) | [scene](dms_vs_dc.py) · [data generator](generate_pendulum_data.py) · [notes](README.md) |
| `CollocationDegree` | **Collocation degree and points**<br>Shows where the collocation points fall inside one interval for degrees 2 to 6 (Legendre: interior, Radau: the last one is the next node), then the dynamics error, number of variables and IPOPT iterations of real solves. | 2 - intermediate | [OdeSolver.COLLOCATION options](../../bioptim/dynamics/ode_solvers.py#L111-L170)<br>[COLLOCATION integrator (points)](../../bioptim/dynamics/integrator.py#L569-L695) | [scene](anim_colloc.py) · [data generator](generate_colloc_data.py) · [notes](notes/colloc.md) |
| `ControlTypes` | **Control interpolation types**<br>Solves the same pendulum swing-up with ControlType CONSTANT, LINEAR_CONTINUOUS and CONSTANT_WITH_LAST_NODE and shows the torque curves and decision-vector sizes (125, 127, 127). | 2 - intermediate | [Example with control_type argument](../../bioptim/examples/getting_started/basic_ocp.py#L33-L127)<br>[ControlType enum](../../bioptim/misc/enums.py#L126-L170)<br>[OptimalControlProgram constructor](../../bioptim/optimization/optimal_control_program.py#L160-L338) | [scene](anim_controls.py) · [data generator](generate_controls_data.py) · [notes](notes/controls.md) |
| `DiscreteMechanics` | **Discrete mechanics (variational)**<br>With VariationalOptimalControlProgram the state is only q_k and the dynamics are the discrete Euler-Lagrange equations. | 3 - advanced | [Example: variational integrator pendulum](../../bioptim/examples/toy_examples/discrete_mechanics_and_optimal_control/example_variational_integrator_pendulum.py#L21-L86)<br>[VariationalOptimalControlProgram](../../bioptim/optimization/variational_optimal_control_program.py#L33-L377)<br>[Discrete Euler-Lagrange equations](../../bioptim/models/biorbd/variational_biorbd_model.py#L163-L232)<br>[OdeSolver.VARIATIONAL (placeholder)](../../bioptim/dynamics/ode_solvers.py#L53-L61) | [scene](anim_variational.py) · [data generator](generate_variational_data.py) · [notes](notes/variational.md) |

#### Objectives and constraints

| Scene (mp4) | Title and content | Level | Example and code | Sources |
| --- | --- | --- | --- | --- |
| `TrackState` | **Tracking a reference**<br>The pendulum angle must follow a sine reference with ObjectiveFcn.Lagrange.TRACK_STATE at weights 1, 30 and 1000, then with a hard ConstraintFcn.TRACK_STATE. A higher weight tracks better but needs more effort. | 1 - introductory | [Example: cyclic tracking objective](../../bioptim/examples/toy_examples/moving_horizon_estimation/multi_cyclic_nmpc_with_parameters.py#L178-L186)<br>[minimize_states (TRACK_STATE)](../../bioptim/limits/penalty.py#L60-L86)<br>[ObjectiveFcn (Lagrange, Mayer)](../../bioptim/limits/objective_functions.py#L362-L522) | [scene](anim_track.py) · [data generator](generate_track_data.py) · [notes](notes/track.md) |
| `ObjectivesNodes` | **Lagrange, Mayer and nodes**<br>Shows on a schematic node grid which nodes the Node enum selects (START, INTERMEDIATES, PENULTIMATE, END, ALL_SHOOTING, ALL), then five real pendulum swing-up solves with increasing Mayer weight on the final angle. | 2 - intermediate | [Example: basic pendulum OCP](../../bioptim/examples/getting_started/basic_ocp.py#L33-L127)<br>[Node enum](../../bioptim/misc/enums.py#L33-L47)<br>[Lagrange and Mayer objective functions](../../bioptim/limits/objective_functions.py#L362-L483)<br>[How nodes are turned into indices](../../bioptim/limits/penalty_option.py#L989-L1046) | [scene](features_scenes.py) · [data generator](generate_features_data.py) · [notes](FEATURES.md) |
| `ConstraintsBounds` | **Bounds on controls and states**<br>The same pendulum swing-up is solved with a torque bound u_bounds that shrinks from 100 to 12 N, then with a bound on the cart position x_bounds. | 2 - intermediate | [Example: bounds on states and controls](../../bioptim/examples/toy_examples/feature_examples/pendulum_constrained_states_controls.py#L35-L126)<br>[Example: pendulum with x_bounds and u_bounds](../../bioptim/examples/getting_started/basic_ocp.py#L33-L127)<br>[Bounds and BoundsList classes](../../bioptim/limits/path_conditions.py#L339-L700) | [scene](features_scenes.py) · [data generator](generate_features_data.py) · [notes](FEATURES.md) |
| `Parameters` | **Optimizing a parameter**<br>A scalar parameter, the peak torque max_tau, is declared with ParameterList and optimized together with the trajectory. | 2 - intermediate | [Example: custom parameters](../../bioptim/examples/getting_started/custom_parameters.py#L89-L252)<br>[Example: minimize the maximum torque with a parameter](../../bioptim/examples/toy_examples/torque_driven_ocp/minimize_maximum_torque_by_extra_parameter.py#L47-L135)<br>[ParameterList class](../../bioptim/optimization/parameters.py#L149-L298) | [scene](features_scenes.py) · [data generator](generate_features_data.py) · [notes](FEATURES.md) |
| `DerivativePenalty` | **Penalty on a control derivative**<br>On a cart-pendulum swing-up, adds MINIMIZE_CONTROL with derivative=True on tau with weights 0, 1, 10 and 100. | 2 - intermediate | [Base example: swing-up OCP](../../bioptim/examples/getting_started/basic_ocp.py#L33-L127)<br>[MINIMIZE_CONTROL penalty](../../bioptim/limits/penalty.py#L89-L113)<br>[derivative=True: difference end minus start](../../bioptim/limits/penalty_option.py#L580-L614)<br>[Alternative: TorqueDerivativeBiorbdModel](../../bioptim/models/biorbd/model_dynamics.py#L269-L293) | [scene](anim_deriv.py) · [data generator](generate_deriv_data.py) · [notes](notes/deriv.md) |
| `MultinodeLink` | **Multinode link**<br>Links variables at distant nodes, here the last node to the first for a cyclic motion, either as a constraint (MultinodeConstraintFcn.STATES_EQUALITY) or as a weighted objective. | 3 - advanced | [Example: multinode constraints](../../bioptim/examples/toy_examples/feature_examples/example_multinode_constraints.py#L67-L204)<br>[Example: multinode objective](../../bioptim/examples/toy_examples/feature_examples/example_multinode_objective.py#L36-L118)<br>[MultinodeConstraintList and Fcn](../../bioptim/limits/multinode_constraint.py#L84-L164)<br>[MultinodeObjectiveList and Fcn](../../bioptim/limits/multinode_objective.py#L55-L122) | [scene](anim_multinode.py) · [data generator](generate_multinode_data.py) · [notes](notes/multinode.md) |

#### Phases and time

| Scene (mp4) | Title and content | Level | Example and code | Sources |
| --- | --- | --- | --- | --- |
| `MultiphaseTransitions` | **Multiphase problems and transitions**<br>A pendulum swing-up split in two phases of different durations (0.5 s and 1.0 s, 12 and 18 nodes). | 2 - intermediate | [Example: multiphase OCP](../../bioptim/examples/getting_started/example_multiphase.py#L39-L189)<br>[PhaseTransitionFcn enum](../../bioptim/limits/phase_transition.py#L355-L375)<br>[CONTINUOUS transition (states equality)](../../bioptim/limits/phase_transition.py#L157-L185)<br>[DISCONTINUOUS transition (no link)](../../bioptim/limits/phase_transition.py#L222-L238) | [scene](features_scenes.py) · [data generator](generate_features_data.py) · [notes](FEATURES.md) |
| `FreeTime` | **Free-time optimization**<br>With ObjectiveFcn.Mayer.MINIMIZE_TIME the phase duration becomes a decision variable; phase_time is only an initial guess. | 2 - intermediate | [Example: minimum-time pendulum](../../bioptim/examples/toy_examples/optimal_time_ocp/pendulum_min_time_Mayer.py#L30-L112)<br>[Mayer minimize_time implementation](../../bioptim/limits/objective_functions.py#L288-L311)<br>[Mayer objective list (MINIMIZE_TIME)](../../bioptim/limits/objective_functions.py#L428-L483) | [scene](features_scenes.py) · [data generator](generate_features_data.py) · [notes](FEATURES.md) |
| `Impact` | **Impact phase transition**<br>A 1 kg point mass falls onto the floor and then slides on it (two phases). | 3 - advanced | [Example using PhaseTransitionFcn.IMPACT](../../bioptim/examples/getting_started/custom_phase_transitions.py#L68-L204)<br>[IMPACT transition implementation](../../bioptim/limits/phase_transition.py#L260-L306)<br>[Model impulse: qdot_from_impact](../../bioptim/models/biorbd/biorbd_model.py#L824-L835) | [scene](features_scenes.py) · [data generator](generate_features_data.py) · [notes](FEATURES.md) |
| `Hopper` | **Hop cycle with impact**<br>A 1-D hopper (foot and body on a vertical axis) does one periodic hop in three phases: flight, stance with a rigid contact and a non-negative force, flight. | 3 - advanced | [Example: phase transitions incl. IMPACT](../../bioptim/examples/getting_started/custom_phase_transitions.py#L68-L204)<br>[Example: rigid contact](../../bioptim/examples/toy_examples/torque_driven_ocp/example_rigid_contact.py#L40-L138)<br>[IMPACT transition implementation](../../bioptim/limits/phase_transition.py#L260-L306)<br>[Rigid contact force penalty](../../bioptim/limits/penalty.py#L768-L817) | [scene](anim_walk.py) · [data generator](generate_walk_data.py) · [notes](notes/walk.md) |

#### Solvers and numerics

| Scene (mp4) | Title and content | Level | Example and code | Sources |
| --- | --- | --- | --- | --- |
| `OnlineIterates` | **Online plots during the solve**<br>Redraws, in Manim, the window shown by Solver.IPOPT(show_online_optim=True) from real IPOPT iterates of a pendulum swing-up: states, controls, a custom plot and IPOPT output, next to the code that enables them. | 2 - intermediate | [Example: custom plotting](../../bioptim/examples/getting_started/custom_plotting.py#L38-L113)<br>[IPOPT solver options](../../bioptim/interfaces/ipopt_options.py#L19-L348)<br>[OnlineOptim enum](../../bioptim/misc/enums.py#L98-L123)<br>[IPOPT output plot](../../bioptim/gui/ipopt_output_plot.py#L10-L63)<br>[OCP.add_plot](../../bioptim/optimization/optimal_control_program.py#L1161-L1206) | [scene](anim_online.py) · [data generator](generate_online_data.py) · [notes](notes/online.md) |
| `IpoptIterates` | **IPOPT iterates**<br>Shows that an IPOPT iterate is generally not a feasible trajectory: for iterations 0 to 51 the objective and the primal infeasibility are displayed with the trajectory. | 2 - intermediate | [IPOPT solver options](../../bioptim/interfaces/ipopt_options.py#L19-L348)<br>[set_maximum_iterations](../../bioptim/interfaces/ipopt_options.py#L240-L241)<br>[Example: pendulum swing-up](../../bioptim/examples/getting_started/basic_ocp.py#L33-L127) | [scene](anim_solver.py) · [data generator](generate_solver_data.py) · [notes](notes/solver.md) |
| `IpoptMultiStart` | **Multi-start: several local minima**<br>The pendulum swing-up is solved from different initial guesses: IPOPT converges each time, but to different local minima (costs from 40.3 to 302.9 are shown). | 2 - intermediate | [Example: multi-start](../../bioptim/examples/getting_started/example_multistart.py#L29-L121)<br>[MultiStart class](../../bioptim/optimization/multi_start.py#L18-L115)<br>[IPOPT: set_maximum_iterations](../../bioptim/interfaces/ipopt_options.py#L240-L241)<br>[IPOPT: set_tol](../../bioptim/interfaces/ipopt_options.py#L216-L217) | [scene](anim_solver.py) · [data generator](generate_solver_data.py) · [notes](notes/solver.md) |
| `VectorCollocation` | **Decision vector with collocation**<br>With direct collocation of degree 3, each state node holds 4 columns (the node plus 3 collocation points) in the flat vector seen by IPOPT. The scene zooms on one node as a 4x4 grid filled column by column, on a small… | 2 - intermediate | [VectorLayout (index_map)](../../bioptim/optimization/vector_layout.py#L110-L286)<br>[OrderingStrategy](../../bioptim/optimization/vector_layout.py#L105-L107)<br>[n_states_decision_steps](../../bioptim/optimization/non_linear_program.py#L433-L445) | [scene](anim_vector.py) · [data generator](generate_vector_data.py) · [notes](notes/vector.md) |
| `AccuracyCheck` | **Checking the solution accuracy**<br>Four solves of the same pendulum swing-up (RK4 with 1 or 5 steps, collocation of degree 3 or 5) are re-integrated from the initial state with Solution.integrate(Shooting.SINGLE, DOP853). | 3 - advanced | [Example: single-shooting reintegration](../../bioptim/examples/getting_started/example_simulation.py)<br>[Solution.integrate](../../bioptim/optimization/solution/solution.py#L785-L866)<br>[Shooting enum](../../bioptim/misc/enums.py#L64-L74)<br>[SolutionIntegrator enum](../../bioptim/misc/enums.py#L184-L194) | [scene](anim_accuracy.py) · [data generator](generate_accuracy_data.py) · [notes](notes/accuracy.md) |
| `PhaseDynamicsScene` | **PhaseDynamics: shared or per node**<br>Compares SHARED_DURING_THE_PHASE (one integrator function for the phase) with ONE_PER_NODE (one per node): measured build and solve times for 30, 60 and 120 nodes, identical optimum, and a multinode constraint on 7… | 3 - advanced | [Example: time-dependent pendulum](../../bioptim/examples/toy_examples/torque_driven_ocp/example_pendulum_time_dependent.py#L82-L166)<br>[PhaseDynamics enum](../../bioptim/misc/enums.py#L5-L7)<br>[One integrator vs one per node](../../bioptim/dynamics/ode_solver_base.py#L268-L339) | [scene](anim_phasedyn.py) · [data generator](generate_phasedyn_data.py) · [notes](notes/phasedyn.md) |
| `Scaling` | **Variable scaling**<br>VariableScalingList (x_scaling, u_scaling) changes only what the solver sees (x divided by scale); bounds and objectives stay in physical units. | 3 - advanced | [Example: variable scaling](../../bioptim/examples/toy_examples/feature_examples/example_variable_scaling.py#L27-L111)<br>[VariableScalingList](../../bioptim/optimization/variable_scaling.py#L67-L136)<br>[VariableScaling](../../bioptim/optimization/variable_scaling.py#L17-L64) | [scene](anim_scaling.py) · [data generator](generate_scaling_data.py) · [notes](notes/scaling.md) |
| `SxVsMx` | **SX versus MX graphs**<br>use_sx=True builds the CasADi graph with SX instead of MX. A schematic of the two graph types is followed by measured times on one pendulum OCP (N = 50): IPOPT time drops from 0.54 to 0.14 s but solver set-up grows… | 3 - advanced | [Example exposing use_sx](../../bioptim/examples/getting_started/basic_ocp.py#L33-L127)<br>[use_sx in OptimalControlProgram](../../bioptim/optimization/optimal_control_program.py#L523-L547) | [scene](anim_sx.py) · [data generator](generate_sx_data.py) · [notes](notes/sx.md) |
| `VectorOrdering` | **Decision vector ordering**<br>Shows the flat vector IPOPT sees (time, states, controls, parameters) and how VectorLayout.index_map locates each block, then reorders the same 24 numbers with OrderingStrategy.TIME_MAJOR. Reading results through… | 3 - advanced | [VectorLayout](../../bioptim/optimization/vector_layout.py#L110-L286)<br>[OrderingStrategy](../../bioptim/optimization/vector_layout.py#L105-L107)<br>[OptimizationVectorHelper](../../bioptim/optimization/optimization_vector.py#L20-L283)<br>[Example: basic OCP](../../bioptim/examples/getting_started/basic_ocp.py#L33-L127) | [scene](anim_vector.py) · [data generator](generate_vector_data.py) · [notes](notes/vector.md) |

#### Models and biomechanics

| Scene (mp4) | Title and content | Level | Example and code | Sources |
| --- | --- | --- | --- | --- |
| `UnilateralContact` | **Unilateral contact force**<br>A vertical leg (5 kg foot, 70 kg body) extends against the floor with a rigid contact. | 2 - intermediate | [Example: rigid contact (older API names)](../../bioptim/examples/toy_examples/torque_driven_ocp/example_rigid_contact.py#L40-L138)<br>[ContactType enum](../../bioptim/misc/enums.py#L219-L238)<br>[TRACK_EXPLICIT_RIGID_CONTACT_FORCES constraint](../../bioptim/limits/constraints.py#L871)<br>[Rigid contact force penalty](../../bioptim/limits/penalty.py#L768-L817)<br>[Model: rigid_contact_forces](../../bioptim/models/biorbd/biorbd_model.py#L1223-L1234) | [scene](anim_contact.py) · [data generator](generate_contact_data.py) · [notes](notes/contact.md) |
| `ExternalForces` | **External forces in the dynamics**<br>A known wind-like force on the hand is described with ExternalForceSetTimeSeries and passed to the model and the dynamics. | 2 - intermediate | [Example: external forces](../../bioptim/examples/getting_started/example_external_forces.py#L30-L118)<br>[ExternalForceSetTimeSeries](../../bioptim/models/biorbd/external_forces.py#L121-L288) | [scene](anim_extforces.py) · [data generator](generate_extforces_data.py) · [notes](notes/extforces.md) |
| `Mapping` | **Sharing variables with a mapping**<br>A double pendulum is solved twice: with two torques, then with a BiMappingList that makes both joints share one torque (decision vector 185 to 155). | 2 - intermediate | [Example: symmetry by mapping](../../bioptim/examples/toy_examples/symmetrical_torque_driven_ocp/symmetry_by_mapping.py#L49-L146)<br>[BiMapping](../../bioptim/misc/mapping.py#L122-L173)<br>[BiMappingList](../../bioptim/misc/mapping.py#L180-L263) | [scene](anim_mapping.py) · [data generator](generate_mapping_data.py) · [notes](notes/mapping.md) |
| `TrackMarkers` | **Tracking measured markers**<br>ObjectiveFcn.Lagrange.TRACK_MARKERS with a target makes model markers follow measured ones; the video shows them converging over IPOPT iterations, then the final error (1.3 cm RMS). | 2 - intermediate | [Example: tracking markers on a pendulum](../../bioptim/examples/toy_examples/torque_driven_ocp/track_markers_2D_pendulum.py#L68-L147)<br>[TRACK_MARKERS in ObjectiveFcn](../../bioptim/limits/objective_functions.py#L362-L483)<br>[minimize_markers penalty](../../bioptim/limits/penalty.py#L281-L331) | [scene](anim_markers.py) · [data generator](generate_markers_data.py) · [notes](notes/markers.md) |
| `MuscFullActivations` | **Muscle activations and joint torque**<br>For a reach with a 2-dof, 6-muscle arm driven only by muscle activations, shows the activation of each muscle with running peaks, and the joint torque the muscles produce compared with a torque-driven solve. | 2 - intermediate | [Example: muscle-driven reach](../../bioptim/examples/toy_examples/muscle_driven_ocp/static_arm.py#L28-L124)<br>[MusclesBiorbdModel](../../bioptim/models/biorbd/model_dynamics.py#L296-L322)<br>[Model: muscle_joint_torque](../../bioptim/models/biorbd/biorbd_model.py#L880-L895) | [scene](anim_muscfull.py) · [data generator](generate_muscfull_data.py) · [notes](notes/muscfull.md) |
| `MuscleReaching` | **Muscle-driven reaching**<br>Replacing TorqueBiorbdModel by MusclesBiorbdModel makes the 6 muscle activations of the arm26 model the controls, bounded in [0, 1], with a Mayer marker objective for the reach. | 2 - intermediate | [Example: static arm reaching](../../bioptim/examples/toy_examples/muscle_driven_ocp/static_arm.py#L28-L124)<br>[MusclesBiorbdModel](../../bioptim/models/biorbd/model_dynamics.py#L296-L322)<br>[MusclesDynamics](../../bioptim/dynamics/state_space_dynamics/muscle_dynamics.py#L13-L171) | [scene](anim_muscle.py) · [data generator](generate_muscle_data.py) · [notes](notes/muscle.md) |
| `SoftContact` | **Soft (compliant) contact**<br>A 1 kg ball is pushed 3 cm into a soft ground declared in the bioMod and enabled with ContactType.SOFT_EXPLICIT. The force comes from soft_contact_forces; for stiffness 1e4, 1e5 and 1e6 the same depth needs far more… | 2 - intermediate | [Example: soft contact](../../bioptim/examples/toy_examples/torque_driven_ocp/example_soft_contact.py#L103-L191)<br>[ContactType enum](../../bioptim/misc/enums.py#L219-L238)<br>[Model: soft_contact_forces](../../bioptim/models/biorbd/biorbd_model.py#L1163-L1181) | [scene](anim_soft.py) · [data generator](generate_soft_data.py) · [notes](notes/soft.md) |
| `ExcitationActivation` | **Excitation to activation**<br>With MusclesWithExcitationsBiorbdModel the muscle excitation is the control and the activation is a state, so activation lags the excitation (about 18 ms up, 14 ms down, measured on the data). | 3 - advanced | [Example: reaching with muscle activations](../../bioptim/examples/toy_examples/muscle_driven_ocp/static_arm.py#L28-L124)<br>[Example: muscle excitations](../../bioptim/examples/toy_examples/muscle_driven_ocp/muscle_excitations_tracker.py#L210-L322)<br>[MusclesDynamicsWithExcitations](../../bioptim/dynamics/state_space_dynamics/muscle_dynamics_with_excitations.py#L13-L190)<br>[MusclesWithExcitationsBiorbdModel](../../bioptim/models/biorbd/model_dynamics.py#L325-L351) | [scene](anim_excitation.py) · [data generator](generate_excitation_data.py) · [notes](notes/excitation.md) |
| `FatigueXia` | **Xia fatigue model on torque**<br>A pendulum swing-up where the torque carries an XiaTauFatigue model: extra states track the active, resting and fatigued fractions of each side. | 3 - advanced | [Example: pendulum with fatigue](../../bioptim/examples/toy_examples/fatigue/pendulum_with_fatigue.py#L39-L194)<br>[XiaFatigue dynamics](../../bioptim/dynamics/fatigue/xia_fatigue.py#L15-L95)<br>[XiaTauFatigue (torque)](../../bioptim/dynamics/fatigue/xia_fatigue.py#L121-L132)<br>[FatigueList](../../bioptim/dynamics/fatigue/fatigue_dynamics.py#L372-L419) | [scene](anim_fatigue.py) · [data generator](generate_fatigue_data.py) · [notes](notes/fatigue.md) |
| `FloatingReorient` | **Free-floating base reorientation**<br>With TorqueFreeFloatingBaseBiorbdModel the root torque is structurally zero; in zero gravity a planar trunk with two arms rotates by 0.80 rad by moving its arms around a closed loop. | 3 - advanced | [Example: free floating base (3D)](../../bioptim/examples/toy_examples/torque_driven_ocp/torque_driven_free_floating_base.py#L23-L172)<br>[TorqueFreeFloatingBaseDynamics](../../bioptim/dynamics/state_space_dynamics/torque_dynamics_free_floating_base.py#L8-L61)<br>[TorqueFreeFloatingBaseBiorbdModel](../../bioptim/models/biorbd/model_dynamics.py#L165-L187)<br>[angular_momentum](../../bioptim/models/biorbd/biorbd_model.py#L414-L425) | [scene](anim_floating.py) · [data generator](generate_floating_data.py) · [notes](notes/floating.md) |
| `HolonomicDoublePendulum` | **Holonomic constraint: double pendulum**<br>A closed-loop constraint superimposing two markers turns two single pendulums into a double pendulum. | 3 - advanced | [Example: two pendulums](../../bioptim/examples/toy_examples/holonomic_constraints/two_pendulums.py#L28-L134)<br>[HolonomicConstraintsFcn.superimpose_markers](../../bioptim/models/protocols/holonomic_constraints.py#L36-L159)<br>[HolonomicTorqueBiorbdModel](../../bioptim/models/biorbd/model_dynamics.py#L120-L135) | [scene](anim_holonomic.py) · [data generator](generate_holonomic_data.py) · [notes](notes/holonomic.md) |
| `MultiBody` | **Two bodies in one OCP**<br>MultiTorqueBiorbdModel puts two independent biorbd models in one phase: q, qdot and tau keep their names and are stacked, and variable_index gives each model's slice. | 3 - advanced | [Example: multi biorbd model](../../bioptim/examples/toy_examples/torque_driven_ocp/example_multi_biorbd_model.py#L20-L74)<br>[MultiBiorbdModel](../../bioptim/models/biorbd/multi_biorbd_model.py#L23-L1088)<br>[variable_index](../../bioptim/models/biorbd/multi_biorbd_model.py#L136-L204)<br>[MultiTorqueBiorbdModel](../../bioptim/models/biorbd/model_dynamics.py#L379-L389) | [scene](anim_multibody.py) · [data generator](generate_multibody_data.py) · [notes](notes/multibody.md) |
| `MuscFullPaths` | **Muscle vs torque reach**<br>The same arm reach is solved once with muscle activations as controls and once with joint torques, and the two hand paths are compared. | 3 - advanced | [Example: muscle-driven static arm](../../bioptim/examples/toy_examples/muscle_driven_ocp/static_arm.py#L28-L124)<br>[MusclesBiorbdModel](../../bioptim/models/biorbd/model_dynamics.py#L296-L322) | [scene](anim_muscfull.py) · [data generator](generate_muscfull_data.py) · [notes](notes/muscfull.md) |

#### Advanced control

| Scene (mp4) | Title and content | Level | Example and code | Sources |
| --- | --- | --- | --- | --- |
| `CyclicNMPC` | **Cyclic NMPC**<br>In a cyclic NMPC the window is one full cycle and advances by a whole cycle at each solve, with the target amplitude changed between cycles. | 3 - advanced | [Example: cyclic NMPC](../../bioptim/examples/toy_examples/moving_horizon_estimation/cyclic_nmpc.py#L35-L95)<br>[CyclicRecedingHorizonOptimization](../../bioptim/optimization/receding_horizon_optimization.py#L448-L637)<br>[CyclicNonlinearModelPredictiveControl](../../bioptim/optimization/receding_horizon_optimization.py#L950-L955) | [scene](anim_cyclic.py) · [data generator](generate_cyclic_data.py) · [notes](notes/cyclic.md) |
| `MHEWindow` | **Moving horizon estimation**<br>A 0.5 s window slides over noisy measurements of the pendulum angle; at each new measurement the target is updated and the OCP is solved again. | 3 - advanced | [Example: moving horizon estimation](../../bioptim/examples/toy_examples/moving_horizon_estimation/mhe.py#L88-L158)<br>[MovingHorizonEstimator](../../bioptim/optimization/receding_horizon_optimization.py#L966-L971)<br>[RecedingHorizonOptimization](../../bioptim/optimization/receding_horizon_optimization.py#L35-L445)<br>[update_objectives_target](../../bioptim/optimization/optimal_control_program.py#L956-L980) | [scene](anim_mhe.py) · [data generator](generate_mhe_data.py) · [notes](notes/mhe.md) |
| `NMPCWindow` | **Receding-horizon NMPC**<br>A 1 s window is solved, only the first node is applied, then the window slides by one node with an updated reference and is solved again from a shifted warm start. | 3 - advanced | [Example: cyclic NMPC (same cart-pendulum)](../../bioptim/examples/toy_examples/moving_horizon_estimation/cyclic_nmpc.py#L35-L95)<br>[NonlinearModelPredictiveControl](../../bioptim/optimization/receding_horizon_optimization.py#L942-L947)<br>[advance_window and bounds update](../../bioptim/optimization/receding_horizon_optimization.py#L288-L327) | [scene](anim_nmpc.py) · [data generator](generate_nmpc_data.py) · [notes](notes/nmpc.md) |
| `RobustPath` | **Robust path constraint (SOCP)**<br>A stochastic OCP optimizes the mean trajectory and the state covariance at each node. | 3 - advanced | [Example: obstacle avoidance (SOCP)](../../bioptim/examples/toy_examples/stochastic_optimal_control/obstacle_avoidance_direct_collocation.py#L368-L533)<br>[Robustified path constraint](../../bioptim/examples/toy_examples/stochastic_optimal_control/obstacle_avoidance_direct_collocation.py#L327-L354)<br>[StochasticOptimalControlProgram](../../bioptim/optimization/stochastic_optimal_control_program.py#L29-L675)<br>[Covariance continuity constraint](../../bioptim/limits/constraints.py#L581-L625) | [scene](anim_socp.py) · [data generator](generate_socp_data.py) · [notes](notes/socp.md) |

#### Library overview

| Scene (mp4) | Title and content | Level | Example and code | Sources |
| --- | --- | --- | --- | --- |
| `ArchitecturePath` | **From inputs to Solution**<br>Follows the path from the user's inputs through OptimalControlProgram, NonLinearProgram, ConfigureProblem, penalties, the decision-vector helper and the solver interface to Solution, then replays it with real sizes of… | 1 - introductory | [NonLinearProgram.declare_shooting_points](../../bioptim/optimization/non_linear_program.py#L357-L366)<br>[ConfigureProblem.initialize](../../bioptim/dynamics/configure_problem.py#L25-L48)<br>[OptimizationVectorHelper](../../bioptim/optimization/optimization_vector.py#L20-L283)<br>[generic_solve](../../bioptim/interfaces/interface_utils.py#L98-L204) | [scene](anim_arch.py) · [data generator](generate_arch_data.py) · [notes](notes/arch.md) |
| `OfflineGraphs` | **Offline graphs**<br>After the solve, sol.graphs(show_bounds=True, save_name=...) draws the same plots that the online window shows during optimization, including custom plots added with ocp.add_plot. | 1 - introductory | [Example: custom plots](../../bioptim/examples/getting_started/custom_plotting.py#L38-L113)<br>[Solution.graphs](../../bioptim/optimization/solution/solution.py#L1219-L1261)<br>[OptimalControlProgram.add_plot](../../bioptim/optimization/optimal_control_program.py#L1161-L1206)<br>[PlotOcp](../../bioptim/gui/plot.py#L162-L1255) | [scene](anim_online.py) · [data generator](generate_online_data.py) · [notes](notes/online.md) |
| `SolutionTour` | **Reading the Solution object**<br>One real two-phase pendulum solve read through decision_states, stepwise_states, interpolate and integrate, with the array shape each returns, plus the SolutionMerge options and sol.cost / detailed_cost. graphs() and… | 1 - introductory | [Solution states and controls accessors](../../bioptim/optimization/solution/solution.py#L544-L635)<br>[Solution.integrate](../../bioptim/optimization/solution/solution.py#L785-L866)<br>[Solution.interpolate](../../bioptim/optimization/solution/solution.py#L1158-L1217)<br>[SolutionMerge](../../bioptim/optimization/solution/solution_data.py#L13-L21) | [scene](anim_solution.py) · [data generator](generate_solution_data.py) · [notes](notes/solution.md) |
| `PenaltyPanorama` | **The penalty library**<br>Lists the penalties of ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer and ConstraintFcn grouped by what they act on, then explains how to choose: Lagrange integrates over intervals, Mayer acts at one node, a constraint… | 2 - intermediate | [ObjectiveFcn (Lagrange, Mayer)](../../bioptim/limits/objective_functions.py#L362-L522)<br>[ConstraintFcn](../../bioptim/limits/constraints.py#L831-L897)<br>[Penalty implementations](../../bioptim/limits/penalty.py) | [scene](anim_panorama.py) · [data generator](generate_panorama_data.py) · [notes](notes/panorama.md) |

<!-- END TABLE -->

## Where are the videos?

**The mp4 files are not stored in the repository** (the repository rule is no binary files; `media/`, `*.mp4` and
`assets/*.png` are git-ignored). Render them yourself (below); the committed material is the scenes, the data
(`data/*.npz`, a few kB each), the catalog and the translations.

## How the videos are made

* **Real data.** Each `generate_*_data.py` builds a Bioptim OCP, solves it with IPOPT and writes a small
  `data/*.npz`, committed so that rendering needs neither Bioptim nor biorbd. There is no fake fallback data; where data
  are synthetic (noisy markers, measurements for MHE) the video and the notes say so.
* **Manim scenes.** `anim_*.py`, `features_scenes.py` (helpers and six scenes on objectives, bounds, phases, time,
  parameters, impact) and `dms_vs_dc.py` (OCP statement and discretization). They use `Text`/`MarkupText` only, no LaTeX.
* **The series layer** (`series_style.py`, `render_series.py`, [SERIES_LAYER.md](SERIES_LAYER.md)) is applied at render
  time without editing any scene: Bioptim logo on every frame, everything **1.6 times slower** with a 0.5 s reading pause
  after new text, plain text transitions (fades instead of letter-by-letter writing), English/French, and a 4.5 s **end card**
  with the links of `catalog.json`. It also audits the frames (logo overlap, text out of frame or overlapping).

### Requirements

* **Rendering** (no Bioptim): Python 3.11 virtual environment with Manim 0.21 (it bundles PyAV: no separate ffmpeg, no
  LaTeX). Fonts: Segoe UI and Consolas on Windows, DejaVu elsewhere.

  ```bash
  python3.11 -m venv .venv-manim
  # Windows: .venv-manim\Scripts\activate      Linux/macOS: source .venv-manim/bin/activate
  pip install "manim==0.21.*" numpy black
  python docs/animations/assets/fetch_logo.py      # downloads assets/bioptim_logo.png (not committed)
  ```

  The logo comes from [pyomeca/biorbd_design](https://github.com/pyomeca/biorbd_design). The first render also tries
  to fetch it; without it the videos are rendered without logo (warning).
* **Regenerating data** (optional): a conda environment with Bioptim, biorbd, casadi and IPOPT
  (`conda env create -f environment.yml` at the repository root, then `conda activate bioptim`). On Windows, if you call the
  environment's `python.exe` without `conda activate`, put the environment folders on `PATH` first, otherwise IPOPT's DLLs
  are not found (`Plugin 'ipopt' is not found`): `<env>\Library\bin`, `<env>\Library\mingw-w64\bin`,
  `<env>\Library\usr\bin`, `<env>\Scripts` and `<env>`.

### Regenerate the data

From the repository root (a few seconds to a few minutes per script):

```bash
PYTHONPATH=. python docs/animations/generate_pendulum_data.py     # OCPStatement ... Comparison (about 1 min)
PYTHONPATH=. python docs/animations/generate_features_data.py     # optionally name experiments: parameters impact
```

The other scripts follow the same pattern (`generate_<topic>_data.py`, see the "Sources" column). IPOPT is deterministic:
the same Bioptim version gives the same numbers. The data of the first five scenes are the pendulum swing-up of
`bioptim/examples/toy_examples/sqp_method/pendulum.py` (N = 20, T = 1 s): RK4 (`rk4*`), Legendre collocation (`col*`) and
Radau (`rad*`), plus early non-converged IPOPT iterates that the animation shows before the gaps close.

### Render the videos

Always through the series layer. From the repository root:

```bash
python docs/animations/render_series.py anim_controls.py ControlTypes --lang en            # one scene, English
python docs/animations/render_series.py anim_controls.py ControlTypes --lang fr --strict   # French, fail if a text is untranslated
python docs/animations/render_series.py --all --lang both --jobs 4 --out my_videos          # the 48 scenes, EN and FR
python docs/animations/render_series.py --all --lang en --quality 480p15 --jobs 4           # quick preview
```

* Output: `<out>/<Scene>_<lang>.mp4` (default `docs/animations/media/series/out`, git-ignored), logs in `<out>/logs`.
* `--quality <height>p<fps>`: `1080p30` (default), `1080p60`, `480p15`. Options: `--no-endcard`, `--no-logo`,
  `--logo-corner auto|br|bl|tr|tl`, `--slow F` (default 1.6), `--min-wait S`, `--media-dir`, `--catalog`.
* `--strict` (French): exit status 3 if a string has no translation.
* `--dry --collect keys.jsonl`: construct only, no video, and write every translatable text key of the scenes
  (`python docs/animations/render_series.py --all --dry --collect keys.jsonl --jobs 4`).
* `python docs/animations/render_series.py --gen-catalog` adds the skeleton of new scenes to `catalog.json`.

Plain Manim still works for a raw preview (`cd docs/animations; manim -ql dms_vs_dc.py MultipleShooting`) but gives
no logo, pacing, French nor end card.

## Adding a new scene

**Standard for new videos:** the authoritative specification and checklist is [STANDARD.md](STANDARD.md) (French: [STANDARD.fr.md](STANDARD.fr.md)); start from [templates/scene_template.py](templates/scene_template.py).

1. Write `anim_<topic>.py` (and `generate_<topic>_data.py` if it needs a solve; commit only the small `.npz`), reusing the
   helpers of `features_scenes.py` and the look of the existing scenes. Add `notes/<topic>.md`: what is real, what is
   simplified, the numbers shown.
2. Add the scene to the catalog: `python docs/animations/render_series.py --gen-catalog`, then fill `title_en/fr`,
   `description_en/fr`, `section`/`section_fr`, `level`, `notes_file` and `links` (example, library lines as
   `"path"` + `"lines": "40-90"`, data generator). Links feed both the end card and the tables of this README.
3. French: `python docs/animations/render_series.py anim_<topic>.py <Scene> --dry --collect keys.jsonl`, translate the
   keys in a new `i18n/fr_<topic>.json` (workflow and glossary in [i18n/README.md](i18n/README.md)), then run the strict
   French render and read the audit warnings (`<out>/logs/<Scene>_<lang>_audit.json`: logo overlap, text out of frame or
   overlapping).
4. Regenerate the tables: `python docs/animations/build_readme_tables.py`.

Checklist for a consistent look:

* code panels have the caption **Bioptim code** above them, in the code font (never translated);
* whole sentences (one idea per screen), short enough for French, which is 15-20 % longer;
* numbers written as plain numbers in the string, so they become `{0}`, `{1}` in translation templates;
* Bioptim identifiers (`OdeSolver`, `ObjectiveFcn.Lagrange...`, `n_shooting`) stay in English;
* end the scene with a 2.5 s hold (`self.wait(2.5)`) before the end card;
* the line ranges in the catalog links are checked against the code of this version.

## Regenerating the tables

`build_readme_tables.py` fills the regions delimited by the HTML comments `BEGIN TABLE` / `END TABLE` (and `BEGIN START` / `END START` for
the "start here" table) of `README.md` and `README.fr.md` from `catalog.json`. It is idempotent and also checks that every relative link
resolves:

```bash
python docs/animations/build_readme_tables.py          # rewrite the tables, then check the links
python docs/animations/build_readme_tables.py --check  # check only
```

Do not edit the tables by hand: edit `catalog.json`.

## Caveats (please read)

This is **teaching material, not a benchmark**.

* **Local minima and warm starts.** Optimal control problems are non-convex. IPOPT returns *a* local minimum that depends
  on the initial guess; several videos use continuation (warm start from the previous weight or bound) and say so. Two
  discretizations of the same problem (RK4 and collocation) may land in different minima with very different costs,
  so their costs are not an accuracy comparison. `IpoptMultiStart` shows this on purpose.
* **Timings are noisy.** Times and iteration counts come from one laptop, a few runs, sometimes with other jobs running;
  iteration counts are more reliable than seconds. No video claims real-time performance.
* **Synthetic data where used.** The measured markers (`TrackMarkers`) and the noisy measurements of `MHEWindow` are
  synthetic (known truth plus stated noise and seed).
* **Simplified models.** The pendulum has one actuated coordinate; the hopper (`Hopper`, `Impact`) is 1-D with an
  inelastic, frictionless point impact and no leg segments; soft contact is a simple compliant model; muscle and fatigue
  models are the small ones of Bioptim's examples. Code panels sometimes show a simplified call (bounds, imports
  omitted): the linked example is the reference.
* **Counts.** Bioptim reports one more decision variable than `(N+1)*4 + N*2` for the RK4 pendulum (125 instead of
  124), not investigated.
* The line ranges linked in the table refer to the Bioptim version of this branch and may drift after later changes.

## Exercises (from the first five videos)

1. Change `polynomial_degree` (2 to 5): how do variables, iterations and solve time evolve?
2. Change `n_shooting` (10, 20, 40) for both transcriptions: size of the NLP, gaps between nodes?
3. Try `OdeSolver.RK4(n_integration_steps=1, 5, 10)`, `RK8()`, `COLLOCATION(method="radau")`, `IRK()`: which are
   shooting-like, which collocation-like?
4. After `sol = ocp.solve(...)`, integrate the optimal controls with `sol.integrate()` and plot the defects
   `F(x_k, u_k) - x_{k+1}`.
5. For a fair comparison, give both problems the same initial guess (or warm start one from the other) and compare
   against a fine simulation.
6. Change the OCP, not the discretization: a free time with `ObjectiveFcn.Mayer.MINIMIZE_TIME`, or bounds on `q`.

## Credits and licence

Bioptim is developed by the [pyomeca](https://github.com/pyomeca/bioptim) community; see [LICENSE](../../LICENSE) for the
licence of this repository. The logo comes from [pyomeca/biorbd_design](https://github.com/pyomeca/biorbd_design) and is
downloaded at render time, not redistributed here. Animations are made with
[Manim Community](https://www.manim.community/).

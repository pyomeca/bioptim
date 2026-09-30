# Apprendre Bioptim en petites animations

*English version: [README.md](README.md).*

Une série de **48 courtes vidéos** (environ 15 à 60 secondes chacune, en français et en anglais) qui enseignent
[Bioptim](https://github.com/pyomeca/bioptim), la bibliothèque de commande optimale pour la biomécanique, une idée à la
fois. Les vidéos sont réalisées avec [Manim Community](https://www.manim.community/) et chaque courbe affichée provient
d'une **vraie résolution Bioptim** (IPOPT), pas d'un schéma.

Elle s'adresse aux étudiants et chercheurs en biomécanique et en commande optimale qui connaissent un peu Python et la
mécanique, mais pas forcément le tir multiple direct. Chaque vidéo renvoie à l'exemple Bioptim correspondant et aux lignes
exactes du code de la bibliothèque : l'animation est une porte d'entrée vers le code, pas un remplacement.

## Comment utiliser la série

1. Regardez une vidéo, ouvrez ses liens **exemple** et **code** (tableau ci-dessous) et exécutez l'exemple.
2. Colonne niveau : **1** = introduction, **2** = intermédiaire, **3** = avancé (suppose les notions précédentes).
3. Suivez les sections dans cet ordre, ou allez directement à ce qu'il vous faut :
   *Fondamentaux* -> *Discrétisation* -> *Objectifs et contraintes* -> *Phases et temps* -> *Solveurs et numérique* ->
   *Modèles et biomécanique* -> *Commande avancée*. *Vue d'ensemble de la bibliothèque* peut se regarder à tout moment.
4. Chaque vidéo a un fichier de notes (ce qui a été calculé, les chiffres, les limites) et un générateur de données que
   vous pouvez modifier et relancer.

### Commencer ici

<!-- BEGIN START -->

| Scène (mp4) | Titre | Pourquoi commencer ici |
| --- | --- | --- |
| `OCPStatement` | **Qu'est-ce qu'un OCP ?** | Le vocabulaire : état, commande, dynamique, coût, contraintes, bornes. |
| `MultipleShooting` | **Tir multiple direct** | Comment Bioptim transforme un OCP en NLP (nœuds, défauts). |
| `ArchitecturePath` | **Des entrées à Solution** | La carte de la bibliothèque : de vos entrées à la Solution. |
| `ObjectivesNodes` | **Lagrange, Mayer et nœuds** | Termes de Lagrange et de Mayer, énumération Node. |
| `SolutionTour` | **Lire l'objet Solution** | Lire et post-traiter un problème résolu. |
| `TrackState` | **Suivre une référence** | Un premier objectif réaliste : suivre une référence. |

<!-- END START -->

Ensuite : `ControlTypes` (interpolation des commandes), `ConstraintsBounds`, `IpoptIterates`, et un modèle de votre domaine
(`MuscleReaching`, `ExternalForces`, `Hopper`...).

## Les 48 vidéos

Le nom de la scène est aussi le nom du fichier mp4 (`<Scène>_fr.mp4`, `<Scène>_en.mp4`). Les liens « Exemple et code »
ouvrent l'exemple Bioptim et les lignes utiles de la bibliothèque ; « Sources » donne la scène Manim, le script qui a
produit les données et les notes. Ce tableau est généré : voir [Régénérer les tableaux](#régénérer-les-tableaux).

<!-- BEGIN TABLE -->

#### Fondamentaux

| Scène (mp4) | Titre et contenu | Niveau | Exemple et code | Sources |
| --- | --- | --- | --- | --- |
| `OCPStatement` | **Qu'est-ce qu'un OCP ?**<br>Écrit le problème de commande optimale (OCP) sous forme de tableau associant chaque élément mathématique (coût de Lagrange et de Mayer, dynamique, valeurs aux limites, bornes, contraintes de chemin) à son appel… | 1 - introduction | [Exemple : balancement du pendule](../../bioptim/examples/toy_examples/sqp_method/pendulum.py#L22-L97)<br>[Classe Objective](../../bioptim/limits/objective_functions.py#L21-L181)<br>[Classe Constraint](../../bioptim/limits/constraints.py#L25-L162)<br>[Constructeur d'OptimalControlProgram](../../bioptim/optimization/optimal_control_program.py#L160-L338) | [scène](dms_vs_dc.py) · [générateur de données](generate_pendulum_data.py) · [notes](README.md) |

#### Discrétisation

| Scène (mp4) | Titre et contenu | Niveau | Exemple et code | Sources |
| --- | --- | --- | --- | --- |
| `TimeGrid` | **Discrétiser le temps en nœuds**<br>L'horizon T est découpé en N intervalles de tir de longueur T/N. Les états sont définis aux nœuds et les commandes sont constantes par morceaux sur chaque intervalle. | 1 - introduction | [Exemple : balancement du pendule (n_shooting)](../../bioptim/examples/toy_examples/sqp_method/pendulum.py#L22-L97)<br>[Énumération Node](../../bioptim/misc/enums.py#L33-L47) | [scène](dms_vs_dc.py) · [générateur de données](generate_pendulum_data.py) · [notes](README.md) |
| `MultipleShooting` | **Tir multiple direct**<br>Sur le balancier du pendule (20 intervalles de tir), montre une fenêtre de nœuds avec OdeSolver.RK4(n_integration_steps=5) : chaque intervalle est intégré à partir de x_k, et la contrainte de continuité F(x_k, u_k) =… | 1 - introduction | [Exemple : balancier du pendule](../../bioptim/examples/toy_examples/sqp_method/pendulum.py#L22-L97)<br>[OdeSolver.RK4](../../bioptim/dynamics/ode_solvers.py#L35-L42)<br>[Pas d'intégration RK4](../../bioptim/dynamics/integrator.py#L366-L379)<br>[Contrainte de continuité des états](../../bioptim/limits/penalty.py#L1224-L1276) | [scène](dms_vs_dc.py) · [générateur de données](generate_pendulum_data.py) · [notes](README.md) |
| `Comparison` | **Tir multiple vs collocation**<br>Un tableau compare le tir multiple direct (OdeSolver.RK4) et la collocation directe (OdeSolver.COLLOCATION) sur le même pendule : inconnues supplémentaires, contraintes par intervalle, évaluations de la dynamique,… | 1 - introduction | [Exemple : pendule avec RK4 ou collocation](../../bioptim/examples/toy_examples/sqp_method/pendulum.py#L22-L97)<br>[Classes OdeSolver (RK4, COLLOCATION)](../../bioptim/dynamics/ode_solvers.py#L12-L357)<br>[Intégrateur COLLOCATION](../../bioptim/dynamics/integrator.py#L569-L695) | [scène](dms_vs_dc.py) · [générateur de données](generate_pendulum_data.py) · [notes](README.md) |
| `DirectCollocation` | **Collocation directe**<br>Présente OdeSolver.COLLOCATION(polynomial_degree=3, method='legendre') : points de collocation, polynôme de Lagrange par intervalle et défauts (pente du polynôme moins dynamique) que IPOPT annule, sur les itérations… | 2 - intermédiaire | [Exemple : balancement du pendule](../../bioptim/examples/toy_examples/sqp_method/pendulum.py#L22-L97)<br>[OdeSolver.COLLOCATION](../../bioptim/dynamics/ode_solvers.py#L111-L214)<br>[Interpolation de Lagrange](../../bioptim/dynamics/lagrange_interpolation.py#L15-L202) | [scène](dms_vs_dc.py) · [générateur de données](generate_pendulum_data.py) · [notes](README.md) |
| `CollocationDegree` | **Degré et points de collocation**<br>Montre où se placent les points de collocation dans un intervalle pour les degrés 2 à 6 (Legendre : intérieurs, Radau : le dernier est le nœud suivant), puis l'erreur de dynamique, le nombre de variables et… | 2 - intermédiaire | [Options d'OdeSolver.COLLOCATION](../../bioptim/dynamics/ode_solvers.py#L111-L170)<br>[Intégrateur COLLOCATION (points)](../../bioptim/dynamics/integrator.py#L569-L695) | [scène](anim_colloc.py) · [générateur de données](generate_colloc_data.py) · [notes](notes/colloc.md) |
| `ControlTypes` | **Types d'interpolation des commandes**<br>Résout le même balancement de pendule avec ControlType CONSTANT, LINEAR_CONTINUOUS et CONSTANT_WITH_LAST_NODE et montre les courbes de couple et les tailles du vecteur de décision (125, 127, 127). | 2 - intermédiaire | [Exemple avec l'argument control_type](../../bioptim/examples/getting_started/basic_ocp.py#L33-L127)<br>[Énumération ControlType](../../bioptim/misc/enums.py#L126-L170)<br>[Constructeur d'OptimalControlProgram](../../bioptim/optimization/optimal_control_program.py#L160-L338) | [scène](anim_controls.py) · [générateur de données](generate_controls_data.py) · [notes](notes/controls.md) |
| `DiscreteMechanics` | **Mécanique discrète (variationnelle)**<br>Avec VariationalOptimalControlProgram, l'état est seulement q_k et la dynamique est donnée par les équations d'Euler-Lagrange discrètes. | 3 - avancé | [Exemple : pendule à intégrateur variationnel](../../bioptim/examples/toy_examples/discrete_mechanics_and_optimal_control/example_variational_integrator_pendulum.py#L21-L86)<br>[VariationalOptimalControlProgram](../../bioptim/optimization/variational_optimal_control_program.py#L33-L377)<br>[Équations d'Euler-Lagrange discrètes](../../bioptim/models/biorbd/variational_biorbd_model.py#L163-L232)<br>[OdeSolver.VARIATIONAL (espace réservé)](../../bioptim/dynamics/ode_solvers.py#L53-L61) | [scène](anim_variational.py) · [générateur de données](generate_variational_data.py) · [notes](notes/variational.md) |

#### Objectifs et contraintes

| Scène (mp4) | Titre et contenu | Niveau | Exemple et code | Sources |
| --- | --- | --- | --- | --- |
| `TrackState` | **Suivre une référence**<br>L'angle du pendule doit suivre une référence sinusoïdale avec ObjectiveFcn.Lagrange.TRACK_STATE aux poids 1, 30 et 1000, puis avec une contrainte ConstraintFcn.TRACK_STATE. Un poids plus élevé suit mieux mais demande… | 1 - introduction | [Exemple : objectif de suivi cyclique](../../bioptim/examples/toy_examples/moving_horizon_estimation/multi_cyclic_nmpc_with_parameters.py#L178-L186)<br>[minimize_states (TRACK_STATE)](../../bioptim/limits/penalty.py#L60-L86)<br>[ObjectiveFcn (Lagrange, Mayer)](../../bioptim/limits/objective_functions.py#L362-L522) | [scène](anim_track.py) · [générateur de données](generate_track_data.py) · [notes](notes/track.md) |
| `ObjectivesNodes` | **Lagrange, Mayer et nœuds**<br>Montre sur une grille schématique quels nœuds sélectionne l'énumération Node (START, INTERMEDIATES, PENULTIMATE, END, ALL_SHOOTING, ALL), puis cinq résolutions réelles du balancement du pendule avec un poids de Mayer… | 2 - intermédiaire | [Exemple : OCP du pendule de base](../../bioptim/examples/getting_started/basic_ocp.py#L33-L127)<br>[Énumération Node](../../bioptim/misc/enums.py#L33-L47)<br>[Fonctions objectif Lagrange et Mayer](../../bioptim/limits/objective_functions.py#L362-L483)<br>[Conversion des nœuds en indices](../../bioptim/limits/penalty_option.py#L989-L1046) | [scène](features_scenes.py) · [générateur de données](generate_features_data.py) · [notes](FEATURES.md) |
| `ConstraintsBounds` | **Bornes sur commandes et états**<br>Le même balancement de pendule est résolu avec une borne de couple u_bounds qui passe de 100 à 12 N, puis avec une borne sur la position du chariot x_bounds. | 2 - intermédiaire | [Exemple : bornes sur états et commandes](../../bioptim/examples/toy_examples/feature_examples/pendulum_constrained_states_controls.py#L35-L126)<br>[Exemple : pendule avec x_bounds et u_bounds](../../bioptim/examples/getting_started/basic_ocp.py#L33-L127)<br>[Classes Bounds et BoundsList](../../bioptim/limits/path_conditions.py#L339-L700) | [scène](features_scenes.py) · [générateur de données](generate_features_data.py) · [notes](FEATURES.md) |
| `Parameters` | **Optimiser un paramètre**<br>Un paramètre scalaire, le couple maximal max_tau, est déclaré avec ParameterList et optimisé avec la trajectoire. | 2 - intermédiaire | [Exemple : paramètres personnalisés](../../bioptim/examples/getting_started/custom_parameters.py#L89-L252)<br>[Exemple : minimiser le couple maximal avec un paramètre](../../bioptim/examples/toy_examples/torque_driven_ocp/minimize_maximum_torque_by_extra_parameter.py#L47-L135)<br>[Classe ParameterList](../../bioptim/optimization/parameters.py#L149-L298) | [scène](features_scenes.py) · [générateur de données](generate_features_data.py) · [notes](FEATURES.md) |
| `DerivativePenalty` | **Pénalité sur la dérivée d'une commande**<br>Sur un balancier de pendule sur chariot, ajoute MINIMIZE_CONTROL avec derivative=True sur tau, avec des poids de 0, 1, 10 et 100. | 2 - intermédiaire | [Exemple de base : OCP de balancier](../../bioptim/examples/getting_started/basic_ocp.py#L33-L127)<br>[Pénalité MINIMIZE_CONTROL](../../bioptim/limits/penalty.py#L89-L113)<br>[derivative=True : différence fin moins début](../../bioptim/limits/penalty_option.py#L580-L614)<br>[Alternative : TorqueDerivativeBiorbdModel](../../bioptim/models/biorbd/model_dynamics.py#L269-L293) | [scène](anim_deriv.py) · [générateur de données](generate_deriv_data.py) · [notes](notes/deriv.md) |
| `MultinodeLink` | **Lien multi-nœuds**<br>Lie des variables de nœuds éloignés, ici le dernier nœud au premier pour un mouvement cyclique, soit par une contrainte (MultinodeConstraintFcn.STATES_EQUALITY), soit par un objectif pondéré. Sur un chariot-pendule,… | 3 - avancé | [Exemple : contraintes multi-nœuds](../../bioptim/examples/toy_examples/feature_examples/example_multinode_constraints.py#L67-L204)<br>[Exemple : objectif multi-nœuds](../../bioptim/examples/toy_examples/feature_examples/example_multinode_objective.py#L36-L118)<br>[MultinodeConstraintList et Fcn](../../bioptim/limits/multinode_constraint.py#L84-L164)<br>[MultinodeObjectiveList et Fcn](../../bioptim/limits/multinode_objective.py#L55-L122) | [scène](anim_multinode.py) · [générateur de données](generate_multinode_data.py) · [notes](notes/multinode.md) |

#### Phases et temps

| Scène (mp4) | Titre et contenu | Niveau | Exemple et code | Sources |
| --- | --- | --- | --- | --- |
| `MultiphaseTransitions` | **Problèmes multiphases et transitions**<br>Un balancier de pendule découpé en deux phases de durées différentes (0,5 s et 1,0 s, 12 et 18 nœuds). | 2 - intermédiaire | [Exemple : OCP multiphase](../../bioptim/examples/getting_started/example_multiphase.py#L39-L189)<br>[Énumération PhaseTransitionFcn](../../bioptim/limits/phase_transition.py#L355-L375)<br>[Transition CONTINUOUS (égalité des états)](../../bioptim/limits/phase_transition.py#L157-L185)<br>[Transition DISCONTINUOUS (aucun lien)](../../bioptim/limits/phase_transition.py#L222-L238) | [scène](features_scenes.py) · [générateur de données](generate_features_data.py) · [notes](FEATURES.md) |
| `FreeTime` | **Optimisation à temps libre**<br>Avec ObjectiveFcn.Mayer.MINIMIZE_TIME, la durée de la phase devient une variable de décision; phase_time n'est qu'une estimation initiale. | 2 - intermédiaire | [Exemple : pendule en temps minimal](../../bioptim/examples/toy_examples/optimal_time_ocp/pendulum_min_time_Mayer.py#L30-L112)<br>[Implémentation de Mayer minimize_time](../../bioptim/limits/objective_functions.py#L288-L311)<br>[Liste des objectifs de Mayer (MINIMIZE_TIME)](../../bioptim/limits/objective_functions.py#L428-L483) | [scène](features_scenes.py) · [générateur de données](generate_features_data.py) · [notes](FEATURES.md) |
| `Impact` | **Transition de phase par impact**<br>Une masse ponctuelle de 1 kg tombe sur le sol puis glisse dessus (deux phases). | 3 - avancé | [Exemple utilisant PhaseTransitionFcn.IMPACT](../../bioptim/examples/getting_started/custom_phase_transitions.py#L68-L204)<br>[Implémentation de la transition IMPACT](../../bioptim/limits/phase_transition.py#L260-L306)<br>[Impulsion du modèle : qdot_from_impact](../../bioptim/models/biorbd/biorbd_model.py#L824-L835) | [scène](features_scenes.py) · [générateur de données](generate_features_data.py) · [notes](FEATURES.md) |
| `Hopper` | **Cycle de saut avec impact**<br>Un sauteur 1-D (pied et corps sur un axe vertical) effectue un saut périodique en trois phases : vol, appui avec un contact rigide et une force non négative, vol. | 3 - avancé | [Exemple : transitions de phase dont IMPACT](../../bioptim/examples/getting_started/custom_phase_transitions.py#L68-L204)<br>[Exemple : contact rigide](../../bioptim/examples/toy_examples/torque_driven_ocp/example_rigid_contact.py#L40-L138)<br>[Implémentation de la transition IMPACT](../../bioptim/limits/phase_transition.py#L260-L306)<br>[Pénalité sur la force de contact rigide](../../bioptim/limits/penalty.py#L768-L817) | [scène](anim_walk.py) · [générateur de données](generate_walk_data.py) · [notes](notes/walk.md) |

#### Solveurs et numérique

| Scène (mp4) | Titre et contenu | Niveau | Exemple et code | Sources |
| --- | --- | --- | --- | --- |
| `OnlineIterates` | **Graphiques en ligne pendant la résolution**<br>Redessine, dans Manim, la fenêtre affichée par Solver.IPOPT(show_online_optim=True) à partir d'itérés réels d'IPOPT sur un balancier de pendule : états, commandes, un graphique personnalisé et la sortie d'IPOPT, à côté… | 2 - intermédiaire | [Exemple : graphiques personnalisés](../../bioptim/examples/getting_started/custom_plotting.py#L38-L113)<br>[Options du solveur IPOPT](../../bioptim/interfaces/ipopt_options.py#L19-L348)<br>[Énumération OnlineOptim](../../bioptim/misc/enums.py#L98-L123)<br>[Graphique de sortie d'IPOPT](../../bioptim/gui/ipopt_output_plot.py#L10-L63)<br>[OCP.add_plot](../../bioptim/optimization/optimal_control_program.py#L1161-L1206) | [scène](anim_online.py) · [générateur de données](generate_online_data.py) · [notes](notes/online.md) |
| `IpoptIterates` | **Itérés d'IPOPT**<br>Montre qu'un itéré d'IPOPT n'est en général pas une trajectoire faisable : pour les itérations 0 à 51, le coût et l'infaisabilité primale sont affichés avec la trajectoire. | 2 - intermédiaire | [Options du solveur IPOPT](../../bioptim/interfaces/ipopt_options.py#L19-L348)<br>[set_maximum_iterations](../../bioptim/interfaces/ipopt_options.py#L240-L241)<br>[Exemple : balancement du pendule](../../bioptim/examples/getting_started/basic_ocp.py#L33-L127) | [scène](anim_solver.py) · [générateur de données](generate_solver_data.py) · [notes](notes/solver.md) |
| `IpoptMultiStart` | **Multi-départ : plusieurs minima locaux**<br>Le balancier du pendule est résolu à partir de différentes estimations initiales : IPOPT converge à chaque fois, mais vers des minima locaux différents (des coûts de 40,3 à 302,9 sont montrés). | 2 - intermédiaire | [Exemple : multi-départ](../../bioptim/examples/getting_started/example_multistart.py#L29-L121)<br>[Classe MultiStart](../../bioptim/optimization/multi_start.py#L18-L115)<br>[IPOPT : set_maximum_iterations](../../bioptim/interfaces/ipopt_options.py#L240-L241)<br>[IPOPT : set_tol](../../bioptim/interfaces/ipopt_options.py#L216-L217) | [scène](anim_solver.py) · [générateur de données](generate_solver_data.py) · [notes](notes/solver.md) |
| `VectorCollocation` | **Vecteur de décision et collocation**<br>Avec la collocation directe de degré 3, chaque nœud d'état contient 4 colonnes (le nœud plus 3 points de collocation) dans le vecteur plat vu par IPOPT. La scène zoome sur un nœud sous forme de grille 4x4 remplie… | 2 - intermédiaire | [VectorLayout (index_map)](../../bioptim/optimization/vector_layout.py#L110-L286)<br>[OrderingStrategy](../../bioptim/optimization/vector_layout.py#L105-L107)<br>[n_states_decision_steps](../../bioptim/optimization/non_linear_program.py#L433-L445) | [scène](anim_vector.py) · [générateur de données](generate_vector_data.py) · [notes](notes/vector.md) |
| `AccuracyCheck` | **Vérifier la précision de la solution**<br>Quatre résolutions du même balancier de pendule (RK4 à 1 ou 5 pas, collocation de degré 3 ou 5) sont réintégrées depuis l'état initial avec Solution.integrate(Shooting.SINGLE, DOP853). | 3 - avancé | [Exemple : réintégration en tir simple](../../bioptim/examples/getting_started/example_simulation.py)<br>[Solution.integrate](../../bioptim/optimization/solution/solution.py#L785-L866)<br>[Énumération Shooting](../../bioptim/misc/enums.py#L64-L74)<br>[Énumération SolutionIntegrator](../../bioptim/misc/enums.py#L184-L194) | [scène](anim_accuracy.py) · [générateur de données](generate_accuracy_data.py) · [notes](notes/accuracy.md) |
| `PhaseDynamicsScene` | **PhaseDynamics : partagée ou par nœud**<br>Compare SHARED_DURING_THE_PHASE (une fonction d'intégration pour la phase) et ONE_PER_NODE (une par nœud) : temps de construction et de résolution mesurés pour 30, 60 et 120 nœuds, optimum identique, et une contrainte… | 3 - avancé | [Exemple : pendule dépendant du temps](../../bioptim/examples/toy_examples/torque_driven_ocp/example_pendulum_time_dependent.py#L82-L166)<br>[Énumération PhaseDynamics](../../bioptim/misc/enums.py#L5-L7)<br>[Un intégrateur ou un par nœud](../../bioptim/dynamics/ode_solver_base.py#L268-L339) | [scène](anim_phasedyn.py) · [générateur de données](generate_phasedyn_data.py) · [notes](notes/phasedyn.md) |
| `Scaling` | **Mise à l'échelle des variables**<br>VariableScalingList (x_scaling, u_scaling) ne change que ce que voit le solveur (x divisé par l'échelle); les bornes et les objectifs restent en unités physiques. | 3 - avancé | [Exemple : mise à l'échelle des variables](../../bioptim/examples/toy_examples/feature_examples/example_variable_scaling.py#L27-L111)<br>[VariableScalingList](../../bioptim/optimization/variable_scaling.py#L67-L136)<br>[VariableScaling](../../bioptim/optimization/variable_scaling.py#L17-L64) | [scène](anim_scaling.py) · [générateur de données](generate_scaling_data.py) · [notes](notes/scaling.md) |
| `SxVsMx` | **Graphes SX ou MX**<br>use_sx=True construit le graphe CasADi avec SX plutôt que MX. Un schéma des deux types de graphes précède des temps mesurés sur un OCP de pendule (N = 50) : le temps d'IPOPT passe de 0,54 à 0,14 s mais la préparation… | 3 - avancé | [Exemple exposant use_sx](../../bioptim/examples/getting_started/basic_ocp.py#L33-L127)<br>[use_sx dans OptimalControlProgram](../../bioptim/optimization/optimal_control_program.py#L523-L547) | [scène](anim_sx.py) · [générateur de données](generate_sx_data.py) · [notes](notes/sx.md) |
| `VectorOrdering` | **Ordre du vecteur de décision**<br>Montre le vecteur plat vu par IPOPT (temps, états, commandes, paramètres) et comment VectorLayout.index_map localise chaque bloc, puis réordonne les mêmes 24 nombres avec OrderingStrategy.TIME_MAJOR. Lire les résultats… | 3 - avancé | [VectorLayout](../../bioptim/optimization/vector_layout.py#L110-L286)<br>[OrderingStrategy](../../bioptim/optimization/vector_layout.py#L105-L107)<br>[OptimizationVectorHelper](../../bioptim/optimization/optimization_vector.py#L20-L283)<br>[Exemple : OCP de base](../../bioptim/examples/getting_started/basic_ocp.py#L33-L127) | [scène](anim_vector.py) · [générateur de données](generate_vector_data.py) · [notes](notes/vector.md) |

#### Modèles et biomécanique

| Scène (mp4) | Titre et contenu | Niveau | Exemple et code | Sources |
| --- | --- | --- | --- | --- |
| `UnilateralContact` | **Force de contact unilatérale**<br>Une jambe verticale (pied de 5 kg, corps de 70 kg) s'étend contre le sol avec un contact rigide. | 2 - intermédiaire | [Exemple : contact rigide (anciens noms d'API)](../../bioptim/examples/toy_examples/torque_driven_ocp/example_rigid_contact.py#L40-L138)<br>[Énumération ContactType](../../bioptim/misc/enums.py#L219-L238)<br>[Contrainte TRACK_EXPLICIT_RIGID_CONTACT_FORCES](../../bioptim/limits/constraints.py#L871)<br>[Pénalité sur la force de contact rigide](../../bioptim/limits/penalty.py#L768-L817)<br>[Modèle : rigid_contact_forces](../../bioptim/models/biorbd/biorbd_model.py#L1223-L1234) | [scène](anim_contact.py) · [générateur de données](generate_contact_data.py) · [notes](notes/contact.md) |
| `ExternalForces` | **Forces externes dans la dynamique**<br>Une force connue, de type vent, appliquée à la main est décrite avec ExternalForceSetTimeSeries et transmise au modèle et à la dynamique. | 2 - intermédiaire | [Exemple : forces externes](../../bioptim/examples/getting_started/example_external_forces.py#L30-L118)<br>[ExternalForceSetTimeSeries](../../bioptim/models/biorbd/external_forces.py#L121-L288) | [scène](anim_extforces.py) · [générateur de données](generate_extforces_data.py) · [notes](notes/extforces.md) |
| `Mapping` | **Partager des variables avec un mapping**<br>Un double pendule est résolu deux fois : avec deux couples, puis avec une BiMappingList qui fait partager un seul couple aux deux articulations (vecteur de décision de 185 à 155). | 2 - intermédiaire | [Exemple : symétrie par mapping](../../bioptim/examples/toy_examples/symmetrical_torque_driven_ocp/symmetry_by_mapping.py#L49-L146)<br>[BiMapping](../../bioptim/misc/mapping.py#L122-L173)<br>[BiMappingList](../../bioptim/misc/mapping.py#L180-L263) | [scène](anim_mapping.py) · [générateur de données](generate_mapping_data.py) · [notes](notes/mapping.md) |
| `TrackMarkers` | **Suivi de marqueurs mesurés**<br>ObjectiveFcn.Lagrange.TRACK_MARKERS avec une cible fait suivre par les marqueurs du modèle les marqueurs mesurés; la vidéo les montre converger au fil des itérations d'IPOPT, puis l'erreur finale (1,3 cm RMS). | 2 - intermédiaire | [Exemple : suivi de marqueurs sur un pendule](../../bioptim/examples/toy_examples/torque_driven_ocp/track_markers_2D_pendulum.py#L68-L147)<br>[TRACK_MARKERS dans ObjectiveFcn](../../bioptim/limits/objective_functions.py#L362-L483)<br>[Pénalité minimize_markers](../../bioptim/limits/penalty.py#L281-L331) | [scène](anim_markers.py) · [générateur de données](generate_markers_data.py) · [notes](notes/markers.md) |
| `MuscFullActivations` | **Activations musculaires et couple articulaire**<br>Pour un mouvement d'atteinte d'un bras à 2 degrés de liberté et 6 muscles commandé uniquement par des activations musculaires, montre l'activation de chaque muscle avec ses pics, et le couple articulaire produit par… | 2 - intermédiaire | [Exemple : atteinte pilotée par muscles](../../bioptim/examples/toy_examples/muscle_driven_ocp/static_arm.py#L28-L124)<br>[MusclesBiorbdModel](../../bioptim/models/biorbd/model_dynamics.py#L296-L322)<br>[Modèle : muscle_joint_torque](../../bioptim/models/biorbd/biorbd_model.py#L880-L895) | [scène](anim_muscfull.py) · [générateur de données](generate_muscfull_data.py) · [notes](notes/muscfull.md) |
| `MuscleReaching` | **Atteinte par actionnement musculaire**<br>Remplacer TorqueBiorbdModel par MusclesBiorbdModel fait des 6 activations musculaires du modèle arm26 les commandes, bornées dans [0, 1], avec un objectif de Mayer sur les marqueurs pour l'atteinte. | 2 - intermédiaire | [Exemple : atteinte du bras statique](../../bioptim/examples/toy_examples/muscle_driven_ocp/static_arm.py#L28-L124)<br>[MusclesBiorbdModel](../../bioptim/models/biorbd/model_dynamics.py#L296-L322)<br>[MusclesDynamics](../../bioptim/dynamics/state_space_dynamics/muscle_dynamics.py#L13-L171) | [scène](anim_muscle.py) · [générateur de données](generate_muscle_data.py) · [notes](notes/muscle.md) |
| `SoftContact` | **Contact souple (compliant)**<br>Une balle de 1 kg est enfoncée de 3 cm dans un sol souple déclaré dans le bioMod et activé par ContactType.SOFT_EXPLICIT. La force vient de soft_contact_forces ; pour des raideurs de 1e4, 1e5 et 1e6, la même profondeur… | 2 - intermédiaire | [Exemple : contact souple](../../bioptim/examples/toy_examples/torque_driven_ocp/example_soft_contact.py#L103-L191)<br>[Énumération ContactType](../../bioptim/misc/enums.py#L219-L238)<br>[Modèle : soft_contact_forces](../../bioptim/models/biorbd/biorbd_model.py#L1163-L1181) | [scène](anim_soft.py) · [générateur de données](generate_soft_data.py) · [notes](notes/soft.md) |
| `ExcitationActivation` | **De l'excitation à l'activation**<br>Avec MusclesWithExcitationsBiorbdModel, l'excitation musculaire est la commande et l'activation est un état : l'activation est donc en retard sur l'excitation (environ 18 ms en montée, 14 ms en descente, mesurés sur… | 3 - avancé | [Exemple : atteinte avec activations musculaires](../../bioptim/examples/toy_examples/muscle_driven_ocp/static_arm.py#L28-L124)<br>[Exemple : excitations musculaires](../../bioptim/examples/toy_examples/muscle_driven_ocp/muscle_excitations_tracker.py#L210-L322)<br>[MusclesDynamicsWithExcitations](../../bioptim/dynamics/state_space_dynamics/muscle_dynamics_with_excitations.py#L13-L190)<br>[MusclesWithExcitationsBiorbdModel](../../bioptim/models/biorbd/model_dynamics.py#L325-L351) | [scène](anim_excitation.py) · [générateur de données](generate_excitation_data.py) · [notes](notes/excitation.md) |
| `FatigueXia` | **Fatigue de Xia sur le couple**<br>Un balancier de pendule où le couple porte un modèle XiaTauFatigue : des états supplémentaires suivent les fractions active, au repos et fatiguée de chaque côté. Sur 1 s la fatigue reste faible (fraction fatiguée d'au… | 3 - avancé | [Exemple : pendule avec fatigue](../../bioptim/examples/toy_examples/fatigue/pendulum_with_fatigue.py#L39-L194)<br>[Dynamique XiaFatigue](../../bioptim/dynamics/fatigue/xia_fatigue.py#L15-L95)<br>[XiaTauFatigue (couple)](../../bioptim/dynamics/fatigue/xia_fatigue.py#L121-L132)<br>[FatigueList](../../bioptim/dynamics/fatigue/fatigue_dynamics.py#L372-L419) | [scène](anim_fatigue.py) · [générateur de données](generate_fatigue_data.py) · [notes](notes/fatigue.md) |
| `FloatingReorient` | **Réorientation à base flottante**<br>Avec TorqueFreeFloatingBaseBiorbdModel, le couple de la base est structurellement nul; en apesanteur, un tronc plan à deux bras tourne de 0,80 rad en déplaçant ses bras selon une boucle fermée. | 3 - avancé | [Exemple : base flottante (3D)](../../bioptim/examples/toy_examples/torque_driven_ocp/torque_driven_free_floating_base.py#L23-L172)<br>[TorqueFreeFloatingBaseDynamics](../../bioptim/dynamics/state_space_dynamics/torque_dynamics_free_floating_base.py#L8-L61)<br>[TorqueFreeFloatingBaseBiorbdModel](../../bioptim/models/biorbd/model_dynamics.py#L165-L187)<br>[angular_momentum](../../bioptim/models/biorbd/biorbd_model.py#L414-L425) | [scène](anim_floating.py) · [générateur de données](generate_floating_data.py) · [notes](notes/floating.md) |
| `HolonomicDoublePendulum` | **Contrainte holonome : double pendule**<br>Une contrainte de boucle fermée superposant deux marqueurs transforme deux pendules simples en un double pendule. | 3 - avancé | [Exemple : deux pendules](../../bioptim/examples/toy_examples/holonomic_constraints/two_pendulums.py#L28-L134)<br>[HolonomicConstraintsFcn.superimpose_markers](../../bioptim/models/protocols/holonomic_constraints.py#L36-L159)<br>[HolonomicTorqueBiorbdModel](../../bioptim/models/biorbd/model_dynamics.py#L120-L135) | [scène](anim_holonomic.py) · [générateur de données](generate_holonomic_data.py) · [notes](notes/holonomic.md) |
| `MultiBody` | **Deux corps dans un même OCP**<br>MultiTorqueBiorbdModel place deux modèles biorbd indépendants dans une même phase : q, qdot et tau gardent leur nom et sont empilés, et variable_index donne la portion de chaque modèle. | 3 - avancé | [Exemple : modèle multi-biorbd](../../bioptim/examples/toy_examples/torque_driven_ocp/example_multi_biorbd_model.py#L20-L74)<br>[MultiBiorbdModel](../../bioptim/models/biorbd/multi_biorbd_model.py#L23-L1088)<br>[variable_index](../../bioptim/models/biorbd/multi_biorbd_model.py#L136-L204)<br>[MultiTorqueBiorbdModel](../../bioptim/models/biorbd/model_dynamics.py#L379-L389) | [scène](anim_multibody.py) · [générateur de données](generate_multibody_data.py) · [notes](notes/multibody.md) |
| `MuscFullPaths` | **Atteinte musculaire vs couples**<br>La même atteinte de bras est résolue une fois avec les activations musculaires comme commandes et une fois avec des couples articulaires, et les deux trajectoires de la main sont comparées. | 3 - avancé | [Exemple : bras statique piloté par muscles](../../bioptim/examples/toy_examples/muscle_driven_ocp/static_arm.py#L28-L124)<br>[MusclesBiorbdModel](../../bioptim/models/biorbd/model_dynamics.py#L296-L322) | [scène](anim_muscfull.py) · [générateur de données](generate_muscfull_data.py) · [notes](notes/muscfull.md) |

#### Commande avancée

| Scène (mp4) | Titre et contenu | Niveau | Exemple et code | Sources |
| --- | --- | --- | --- | --- |
| `CyclicNMPC` | **NMPC cyclique**<br>Dans un NMPC cyclique, la fenêtre est un cycle complet et avance d'un cycle entier à chaque résolution, avec une amplitude cible modifiée entre les cycles. | 3 - avancé | [Exemple : NMPC cyclique](../../bioptim/examples/toy_examples/moving_horizon_estimation/cyclic_nmpc.py#L35-L95)<br>[CyclicRecedingHorizonOptimization](../../bioptim/optimization/receding_horizon_optimization.py#L448-L637)<br>[CyclicNonlinearModelPredictiveControl](../../bioptim/optimization/receding_horizon_optimization.py#L950-L955) | [scène](anim_cyclic.py) · [générateur de données](generate_cyclic_data.py) · [notes](notes/cyclic.md) |
| `MHEWindow` | **Estimation à horizon glissant**<br>Une fenêtre de 0,5 s glisse sur des mesures bruitées de l'angle du pendule ; à chaque nouvelle mesure la cible est mise à jour et l'OCP est résolu de nouveau. | 3 - avancé | [Exemple : estimation à horizon glissant](../../bioptim/examples/toy_examples/moving_horizon_estimation/mhe.py#L88-L158)<br>[MovingHorizonEstimator](../../bioptim/optimization/receding_horizon_optimization.py#L966-L971)<br>[RecedingHorizonOptimization](../../bioptim/optimization/receding_horizon_optimization.py#L35-L445)<br>[update_objectives_target](../../bioptim/optimization/optimal_control_program.py#L956-L980) | [scène](anim_mhe.py) · [générateur de données](generate_mhe_data.py) · [notes](notes/mhe.md) |
| `NMPCWindow` | **NMPC à horizon glissant**<br>Une fenêtre de 1 s est résolue, seul le premier nœud est appliqué, puis la fenêtre glisse d'un nœud avec une référence mise à jour et est résolue de nouveau depuis une initialisation à chaud décalée. | 3 - avancé | [Exemple : NMPC cyclique (même chariot-pendule)](../../bioptim/examples/toy_examples/moving_horizon_estimation/cyclic_nmpc.py#L35-L95)<br>[NonlinearModelPredictiveControl](../../bioptim/optimization/receding_horizon_optimization.py#L942-L947)<br>[advance_window et mise à jour des bornes](../../bioptim/optimization/receding_horizon_optimization.py#L288-L327) | [scène](anim_nmpc.py) · [générateur de données](generate_nmpc_data.py) · [notes](notes/nmpc.md) |
| `RobustPath` | **Contrainte de chemin robuste (SOCP)**<br>Un OCP stochastique optimise la trajectoire moyenne et la covariance de l'état à chaque nœud. | 3 - avancé | [Exemple : évitement d'obstacles (SOCP)](../../bioptim/examples/toy_examples/stochastic_optimal_control/obstacle_avoidance_direct_collocation.py#L368-L533)<br>[Contrainte de chemin robustifiée](../../bioptim/examples/toy_examples/stochastic_optimal_control/obstacle_avoidance_direct_collocation.py#L327-L354)<br>[StochasticOptimalControlProgram](../../bioptim/optimization/stochastic_optimal_control_program.py#L29-L675)<br>[Contrainte de continuité de la covariance](../../bioptim/limits/constraints.py#L581-L625) | [scène](anim_socp.py) · [générateur de données](generate_socp_data.py) · [notes](notes/socp.md) |

#### Vue d'ensemble de la bibliothèque

| Scène (mp4) | Titre et contenu | Niveau | Exemple et code | Sources |
| --- | --- | --- | --- | --- |
| `ArchitecturePath` | **Des entrées à Solution**<br>Suit le chemin des entrées de l'utilisateur vers Solution, en passant par OptimalControlProgram, NonLinearProgram, ConfigureProblem, les pénalités, l'assembleur du vecteur de décision et l'interface du solveur, puis le… | 1 - introduction | [NonLinearProgram.declare_shooting_points](../../bioptim/optimization/non_linear_program.py#L357-L366)<br>[ConfigureProblem.initialize](../../bioptim/dynamics/configure_problem.py#L25-L48)<br>[OptimizationVectorHelper](../../bioptim/optimization/optimization_vector.py#L20-L283)<br>[generic_solve](../../bioptim/interfaces/interface_utils.py#L98-L204) | [scène](anim_arch.py) · [générateur de données](generate_arch_data.py) · [notes](notes/arch.md) |
| `OfflineGraphs` | **Graphiques hors ligne**<br>Après la résolution, sol.graphs(show_bounds=True, save_name=...) trace les mêmes graphiques que la fenêtre en ligne pendant l'optimisation, y compris les graphiques personnalisés ajoutés avec ocp.add_plot. | 1 - introduction | [Exemple : graphiques personnalisés](../../bioptim/examples/getting_started/custom_plotting.py#L38-L113)<br>[Solution.graphs](../../bioptim/optimization/solution/solution.py#L1219-L1261)<br>[OptimalControlProgram.add_plot](../../bioptim/optimization/optimal_control_program.py#L1161-L1206)<br>[PlotOcp](../../bioptim/gui/plot.py#L162-L1255) | [scène](anim_online.py) · [générateur de données](generate_online_data.py) · [notes](notes/online.md) |
| `SolutionTour` | **Lire l'objet Solution**<br>Une résolution réelle de pendule à deux phases lue avec decision_states, stepwise_states, interpolate et integrate, avec la forme du tableau retourné par chacun, ainsi que les options de SolutionMerge et sol.cost /… | 1 - introduction | [Accesseurs d'états et de commandes de Solution](../../bioptim/optimization/solution/solution.py#L544-L635)<br>[Solution.integrate](../../bioptim/optimization/solution/solution.py#L785-L866)<br>[Solution.interpolate](../../bioptim/optimization/solution/solution.py#L1158-L1217)<br>[SolutionMerge](../../bioptim/optimization/solution/solution_data.py#L13-L21) | [scène](anim_solution.py) · [générateur de données](generate_solution_data.py) · [notes](notes/solution.md) |
| `PenaltyPanorama` | **La bibliothèque de pénalités**<br>Liste les pénalités de ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer et ConstraintFcn regroupées selon ce sur quoi elles agissent, puis explique comment choisir : Lagrange intègre sur les intervalles, Mayer agit à un nœud,… | 2 - intermédiaire | [ObjectiveFcn (Lagrange, Mayer)](../../bioptim/limits/objective_functions.py#L362-L522)<br>[ConstraintFcn](../../bioptim/limits/constraints.py#L831-L897)<br>[Implémentations des pénalités](../../bioptim/limits/penalty.py) | [scène](anim_panorama.py) · [générateur de données](generate_panorama_data.py) · [notes](notes/panorama.md) |

<!-- END TABLE -->

## Où sont les vidéos ?

**Les fichiers mp4 ne sont pas stockés dans le dépôt** (règle du dépôt : pas de fichiers binaires ; `media/`, `*.mp4` et
`assets/*.png` sont ignorés par git). Générez-les vous-même (ci-dessous) ; ce qui est versionné, ce sont les scènes, les
données (`data/*.npz`, quelques ko), le catalogue et les traductions.

## Comment les vidéos sont fabriquées

* **Données réelles.** Chaque `generate_*_data.py` construit un OCP Bioptim, le résout avec IPOPT et écrit un petit
  `data/*.npz`, versionné pour que le rendu ne nécessite ni Bioptim ni biorbd. Il n'y a aucune donnée de secours factice ;
  quand des données sont synthétiques (marqueurs bruités, mesures pour la MHE), la vidéo et les notes le disent.
* **Scènes Manim.** `anim_*.py`, `features_scenes.py` (outils communs et six scènes sur les objectifs, bornes, phases,
  temps, paramètres, impact) et `dms_vs_dc.py` (énoncé de l'OCP et discrétisation). Elles n'utilisent que
  `Text`/`MarkupText`, sans LaTeX.
* **La couche de série** (`series_style.py`, `render_series.py`, [SERIES_LAYER.md](SERIES_LAYER.md)) est appliquée au rendu
  sans modifier aucune scène : logo Bioptim sur chaque image, tout **1,6 fois plus lent** avec une pause de lecture de
  0,5 s après un nouveau texte, transitions de texte simples (fondus au lieu de l'écriture lettre par lettre),
  français/anglais, et une **carte de fin** de 4,5 s avec les liens de `catalog.json`. Elle audite aussi les images
  (chevauchement du logo, texte hors cadre ou qui se chevauche).

### Prérequis

* **Rendu** (sans Bioptim) : environnement virtuel Python 3.11 avec Manim 0.21 (il embarque PyAV : ni ffmpeg séparé, ni
  LaTeX). Polices : Segoe UI et Consolas sous Windows, DejaVu ailleurs.

  ```bash
  python3.11 -m venv .venv-manim
  # Windows : .venv-manim\Scripts\activate      Linux/macOS : source .venv-manim/bin/activate
  pip install "manim==0.21.*" numpy black
  python docs/animations/assets/fetch_logo.py      # télécharge assets/bioptim_logo.png (non versionné)
  ```

  Le logo vient de [pyomeca/biorbd_design](https://github.com/pyomeca/biorbd_design). Le premier rendu tente aussi de le
  télécharger ; sans lui, les vidéos sont produites sans logo (avertissement).
* **Régénérer les données** (facultatif) : un environnement conda avec Bioptim, biorbd, casadi et IPOPT
  (`conda env create -f environment.yml` à la racine du dépôt, puis `conda activate bioptim`). Sous Windows, si vous
  appelez directement le `python.exe` de l'environnement sans `conda activate`, ajoutez d'abord les dossiers de
  l'environnement au `PATH`, sinon les DLL d'IPOPT ne sont pas trouvées (`Plugin 'ipopt' is not found`) :
  `<env>\Library\bin`, `<env>\Library\mingw-w64\bin`, `<env>\Library\usr\bin`, `<env>\Scripts` et `<env>`.

### Régénérer les données

Depuis la racine du dépôt (de quelques secondes à quelques minutes par script) :

```bash
PYTHONPATH=. python docs/animations/generate_pendulum_data.py     # OCPStatement ... Comparison (environ 1 min)
PYTHONPATH=. python docs/animations/generate_features_data.py     # on peut nommer les expériences : parameters impact
```

Les autres scripts suivent le même schéma (`generate_<sujet>_data.py`, voir la colonne « Sources »). IPOPT est
déterministe : la même version de Bioptim donne les mêmes nombres. Les données des cinq premières scènes sont le balancement
du pendule de `bioptim/examples/toy_examples/sqp_method/pendulum.py` (N = 20, T = 1 s) : RK4 (`rk4*`), collocation de
Legendre (`col*`) et de Radau (`rad*`), plus des itérés IPOPT précoces, non convergés, que l'animation montre avant la
fermeture des écarts.

### Générer les vidéos

Toujours via la couche de série. Depuis la racine du dépôt :

```bash
python docs/animations/render_series.py anim_controls.py ControlTypes --lang fr            # une scène, en français
python docs/animations/render_series.py anim_controls.py ControlTypes --lang fr --strict   # échoue si un texte n'est pas traduit
python docs/animations/render_series.py --all --lang both --jobs 4 --out mes_videos         # les 48 scènes, FR et EN
python docs/animations/render_series.py --all --lang fr --quality 480p15 --jobs 4           # aperçu rapide
```

* Sortie : `<out>/<Scène>_<langue>.mp4` (par défaut `docs/animations/media/series/out`, ignoré par git), journaux dans
  `<out>/logs`.
* `--quality <hauteur>p<images/s>` : `1080p30` (défaut), `1080p60`, `480p15`. Options : `--no-endcard`, `--no-logo`,
  `--logo-corner auto|br|bl|tr|tl`, `--slow F` (défaut 1,6), `--min-wait S`, `--media-dir`, `--catalog`.
* `--strict` (français) : code de sortie 3 si une chaîne n'a pas de traduction.
* `--dry --collect keys.jsonl` : construction seule, sans vidéo, et écriture de toutes les clés de texte traduisibles
  (`python docs/animations/render_series.py --all --dry --collect keys.jsonl --jobs 4`).
* `python docs/animations/render_series.py --gen-catalog` ajoute le squelette des nouvelles scènes à `catalog.json`.

Manim seul fonctionne encore pour un aperçu brut (`cd docs/animations; manim -ql dms_vs_dc.py MultipleShooting`) mais sans
logo, rythme, français ni carte de fin.

## Ajouter une scène

1. Écrire `anim_<sujet>.py` (et `generate_<sujet>_data.py` si une résolution est nécessaire ; ne versionner que le petit
   `.npz`), en réutilisant les outils de `features_scenes.py` et l'aspect des scènes existantes. Ajouter
   `notes/<sujet>.md` : ce qui est réel, ce qui est simplifié, les chiffres montrés.
2. Ajouter la scène au catalogue : `python docs/animations/render_series.py --gen-catalog`, puis renseigner `title_en/fr`,
   `description_en/fr`, `section`/`section_fr`, `level`, `notes_file` et `links` (exemple, lignes de la bibliothèque sous
   la forme `"path"` + `"lines": "40-90"`, générateur de données). Les liens alimentent la carte de fin et les tableaux
   de ce README.
3. Français : `python docs/animations/render_series.py anim_<sujet>.py <Scène> --dry --collect keys.jsonl`, traduire les
   clés dans un nouveau `i18n/fr_<sujet>.json` (méthode et glossaire dans [i18n/README.md](i18n/README.md)), puis lancer le
   rendu français avec `--strict` et lire les avertissements d'audit (`<out>/logs/<Scène>_<langue>_audit.json` :
   chevauchement du logo, texte hors cadre ou qui se chevauche).
4. Régénérer les tableaux : `python docs/animations/build_readme_tables.py`.

Liste de contrôle pour un aspect homogène :

* les panneaux de code portent la légende **Bioptim code** au-dessus, en police de code (jamais traduits) ;
* des phrases entières (une idée par écran), assez courtes pour le français, qui est plus long de 15 à 20 % ;
* les nombres écrits comme de simples nombres dans la chaîne, pour devenir `{0}`, `{1}` dans les modèles de traduction ;
* les identifiants Bioptim (`OdeSolver`, `ObjectiveFcn.Lagrange...`, `n_shooting`) restent en anglais ;
* terminer la scène par un maintien de 2,5 s (`self.wait(2.5)`) avant la carte de fin ;
* les plages de lignes des liens du catalogue sont vérifiées sur le code de cette version.

Glossaire principal : commandes (controls), états (states), nœud (node), tir multiple (multiple shooting), collocation
directe (direct collocation), bornes (bounds), coût (cost), contrainte (constraint), initialisation à chaud (warm start),
estimation initiale (initial guess), poids (weight). Le glossaire complet est dans [i18n/README.md](i18n/README.md).

## Régénérer les tableaux

`build_readme_tables.py` remplit les zones délimitées par les commentaires HTML `BEGIN TABLE` / `END TABLE` (et `BEGIN START` / `END START`
pour le tableau « commencer ici ») de `README.md` et `README.fr.md` à partir de `catalog.json`. Il est idempotent et vérifie aussi que chaque lien
relatif existe :

```bash
python docs/animations/build_readme_tables.py          # réécrit les tableaux, puis vérifie les liens
python docs/animations/build_readme_tables.py --check  # vérification seule
```

Ne modifiez pas les tableaux à la main : modifiez `catalog.json`.

## Réserves (à lire)

C'est du **matériel pédagogique, pas un benchmark**.

* **Minima locaux et initialisation à chaud.** Les problèmes de commande optimale sont non convexes. IPOPT renvoie *un*
  minimum local qui dépend de l'estimation initiale ; plusieurs vidéos utilisent une continuation (initialisation à chaud
  à partir du poids ou de la borne précédents) et le disent. Deux discrétisations du même problème (RK4 et collocation)
  peuvent aboutir à des minima différents avec des coûts très différents : leurs coûts ne sont donc pas une comparaison de
  précision. `IpoptMultiStart` le montre volontairement.
* **Les temps sont bruités.** Temps et nombres d'itérations viennent d'un seul portable, de quelques essais, parfois avec
  d'autres tâches en cours ; les itérations sont plus fiables que les secondes. Aucune vidéo ne revendique du temps réel.
* **Données synthétiques quand c'est le cas.** Les marqueurs mesurés (`TrackMarkers`) et les mesures bruitées de
  `MHEWindow` sont synthétiques (vérité connue, bruit et graine indiqués).
* **Modèles simplifiés.** Le pendule n'a qu'une coordonnée actionnée ; le sauteur (`Hopper`, `Impact`) est 1-D avec un
  impact ponctuel inélastique et sans frottement, sans segments de jambe ; le contact souple est un modèle compliant
  simple ; les modèles de muscles et de fatigue sont les petits modèles des exemples de Bioptim. Les panneaux de code
  montrent parfois un appel simplifié (bornes, imports omis) : l'exemple lié fait référence.
* **Comptes.** Bioptim déclare une variable de décision de plus que `(N+1)*4 + N*2` pour le pendule RK4 (125 au lieu de
  124), non investigué.
* Les plages de lignes liées dans le tableau correspondent à la version de Bioptim de cette branche et peuvent dériver
  après des modifications ultérieures.

## Exercices (issus des cinq premières vidéos)

1. Changer `polynomial_degree` (2 à 5) : comment évoluent variables, itérations et temps de résolution ?
2. Changer `n_shooting` (10, 20, 40) pour les deux transcriptions : taille du NLP, écarts entre nœuds ?
3. Essayer `OdeSolver.RK4(n_integration_steps=1, 5, 10)`, `RK8()`, `COLLOCATION(method="radau")`, `IRK()` : lesquels
   ressemblent au tir, lesquels à la collocation ?
4. Après `sol = ocp.solve(...)`, intégrer les commandes optimales avec `sol.integrate()` et tracer les défauts
   `F(x_k, u_k) - x_{k+1}`.
5. Pour une comparaison équitable, donner la même estimation initiale aux deux problèmes (ou initialiser l'un à chaud à
   partir de l'autre) et comparer à une simulation fine.
6. Changer l'OCP, pas la discrétisation : un temps libre avec `ObjectiveFcn.Mayer.MINIMIZE_TIME`, ou des bornes sur `q`.

## Crédits et licence

Bioptim est développé par la communauté [pyomeca](https://github.com/pyomeca/bioptim) ; voir [LICENSE](../../LICENSE)
pour la licence de ce dépôt. Le logo vient de [pyomeca/biorbd_design](https://github.com/pyomeca/biorbd_design) et est
téléchargé au moment du rendu, non redistribué ici. Les animations sont réalisées avec
[Manim Community](https://www.manim.community/).

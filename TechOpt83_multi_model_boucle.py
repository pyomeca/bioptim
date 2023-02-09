"""
The goal of this program is to optimize the movement to achieve a rudi out pike (803<).
Simultaneously for two anthropometric models.
"""
import numpy as np
import biorbd_casadi as biorbd
from typing import Union
import casadi as cas
import sys
import argparse

sys.path.append('/home/lim/Documents/Stage_Lisa/bioptim/')
from bioptim import (
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    ObjectiveList,
    ObjectiveFcn,
    BoundsList,
    QAndQDotBounds,
    InitialGuessList,
    InterpolationType,
    OdeSolver,
    Node,
    Solver,
    BiMappingList,
    CostType,
    ConstraintList,
    ConstraintFcn,
    PenaltyNodeList,
    BiorbdInterface,
    NonLinearProgram,
    ConfigureProblem,
    DynamicsFunctions,
    PhaseTransitionList,
    PhaseTransitionFcn,
    NodeMappingList,
)

import time

try:
    import IPython
    IPYTHON = True
except ImportError:
    print("No IPython.")
    IPYTHON = False


def minimize_dofs(all_pn: PenaltyNodeList, dofs: list, targets: list) -> cas.MX:
    diff = 0
    for i, dof in enumerate(dofs):
        diff += (all_pn.nlp.states['q'].mx[dof] - targets[i])**2
    return BiorbdInterface.mx_to_cx('minimize_dofs', diff, all_pn.nlp.states['q'])


def set_fancy_names_index(nb_q):
    """
    For readability
    """
    fancy_names_index = {}
    fancy_names_index["X"] = 0
    fancy_names_index["Y"] = 1
    fancy_names_index["Z"] = 2
    fancy_names_index["Xrot"] = 3
    fancy_names_index["Yrot"] = 4
    fancy_names_index["Zrot"] = 5
    fancy_names_index["ZrotBD"] = 6
    fancy_names_index["YrotBD"] = 7
    fancy_names_index["ZrotABD"] = 8
    fancy_names_index["XrotABD"] = 9
    fancy_names_index["ZrotBG"] = 10
    fancy_names_index["YrotBG"] = 11
    fancy_names_index["ZrotABG"] = 12
    fancy_names_index["XrotABG"] = 13
    fancy_names_index["XrotC"] = 14
    fancy_names_index["YrotC"] = 15
    fancy_names_index["vX"] = 0 + nb_q
    fancy_names_index["vY"] = 1 + nb_q
    fancy_names_index["vZ"] = 2 + nb_q
    fancy_names_index["vXrot"] = 3 + nb_q
    fancy_names_index["vYrot"] = 4 + nb_q
    fancy_names_index["vZrot"] = 5 + nb_q
    fancy_names_index["vZrotBD"] = 6 + nb_q
    fancy_names_index["vYrotBD"] = 7 + nb_q
    fancy_names_index["vZrotABD"] = 8 + nb_q
    fancy_names_index["vYrotABD"] = 9 + nb_q
    fancy_names_index["vZrotBG"] = 10 + nb_q
    fancy_names_index["vYrotBG"] = 11 + nb_q
    fancy_names_index["vZrotABG"] = 12 + nb_q
    fancy_names_index["vYrotABG"] = 13 + nb_q
    fancy_names_index["vXrotC"] = 14 + nb_q
    fancy_names_index["vYrotC"] = 15 + nb_q

    return fancy_names_index

def set_x_bounds(biorbd_model, fancy_names_index, final_time):

    nb_q = biorbd_model[0].nbQ()
    nb_qdot = biorbd_model[0].nbQ()

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[1]))
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[2]))
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[3]))
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[4]))
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[5]))
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[6]))
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[7]))
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[8]))
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[9]))

    # Pour la lisibilite
    DEBUT, MILIEU, FIN = 0, 1, 2

    #
    # Contraintes de position: PHASE 0 la montee en carpe
    #

    zmax = 9.81 / 8 * final_time**2 + 1  # une petite marge

    # deplacement
    x_bounds[0].min[fancy_names_index["X"], :] = -.1
    x_bounds[0].max[fancy_names_index["X"], :] = .1
    x_bounds[0].min[fancy_names_index["Y"], :] = -1.
    x_bounds[0].max[fancy_names_index["Y"], :] = 1.
    x_bounds[0].min[:fancy_names_index["Z"]+1, DEBUT] = 0
    x_bounds[0].max[:fancy_names_index["Z"]+1, DEBUT] = 0
    x_bounds[0].min[fancy_names_index["Z"], MILIEU:] = 0
    x_bounds[0].max[fancy_names_index["Z"], MILIEU:] = zmax  # beaucoup plus que necessaire, juste pour que la parabole fonctionne

    # le salto autour de x
    x_bounds[0].min[fancy_names_index["Xrot"], DEBUT] = .50  # penche vers l'avant un peu carpe
    x_bounds[0].max[fancy_names_index["Xrot"], DEBUT] = .50
    x_bounds[0].min[fancy_names_index["Xrot"], MILIEU:] = 0
    x_bounds[0].max[fancy_names_index["Xrot"], MILIEU:] = 4 * 3.14 + .1  # salto

    # limitation du tilt autour de y
    x_bounds[0].min[fancy_names_index["Yrot"], DEBUT] = 0
    x_bounds[0].max[fancy_names_index["Yrot"], DEBUT] = 0
    x_bounds[0].min[fancy_names_index["Yrot"], MILIEU:] = - 3.14 / 16  # vraiment pas suppose tilte
    x_bounds[0].max[fancy_names_index["Yrot"], MILIEU:] = 3.14 / 16

    # la vrille autour de z
    x_bounds[0].min[fancy_names_index["Zrot"], DEBUT] = 0
    x_bounds[0].max[fancy_names_index["Zrot"], DEBUT] = 0
    x_bounds[0].min[fancy_names_index["Zrot"], MILIEU:] = -.1  # pas de vrille dans cette phase
    x_bounds[0].max[fancy_names_index["Zrot"], MILIEU:] = .1

    # bras droit
    x_bounds[0].min[fancy_names_index["YrotBD"], DEBUT] = 2.9  # debut bras aux oreilles
    x_bounds[0].max[fancy_names_index["YrotBD"], DEBUT] = 2.9
    x_bounds[0].min[fancy_names_index["ZrotBD"], DEBUT] = 0
    x_bounds[0].max[fancy_names_index["ZrotBD"], DEBUT] = 0

    # bras gauche
    x_bounds[0].min[fancy_names_index["YrotBG"], DEBUT] = -2.9  # debut bras aux oreilles
    x_bounds[0].max[fancy_names_index["YrotBG"], DEBUT] = -2.9
    x_bounds[0].min[fancy_names_index["ZrotBG"], DEBUT] = 0
    x_bounds[0].max[fancy_names_index["ZrotBG"], DEBUT] = 0

    # coude droit
    x_bounds[0].min[fancy_names_index["ZrotABD"]:fancy_names_index["XrotABD"]+1, DEBUT] = 0
    x_bounds[0].max[fancy_names_index["ZrotABD"]:fancy_names_index["XrotABD"]+1, DEBUT] = 0

    # coude gauche
    x_bounds[0].min[fancy_names_index["ZrotABG"]:fancy_names_index["XrotABG"]+1, DEBUT] = 0
    x_bounds[0].max[fancy_names_index["ZrotABG"]:fancy_names_index["XrotABG"]+1, DEBUT] = 0

    # le carpe
    x_bounds[0].min[fancy_names_index["XrotC"], DEBUT] = -.50  # depart un peu ferme aux hanches
    x_bounds[0].max[fancy_names_index["XrotC"], DEBUT] = -.50
    x_bounds[0].min[fancy_names_index["XrotC"], FIN] = -2.35
    x_bounds[0].max[fancy_names_index["XrotC"], FIN] = -2.35

    # le dehanchement
    x_bounds[0].min[fancy_names_index["YrotC"], DEBUT] = 0
    x_bounds[0].max[fancy_names_index["YrotC"], DEBUT] = 0
    x_bounds[0].min[fancy_names_index["YrotC"], MILIEU:] = -.1
    x_bounds[0].max[fancy_names_index["YrotC"], MILIEU:] = .1

    # Contraintes de vitesse: PHASE 0 la montee en carpe

    vzinit = 9.81 / 2 * final_time  # vitesse initiale en z du CoM pour revenir a terre au temps final

    # decalage entre le bassin et le CoM
    # AUJO
    CoM_Q_sym = cas.MX.sym('CoM', nb_q)
    CoM_Q_init = x_bounds[0].min[:nb_q, DEBUT]  # min ou max ne change rien a priori, au DEBUT ils sont egaux normalement
    CoM_Q_func = cas.Function('CoM_Q_func', [CoM_Q_sym], [biorbd_model[0].CoM(CoM_Q_sym).to_mx()])
    bassin_Q_func = cas.Function('bassin_Q_func', [CoM_Q_sym],
                             [biorbd_model[0].globalJCS(0).to_mx()])  # retourne la RT du bassin

    r = np.array(CoM_Q_func(CoM_Q_init)).reshape(1, 3) - np.array(bassin_Q_func(CoM_Q_init))[-1, :3]  # selectionne seulement la translation de la RT

    #JECH

    # en xy bassin
    x_bounds[0].min[fancy_names_index["vX"]:fancy_names_index["vY"]+1, :] = -10
    x_bounds[0].max[fancy_names_index["vX"]:fancy_names_index["vY"]+1, :] = 10
    x_bounds[0].min[fancy_names_index["vX"]:fancy_names_index["vY"]+1, DEBUT] = -.5
    x_bounds[0].max[fancy_names_index["vX"]:fancy_names_index["vY"]+1, DEBUT] = .5

    # z bassin
    x_bounds[0].min[fancy_names_index["vZ"], :] = -100
    x_bounds[0].max[fancy_names_index["vZ"], :] = 100
    x_bounds[0].min[fancy_names_index["vZ"], DEBUT] = vzinit - .5
    x_bounds[0].max[fancy_names_index["vZ"], DEBUT] = vzinit + .5

    # autour de x
    x_bounds[0].min[fancy_names_index["vXrot"], :] = .5  # d'apres une observation video
    x_bounds[0].max[fancy_names_index["vXrot"], :] = 20  # aussi vite que nécessaire, mais ne devrait pas atteindre cette vitesse

    # autour de y
    x_bounds[0].min[fancy_names_index["vYrot"], :] = -100
    x_bounds[0].max[fancy_names_index["vYrot"], :] = 100
    x_bounds[0].min[fancy_names_index["vYrot"], DEBUT] = 0
    x_bounds[0].max[fancy_names_index["vYrot"], DEBUT] = 0

    # autour de z
    x_bounds[0].min[fancy_names_index["vZrot"], :] = -100
    x_bounds[0].max[fancy_names_index["vZrot"], :] = 100
    x_bounds[0].min[fancy_names_index["vZrot"], DEBUT] = 0
    x_bounds[0].max[fancy_names_index["vZrot"], DEBUT] = 0


    # tenir compte du decalage entre bassin et CoM avec la rotation
    # Qtransdot = Qtransdot + v cross Qrotdot
    borne_inf = ( x_bounds[0].min[fancy_names_index["vX"]:fancy_names_index["vZ"]+1, DEBUT] + np.cross(r, x_bounds[0].min[fancy_names_index["vXrot"]:fancy_names_index["vZrot"]+1, DEBUT]) )[0]
    borne_sup = ( x_bounds[0].max[fancy_names_index["vX"]:fancy_names_index["vZ"]+1, DEBUT] + np.cross(r, x_bounds[0].max[fancy_names_index["vXrot"]:fancy_names_index["vZrot"]+1, DEBUT]) )[0]
    x_bounds[0].min[fancy_names_index["vX"]:fancy_names_index["vZ"]+1, DEBUT] = min(borne_sup[0], borne_inf[0]), min(borne_sup[1], borne_inf[1]), min(borne_sup[2], borne_inf[2])
    x_bounds[0].max[fancy_names_index["vX"]:fancy_names_index["vZ"]+1, DEBUT] = max(borne_sup[0], borne_inf[0]), max(borne_sup[1], borne_inf[1]), max(borne_sup[2], borne_inf[2])

    # bras droit
    x_bounds[0].min[fancy_names_index["vZrotBD"]:fancy_names_index["vYrotBD"]+1, :] = -100
    x_bounds[0].max[fancy_names_index["vZrotBD"]:fancy_names_index["vYrotBD"]+1, :] = 100
    x_bounds[0].min[fancy_names_index["vZrotBD"]:fancy_names_index["vYrotBD"]+1, DEBUT] = 0
    x_bounds[0].max[fancy_names_index["vZrotBD"]:fancy_names_index["vYrotBD"]+1, DEBUT] = 0

    # bras droit
    x_bounds[0].min[fancy_names_index["vZrotBG"]:fancy_names_index["vYrotBG"]+1, :] = -100
    x_bounds[0].max[fancy_names_index["vZrotBG"]:fancy_names_index["vYrotBG"]+1, :] = 100
    x_bounds[0].min[fancy_names_index["vZrotBG"]:fancy_names_index["vYrotBG"]+1, DEBUT] = 0
    x_bounds[0].max[fancy_names_index["vZrotBG"]:fancy_names_index["vYrotBG"]+1, DEBUT] = 0

    # coude droit
    x_bounds[0].min[fancy_names_index["vZrotABD"]:fancy_names_index["vYrotABD"]+1, :] = -100
    x_bounds[0].max[fancy_names_index["vZrotABD"]:fancy_names_index["vYrotABD"]+1, :] = 100
    x_bounds[0].min[fancy_names_index["vZrotABD"]:fancy_names_index["vYrotABD"]+1, DEBUT] = 0
    x_bounds[0].max[fancy_names_index["vZrotABD"]:fancy_names_index["vYrotABD"]+1, DEBUT] = 0

    # coude gauche
    x_bounds[0].min[fancy_names_index["vZrotABD"]:fancy_names_index["vYrotABG"]+1, :] = -100
    x_bounds[0].max[fancy_names_index["vZrotABD"]:fancy_names_index["vYrotABG"]+1, :] = 100
    x_bounds[0].min[fancy_names_index["vZrotABG"]:fancy_names_index["vYrotABG"]+1, DEBUT] = 0
    x_bounds[0].max[fancy_names_index["vZrotABG"]:fancy_names_index["vYrotABG"]+1, DEBUT] = 0

    # du carpe
    x_bounds[0].min[fancy_names_index["vXrotC"], :] = -100
    x_bounds[0].max[fancy_names_index["vXrotC"], :] = 100
    x_bounds[0].min[fancy_names_index["vXrotC"], DEBUT] = 0
    x_bounds[0].max[fancy_names_index["vXrotC"], DEBUT] = 0

    # du dehanchement
    x_bounds[0].min[fancy_names_index["vYrotC"], :] = -100
    x_bounds[0].max[fancy_names_index["vYrotC"], :] = 100
    x_bounds[0].min[fancy_names_index["vYrotC"], DEBUT] = 0
    x_bounds[0].max[fancy_names_index["vYrotC"], DEBUT] = 0

    #
    # Contraintes de position: PHASE 1 le salto carpe
    #

    # deplacement
    x_bounds[1].min[fancy_names_index["X"], :] = -.1
    x_bounds[1].max[fancy_names_index["X"], :] = .1
    x_bounds[1].min[fancy_names_index["Y"], :] = -1.
    x_bounds[1].max[fancy_names_index["Y"], :] = 1.
    x_bounds[1].min[fancy_names_index["Z"], :] = 0
    x_bounds[1].max[fancy_names_index["Z"], :] = zmax  # beaucoup plus que necessaire, juste pour que la parabole fonctionne

    # le salto autour de x
    x_bounds[1].min[fancy_names_index["Xrot"], :] = 0
    x_bounds[1].max[fancy_names_index["Xrot"], :] = 4 * 3.14
    x_bounds[1].min[fancy_names_index["Xrot"], FIN] = 2 * 3.14 - .1

    # limitation du tilt autour de y
    x_bounds[1].min[fancy_names_index["Yrot"], :] = - 3.14 / 16
    x_bounds[1].max[fancy_names_index["Yrot"], :] = 3.14 / 16

    # la vrille autour de z
    x_bounds[1].min[fancy_names_index["Zrot"], :] = -.1
    x_bounds[1].max[fancy_names_index["Zrot"], :] = .1

    # bras f4a a l'ouverture

    # le carpe
    x_bounds[1].min[fancy_names_index["XrotC"], :] = -2.35 - 0.1
    x_bounds[1].max[fancy_names_index["XrotC"], :] = -2.35 + 0.1

    # le dehanchement
    x_bounds[1].min[fancy_names_index["YrotC"], DEBUT] = -.1
    x_bounds[1].max[fancy_names_index["YrotC"], DEBUT] = .1



    # Contraintes de vitesse: PHASE 1 le salto carpe

    # en xy bassin
    x_bounds[1].min[fancy_names_index["vX"]:fancy_names_index["vY"] + 1, :] = -10
    x_bounds[1].max[fancy_names_index["vX"]:fancy_names_index["vY"] + 1, :] = 10

    # z bassin
    x_bounds[1].min[fancy_names_index["vZ"], :] = -100
    x_bounds[1].max[fancy_names_index["vZ"], :] = 100

    # autour de x
    x_bounds[1].min[fancy_names_index["vXrot"], :] = -100
    x_bounds[1].max[fancy_names_index["vXrot"], :] = 100

    # autour de y
    x_bounds[1].min[fancy_names_index["vYrot"], :] = -100
    x_bounds[1].max[fancy_names_index["vYrot"], :] = 100

    # autour de z
    x_bounds[1].min[fancy_names_index["vZrot"], :] = -100
    x_bounds[1].max[fancy_names_index["vZrot"], :] = 100

    # bras droit
    x_bounds[1].min[fancy_names_index["vZrotBD"]:fancy_names_index["vYrotBD"] + 1, :] = -100
    x_bounds[1].max[fancy_names_index["vZrotBD"]:fancy_names_index["vYrotBD"] + 1, :] = 100

    # bras droit
    x_bounds[1].min[fancy_names_index["vZrotBG"]:fancy_names_index["vYrotBG"] + 1, :] = -100
    x_bounds[1].max[fancy_names_index["vZrotBG"]:fancy_names_index["vYrotBG"] + 1, :] = 100

    # coude droit
    x_bounds[1].min[fancy_names_index["vZrotABD"]:fancy_names_index["vYrotABD"] + 1, :] = -100
    x_bounds[1].max[fancy_names_index["vZrotABD"]:fancy_names_index["vYrotABD"] + 1, :] = 100
    # coude gauche
    x_bounds[1].min[fancy_names_index["vZrotABD"]:fancy_names_index["vYrotABG"] + 1, :] = -100
    x_bounds[1].max[fancy_names_index["vZrotABD"]:fancy_names_index["vYrotABG"] + 1, :] = 100

    # du carpe
    x_bounds[1].min[fancy_names_index["vXrotC"], :] = -100
    x_bounds[1].max[fancy_names_index["vXrotC"], :] = 100

    # du dehanchement
    x_bounds[1].min[fancy_names_index["vYrotC"], :] = -100
    x_bounds[1].max[fancy_names_index["vYrotC"], :] = 100

    #
    # Contraintes de position: PHASE 2 l'ouverture
    #

    # deplacement
    x_bounds[2].min[fancy_names_index["X"], :] = -.2
    x_bounds[2].max[fancy_names_index["X"], :] = .2
    x_bounds[2].min[fancy_names_index["Y"], :] = -1.
    x_bounds[2].max[fancy_names_index["Y"], :] = 1.
    x_bounds[2].min[fancy_names_index["Z"], :] = 0
    x_bounds[2].max[fancy_names_index["Z"], :] = zmax  # beaucoup plus que necessaire, juste pour que la parabole fonctionne

    # le salto autour de x
    x_bounds[2].min[fancy_names_index["Xrot"], :] = 2 * 3.14 + .1  # 1 salto 3/4
    x_bounds[2].max[fancy_names_index["Xrot"], :] = 4 * 3.14

    # limitation du tilt autour de y
    x_bounds[2].min[fancy_names_index["Yrot"], :] = - 3.14 / 4
    x_bounds[2].max[fancy_names_index["Yrot"], :] = 3.14 / 4

    # la vrille autour de z
    x_bounds[2].min[fancy_names_index["Zrot"], :] = 0
    x_bounds[2].max[fancy_names_index["Zrot"], :] = 3 * 3.14

    # bras f4a a l'ouverture

    # le carpe
    x_bounds[2].min[fancy_names_index["XrotC"], FIN] = -.4

    # le dehanchement f4a a l'ouverture

    # Contraintes de vitesse: PHASE 2 l'ouverture

    # en xy bassin
    x_bounds[2].min[fancy_names_index["vX"]:fancy_names_index["vY"] + 1, :] = -10
    x_bounds[2].max[fancy_names_index["vX"]:fancy_names_index["vY"] + 1, :] = 10

    # z bassin
    x_bounds[2].min[fancy_names_index["vZ"], :] = -100
    x_bounds[2].max[fancy_names_index["vZ"], :] = 100

    # autour de x
    x_bounds[2].min[fancy_names_index["vXrot"], :] = -100
    x_bounds[2].max[fancy_names_index["vXrot"], :] = 100

    # autour de y
    x_bounds[2].min[fancy_names_index["vYrot"], :] = -100
    x_bounds[2].max[fancy_names_index["vYrot"], :] = 100

    # autour de z
    x_bounds[2].min[fancy_names_index["vZrot"], :] = -100
    x_bounds[2].max[fancy_names_index["vZrot"], :] = 100

    # bras droit
    x_bounds[2].min[fancy_names_index["vZrotBD"]:fancy_names_index["vYrotBD"] + 1, :] = -100
    x_bounds[2].max[fancy_names_index["vZrotBD"]:fancy_names_index["vYrotBD"] + 1, :] = 100

    # bras droit
    x_bounds[2].min[fancy_names_index["vZrotBG"]:fancy_names_index["vYrotBG"] + 1, :] = -100
    x_bounds[2].max[fancy_names_index["vZrotBG"]:fancy_names_index["vYrotBG"] + 1, :] = 100

    # coude droit
    x_bounds[2].min[fancy_names_index["vZrotABD"]:fancy_names_index["vYrotABD"] + 1, :] = -100
    x_bounds[2].max[fancy_names_index["vZrotABD"]:fancy_names_index["vYrotABD"] + 1, :] = 100

    # coude gauche
    x_bounds[2].min[fancy_names_index["vZrotABD"]:fancy_names_index["vYrotABG"] + 1, :] = -100
    x_bounds[2].max[fancy_names_index["vZrotABD"]:fancy_names_index["vYrotABG"] + 1, :] = 100

    # du carpe
    x_bounds[2].min[fancy_names_index["vXrotC"], :] = -100
    x_bounds[2].max[fancy_names_index["vXrotC"], :] = 100

    # du dehanchement
    x_bounds[2].min[fancy_names_index["vYrotC"], :] = -100
    x_bounds[2].max[fancy_names_index["vYrotC"], :] = 100

    #
    # Contraintes de position: PHASE 3 la vrille et demie
    #

    # deplacement
    x_bounds[3].min[fancy_names_index["X"], :] = -.2
    x_bounds[3].max[fancy_names_index["X"], :] = .2
    x_bounds[3].min[fancy_names_index["Y"], :] = -1.
    x_bounds[3].max[fancy_names_index["Y"], :] = 1.
    x_bounds[3].min[fancy_names_index["Z"], :] = 0
    x_bounds[3].max[fancy_names_index["Z"], :] = zmax  # beaucoup plus que necessaire, juste pour que la parabole fonctionne

    # le salto autour de x
    x_bounds[3].min[fancy_names_index["Xrot"], :] = 2 * 3.14 - .1
    x_bounds[3].max[fancy_names_index["Xrot"], :] = 2 * 3.14 + 3/2 * 3.14 + .1  # 1 salto 3/4
    x_bounds[3].min[fancy_names_index["Xrot"], FIN] = 2 * 3.14 + 3/2 * 3.14 - .1
    x_bounds[3].max[fancy_names_index["Xrot"], FIN] = 2 * 3.14 + 3/2 * 3.14 + .1  # 1 salto 3/4

    # limitation du tilt autour de y
    x_bounds[3].min[fancy_names_index["Yrot"], :] = - 3.14 / 4
    x_bounds[3].max[fancy_names_index["Yrot"], :] = 3.14 / 4
    x_bounds[3].min[fancy_names_index["Yrot"], FIN] = - 3.14 / 8
    x_bounds[3].max[fancy_names_index["Yrot"], FIN] = 3.14 / 8

    # la vrille autour de z
    x_bounds[3].min[fancy_names_index["Zrot"], :] = 0
    x_bounds[3].max[fancy_names_index["Zrot"], :] = 3 * 3.14
    x_bounds[3].min[fancy_names_index["Zrot"], FIN] = 3 * 3.14 - .1  # complete la vrille
    x_bounds[3].max[fancy_names_index["Zrot"], FIN] = 3 * 3.14 + .1

    # bras f4a la vrille

    # le carpe
    x_bounds[3].min[fancy_names_index["XrotC"], :] = -.4

    # le dehanchement f4a la vrille

    # Contraintes de vitesse: PHASE 3 la vrille et demie

    # en xy bassin
    x_bounds[3].min[fancy_names_index["vX"]:fancy_names_index["vY"] + 1, :] = -10
    x_bounds[3].max[fancy_names_index["vX"]:fancy_names_index["vY"] + 1, :] = 10

    # z bassin
    x_bounds[3].min[fancy_names_index["vZ"], :] = -100
    x_bounds[3].max[fancy_names_index["vZ"], :] = 100

    # autour de x
    x_bounds[3].min[fancy_names_index["vXrot"], :] = -100
    x_bounds[3].max[fancy_names_index["vXrot"], :] = 100

    # autour de y
    x_bounds[3].min[fancy_names_index["vYrot"], :] = -100
    x_bounds[3].max[fancy_names_index["vYrot"], :] = 100

    # autour de z
    x_bounds[3].min[fancy_names_index["vZrot"], :] = -100
    x_bounds[3].max[fancy_names_index["vZrot"], :] = 100

    # bras droit
    x_bounds[3].min[fancy_names_index["vZrotBD"]:fancy_names_index["vYrotBD"] + 1, :] = -100
    x_bounds[3].max[fancy_names_index["vZrotBD"]:fancy_names_index["vYrotBD"] + 1, :] = 100

    # bras droit
    x_bounds[3].min[fancy_names_index["vZrotBG"]:fancy_names_index["vYrotBG"] + 1, :] = -100
    x_bounds[3].max[fancy_names_index["vZrotBG"]:fancy_names_index["vYrotBG"] + 1, :] = 100

    # coude droit
    x_bounds[3].min[fancy_names_index["vZrotABD"]:fancy_names_index["vYrotABD"] + 1, :] = -100
    x_bounds[3].max[fancy_names_index["vZrotABD"]:fancy_names_index["vYrotABD"] + 1, :] = 100

    # coude gauche
    x_bounds[3].min[fancy_names_index["vZrotABD"]:fancy_names_index["vYrotABG"] + 1, :] = -100
    x_bounds[3].max[fancy_names_index["vZrotABD"]:fancy_names_index["vYrotABG"] + 1, :] = 100

    # du carpe
    x_bounds[3].min[fancy_names_index["vXrotC"], :] = -100
    x_bounds[3].max[fancy_names_index["vXrotC"], :] = 100

    # du dehanchement
    x_bounds[3].min[fancy_names_index["vYrotC"], :] = -100
    x_bounds[3].max[fancy_names_index["vYrotC"], :] = 100

    #
    # Contraintes de position: PHASE 4 la reception
    #

    # deplacement
    x_bounds[4].min[fancy_names_index["X"], :] = -.1
    x_bounds[4].max[fancy_names_index["X"], :] = .1
    x_bounds[4].min[fancy_names_index["Y"], FIN] = -.1
    x_bounds[4].max[fancy_names_index["Y"], FIN] = .1
    x_bounds[4].min[fancy_names_index["Z"], :] = 0
    x_bounds[4].max[fancy_names_index["Z"], :] = zmax  # beaucoup plus que necessaire, juste pour que la parabole fonctionne
    x_bounds[4].min[fancy_names_index["Z"], FIN] = 0
    x_bounds[4].max[fancy_names_index["Z"], FIN] = .1

    # le salto autour de x
    x_bounds[4].min[fancy_names_index["Xrot"], :] = 2 * 3.14 + 3 / 2 * 3.14 - .2  # penche vers avant -> moins de salto
    x_bounds[4].max[fancy_names_index["Xrot"], :] = -.50 + 4 * 3.14  # un peu carpe a la fin
    x_bounds[4].min[fancy_names_index["Xrot"], FIN] = -.50 + 4 * 3.14 - .1
    x_bounds[4].max[fancy_names_index["Xrot"], FIN] = -.50 + 4 * 3.14 + .1  # 2 salto fin un peu carpe

    # limitation du tilt autour de y
    x_bounds[4].min[fancy_names_index["Yrot"], :] = - 3.14 / 16
    x_bounds[4].max[fancy_names_index["Yrot"], :] = 3.14 / 16

    # la vrille autour de z
    x_bounds[4].min[fancy_names_index["Zrot"], :] = 3 * 3.14 - .1  # complete la vrille
    x_bounds[4].max[fancy_names_index["Zrot"], :] = 3 * 3.14 + .1

    # bras droit
    x_bounds[4].min[fancy_names_index["YrotBD"], FIN] = 2.9 - .1  # debut bras aux oreilles
    x_bounds[4].max[fancy_names_index["YrotBD"], FIN] = 2.9 + .1
    x_bounds[4].min[fancy_names_index["ZrotBD"], FIN] = -.1
    x_bounds[4].max[fancy_names_index["ZrotBD"], FIN] = .1

    # bras gauche
    x_bounds[4].min[fancy_names_index["YrotBG"], FIN] = -2.9 - .1  # debut bras aux oreilles
    x_bounds[4].max[fancy_names_index["YrotBG"], FIN] = -2.9 + .1
    x_bounds[4].min[fancy_names_index["ZrotBG"], FIN] = -.1
    x_bounds[4].max[fancy_names_index["ZrotBG"], FIN] = .1

    # coude droit
    x_bounds[4].min[fancy_names_index["ZrotABD"]:fancy_names_index["XrotABD"] + 1, FIN] = -.1
    x_bounds[4].max[fancy_names_index["ZrotABD"]:fancy_names_index["XrotABD"] + 1, FIN] = .1

    # coude gauche
    x_bounds[4].min[fancy_names_index["ZrotABG"]:fancy_names_index["XrotABG"] + 1, FIN] = -.1
    x_bounds[4].max[fancy_names_index["ZrotABG"]:fancy_names_index["XrotABG"] + 1, FIN] = .1

    # le carpe
    x_bounds[4].min[fancy_names_index["XrotC"], :] = -.4
    x_bounds[4].min[fancy_names_index["XrotC"], FIN] = -.60
    x_bounds[4].max[fancy_names_index["XrotC"], FIN] = -.40  # fin un peu carpe

    # le dehanchement
    x_bounds[4].min[fancy_names_index["YrotC"], FIN] = -.1
    x_bounds[4].max[fancy_names_index["YrotC"], FIN] = .1

    # Contraintes de vitesse: PHASE 4 la reception

    # en xy bassin
    x_bounds[4].min[fancy_names_index["vX"]:fancy_names_index["vY"] + 1, :] = -10
    x_bounds[4].max[fancy_names_index["vX"]:fancy_names_index["vY"] + 1, :] = 10

    # z bassin
    x_bounds[4].min[fancy_names_index["vZ"], :] = -100
    x_bounds[4].max[fancy_names_index["vZ"], :] = 100

    # autour de x
    x_bounds[4].min[fancy_names_index["vXrot"], :] = -100
    x_bounds[4].max[fancy_names_index["vXrot"], :] = 100

    # autour de y
    x_bounds[4].min[fancy_names_index["vYrot"], :] = -100
    x_bounds[4].max[fancy_names_index["vYrot"], :] = 100

    # autour de z
    x_bounds[4].min[fancy_names_index["vZrot"], :] = -100
    x_bounds[4].max[fancy_names_index["vZrot"], :] = 100

    # bras droit
    x_bounds[4].min[fancy_names_index["vZrotBD"]:fancy_names_index["vYrotBD"] + 1, :] = -100
    x_bounds[4].max[fancy_names_index["vZrotBD"]:fancy_names_index["vYrotBD"] + 1, :] = 100

    # bras droit
    x_bounds[4].min[fancy_names_index["vZrotBG"]:fancy_names_index["vYrotBG"] + 1, :] = -100
    x_bounds[4].max[fancy_names_index["vZrotBG"]:fancy_names_index["vYrotBG"] + 1, :] = 100

    # coude droit
    x_bounds[4].min[fancy_names_index["vZrotABD"]:fancy_names_index["vYrotABD"] + 1, :] = -100
    x_bounds[4].max[fancy_names_index["vZrotABD"]:fancy_names_index["vYrotABD"] + 1, :] = 100

    # coude gauche
    x_bounds[4].min[fancy_names_index["vZrotABD"]:fancy_names_index["vYrotABG"] + 1, :] = -100
    x_bounds[4].max[fancy_names_index["vZrotABD"]:fancy_names_index["vYrotABG"] + 1, :] = 100

    # du carpe
    x_bounds[4].min[fancy_names_index["vXrotC"], :] = -100
    x_bounds[4].max[fancy_names_index["vXrotC"], :] = 100

    # du dehanchement
    x_bounds[4].min[fancy_names_index["vYrotC"], :] = -100
    x_bounds[4].max[fancy_names_index["vYrotC"], :] = 100

    # deplacement
    x_bounds[5].min[fancy_names_index["X"], :] = -.1
    x_bounds[5].max[fancy_names_index["X"], :] = .1
    x_bounds[5].min[fancy_names_index["Y"], :] = -1.
    x_bounds[5].max[fancy_names_index["Y"], :] = 1.
    x_bounds[5].min[:fancy_names_index["Z"] + 1, DEBUT] = 0
    x_bounds[5].max[:fancy_names_index["Z"] + 1, DEBUT] = 0
    x_bounds[5].min[fancy_names_index["Z"], MILIEU:] = 0
    x_bounds[5].max[fancy_names_index["Z"],
    MILIEU:] = zmax  # beaucoup plus que necessaire, juste pour que la parabole fonctionne

    # le salto autour de x
    x_bounds[5].min[fancy_names_index["Xrot"], DEBUT] = .50  # penche vers l'avant un peu carpe
    x_bounds[5].max[fancy_names_index["Xrot"], DEBUT] = .50
    x_bounds[5].min[fancy_names_index["Xrot"], MILIEU:] = 0
    x_bounds[5].max[fancy_names_index["Xrot"], MILIEU:] = 4 * 3.14 + .1  # salto

    # limitation du tilt autour de y
    x_bounds[5].min[fancy_names_index["Yrot"], DEBUT] = 0
    x_bounds[5].max[fancy_names_index["Yrot"], DEBUT] = 0
    x_bounds[5].min[fancy_names_index["Yrot"], MILIEU:] = - 3.14 / 16  # vraiment pas suppose tilte
    x_bounds[5].max[fancy_names_index["Yrot"], MILIEU:] = 3.14 / 16

    # la vrille autour de z
    x_bounds[5].min[fancy_names_index["Zrot"], DEBUT] = 0
    x_bounds[5].max[fancy_names_index["Zrot"], DEBUT] = 0
    x_bounds[5].min[fancy_names_index["Zrot"], MILIEU:] = -.1  # pas de vrille dans cette phase
    x_bounds[5].max[fancy_names_index["Zrot"], MILIEU:] = .1

    # bras droit
    x_bounds[5].min[fancy_names_index["YrotBD"], DEBUT] = 2.9  # debut bras aux oreilles
    x_bounds[5].max[fancy_names_index["YrotBD"], DEBUT] = 2.9
    x_bounds[5].min[fancy_names_index["ZrotBD"], DEBUT] = 0
    x_bounds[5].max[fancy_names_index["ZrotBD"], DEBUT] = 0

    # bras gauche
    x_bounds[5].min[fancy_names_index["YrotBG"], DEBUT] = -2.9  # debut bras aux oreilles
    x_bounds[5].max[fancy_names_index["YrotBG"], DEBUT] = -2.9
    x_bounds[5].min[fancy_names_index["ZrotBG"], DEBUT] = 0
    x_bounds[5].max[fancy_names_index["ZrotBG"], DEBUT] = 0

    # coude droit
    x_bounds[5].min[fancy_names_index["ZrotABD"]:fancy_names_index["XrotABD"] + 1, DEBUT] = 0
    x_bounds[5].max[fancy_names_index["ZrotABD"]:fancy_names_index["XrotABD"] + 1, DEBUT] = 0

    # coude gauche
    x_bounds[5].min[fancy_names_index["ZrotABG"]:fancy_names_index["XrotABG"] + 1, DEBUT] = 0
    x_bounds[5].max[fancy_names_index["ZrotABG"]:fancy_names_index["XrotABG"] + 1, DEBUT] = 0

    # le carpe
    x_bounds[5].min[fancy_names_index["XrotC"], DEBUT] = -.50  # depart un peu ferme aux hanches
    x_bounds[5].max[fancy_names_index["XrotC"], DEBUT] = -.50
    x_bounds[5].min[fancy_names_index["XrotC"], FIN] = -2.35
    x_bounds[5].max[fancy_names_index["XrotC"], FIN] = -2.35

    # le dehanchement
    x_bounds[5].min[fancy_names_index["YrotC"], DEBUT] = 0
    x_bounds[5].max[fancy_names_index["YrotC"], DEBUT] = 0
    x_bounds[5].min[fancy_names_index["YrotC"], MILIEU:] = -.1
    x_bounds[5].max[fancy_names_index["YrotC"], MILIEU:] = .1

    # Contraintes de vitesse: PHASE 0 la montee en carpe

    vzinit = 9.81 / 2 * final_time  # vitesse initiale en z du CoM pour revenir a terre au temps final

    # decalage entre le bassin et le CoM
    # AUJO
    CoM_Q_sym = cas.MX.sym('CoM', nb_q)
    CoM_Q_init = x_bounds[5].min[:nb_q,
                 DEBUT]  # min ou max ne change rien a priori, au DEBUT ils sont egaux normalement
    CoM_Q_func = cas.Function('CoM_Q_func', [CoM_Q_sym], [biorbd_model[5].CoM(CoM_Q_sym).to_mx()])
    bassin_Q_func = cas.Function('bassin_Q_func', [CoM_Q_sym],
                                 [biorbd_model[5].globalJCS(0).to_mx()])  # retourne la RT du bassin

    r = np.array(CoM_Q_func(CoM_Q_init)).reshape(1, 3) - np.array(bassin_Q_func(CoM_Q_init))[-1,
                                                         :3]  # selectionne seulement la translation de la RT

    # JECH

    # en xy bassin
    x_bounds[5].min[fancy_names_index["vX"]:fancy_names_index["vY"] + 1, :] = -10
    x_bounds[5].max[fancy_names_index["vX"]:fancy_names_index["vY"] + 1, :] = 10
    x_bounds[5].min[fancy_names_index["vX"]:fancy_names_index["vY"] + 1, DEBUT] = -.5
    x_bounds[5].max[fancy_names_index["vX"]:fancy_names_index["vY"] + 1, DEBUT] = .5

    # z bassin
    x_bounds[5].min[fancy_names_index["vZ"], :] = -100
    x_bounds[5].max[fancy_names_index["vZ"], :] = 100
    x_bounds[5].min[fancy_names_index["vZ"], DEBUT] = vzinit - .5
    x_bounds[5].max[fancy_names_index["vZ"], DEBUT] = vzinit + .5

    # autour de x
    x_bounds[5].min[fancy_names_index["vXrot"], :] = .5  # d'apres une observation video
    x_bounds[5].max[fancy_names_index["vXrot"],
    :] = 20  # aussi vite que nécessaire, mais ne devrait pas atteindre cette vitesse

    # autour de y
    x_bounds[5].min[fancy_names_index["vYrot"], :] = -100
    x_bounds[5].max[fancy_names_index["vYrot"], :] = 100
    x_bounds[5].min[fancy_names_index["vYrot"], DEBUT] = 0
    x_bounds[5].max[fancy_names_index["vYrot"], DEBUT] = 0

    # autour de z
    x_bounds[5].min[fancy_names_index["vZrot"], :] = -100
    x_bounds[5].max[fancy_names_index["vZrot"], :] = 100
    x_bounds[5].min[fancy_names_index["vZrot"], DEBUT] = 0
    x_bounds[5].max[fancy_names_index["vZrot"], DEBUT] = 0

    # tenir compte du decalage entre bassin et CoM avec la rotation
    # Qtransdot = Qtransdot + v cross Qrotdot
    borne_inf = (x_bounds[5].min[fancy_names_index["vX"]:fancy_names_index["vZ"] + 1, DEBUT] + np.cross(r,
                                                                                                        x_bounds[5].min[
                                                                                                        fancy_names_index[
                                                                                                            "vXrot"]:
                                                                                                        fancy_names_index[
                                                                                                            "vZrot"] + 1,
                                                                                                        DEBUT]))[0]
    borne_sup = (x_bounds[5].max[fancy_names_index["vX"]:fancy_names_index["vZ"] + 1, DEBUT] + np.cross(r,
                                                                                                        x_bounds[5].max[
                                                                                                        fancy_names_index[
                                                                                                            "vXrot"]:
                                                                                                        fancy_names_index[
                                                                                                            "vZrot"] + 1,
                                                                                                        DEBUT]))[0]
    x_bounds[5].min[fancy_names_index["vX"]:fancy_names_index["vZ"] + 1, DEBUT] = min(borne_sup[0], borne_inf[0]), min(
        borne_sup[1], borne_inf[1]), min(borne_sup[2], borne_inf[2])
    x_bounds[5].max[fancy_names_index["vX"]:fancy_names_index["vZ"] + 1, DEBUT] = max(borne_sup[0], borne_inf[0]), max(
        borne_sup[1], borne_inf[1]), max(borne_sup[2], borne_inf[2])

    # bras droit
    x_bounds[5].min[fancy_names_index["vZrotBD"]:fancy_names_index["vYrotBD"] + 1, :] = -100
    x_bounds[5].max[fancy_names_index["vZrotBD"]:fancy_names_index["vYrotBD"] + 1, :] = 100
    x_bounds[5].min[fancy_names_index["vZrotBD"]:fancy_names_index["vYrotBD"] + 1, DEBUT] = 0
    x_bounds[5].max[fancy_names_index["vZrotBD"]:fancy_names_index["vYrotBD"] + 1, DEBUT] = 0

    # bras droit
    x_bounds[5].min[fancy_names_index["vZrotBG"]:fancy_names_index["vYrotBG"] + 1, :] = -100
    x_bounds[5].max[fancy_names_index["vZrotBG"]:fancy_names_index["vYrotBG"] + 1, :] = 100
    x_bounds[5].min[fancy_names_index["vZrotBG"]:fancy_names_index["vYrotBG"] + 1, DEBUT] = 0
    x_bounds[5].max[fancy_names_index["vZrotBG"]:fancy_names_index["vYrotBG"] + 1, DEBUT] = 0

    # coude droit
    x_bounds[5].min[fancy_names_index["vZrotABD"]:fancy_names_index["vYrotABD"] + 1, :] = -100
    x_bounds[5].max[fancy_names_index["vZrotABD"]:fancy_names_index["vYrotABD"] + 1, :] = 100
    x_bounds[5].min[fancy_names_index["vZrotABD"]:fancy_names_index["vYrotABD"] + 1, DEBUT] = 0
    x_bounds[5].max[fancy_names_index["vZrotABD"]:fancy_names_index["vYrotABD"] + 1, DEBUT] = 0

    # coude gauche
    x_bounds[5].min[fancy_names_index["vZrotABD"]:fancy_names_index["vYrotABG"] + 1, :] = -100
    x_bounds[5].max[fancy_names_index["vZrotABD"]:fancy_names_index["vYrotABG"] + 1, :] = 100
    x_bounds[5].min[fancy_names_index["vZrotABG"]:fancy_names_index["vYrotABG"] + 1, DEBUT] = 0
    x_bounds[5].max[fancy_names_index["vZrotABG"]:fancy_names_index["vYrotABG"] + 1, DEBUT] = 0

    # du carpe
    x_bounds[5].min[fancy_names_index["vXrotC"], :] = -100
    x_bounds[5].max[fancy_names_index["vXrotC"], :] = 100
    x_bounds[5].min[fancy_names_index["vXrotC"], DEBUT] = 0
    x_bounds[5].max[fancy_names_index["vXrotC"], DEBUT] = 0

    # du dehanchement
    x_bounds[5].min[fancy_names_index["vYrotC"], :] = -100
    x_bounds[5].max[fancy_names_index["vYrotC"], :] = 100
    x_bounds[5].min[fancy_names_index["vYrotC"], DEBUT] = 0
    x_bounds[5].max[fancy_names_index["vYrotC"], DEBUT] = 0

    #
    # Contraintes de position: PHASE 1 le salto carpe
    #

    # # deplacement
    # x_bounds[1].min[fancy_names_index["X"], :] = -.1
    # x_bounds[1].max[fancy_names_index["X"], :] = .1
    # x_bounds[1].min[fancy_names_index["Y"], :] = -1.
    # x_bounds[1].max[fancy_names_index["Y"], :] = 1.
    # x_bounds[1].min[fancy_names_index["Z"], :] = 0
    # x_bounds[1].max[fancy_names_index["Z"],
    # :] = zmax  # beaucoup plus que necessaire, juste pour que la parabole fonctionne
    #
    # # le salto autour de x
    # x_bounds[1].min[fancy_names_index["Xrot"], :] = 0
    # x_bounds[1].max[fancy_names_index["Xrot"], :] = 4 * 3.14
    # x_bounds[1].min[fancy_names_index["Xrot"], FIN] = 2 * 3.14 - .1
    #
    # # limitation du tilt autour de y
    # x_bounds[1].min[fancy_names_index["Yrot"], :] = - 3.14 / 16
    # x_bounds[1].max[fancy_names_index["Yrot"], :] = 3.14 / 16
    #
    # # la vrille autour de z
    # x_bounds[1].min[fancy_names_index["Zrot"], :] = -.1
    # x_bounds[1].max[fancy_names_index["Zrot"], :] = .1
    #
    # # bras f4a a l'ouverture
    #
    # # le carpe
    # x_bounds[1].min[fancy_names_index["XrotC"], :] = -2.35 - 0.1
    # x_bounds[1].max[fancy_names_index["XrotC"], :] = -2.35 + 0.1
    #
    # # le dehanchement
    # x_bounds[1].min[fancy_names_index["YrotC"], DEBUT] = -.1
    # x_bounds[1].max[fancy_names_index["YrotC"], DEBUT] = .1
    #
    # # Contraintes de vitesse: PHASE 1 le salto carpe
    #
    # # en xy bassin
    # x_bounds[1].min[fancy_names_index["vX"]:fancy_names_index["vY"] + 1, :] = -10
    # x_bounds[1].max[fancy_names_index["vX"]:fancy_names_index["vY"] + 1, :] = 10
    #
    # # z bassin
    # x_bounds[1].min[fancy_names_index["vZ"], :] = -100
    # x_bounds[1].max[fancy_names_index["vZ"], :] = 100
    #
    # # autour de x
    # x_bounds[1].min[fancy_names_index["vXrot"], :] = -100
    # x_bounds[1].max[fancy_names_index["vXrot"], :] = 100
    #
    # # autour de y
    # x_bounds[1].min[fancy_names_index["vYrot"], :] = -100
    # x_bounds[1].max[fancy_names_index["vYrot"], :] = 100
    #
    # # autour de z
    # x_bounds[1].min[fancy_names_index["vZrot"], :] = -100
    # x_bounds[1].max[fancy_names_index["vZrot"], :] = 100
    #
    # # bras droit
    # x_bounds[1].min[fancy_names_index["vZrotBD"]:fancy_names_index["vYrotBD"] + 1, :] = -100
    # x_bounds[1].max[fancy_names_index["vZrotBD"]:fancy_names_index["vYrotBD"] + 1, :] = 100
    #
    # # bras droit
    # x_bounds[1].min[fancy_names_index["vZrotBG"]:fancy_names_index["vYrotBG"] + 1, :] = -100
    # x_bounds[1].max[fancy_names_index["vZrotBG"]:fancy_names_index["vYrotBG"] + 1, :] = 100
    #
    # # coude droit
    # x_bounds[1].min[fancy_names_index["vZrotABD"]:fancy_names_index["vYrotABD"] + 1, :] = -100
    # x_bounds[1].max[fancy_names_index["vZrotABD"]:fancy_names_index["vYrotABD"] + 1, :] = 100
    # # coude gauche
    # x_bounds[1].min[fancy_names_index["vZrotABD"]:fancy_names_index["vYrotABG"] + 1, :] = -100
    # x_bounds[1].max[fancy_names_index["vZrotABD"]:fancy_names_index["vYrotABG"] + 1, :] = 100
    #
    # # du carpe
    # x_bounds[1].min[fancy_names_index["vXrotC"], :] = -100
    # x_bounds[1].max[fancy_names_index["vXrotC"], :] = 100
    #
    # # du dehanchement
    # x_bounds[1].min[fancy_names_index["vYrotC"], :] = -100
    # x_bounds[1].max[fancy_names_index["vYrotC"], :] = 100
    #
    # #
    # # Contraintes de position: PHASE 2 l'ouverture
    # #
    #
    # # deplacement
    # x_bounds[2].min[fancy_names_index["X"], :] = -.2
    # x_bounds[2].max[fancy_names_index["X"], :] = .2
    # x_bounds[2].min[fancy_names_index["Y"], :] = -1.
    # x_bounds[2].max[fancy_names_index["Y"], :] = 1.
    # x_bounds[2].min[fancy_names_index["Z"], :] = 0
    # x_bounds[2].max[fancy_names_index["Z"],
    # :] = zmax  # beaucoup plus que necessaire, juste pour que la parabole fonctionne
    #
    # # le salto autour de x
    # x_bounds[2].min[fancy_names_index["Xrot"], :] = 2 * 3.14 + .1  # 1 salto 3/4
    # x_bounds[2].max[fancy_names_index["Xrot"], :] = 4 * 3.14
    #
    # # limitation du tilt autour de y
    # x_bounds[2].min[fancy_names_index["Yrot"], :] = - 3.14 / 4
    # x_bounds[2].max[fancy_names_index["Yrot"], :] = 3.14 / 4
    #
    # # la vrille autour de z
    # x_bounds[2].min[fancy_names_index["Zrot"], :] = 0
    # x_bounds[2].max[fancy_names_index["Zrot"], :] = 3 * 3.14
    #
    # # bras f4a a l'ouverture
    #
    # # le carpe
    # x_bounds[2].min[fancy_names_index["XrotC"], FIN] = -.4
    #
    # # le dehanchement f4a a l'ouverture
    #
    # # Contraintes de vitesse: PHASE 2 l'ouverture
    #
    # # en xy bassin
    # x_bounds[2].min[fancy_names_index["vX"]:fancy_names_index["vY"] + 1, :] = -10
    # x_bounds[2].max[fancy_names_index["vX"]:fancy_names_index["vY"] + 1, :] = 10
    #
    # # z bassin
    # x_bounds[2].min[fancy_names_index["vZ"], :] = -100
    # x_bounds[2].max[fancy_names_index["vZ"], :] = 100
    #
    # # autour de x
    # x_bounds[2].min[fancy_names_index["vXrot"], :] = -100
    # x_bounds[2].max[fancy_names_index["vXrot"], :] = 100
    #
    # # autour de y
    # x_bounds[2].min[fancy_names_index["vYrot"], :] = -100
    # x_bounds[2].max[fancy_names_index["vYrot"], :] = 100
    #
    # # autour de z
    # x_bounds[2].min[fancy_names_index["vZrot"], :] = -100
    # x_bounds[2].max[fancy_names_index["vZrot"], :] = 100
    #
    # # bras droit
    # x_bounds[2].min[fancy_names_index["vZrotBD"]:fancy_names_index["vYrotBD"] + 1, :] = -100
    # x_bounds[2].max[fancy_names_index["vZrotBD"]:fancy_names_index["vYrotBD"] + 1, :] = 100
    #
    # # bras droit
    # x_bounds[2].min[fancy_names_index["vZrotBG"]:fancy_names_index["vYrotBG"] + 1, :] = -100
    # x_bounds[2].max[fancy_names_index["vZrotBG"]:fancy_names_index["vYrotBG"] + 1, :] = 100
    #
    # # coude droit
    # x_bounds[2].min[fancy_names_index["vZrotABD"]:fancy_names_index["vYrotABD"] + 1, :] = -100
    # x_bounds[2].max[fancy_names_index["vZrotABD"]:fancy_names_index["vYrotABD"] + 1, :] = 100
    #
    # # coude gauche
    # x_bounds[2].min[fancy_names_index["vZrotABD"]:fancy_names_index["vYrotABG"] + 1, :] = -100
    # x_bounds[2].max[fancy_names_index["vZrotABD"]:fancy_names_index["vYrotABG"] + 1, :] = 100
    #
    # # du carpe
    # x_bounds[2].min[fancy_names_index["vXrotC"], :] = -100
    # x_bounds[2].max[fancy_names_index["vXrotC"], :] = 100
    #
    # # du dehanchement
    # x_bounds[2].min[fancy_names_index["vYrotC"], :] = -100
    # x_bounds[2].max[fancy_names_index["vYrotC"], :] = 100
    #
    # #
    # # Contraintes de position: PHASE 3 la vrille et demie
    # #
    #
    # # deplacement
    # x_bounds[3].min[fancy_names_index["X"], :] = -.2
    # x_bounds[3].max[fancy_names_index["X"], :] = .2
    # x_bounds[3].min[fancy_names_index["Y"], :] = -1.
    # x_bounds[3].max[fancy_names_index["Y"], :] = 1.
    # x_bounds[3].min[fancy_names_index["Z"], :] = 0
    # x_bounds[3].max[fancy_names_index["Z"],
    # :] = zmax  # beaucoup plus que necessaire, juste pour que la parabole fonctionne
    #
    # # le salto autour de x
    # x_bounds[3].min[fancy_names_index["Xrot"], :] = 2 * 3.14 - .1
    # x_bounds[3].max[fancy_names_index["Xrot"], :] = 2 * 3.14 + 3 / 2 * 3.14 + .1  # 1 salto 3/4
    # x_bounds[3].min[fancy_names_index["Xrot"], FIN] = 2 * 3.14 + 3 / 2 * 3.14 - .1
    # x_bounds[3].max[fancy_names_index["Xrot"], FIN] = 2 * 3.14 + 3 / 2 * 3.14 + .1  # 1 salto 3/4
    #
    # # limitation du tilt autour de y
    # x_bounds[3].min[fancy_names_index["Yrot"], :] = - 3.14 / 4
    # x_bounds[3].max[fancy_names_index["Yrot"], :] = 3.14 / 4
    # x_bounds[3].min[fancy_names_index["Yrot"], FIN] = - 3.14 / 8
    # x_bounds[3].max[fancy_names_index["Yrot"], FIN] = 3.14 / 8
    #
    # # la vrille autour de z
    # x_bounds[3].min[fancy_names_index["Zrot"], :] = 0
    # x_bounds[3].max[fancy_names_index["Zrot"], :] = 3 * 3.14
    # x_bounds[3].min[fancy_names_index["Zrot"], FIN] = 3 * 3.14 - .1  # complete la vrille
    # x_bounds[3].max[fancy_names_index["Zrot"], FIN] = 3 * 3.14 + .1
    #
    # # bras f4a la vrille
    #
    # # le carpe
    # x_bounds[3].min[fancy_names_index["XrotC"], :] = -.4
    #
    # # le dehanchement f4a la vrille
    #
    # # Contraintes de vitesse: PHASE 3 la vrille et demie
    #
    # # en xy bassin
    # x_bounds[3].min[fancy_names_index["vX"]:fancy_names_index["vY"] + 1, :] = -10
    # x_bounds[3].max[fancy_names_index["vX"]:fancy_names_index["vY"] + 1, :] = 10
    #
    # # z bassin
    # x_bounds[3].min[fancy_names_index["vZ"], :] = -100
    # x_bounds[3].max[fancy_names_index["vZ"], :] = 100
    #
    # # autour de x
    # x_bounds[3].min[fancy_names_index["vXrot"], :] = -100
    # x_bounds[3].max[fancy_names_index["vXrot"], :] = 100
    #
    # # autour de y
    # x_bounds[3].min[fancy_names_index["vYrot"], :] = -100
    # x_bounds[3].max[fancy_names_index["vYrot"], :] = 100
    #
    # # autour de z
    # x_bounds[3].min[fancy_names_index["vZrot"], :] = -100
    # x_bounds[3].max[fancy_names_index["vZrot"], :] = 100
    #
    # # bras droit
    # x_bounds[3].min[fancy_names_index["vZrotBD"]:fancy_names_index["vYrotBD"] + 1, :] = -100
    # x_bounds[3].max[fancy_names_index["vZrotBD"]:fancy_names_index["vYrotBD"] + 1, :] = 100
    #
    # # bras droit
    # x_bounds[3].min[fancy_names_index["vZrotBG"]:fancy_names_index["vYrotBG"] + 1, :] = -100
    # x_bounds[3].max[fancy_names_index["vZrotBG"]:fancy_names_index["vYrotBG"] + 1, :] = 100
    #
    # # coude droit
    # x_bounds[3].min[fancy_names_index["vZrotABD"]:fancy_names_index["vYrotABD"] + 1, :] = -100
    # x_bounds[3].max[fancy_names_index["vZrotABD"]:fancy_names_index["vYrotABD"] + 1, :] = 100
    #
    # # coude gauche
    # x_bounds[3].min[fancy_names_index["vZrotABD"]:fancy_names_index["vYrotABG"] + 1, :] = -100
    # x_bounds[3].max[fancy_names_index["vZrotABD"]:fancy_names_index["vYrotABG"] + 1, :] = 100
    #
    # # du carpe
    # x_bounds[3].min[fancy_names_index["vXrotC"], :] = -100
    # x_bounds[3].max[fancy_names_index["vXrotC"], :] = 100
    #
    # # du dehanchement
    # x_bounds[3].min[fancy_names_index["vYrotC"], :] = -100
    # x_bounds[3].max[fancy_names_index["vYrotC"], :] = 100
    #
    # #
    # # Contraintes de position: PHASE 4 la reception
    # #
    #
    # # deplacement
    # x_bounds[4].min[fancy_names_index["X"], :] = -.1
    # x_bounds[4].max[fancy_names_index["X"], :] = .1
    # x_bounds[4].min[fancy_names_index["Y"], FIN] = -.1
    # x_bounds[4].max[fancy_names_index["Y"], FIN] = .1
    # x_bounds[4].min[fancy_names_index["Z"], :] = 0
    # x_bounds[4].max[fancy_names_index["Z"],
    # :] = zmax  # beaucoup plus que necessaire, juste pour que la parabole fonctionne
    # x_bounds[4].min[fancy_names_index["Z"], FIN] = 0
    # x_bounds[4].max[fancy_names_index["Z"], FIN] = .1
    #
    # # le salto autour de x
    # x_bounds[4].min[fancy_names_index["Xrot"], :] = 2 * 3.14 + 3 / 2 * 3.14 - .2  # penche vers avant -> moins de salto
    # x_bounds[4].max[fancy_names_index["Xrot"], :] = -.50 + 4 * 3.14  # un peu carpe a la fin
    # x_bounds[4].min[fancy_names_index["Xrot"], FIN] = -.50 + 4 * 3.14 - .1
    # x_bounds[4].max[fancy_names_index["Xrot"], FIN] = -.50 + 4 * 3.14 + .1  # 2 salto fin un peu carpe
    #
    # # limitation du tilt autour de y
    # x_bounds[4].min[fancy_names_index["Yrot"], :] = - 3.14 / 16
    # x_bounds[4].max[fancy_names_index["Yrot"], :] = 3.14 / 16
    #
    # # la vrille autour de z
    # x_bounds[4].min[fancy_names_index["Zrot"], :] = 3 * 3.14 - .1  # complete la vrille
    # x_bounds[4].max[fancy_names_index["Zrot"], :] = 3 * 3.14 + .1
    #
    # # bras droit
    # x_bounds[4].min[fancy_names_index["YrotBD"], FIN] = 2.9 - .1  # debut bras aux oreilles
    # x_bounds[4].max[fancy_names_index["YrotBD"], FIN] = 2.9 + .1
    # x_bounds[4].min[fancy_names_index["ZrotBD"], FIN] = -.1
    # x_bounds[4].max[fancy_names_index["ZrotBD"], FIN] = .1
    #
    # # bras gauche
    # x_bounds[4].min[fancy_names_index["YrotBG"], FIN] = -2.9 - .1  # debut bras aux oreilles
    # x_bounds[4].max[fancy_names_index["YrotBG"], FIN] = -2.9 + .1
    # x_bounds[4].min[fancy_names_index["ZrotBG"], FIN] = -.1
    # x_bounds[4].max[fancy_names_index["ZrotBG"], FIN] = .1
    #
    # # coude droit
    # x_bounds[4].min[fancy_names_index["ZrotABD"]:fancy_names_index["XrotABD"] + 1, FIN] = -.1
    # x_bounds[4].max[fancy_names_index["ZrotABD"]:fancy_names_index["XrotABD"] + 1, FIN] = .1
    #
    # # coude gauche
    # x_bounds[4].min[fancy_names_index["ZrotABG"]:fancy_names_index["XrotABG"] + 1, FIN] = -.1
    # x_bounds[4].max[fancy_names_index["ZrotABG"]:fancy_names_index["XrotABG"] + 1, FIN] = .1
    #
    # # le carpe
    # x_bounds[4].min[fancy_names_index["XrotC"], :] = -.4
    # x_bounds[4].min[fancy_names_index["XrotC"], FIN] = -.60
    # x_bounds[4].max[fancy_names_index["XrotC"], FIN] = -.40  # fin un peu carpe
    #
    # # le dehanchement
    # x_bounds[4].min[fancy_names_index["YrotC"], FIN] = -.1
    # x_bounds[4].max[fancy_names_index["YrotC"], FIN] = .1
    #
    # # Contraintes de vitesse: PHASE 4 la reception
    #
    # # en xy bassin
    # x_bounds[4].min[fancy_names_index["vX"]:fancy_names_index["vY"] + 1, :] = -10
    # x_bounds[4].max[fancy_names_index["vX"]:fancy_names_index["vY"] + 1, :] = 10
    #
    # # z bassin
    # x_bounds[4].min[fancy_names_index["vZ"], :] = -100
    # x_bounds[4].max[fancy_names_index["vZ"], :] = 100
    #
    # # autour de x
    # x_bounds[4].min[fancy_names_index["vXrot"], :] = -100
    # x_bounds[4].max[fancy_names_index["vXrot"], :] = 100
    #
    # # autour de y
    # x_bounds[4].min[fancy_names_index["vYrot"], :] = -100
    # x_bounds[4].max[fancy_names_index["vYrot"], :] = 100
    #
    # # autour de z
    # x_bounds[4].min[fancy_names_index["vZrot"], :] = -100
    # x_bounds[4].max[fancy_names_index["vZrot"], :] = 100
    #
    # # bras droit
    # x_bounds[4].min[fancy_names_index["vZrotBD"]:fancy_names_index["vYrotBD"] + 1, :] = -100
    # x_bounds[4].max[fancy_names_index["vZrotBD"]:fancy_names_index["vYrotBD"] + 1, :] = 100
    #
    # # bras droit
    # x_bounds[4].min[fancy_names_index["vZrotBG"]:fancy_names_index["vYrotBG"] + 1, :] = -100
    # x_bounds[4].max[fancy_names_index["vZrotBG"]:fancy_names_index["vYrotBG"] + 1, :] = 100
    #
    # # coude droit
    # x_bounds[4].min[fancy_names_index["vZrotABD"]:fancy_names_index["vYrotABD"] + 1, :] = -100
    # x_bounds[4].max[fancy_names_index["vZrotABD"]:fancy_names_index["vYrotABD"] + 1, :] = 100
    #
    # # coude gauche
    # x_bounds[4].min[fancy_names_index["vZrotABD"]:fancy_names_index["vYrotABG"] + 1, :] = -100
    # x_bounds[4].max[fancy_names_index["vZrotABD"]:fancy_names_index["vYrotABG"] + 1, :] = 100
    #
    # # du carpe
    # x_bounds[4].min[fancy_names_index["vXrotC"], :] = -100
    # x_bounds[4].max[fancy_names_index["vXrotC"], :] = 100
    #
    # # du dehanchement
    # x_bounds[4].min[fancy_names_index["vYrotC"], :] = -100
    # x_bounds[4].max[fancy_names_index["vYrotC"], :] = 100

    # x_bounds[5] = x_bounds[0]
    # x_bounds[6] = x_bounds[1]
    # x_bounds[7] = x_bounds[2]
    # x_bounds[8] = x_bounds[0]
    # x_bounds[9] = x_bounds[4]
    # x_bounds[5].min[:] = x_bounds[0].min[:]
    # x_bounds[5].max[:] = x_bounds[0].max[:]
    x_bounds[6].min[:] = x_bounds[1].min[:]
    x_bounds[6].max[:] = x_bounds[1].max[:]
    x_bounds[7].min[:] = x_bounds[2].min[:]
    x_bounds[7].max[:] = x_bounds[2].max[:]
    x_bounds[8].min[:] = x_bounds[3].min[:]
    x_bounds[8].max[:] = x_bounds[3].max[:]
    x_bounds[9].min[:] = x_bounds[4].min[:]
    x_bounds[9].max[:] = x_bounds[4].max[:]

    return x_bounds

def set_x_init(biorbd_model, fancy_names_index):
    index_roots_x = [0,1,2,3,4,5]
    nb_q = biorbd_model[0].nbQ()
    nb_qdot = biorbd_model[0].nbQdot()
    l = len(index_roots_x)
    x0 = np.vstack((np.zeros((nb_q, 2)), np.zeros((nb_qdot, 2))))
    x1 = np.vstack((np.zeros((nb_q, 2)), np.zeros((nb_qdot, 2))))
    x2 = np.vstack((np.zeros((nb_q, 2)), np.zeros((nb_qdot, 2))))
    x3 = np.vstack((np.zeros((nb_q, 2)), np.zeros((nb_qdot, 2))))
    x4 = np.vstack((np.zeros((nb_q, 2)), np.zeros((nb_qdot, 2))))
    x5 = np.vstack((np.zeros((l, 2)), np.zeros((l, 2))))
    x6 = np.vstack((np.zeros((l, 2)), np.zeros((l, 2))))
    x7 = np.vstack((np.zeros((l, 2)), np.zeros((l, 2))))
    x8 = np.vstack((np.zeros((l, 2)), np.zeros((l, 2))))
    x9 = np.vstack((np.zeros((l, 2)), np.zeros((l, 2))))

    x0[fancy_names_index["Xrot"], 0] = .50
    x0[fancy_names_index["ZrotBG"]] = -.75
    x0[fancy_names_index["ZrotBD"]] = .75
    x0[fancy_names_index["YrotBG"], 0] = -2.9
    x0[fancy_names_index["YrotBD"], 0] = 2.9
    x0[fancy_names_index["YrotBG"], 1] = -1.35
    x0[fancy_names_index["YrotBD"], 1] = 1.35
    x0[fancy_names_index["XrotC"], 0] = -.5
    x0[fancy_names_index["XrotC"], 1] = -2.6

    x1[fancy_names_index["ZrotBG"]] = -.75
    x1[fancy_names_index["ZrotBD"]] = .75
    x1[fancy_names_index["Xrot"], 1] = 2 * 3.14
    x1[fancy_names_index["YrotBG"]] = -1.35
    x1[fancy_names_index["YrotBD"]] = 1.35
    x1[fancy_names_index["XrotC"]] = -2.6

    x2[fancy_names_index["Xrot"]] = 2 * 3.14
    x2[fancy_names_index["Zrot"], 1] = 3.14
    x2[fancy_names_index["ZrotBG"], 0] = -.75
    x2[fancy_names_index["ZrotBD"], 0] = .75
    x2[fancy_names_index["YrotBG"], 0] = -1.35
    x2[fancy_names_index["YrotBD"], 0] = 1.35
    x2[fancy_names_index["XrotC"], 0] = -2.6

    x3[fancy_names_index["Xrot"], 0] = 2 * 3.14
    x3[fancy_names_index["Xrot"], 1] = 2 * 3.14 + 3/2 * 3.14
    x3[fancy_names_index["Zrot"], 0] = 3.14
    x3[fancy_names_index["Zrot"], 1] = 3 * 3.14

    x4[fancy_names_index["Xrot"], 0] = 2 * 3.14 + 3/2 * 3.14
    x4[fancy_names_index["Xrot"], 1] = 4 * 3.14
    x4[fancy_names_index["Zrot"]] = 3 * 3.14
    x4[fancy_names_index["XrotC"], 1] = -.5

    # for i in range(32) :
    #     x5[i,:] = x0[i,:]
    #     x6[i,:] = x1[i,:]
    #     x7[i,:] = x2[i,:]
    #     x8[i,:] = x3[i,:]
    #     x9[i,:] = x4[i,:]
    for i in index_roots_x:
        x5[i,:] = x0[i,:]
        x6[i,:] = x1[i,:]
        x7[i,:] = x2[i,:]
        x8[i,:] = x3[i,:]
        x9[i,:] = x4[i,:]
    for i in np.array(index_roots_x): #qdot
        x5[i, :] = x0[i+nb_q, :]
        x6[i, :] = x1[i+nb_q, :]
        x7[i, :] = x2[i+nb_q, :]
        x8[i, :] = x3[i+nb_q, :]
        x9[i, :] = x4[i+nb_q, :]

    x_init = InitialGuessList()
    x_init.add(x0, interpolation=InterpolationType.LINEAR)
    x_init.add(x1, interpolation=InterpolationType.LINEAR)
    x_init.add(x2, interpolation=InterpolationType.LINEAR)
    x_init.add(x3, interpolation=InterpolationType.LINEAR)
    x_init.add(x4, interpolation=InterpolationType.LINEAR)
    x_init.add(x5, interpolation=InterpolationType.LINEAR)
    x_init.add(x6, interpolation=InterpolationType.LINEAR)
    x_init.add(x7, interpolation=InterpolationType.LINEAR)
    x_init.add(x8, interpolation=InterpolationType.LINEAR)
    x_init.add(x9, interpolation=InterpolationType.LINEAR)

    return x_init


# def root_explicit_dynamic(
#     states: Union[cas.MX, cas.SX],
#     controls: Union[cas.MX, cas.SX],
#     parameters: Union[cas.MX, cas.SX],
#     nlp: NonLinearProgram,
# ) -> tuple:
#
#     DynamicsFunctions.apply_parameters(parameters, nlp)
#     q = DynamicsFunctions.get(nlp.states["q"], states)
#     qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
#     nb_root = nlp.model.nbRoot()
#
#     qddot_joints = DynamicsFunctions.get(nlp.controls["qddot_joints"], controls)
#
#     mass_matrix_nl_effects = nlp.model.InverseDynamics(
#         q, qdot, cas.vertcat(cas.MX.zeros((nb_root, 1)), qddot_joints)
#     ).to_mx()[:nb_root]
#
#     mass_matrix = nlp.model.massMatrix(q).to_mx()
#     mass_matrix_nl_effects_func = Function(
#         "mass_matrix_nl_effects_func", [q, qdot, qddot_joints], [mass_matrix_nl_effects[:nb_root]]
#     ).expand()
#
#     M_66 = mass_matrix[:nb_root, :nb_root]
#     M_66_func = Function("M66_func", [q], [M_66]).expand()
#
#     qddot_root = solve(-M_66_func(q), mass_matrix_nl_effects_func(q, qdot, qddot_joints), "ldl")
#
#     return cas.vertcat(qdot, cas.vertcat(qddot_root, qddot_joints))


# def custom_configure_root_explicit(ocp: OptimalControlProgram, nlp: NonLinearProgram):
#     ConfigureProblem.configure_q(nlp, as_states=True, as_controls=False)
#     ConfigureProblem.configure_qdot(nlp, as_states=True, as_controls=False)
#     configure_qddot_joint(nlp, as_states=False, as_controls=True)
#     ConfigureProblem.configure_dynamics_function(ocp, nlp, root_explicit_dynamic, expand=False)


# def configure_qddot_joint(nlp, as_states: bool, as_controls: bool):
#     nb_root = nlp.model.nbRoot()
#     name_qddot_joint = [str(i + nb_root) for i in range(nlp.model.nbQddot() - nb_root)]
#     ConfigureProblem.configure_new_variable("qddot_joints", name_qddot_joint, nlp, as_states, as_controls)


def prepare_ocp(
        biorbd_model_path_AuJo: str,
        biorbd_model_path_JeCh: str,
        n_shooting: int,
        final_time: float,
        n_threads: int,
        ode_solver: OdeSolver = OdeSolver.RK4()
) -> OptimalControlProgram:
    """
    Prepare the ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the bioMod file
    n_shooting: int
        The number of shooting points
    final_time: float
        The time at the final node
    ode_solver: OdeSolver
        The ode solver to use

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    biorbd_model = ( biorbd.Model(biorbd_model_path_AuJo),
                     biorbd.Model(biorbd_model_path_AuJo),
                     biorbd.Model(biorbd_model_path_AuJo),
                     biorbd.Model(biorbd_model_path_AuJo),
                     biorbd.Model(biorbd_model_path_AuJo),
                     biorbd.Model(biorbd_model_path_JeCh),
                     biorbd.Model(biorbd_model_path_JeCh),
                     biorbd.Model(biorbd_model_path_JeCh),
                     biorbd.Model(biorbd_model_path_JeCh),
                     biorbd.Model(biorbd_model_path_JeCh),
                     )

    nb_q = biorbd_model[0].nbQ()
    nb_qdot = biorbd_model[0].nbQdot()
    nb_qddot_joints = nb_q - biorbd_model[0].nbRoot()

    fancy_names_index = set_fancy_names_index(nb_q)

    # Phase mapping
    # controls
    node_mappings = NodeMappingList()
    node_mappings.add("qddot_joints", map_controls=True, phase_pre=0, phase_post=5)
    node_mappings.add("qddot_joints", map_controls=True, phase_pre=1, phase_post=6)
    node_mappings.add("qddot_joints", map_controls=True, phase_pre=2, phase_post=7)
    node_mappings.add("qddot_joints", map_controls=True, phase_pre=3, phase_post=8)
    node_mappings.add("qddot_joints", map_controls=True, phase_pre=4, phase_post=9)
    # states

    node_mappings.add("q", map_states=True, phase_pre=0, phase_post=5, index= [0,1,2,3,4,5])
    node_mappings.add("qdot", map_states=True, phase_pre=0, phase_post=5, index= [16,17,18,19,20,21])
    node_mappings.add("q", map_states=True, phase_pre=1, phase_post=6, index= [0,1,2,3,4,5])
    node_mappings.add("qdot", map_states=True, phase_pre=1, phase_post=6, index= [16,17,18,19,20,21])
    node_mappings.add("q", map_states=True, phase_pre=2, phase_post=7, index = [0,1,2,3,4,5])
    node_mappings.add("qdot", map_states=True, phase_pre=2, phase_post=7, index = [16,17,18,19,20,21])
    node_mappings.add("q", map_states=True, phase_pre=3, phase_post=8, index =[0,1,2,3,4,5])
    node_mappings.add("qdot", map_states=True, phase_pre=3, phase_post=8, index =[16,17,18,19,20,21])
    node_mappings.add("q", map_states=True, phase_pre=4, phase_post=9, index =[0,1,2,3,4,5])
    node_mappings.add("qdot", map_states=True, phase_pre=4, phase_post=9, index =[16,17,18,19,20,21])

    #node_mappings.add("qdot", map_states=True, phase_pre=0, phase_post=5)
   # node_mappings.add("qdot", map_states=True, phase_pre=1, phase_post=6)
   #node_mappings.add("qdot", map_states=True, phase_pre=2, phase_post=7)
    #node_mappings.add("qdot", map_states=True, phase_pre=3, phase_post=8)
    #node_mappings.add("qdot", map_states=True, phase_pre=4, phase_post=9)




    # Add objective functions
    objective_functions = ObjectiveList()
    # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_MARKERS, marker_index=1, weight=-1)
    ## AuJo
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qddot_joints", node=Node.ALL_SHOOTING, weight=1, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qddot_joints", node=Node.ALL_SHOOTING, weight=1, phase=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qddot_joints", node=Node.ALL_SHOOTING, weight=1, phase=2)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qddot_joints", node=Node.ALL_SHOOTING, weight=1, phase=3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qddot_joints", node=Node.ALL_SHOOTING, weight=1, phase=4)
    ## JeCh
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qddot_joints", node=Node.ALL_SHOOTING, weight=1, phase=5)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qddot_joints", node=Node.ALL_SHOOTING, weight=1, phase=6)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qddot_joints", node=Node.ALL_SHOOTING, weight=1, phase=7)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qddot_joints", node=Node.ALL_SHOOTING, weight=1, phase=8)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qddot_joints", node=Node.ALL_SHOOTING, weight=1, phase=9)

    ## AuJo
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, min_bound=.0, max_bound=final_time, weight=100000, phase=0)
    ## JeCh
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, min_bound=.0, max_bound=final_time, weight=100000, phase=5)

    # Les hanches sont fixes a +-0.2 en bounds, mais les mains doivent quand meme être proches des jambes
    ## AuJo
    objective_functions.add(ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS, node=Node.END, first_marker='MidMainG', second_marker='CibleMainG', weight=1000, phase=0)
    objective_functions.add(ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS, node=Node.END, first_marker='MidMainD', second_marker='CibleMainD', weight=1000, phase=0)
    # JeCh
    objective_functions.add(ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS, node=Node.END, first_marker='MidMainG', second_marker='CibleMainG', weight=1000, phase=5)
    objective_functions.add(ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS, node=Node.END, first_marker='MidMainD', second_marker='CibleMainD', weight=1000, phase=5)


    # arrete de gigoter les bras
    les_bras = [fancy_names_index["ZrotBD"], fancy_names_index["YrotBD"], fancy_names_index["ZrotABD"],
                fancy_names_index["XrotABD"], fancy_names_index["ZrotBG"], fancy_names_index["YrotBG"],
                fancy_names_index["ZrotABG"], fancy_names_index["XrotABG"]]
    les_coudes = [fancy_names_index["ZrotABD"], fancy_names_index["XrotABD"],
                  fancy_names_index["ZrotABG"], fancy_names_index["XrotABG"]]
    ## AuJo
    objective_functions.add(minimize_dofs, custom_type=ObjectiveFcn.Lagrange, node=Node.ALL_SHOOTING, dofs=les_coudes, targets=np.zeros(len(les_coudes)), weight=10000, phase=0)
    objective_functions.add(minimize_dofs, custom_type=ObjectiveFcn.Lagrange, node=Node.ALL_SHOOTING, dofs=les_bras, targets=np.zeros(len(les_bras)), weight=10000, phase=2)
    objective_functions.add(minimize_dofs, custom_type=ObjectiveFcn.Lagrange, node=Node.ALL_SHOOTING, dofs=les_bras, targets=np.zeros(len(les_bras)), weight=10000, phase=3)
    objective_functions.add(minimize_dofs, custom_type=ObjectiveFcn.Lagrange, node=Node.ALL_SHOOTING, dofs=les_coudes, targets=np.zeros(len(les_coudes)), weight=10000, phase=4)
    ##JeCh
    objective_functions.add(minimize_dofs, custom_type=ObjectiveFcn.Lagrange, node=Node.ALL_SHOOTING, dofs=les_coudes, targets=np.zeros(len(les_coudes)), weight=10000, phase=5)
    objective_functions.add(minimize_dofs, custom_type=ObjectiveFcn.Lagrange, node=Node.ALL_SHOOTING, dofs=les_bras, targets=np.zeros(len(les_bras)), weight=10000, phase=7)
    objective_functions.add(minimize_dofs, custom_type=ObjectiveFcn.Lagrange, node=Node.ALL_SHOOTING, dofs=les_bras, targets=np.zeros(len(les_bras)), weight=10000, phase=8)
    objective_functions.add(minimize_dofs, custom_type=ObjectiveFcn.Lagrange, node=Node.ALL_SHOOTING, dofs=les_coudes, targets=np.zeros(len(les_coudes)), weight=10000, phase=9)
    #argparse
    #parser = argparse.ArgumentParser()
    #parser.add_argument("model", type=str, help="the bioMod file")
   # parser.add_argument("--no-hsl", dest='with_hsl', action='store_false', help="do not use libhsl")
    #parser.add_argument("-j", default=1, dest='n_threads', type=int, help="number of threads in the solver")
   # parser.add_argument("--no-sol", action='store_false', dest='savesol', help="do not save the solution")
   # parser.add_argument("--no-show-online", action='store_false', dest='show_online', help="do not show graphs during optimization")
   # parser.add_argument("--print-ocp", action='store_true', dest='print_ocp', help="print the ocp")
    #args = parser.parse_args()
    # ouvre les hanches rapidement apres la vrille
    ## AuJo
    objective_functions.add(minimize_dofs, custom_type=ObjectiveFcn.Mayer, node=Node.END, dofs=[fancy_names_index["XrotC"]], targets=[0, 0], weight=10000, phase=3)
    ## JeCh
    objective_functions.add(minimize_dofs, custom_type=ObjectiveFcn.Mayer, node=Node.END, dofs=[fancy_names_index["XrotC"]], targets=[0, 0], weight=10000, phase=8)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.JOINTS_ACCELERATION_DRIVEN)
    dynamics.add(DynamicsFcn.JOINTS_ACCELERATION_DRIVEN)
    dynamics.add(DynamicsFcn.JOINTS_ACCELERATION_DRIVEN)
    dynamics.add(DynamicsFcn.JOINTS_ACCELERATION_DRIVEN)
    dynamics.add(DynamicsFcn.JOINTS_ACCELERATION_DRIVEN)
    dynamics.add(DynamicsFcn.JOINTS_ACCELERATION_DRIVEN)
    dynamics.add(DynamicsFcn.JOINTS_ACCELERATION_DRIVEN)
    dynamics.add(DynamicsFcn.JOINTS_ACCELERATION_DRIVEN)
    dynamics.add(DynamicsFcn.JOINTS_ACCELERATION_DRIVEN)
    dynamics.add(DynamicsFcn.JOINTS_ACCELERATION_DRIVEN)

    x_bounds = set_x_bounds(biorbd_model, fancy_names_index, final_time)

    qddot_joints_min, qddot_joints_max, qddot_joints_init = -500, 500, 0
    u_bounds = BoundsList()
    u_bounds.add([qddot_joints_min] * nb_qddot_joints, [qddot_joints_max] * nb_qddot_joints)
    u_bounds.add([qddot_joints_min] * nb_qddot_joints, [qddot_joints_max] * nb_qddot_joints)
    u_bounds.add([qddot_joints_min] * nb_qddot_joints, [qddot_joints_max] * nb_qddot_joints)
    u_bounds.add([qddot_joints_min] * nb_qddot_joints, [qddot_joints_max] * nb_qddot_joints)
    u_bounds.add([qddot_joints_min] * nb_qddot_joints, [qddot_joints_max] * nb_qddot_joints)
    u_bounds.add([qddot_joints_min] * nb_qddot_joints, [qddot_joints_max] * nb_qddot_joints)
    u_bounds.add([qddot_joints_min] * nb_qddot_joints, [qddot_joints_max] * nb_qddot_joints)
    u_bounds.add([qddot_joints_min] * nb_qddot_joints, [qddot_joints_max] * nb_qddot_joints)
    u_bounds.add([qddot_joints_min] * nb_qddot_joints, [qddot_joints_max] * nb_qddot_joints)
    u_bounds.add([qddot_joints_min] * nb_qddot_joints, [qddot_joints_max] * nb_qddot_joints)

    u_init = InitialGuessList()
    u_init.add([qddot_joints_init] * nb_qddot_joints)
    u_init.add([qddot_joints_init] * nb_qddot_joints)
    u_init.add([qddot_joints_init] * nb_qddot_joints)
    u_init.add([qddot_joints_init] * nb_qddot_joints)
    u_init.add([qddot_joints_init] * nb_qddot_joints)
   # u_init.add([qddot_joints_init] * nb_qddot_joints)
   # u_init.add([qddot_joints_init] * nb_qddot_joints)
    #u_init.add([qddot_joints_init] * nb_qddot_joints)
   # u_init.add([qddot_joints_init] * nb_qddot_joints)
    #u_init.add([qddot_joints_init] * nb_qddot_joints)

    x_init = set_x_init(biorbd_model, fancy_names_index)

    constraints = ConstraintList()
#    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=0, max_bound=final_time, phase=0)
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=1e-4, max_bound=final_time, phase=1)
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=1e-4, max_bound=final_time, phase=2)
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=1e-4, max_bound=final_time, phase=3)
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=1e-4, max_bound=final_time, phase=4)
   # constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=1e-4, max_bound=final_time, phase=5)
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=1e-4, max_bound=final_time, phase=6)
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=1e-4, max_bound=final_time, phase=7)
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=1e-4, max_bound=final_time, phase=8)
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=1e-4, max_bound=final_time, phase=9)

    phase_transitions = PhaseTransitionList()
    phase_transitions.add(PhaseTransitionFcn.CONTINUOUS, phase_pre_idx=0)  # 0-1
    phase_transitions.add(PhaseTransitionFcn.CONTINUOUS, phase_pre_idx=1)  # 1-2
    phase_transitions.add(PhaseTransitionFcn.CONTINUOUS, phase_pre_idx=2)  # 2-3
    phase_transitions.add(PhaseTransitionFcn.CONTINUOUS, phase_pre_idx=3)  # 3-4
    phase_transitions.add(
        PhaseTransitionFcn.DISCONTINUOUS,
        phase_pre_idx=4,
    )  # 4-5
    phase_transitions.add(PhaseTransitionFcn.CONTINUOUS, phase_pre_idx=5)  # 5-6
    phase_transitions.add(PhaseTransitionFcn.CONTINUOUS, phase_pre_idx=6)  # 6-7
    phase_transitions.add(PhaseTransitionFcn.CONTINUOUS, phase_pre_idx=7)  # 7-8
    phase_transitions.add(PhaseTransitionFcn.CONTINUOUS, phase_pre_idx=8)  # 8-9

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        [final_time/len(biorbd_model)] * len(biorbd_model),
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        constraints,
        ode_solver=ode_solver,
        n_threads=n_threads,
        node_mappings=node_mappings,
        phase_transitions=phase_transitions,
    )


def main():

    model_path_AuJo = "Models/AuJo_TechOpt83.bioMod"
    model_path_JeCh = "Models/JeCh_TechOpt83.bioMod"
    n_threads = 4
    print_ocp_FLAG = False  # True.

    show_online_FLAG = False  # True
    HSL_FLAG = True
    save_sol_FLAG = True
    # n_shooting = (40, 100, 100, 100, 40,
    #               40, 100, 100, 100, 40)
    n_shooting = (4, 10, 10, 10, 4,
                  4, 10, 10, 10, 4)

    ocp = prepare_ocp(model_path_AuJo,model_path_JeCh, n_shooting=n_shooting, n_threads=n_threads, final_time=1.87*2)
    #ocp.add_plot_penalty(CostType.ALL)
    if print_ocp_FLAG:
        ocp.print(to_graph=True)
    solver = Solver.IPOPT(show_online_optim=show_online_FLAG, show_options=dict(show_bounds=True))
    if HSL_FLAG:
         solver.set_linear_solver('ma57')
    else:
        print("Not using ma57")
    solver.set_maximum_iterations(10)
    solver.set_convergence_tolerance(1e-4)
    sol = ocp.solve(solver)

    temps = time.strftime("%Y-%m-%d-%H%M")
    nom = model_path_AuJo.split('/')[-1].removesuffix('.bioMod')
    qs = sol.states[0]['q']
    qdots = sol.states[0]['qdot']

    for i in range(1, len(sol.states)):
        qs = np.hstack((qs, sol.states[i]['q']))
        qdots = np.hstack((qdots, sol.states[i]['qdot']))
    if save_sol_FLAG:  # switch manuelle
        np.save(f"Solutions/{nom}-{str(n_shooting).replace(', ', '_')}-{temps}-q.npy", qs)
        np.save(f"Solutions/{nom}-{str(n_shooting).replace(', ', '_')}-{temps}-qdot.npy", qdots)
        np.save(f"Solutions/{nom}-{str(n_shooting).replace(', ', '_')}-{temps}-t.npy", sol.phase_time)

    if IPYTHON:
        IPython.embed()  # afin de pouvoir explorer plus en details la solution

    # Print the last solution
    #sol.animate(n_frames=-1, show_floor=False)

    # sol.graphs(show_bounds=True)

if __name__ == "__main__":
    main()
    #main()


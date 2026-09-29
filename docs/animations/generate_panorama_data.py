"""
Panorama of the penalty library: the names of ``ObjectiveFcn.Lagrange``, ``ObjectiveFcn.Mayer`` and ``ConstraintFcn``
are read from the enums of THIS bioptim version (no solver involved) and grouped by what they act on. Stored in
``data/panorama_library.npz``: for each family, the names, the group of each name and the penalty function each name
points to (several names can be aliases of the same function, e.g. MINIMIZE_CONTROL and TRACK_CONTROL).

Usage, from the repo root:  PYTHONPATH=. python docs/animations/generate_panorama_data.py
"""

import re
from pathlib import Path

import numpy as np
from bioptim import ConstraintFcn, ConstraintList, Node, ObjectiveFcn, ObjectiveList

OUT = Path(__file__).parent / "data"

# first matching rule wins
GROUPS = [
    ("stochastic", r"STOCHASTIC|SYMMETRIC_MATRIX|SEMIDEFINITE"),
    ("continuity", r"CONTINUITY|FIRST_COLLOCATION"),
    ("time", r"TIME"),
    ("markers", r"MARKER"),
    ("segments", r"SEGMENT"),
    ("center of mass", r"COM_|_COM|MOMENTUM"),
    ("contacts / forces", r"CONTACT|REACTION_FORCES|CENTER_OF_PRESSURE|NON_SLIPPING"),
    ("power / energy", r"POWER|FATIGUE|TORQUE_MAX"),
    ("controls", r"CONTROL|QDDOT"),
    ("states", r"STATE"),
]
OTHER = "other"
FAMILIES = {
    "Lagrange": ObjectiveFcn.Lagrange,
    "Mayer": ObjectiveFcn.Mayer,
    "Constraint": ConstraintFcn,
}


def group_of(name: str) -> str:
    for group, pattern in GROUPS:
        if re.search(pattern, name):
            return group
    return OTHER


def check_code_lines():
    """Build the exact lines shown in the animation (they raise if a name or an argument is wrong)."""
    objectives = ObjectiveList()
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1e-2)
    objectives.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="qdot", node=Node.END, weight=100)
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="hand", second_marker="target")
    print(
        "code lines OK, nodes before the OCP resolves them:",
        [o.node for ph in objectives for o in ph],
        [c.node for ph in constraints for c in ph],
    )


def main():
    check_code_lines()
    out = {"group_names": np.array([g for g, _ in GROUPS] + [OTHER])}
    for fam, enum in FAMILIES.items():
        names = sorted(enum.__members__.keys())  # __members__ also lists the aliases
        canonical = [enum.__members__[n].name for n in names]  # first name of the (aliased) member
        out[f"{fam}_names"] = np.array(names)
        out[f"{fam}_groups"] = np.array([group_of(n) for n in names])
        out[f"{fam}_canonical"] = np.array(canonical)
        print(f"{fam}: {len(names)} names, {len(set(canonical))} distinct functions")
        for g in out["group_names"]:
            sel = [n for n, gg in zip(names, out[f"{fam}_groups"]) if gg == g]
            print(f"   {g:20s} {len(sel):3d}  {sel if g == OTHER else ''}")
    np.savez(OUT / "panorama_library.npz", **out)


if __name__ == "__main__":
    main()

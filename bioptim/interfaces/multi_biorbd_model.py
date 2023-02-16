from typing import Any

import biorbd_casadi as biorbd
from casadi import MX, vertcat

from ..misc.mapping import BiMapping, BiMappingList
from ..misc.utils import check_version
from ..limits.path_conditions import Bounds

check_version(biorbd, "1.9.9", "1.10.0")


class MultiBiorbdModel:
    def __init__(self, bio_models: tuple[str | biorbd.Model, ...]):
        self.models = []
        for bio_model in bio_models:
            if isinstance(bio_model, str):
                self.models.append(biorbd.Model(bio_model))
            elif isinstance(bio_model, biorbd.Model):
                self.models.append(bio_model)
            else:
                raise RuntimeError("Type must be a 'str' or a 'biorbd.Model'")

    def serialize(self):
        return "Can't reload"

    @property
    def nb_tau(self) -> int:
        return sum(model.nbGeneralizedTorque() for model in self.models)

    @property
    def nb_q(self) -> int:
        return sum(model.nbQ() for model in self.models)

    @property
    def nb_qdot(self) -> int:
        return sum(model.nbQdot() for model in self.models)

    @property
    def nb_qddot(self) -> int:
        return sum(model.nbQddot() for model in self.models)

    @property
    def nb_root(self) -> int:
        return self.models[0].nbRoot()

    def center_of_mass(self, q) -> MX:
        return vertcat(*(model.CoM(q, True).to_mx() for model in self.models))

    @property
    def name_dof(self) -> tuple[str, ...]:
        out = []
        for model in self.models:
            for s in model.nameDof():
                out.append(s.to_string())
        return tuple(out)

    def forward_dynamics(self, q, qdot, tau, external_forces=None, f_contacts=None) -> MX:
        if external_forces is not None:
            raise RuntimeError("Coucou")
        return vertcat(*(model.ForwardDynamics(q, qdot, tau, external_forces, f_contacts).to_mx() for model in self.models))

    def inverse_dynamics(self, q, qdot, tau, external_forces=None, f_contacts=None) -> MX:
        return vertcat(*(model.InverseDynamics(q, qdot, tau, external_forces, f_contacts).to_mx() for model in self.models))

    def markers(self, q) -> Any | list[MX]:
        out = []
        for model in self.models:
            for m in model.markers(q):
                out.append(m.to_mx())
        return out

    @property
    def nb_markers(self) -> int:
        return sum(model.nbMarkers() for model in self.models)

    @property
    def nb_dof(self) -> int:
        return sum(model.nbDof() for model in self.models)

    def reshape_qdot(self, q, qdot, k_stab=1) -> MX:
        return vertcat(*(model.computeQdot(q, qdot, k_stab).to_mx() for model in self.models))

    @staticmethod
    def _q_mapping(model, mapping: BiMapping = None) -> BiMapping:
        if mapping is None:
            mapping = {}
        if "q" not in mapping:
            mapping["q"] = BiMapping(range(model.nbQ()), range(model.nbQ()))
        return mapping

    @staticmethod
    def _qdot_mapping(model, mapping: BiMapping = None) -> BiMapping:
        if mapping is None:
            mapping = {}
        if "qdot" not in mapping:
            if "q" not in mapping:
                mapping["q"] = BiMapping(range(model.nbQ()), range(model.nbQ()))
            mapping["qdot"] = mapping["q"]
        return mapping

    def bounds_from_ranges(self, variables: str | list[str, ...], mapping: BiMapping | BiMappingList = None) -> Bounds:
        out = Bounds()
        q_ranges = []
        qdot_ranges = []

        for model in self.models:
            for i in range(model.nbSegment()):
                segment = model.segment(i)
                for var in variables:
                    if var == "q":
                        q_ranges += [q_range for q_range in segment.QRanges()]

            for var in variables:
                if var == "q":
                    q_mapping = self._q_mapping(model, mapping)
                    mapping = q_mapping
                    x_min = [q_ranges[i].min() for i in q_mapping["q"].to_first.map_idx]
                    x_max = [q_ranges[i].max() for i in q_mapping["q"].to_first.map_idx]
                    out.concatenate(Bounds(min_bound=x_min, max_bound=x_max))

        for model in self.models:
            for i in range(model.nbSegment()):
                segment = model.segment(i)
                for var in variables:
                    if var == "qdot":
                        qdot_ranges += [qdot_range for qdot_range in segment.QDotRanges()]

            for var in variables:
                if var == "qdot":
                    qdot_mapping = self._qdot_mapping(model, mapping)
                    mapping = qdot_mapping
                    x_min = [qdot_ranges[i].min() for i in qdot_mapping["qdot"].to_first.map_idx]
                    x_max = [qdot_ranges[i].max() for i in qdot_mapping["qdot"].to_first.map_idx]
                    out.concatenate(Bounds(min_bound=x_min, max_bound=x_max))

        if out.shape[0] == 0:
            raise ValueError(f"Unrecognized variable ({variables}), only 'q', 'qdot' and 'qddot' are allowed")

        return out

    # To complete from here
    @property
    def nb_soft_contacts(self):
        return 0

    def soft_contact_forces(self, q, qdot):
        return MX()

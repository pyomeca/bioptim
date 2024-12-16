from typing import Callable

from casadi import SX, MX, vertcat, horzcat, norm_fro, Function
import l4casadi as l4c
import numpy as np
import torch

# """
# INSTALLATION:
# First, make sure pytorch is installed

#     pip install torch>=2.0 --index-url https://download.pytorch.org/whl/cpu/torch_stable.html
# setuptools>=68.1
# scikit-build>=0.17
# cmake>=3.27
# ninja>=1.11
# Then, install l4casadi as the interface between CasADi and PyTorch

#     pip install l4casadi --no-build-isolation

# """


class TorchModel:
    """
    This class wraps a pytorch model and allows the user to call some useful functions on it.
    """

    def __init__(self, torch_model: torch.nn.Module):
        self._dynamic_model = l4c.L4CasADi(torch_model, device="cpu")  # device='cuda' for GPU

        self._nb_dof = torch_model.size_in // 3
        self._symbolic_variables()

    def _symbolic_variables(self):
        """Declaration of MX variables of the right shape for the creation of CasADi Functions"""
        self.q = MX.sym("q_mx", self.nb_dof, 1)
        self.qdot = MX.sym("qdot_mx", self.nb_dof, 1)
        self.tau = MX.sym("tau_mx", self.nb_dof, 1)
        self.external_forces = MX.sym("external_forces_mx", 0, 1)
        self.parameters = MX.sym("parameters_mx", 0, 1)

    @property
    def name(self) -> str:
        # parse the path and split to get the .bioMod name
        return "forward_dynamics_torch_model"

    @property
    def name_dof(self) -> list[str]:
        return [f"q_{i}" for i in range(self.nb_dof)]

    @property
    def nb_dof(self) -> int:
        return self._nb_dof

    @property
    def nb_q(self) -> int:
        return self.nb_dof

    @property
    def nb_qdot(self) -> int:
        return self.nb_dof

    @property
    def nb_tau(self) -> int:
        return self.nb_dof

    def forward_dynamics(self, with_contact: bool = False) -> Function:
        return Function(
            "forward_dynamics",
            [self.q, self.qdot, self.tau, self.external_forces, self.parameters],
            [self._dynamic_model(vertcat(self.q, self.qdot, self.tau).T).T],
            ["q", "qdot", "tau", "external_forces", "parameters"],
            ["qddot"],
        ).expand()

    @property
    def nb_contacts(self) -> int:
        return 0

    @property
    def nb_soft_contacts(self) -> int:
        return 0

    def reshape_qdot(self, k_stab=1) -> Function:
        return Function(
            "reshape_qdot",
            [self.q, self.qdot, self.parameters],
            [self.qdot],
            ["q", "qdot", "parameters"],
            ["Reshaped qdot"],
        ).expand()

    def soft_contact_forces(self) -> Function:
        return Function(
            "soft_contact_forces",
            [self.q, self.qdot, self.parameters],
            [MX(0)],
            ["q", "qdot", "parameters"],
            ["Soft contact forces"],
        ).expand()

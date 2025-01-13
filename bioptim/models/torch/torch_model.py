"""
This wrapper can be used to wrap a PyTorch model and use it in bioptim. This is a much incomplete class though as compared
to the BiorbdModel class. The main reason is that as opposed to Biorbd, the dynamics produced by a PyTorch model can be
of any nature. This means this wrapper be more viewed as an example of how to wrap a PyTorch model in bioptim than an actual
generic wrapper.

This wrapper is based on the l4casadi library (https://github.com/Tim-Salzmann/l4casadi) which is a bridge between CasADi 
and PyTorch. 

INSTALLATION:
Note these instructions may be outdated. Please refer to the l4casadi documentation for the most up-to-date instructions.

First, make sure pytorch is installed by running the following command:
    pip install torch>=2.0
Please note that some depencecies are required. At the time of writing, the following packages were required:
    pip install setuptools>=68.1 scikit-build>=0.17 cmake>=3.27 ninja>=1.11
Then, install l4casadi as the interface between CasADi and PyTorch, by running the following command:
    pip install l4casadi --no-build-isolation

    
LIMITATIONS:
Since pytorch is wrapped using L4casadi, the casadi functions are generated using External. This means SX variables and
expanding functions are not supported. This will be computationally intensive when solving, making such approach rather slow when
compared to a programmatic approach. Still, it uses much less memory than the symbolic approach, so it has its own advantages.



KNOWN ISSUES: 
    On Windows (and possibly other platforms), you may randomly get the following error when running this example:
        ```python
        OMP: Error #15: Initializing libomp.dll, but found libiomp5md.dll already initialized.
        ```
    This error comes from the fact that installing the libraries copies the libiomp5md.dll file at two different locations.
    When it tries to load them, it notices that "another" library is already loaded. If you are 100% sure that both libraries
    are the exact same version, you can safely ignore this error. To do so, you can add the following lines at the beginning
    of your script:
        ```python
        import os
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        ```
    That being said, this can be very problematic if the two libraries are not the exact same version. The safer approach is
    to delete one of the file. To do so, simply navigate to the folders where the files libiomp5md.dll are located and delete it 
    (or back-it up by adding ".bak" to the name) keeping only one (keeping the one in site-packages seems to work fine).
"""

from casadi import MX, vertcat, Function
import l4casadi as l4c
import torch


class TorchModel:
    """
    This class wraps a pytorch model and allows the user to call some useful functions on it.
    """

    def __init__(self, torch_model: torch.nn.Module, device="cuda" if torch.cuda.is_available() else "cpu"):
        self._dynamic_model = l4c.L4CasADi(
            torch_model, device=device, generate_jac_jac=True, generate_adj1=False, generate_jac_adj1=False
        )

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
        if with_contact:
            raise NotImplementedError("Contact dynamics are not implemented for torch models")

        return Function(
            "forward_dynamics",
            [self.q, self.qdot, self.tau, self.external_forces, self.parameters],
            [self._dynamic_model(vertcat(self.q, self.qdot, self.tau).T).T],
            ["q", "qdot", "tau", "external_forces", "parameters"],
            ["qddot"],
        )

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

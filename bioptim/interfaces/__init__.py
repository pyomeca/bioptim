from .ipopt_options import IPOPT
from .acados_options import ACADOS
from .sqp_options import SQP_METHOD
from .casadi_function_interface import CasadiFunctionInterface


class Solver:
    IPOPT = IPOPT
    SQP_METHOD = SQP_METHOD
    ACADOS = ACADOS

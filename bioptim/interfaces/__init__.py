from .ipopt_options import IPOPT
from .fratrop_options import FATROP
from .acados_options import ACADOS
from .sqp_options import SQP_METHOD
from .casadi_function_interface import CasadiFunctionInterface


class Solver:
    IPOPT = IPOPT
    FATROP = FATROP
    SQP_METHOD = SQP_METHOD
    ACADOS = ACADOS

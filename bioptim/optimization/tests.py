

import casadi as cas
import numpy as np

# Q = cas.MX.sym('Q', 10, 10)
Q_1 = cas.MX.sym('Q', 5)
Q_2 = cas.MX.sym('Q', 5)
Q = cas.vertcat(Q_1, Q_2)

func = cas.Function('test', [Q], [Q**2])
print(func(np.ones((10,))))


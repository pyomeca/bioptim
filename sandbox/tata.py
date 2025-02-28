import biorbd
import numpy as np

model = biorbd.Model("/home/pariterre/Bureau/Model_Sujet04.txt")

q = np.ones((model.nbQ(),))
qdot = np.ones((model.nbQ(),))
calc = np.array([segment.to_array() for segment in model.CalcSegmentsAngularMomentum(q, qdot, True)])
print(calc)

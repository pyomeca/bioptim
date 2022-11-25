"""
This script implements a custom model to work with bioptim. Bioptim has a deep connection with biorbd,
but it is possible to use bioptim without biorbd. This is an example of how to use bioptim with a custom model.
"""
import numpy as np
from casadi import sin, MX

from bioptim import (
    Model,
)


class MyModel(Model):
    """
    This is a custom model that inherits from bioptim.Model
    As Model is an abstract class, we need to implement all the following methods,
    otherwise it will raise an error
    """

    def __init__(self):
        self.com = MX(np.array([-0.0005, 0.0688, -0.9542]))
        self.inertia = MX(0.0391)

    def getGravity(self):
        raise -9.81

    def nbGeneralizedTorque(self):
        return 1

    def nbSegment(self):
        return 1

    def nbQuat(self):
        return 0

    def nbQ(self):
        return 1

    def nbQdot(self):
        return 1

    def nbQddot(self):
        return 1

    def nbRoot(self):
        return 0

    def mass(self):
        return 1

    def massMatrix(self, Q, updateKin=True):
        return self.mass() * self.com[2] ** 2

    def nameDof(self):
        return ["rotx"]

    def nbRigidContacts(self):
        return 0

    def path(self):
        return None

    def ForwardDynamics(self, q, qdot, tau, fext=None, f_contacts=None):
        d = 0  # damping
        L = self.com[2]
        I  = self.inertia
        m = self.mass()
        g = 9.81
        return 1/(I + m * L ** 2) \
               * (- qdot[0] * d - g * m * L * sin(q[0]) + tau[0])

    def InverseDynamics(self, q, qdot, qddot, f_ext=None, f_contacts=None):
        return self.mass() * self.com[2] ** 2 * qddot[0] + self.mass() * -9.81 * self.com[2] * sin(q[0])

    def computeQdot(self, Q, QDot, k_stab=1):
        raise NotImplementedError("computeQdot is not implemented")

    def DeepCopy(self, *args):
        raise NotImplementedError("DeepCopy is not implemented")

    def AddSegment(self, *args):
        raise NotImplementedError("AddSegment is not implemented")

    def setGravity(self, newGravity):
        raise NotImplementedError("setGravity is not implemented")

    def getBodyBiorbdId(self, segmentName):
        raise NotImplementedError("getBodyBiorbdId is not implemented")

    def getBodyRbdlId(self, segmentName):
        raise NotImplementedError("getBodyRbdlId is not implemented")

    def getBodyRbdlIdToBiorbdId(self, idx):
        raise NotImplementedError("getBodyRbdlIdToBiorbdId is not implemented")

    def getBodyBiorbdIdToRbdlId(self, idx):
        raise NotImplementedError("getBodyBiorbdIdToRbdlId is not implemented")

    def getDofSubTrees(self):
        raise NotImplementedError("getDofSubTrees is not implemented")

    def getDofIndex(self, SegmentName, dofName):
        raise NotImplementedError("getDofIndex is not implemented")

    def updateSegmentCharacteristics(self, idx, characteristics):
        raise NotImplementedError("updateSegmentCharacteristics is not implemented")

    def segment(self, *args):
        raise NotImplementedError("segment is not implemented")

    def segments(self, i):
        raise NotImplementedError("segments is not implemented")

    def dispatchedForce(self, *args):
        raise NotImplementedError("dispatchedForce is not implemented")

    def UpdateKinematicsCustom(self, Q=None, Qdot=None, Qddot=None):
        raise NotImplementedError("UpdateKinematicsCustom is not implemented")

    def allGlobalJCS(self, *args):
        raise NotImplementedError("allGlobalJCS is not implemented")

    def globalJCS(self, *args):
        raise NotImplementedError("globalJCS is not implemented")

    def localJCS(self, *args):
        raise NotImplementedError("localJCS is not implemented")

    def projectPoint(self, *args):
        raise NotImplementedError("projectPoint is not implemented")

    def projectPointJacobian(self, *args):
        raise NotImplementedError("projectPointJacobian is not implemented")

    def CoM(self, Q, updateKin=True):
        raise NotImplementedError("CoM is not implemented")

    def CoMbySegmentInMatrix(self, Q, updateKin=True):
        raise NotImplementedError("CoMbySegmentInMatrix is not implemented")

    def CoMbySegment(self, *args):
        raise NotImplementedError("CoMbySegment is not implemented")

    def CoMdot(self, Q, Qdot, updateKin=True):
        raise NotImplementedError("CoMdot is not implemented")

    def CoMddot(self, Q, Qdot, Qddot, updateKin=True):
        raise NotImplementedError("CoMddot is not implemented")

    def CoMdotBySegment(self, *args):
        raise NotImplementedError("CoMdotBySegment is not implemented")

    def CoMddotBySegment(self, *args):
        raise NotImplementedError("CoMddotBySegment is not implemented")

    def CoMJacobian(self, Q, updateKin=True):
        raise NotImplementedError("CoMJacobian is not implemented")

    def meshPoints(self, *args):
        raise NotImplementedError("meshPoints is not implemented")

    def meshPointsInMatrix(self, Q, updateKin=True):
        raise NotImplementedError("meshPointsInMatrix is not implemented")

    def meshFaces(self, *args):
        raise NotImplementedError("meshFaces is not implemented")

    def mesh(self, *args):
        raise NotImplementedError("mesh is not implemented")

    def angularMomentum(self, Q, Qdot, updateKin=True):
        raise NotImplementedError("angularMomentum is not implemented")

    def massMatrixInverse(self, Q, updateKin=True):
        raise NotImplementedError("massMatrixInverse is not implemented")

    def CalcAngularMomentum(self, *args):
        raise NotImplementedError("CalcAngularMomentum is not implemented")

    def CalcSegmentsAngularMomentum(self, *args):
        raise NotImplementedError("CalcSegmentsAngularMomentum is not implemented")

    def bodyAngularVelocity(self, Q, Qdot, updateKin=True):
        raise NotImplementedError("bodyAngularVelocity is not implemented")

    def CalcMatRotJacobian(self, Q, segmentIdx, rotation, G, updateKin):
        raise NotImplementedError("CalcMatRotJacobian is not implemented")

    def JacobianSegmentRotMat(self, Q, segmentIdx, updateKin):
        raise NotImplementedError("JacobianSegmentRotMat is not implemented")

    def segmentAngularVelocity(self, Q, Qdot, idx, updateKin=True):
        raise NotImplementedError("segmentAngularVelocity is not implemented")

    def CalcKineticEnergy(self, Q, QDot, updateKin=True):
        raise NotImplementedError("CalcKineticEnergy is not implemented")

    def CalcPotentialEnergy(self, Q, updateKin=True):
        raise NotImplementedError("CalcPotentialEnergy is not implemented")

    def contactNames(self):
        raise NotImplementedError("contactNames is not implemented")

    def nbSoftContacts(self):
        return 0

    def softContactNames(self):
        raise NotImplementedError("softContactNames is not implemented")

    def muscleNames(self):
        raise NotImplementedError("muscleNames is not implemented")

    def torque(self, tau_activations, q, qdot):
        raise NotImplementedError("torque is not implemented")

    def ForwardDynamicsFreeFloatingBase(self, q, qdot, qddot_joints):
        raise NotImplementedError("ForwardDynamicsFreeFloatingBase is not implemented")

    def ForwardDynamicsConstraintsDirect(self, *args):
        raise NotImplementedError("ForwardDynamicsConstraintsDirect is not implemented")

    def NonLinearEffect(self, Q, QDot, f_ext=None, f_contacts=None):
        raise NotImplementedError("NonLinearEffect is not implemented")

    def ContactForcesFromForwardDynamicsConstraintsDirect(self, Q, QDot, Tau, f_ext=None):
        raise NotImplementedError("ContactForcesFromForwardDynamicsConstraintsDirect is not implemented")

    def bodyInertia(self, Q, updateKin=True):
        raise NotImplementedError("bodyInertia is not implemented")

    def ComputeConstraintImpulsesDirect(self, Q, QDotPre):
        raise NotImplementedError("ComputeConstraintImpulsesDirect is not implemented")

    def checkGeneralizedDimensions(self, Q=None, Qdot=None, Qddot=None, torque=None):
        raise NotImplementedError("checkGeneralizedDimensions is not implemented")

    def stateSet(self):
        raise NotImplementedError("stateSet is not implemented")

    def activationDot(self, muscle_states):
        raise NotImplementedError("activationDot is not implemented")

    def muscularJointTorque(self, muscle_states, q, qdot):
        raise NotImplementedError("muscularJointTorque is not implemented")

    def getConstraints(self):
        raise NotImplementedError("markers is not implemented")

    def markers(self, Q, updateKin=True):
        raise NotImplementedError("markers is not implemented")


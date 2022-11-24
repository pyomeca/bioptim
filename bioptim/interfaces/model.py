import biorbd_casadi as biorbd
from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def DeepCopy(self, *args):
        """Deep copy of the model"""

    @abstractmethod
    def AddSegment(self, *args):
        """Add a segment to the model"""

    @abstractmethod
    def getGravity(self):
        """Get the gravity vector"""

    @abstractmethod
    def setGravity(self, newGravity):
        """Set the gravity vector"""

    @abstractmethod
    def getBodyBiorbdId(self, segmentName):
        """Get the biorbd id of a body"""

    @abstractmethod
    def getBodyRbdlId(self, segmentName):
        """Get the rbdl id of a body"""

    @abstractmethod
    def getBodyRbdlIdToBiorbdId(self, idx):
        """Get the biorbd id of a body from its rbdl id"""

    @abstractmethod
    def getBodyBiorbdIdToRbdlId(self, idx):
        """Get the rbdl id of a body from its biorbd id"""

    @abstractmethod
    def getDofSubTrees(self):
        """Get the dof sub trees"""

    @abstractmethod
    def getDofIndex(self, SegmentName, dofName):
        """Get the dof index"""

    @abstractmethod
    def nbGeneralizedTorque(self):
        """Get the number of generalized torque"""

    @abstractmethod
    def nbSegment(self):
        """Get the number of segment"""

    @abstractmethod
    def nbQuat(self):
        """Get the number of quaternion"""

    @abstractmethod
    def nbQ(self):
        """Get the number of Q"""

    @abstractmethod
    def nbQdot(self):
        """Get the number of Qdot"""

    @abstractmethod
    def nbQddot(self):
        """Get the number of Qddot"""

    @abstractmethod
    def nbRoot(self):
        """Get the number of root Dof"""

    @abstractmethod
    def updateSegmentCharacteristics(self, idx, characteristics):
        """Update the segment characteristics"""

    @abstractmethod
    def segment(self, *args):
        """Get a segment"""

    @abstractmethod
    def segments(self, i):
        """Get a segment"""

    @abstractmethod
    def dispatchedForce(self, *args):
        """Get the dispatched force"""

    @abstractmethod
    def UpdateKinematicsCustom(self, Q=None, Qdot=None, Qddot=None):
        """Update the kinematics of the model"""

    @abstractmethod
    def allGlobalJCS(self, *args):
        """Get all the Rototranslation matrix"""

    @abstractmethod
    def globalJCS(self, *args):
        """Get the Rototranslation matrix"""

    @abstractmethod
    def localJCS(self, *args):
        """Get the Rototranslation matrix"""

    @abstractmethod
    def projectPoint(self, *args):
        """Project a point on the segment"""

    @abstractmethod
    def projectPointJacobian(self, *args):
        """Project a point on the segment"""

    @abstractmethod
    def mass(self):
        """Get the mass of the model"""

    @abstractmethod
    def CoM(self, Q, updateKin=True):
        """Get the center of mass of the model"""

    @abstractmethod
    def CoMbySegmentInMatrix(self, Q, updateKin=True):
        """Get the center of mass of the model"""

    @abstractmethod
    def CoMbySegment(self, *args):
        """Get the center of mass of the model"""

    @abstractmethod
    def CoMdot(self, Q, Qdot, updateKin=True):
        """Get the center of mass velocity of the model"""

    @abstractmethod
    def CoMddot(self, Q, Qdot, Qddot, updateKin=True):
        """Get the center of mass acceleration of the model"""

    @abstractmethod
    def CoMdotBySegment(self, *args):
        """Get the center of mass velocity of the model"""

    @abstractmethod
    def CoMddotBySegment(self, *args):
        """Get the center of mass acceleration of the model"""

    @abstractmethod
    def CoMJacobian(self, Q, updateKin=True):
        """Get the center of mass Jacobian of the model"""

    @abstractmethod
    def meshPoints(self, *args):
        """Get the mesh points of the model"""

    @abstractmethod
    def meshPointsInMatrix(self, Q, updateKin=True):
        """Get the mesh points of the model"""

    @abstractmethod
    def meshFaces(self, *args):
        """Get the mesh faces of the model"""

    @abstractmethod
    def mesh(self, *args):
        """Get the mesh of the model"""

    @abstractmethod
    def angularMomentum(self, Q, Qdot, updateKin=True):
        """Get the angular momentum of the model"""

    @abstractmethod
    def massMatrix(self, Q, updateKin=True):
        """Get the mass matrix of the model"""

    @abstractmethod
    def massMatrixInverse(self, Q, updateKin=True):
        """Get the inverse of the mass matrix of the model"""

    @abstractmethod
    def CalcAngularMomentum(self, *args):
        """Get the angular momentum of the model"""

    @abstractmethod
    def CalcSegmentsAngularMomentum(self, *args):
        """Get the angular momentum of the model"""

    @abstractmethod
    def bodyAngularVelocity(self, Q, Qdot, updateKin=True):
        """Get the body angular velocity of the model"""

    @abstractmethod
    def CalcMatRotJacobian(self, Q, segmentIdx, rotation, G, updateKin):
        """Get the body angular velocity of the model"""

    @abstractmethod
    def JacobianSegmentRotMat(self, Q, segmentIdx, updateKin):
        """Get the body angular velocity of the model"""

    @abstractmethod
    def computeQdot(self, Q, QDot, k_stab=1):
        """Get the body angular velocity of the model"""

    @abstractmethod
    def segmentAngularVelocity(self, Q, Qdot, idx, updateKin=True):
        """Get the body angular velocity of the model"""

    @abstractmethod
    def CalcKineticEnergy(self, Q, QDot, updateKin=True):
        """Get the kinetic energy of the model"""

    @abstractmethod
    def CalcPotentialEnergy(self, Q, updateKin=True):
        """Get the potential energy of the model"""

    @abstractmethod
    def nameDof(self):
        """Get the name of the dof"""

    @abstractmethod
    def contactNames(self):
        """Get the contact names"""

    @abstractmethod
    def nbSoftContacts(self):
        """Get the number of soft contacts"""

    @abstractmethod
    def softContactNames(self):
        """Get the soft contact names"""

    @abstractmethod
    def muscleNames(self):
        """Get the muscle names"""

    @abstractmethod
    def torque(self, tau_activations, q, qdot):
        """Get the muscle torque"""

    @abstractmethod
    def ForwardDynamicsFreeFloatingBase(self, q, qdot, qddot_joints):
        """compute the free floating base forward dynamics"""

    @abstractmethod
    def ForwardDynamics(self, q, qdot, tau, fext=None, f_contacts=None):
        """compute the forward dynamics"""

    @abstractmethod
    def ForwardDynamicsConstraintsDirect(self, *args):
        """compute the forward dynamics with constraints"""

    @abstractmethod
    def InverseDynamics(self, q, qdot, qddot, f_ext=None, f_contacts=None):
        """compute the inverse dynamics"""

    @abstractmethod
    def NonLinearEffect(self, Q, QDot, f_ext=None, f_contacts=None):
        """compute the non linear effect"""

    @abstractmethod
    def ContactForcesFromForwardDynamicsConstraintsDirect(self, Q, QDot, Tau, f_ext=None):
        """compute the contact forces"""

    @abstractmethod
    def bodyInertia(self, Q, updateKin=True):
        """Get the inertia of the model"""

    @abstractmethod
    def ComputeConstraintImpulsesDirect(self, Q, QDotPre):
        """compute the constraint impulses"""

    @abstractmethod
    def checkGeneralizedDimensions(self, Q=None, Qdot=None, Qddot=None, torque=None):
        """check the dimensions of the generalized coordinates"""

    @abstractmethod
    def stateSet(self):
        """Get the state set of the model"""

    @abstractmethod
    def activationDot(self, muscle_states):
        """Get the activation derivative"""

    @abstractmethod
    def muscularJointTorque(self, muscle_states, q, qdot):
        """Get the muscular joint torque"""

    @abstractmethod
    def getConstraints(self):
        """Get the constraints of the model"""

    @abstractmethod
    def markers(self, Q, updateKin=True):
        """Get the markers of the model"""

    @abstractmethod
    def nbRigidContacts(self):
        """Get the number of rigid contacts"""

    @abstractmethod
    def path(self):
        """Get the path of the model"""


class BiorbdModel(Model):
    def __init__(self, biorbd_model: str | biorbd.Model):
        if isinstance(biorbd_model, str):
            self.model = biorbd.Model(biorbd_model)
        else:
            self.model = biorbd_model

    def DeepCopy(self, *args):
        return self.model.DeepCopy(*args)

    def AddSegment(self, *args):
        return self.model.AddSegment(self, *args)

    def getGravity(self):
        return self.model.getGravity()

    def setGravity(self, newGravity):
        return self.model.setGravity(newGravity)

    def getBodyBiorbdId(self, segmentName):
        return self.model.getBodyBiorbdId(segmentName)

    def getBodyRbdlId(self, segmentName):
        return self.model.getBodyRbdlId(segmentName)

    def getBodyRbdlIdToBiorbdId(self, idx):
        return self.model.getBodyRbdlIdToBiorbdId(idx)

    def getBodyBiorbdIdToRbdlId(self, idx):
        return self.model.getBodyBiorbdIdToRbdlId(idx)

    def getDofSubTrees(self):
        return self.model.getDofSubTrees()

    def getDofIndex(self, SegmentName, dofName):
        return self.model.getDofIndex(SegmentName, dofName)

    def nbGeneralizedTorque(self):
        return self.model.nbGeneralizedTorque()

    def nbSegment(self):
        return self.model.nbSegment()

    def nbQuat(self):
        return self.model.nbQuat()

    def nbQ(self):
        return self.model.nbQ()

    def nbQdot(self):
        return self.model.nbQdot()

    def nbQddot(self):
        return self.model.nbQddot()

    def nbRoot(self):
        return self.model.nbRoot()

    def updateSegmentCharacteristics(self, idx, characteristics):
        return self.model.updateSegmentCharacteristics(idx, characteristics)

    def segment(self, *args):
        return self.model.segment(*args)

    def segments(self, i):
        return self.model.segments()

    def dispatchedForce(self, *args):
        return self.model.dispatchedForce(*args)

    def UpdateKinematicsCustom(self, Q=None, Qdot=None, Qddot=None):
        return self.model.UpdateKinematicsCustom(Q, Qdot, Qddot)

    def allGlobalJCS(self, *args):
        return self.model.allGlobalJCS(*args)

    def globalJCS(self, *args):
        return self.model.globalJCS(*args)

    def localJCS(self, *args):
        return self.model.localJCS(*args)

    def projectPoint(self, *args):
        return self.model.projectPoint(*args)

    def projectPointJacobian(self, *args):
        return self.model.projectPointJacobian(*args)

    def mass(self):
        return self.model.mass()

    def CoM(self, Q, updateKin=True):
        return self.model.CoM(Q, updateKin)

    def CoMbySegmentInMatrix(self, Q, updateKin=True):
        return self.model.CoMbySegmentInMatrix(Q, updateKin)

    def CoMbySegment(self, *args):
        return self.model.CoMbySegment(*args)

    def CoMdot(self, Q, Qdot, updateKin=True):
        return self.model.CoMdot(Q, Qdot, updateKin)

    def CoMddot(self, Q, Qdot, Qddot, updateKin=True):
        return self.model.CoMddot(Q, Qdot, Qddot, updateKin)

    def CoMdotBySegment(self, *args):
        return self.model.CoMdotBySegment(*args)

    def CoMddotBySegment(self, *args):
        return self.model.CoMddotBySegment(*args)

    def CoMJacobian(self, Q, updateKin=True):
        return self.model.CoMJacobian(Q, updateKin)

    def meshPoints(self, *args):
        return self.model.meshPoints(*args)

    def meshPointsInMatrix(self, Q, updateKin=True):
        return self.model.meshPointsInMatrix(Q, updateKin)

    def meshFaces(self, *args):
        return self.model.meshFaces(*args)

    def mesh(self, *args):
        return self.model.mesh(*args)

    def angularMomentum(self, Q, Qdot, updateKin=True):
        return self.model.angularMomentum(Q, Qdot, updateKin)

    def massMatrix(self, Q, updateKin=True):
        return self.model.massMatrix(Q, updateKin)

    def massMatrixInverse(self, Q, updateKin=True):
        return self.model.massMatrixInverse(Q, updateKin)

    def CalcAngularMomentum(self, *args):
        return self.model.CalcAngularMomentum(*args)

    def CalcSegmentsAngularMomentum(self, *args):
        return self.model.CalcSegmentsAngularMomentum(*args)

    def bodyAngularVelocity(self, Q, Qdot, updateKin=True):
        return self.model.bodyAngularVelocity(Q, Qdot, updateKin)

    def CalcMatRotJacobian(self, Q, segmentIdx, rotation, G, updateKin):
        return self.model.CalcMatRotJacobian(Q, segmentIdx, rotation, G, updateKin)

    def JacobianSegmentRotMat(self, Q, segmentIdx, updateKin):
        return self.model.JacobianSegmentRotMat(Q, segmentIdx, updateKin)

    def computeQdot(self, Q, QDot, k_stab=1):
        return self.model.computeQdot(Q, QDot, k_stab)

    def segmentAngularVelocity(self, Q, Qdot, idx, updateKin=True):
        return self.model.segmentAngularVelocity(Q, Qdot, idx, updateKin)

    def CalcKineticEnergy(self, Q, QDot, updateKin=True):
        return self.model.CalcKineticEnergy(Q, QDot, updateKin)

    def CalcPotentialEnergy(self, Q, updateKin=True):
        return self.model.CalcPotentialEnergy(Q, updateKin)

    def nameDof(self):
        return self.model.nameDof()

    def contactNames(self):
        return self.model.contactNames()

    def nbSoftContacts(self):
        return self.model.nbSoftContacts()

    def softContactNames(self):
        return self.model.softContactNames()

    def muscleNames(self):
        return self.model.muscleNames()

    def torque(self, tau_activations, q, qdot):
        return self.model.torque(tau_activations, q, qdot)

    def ForwardDynamicsFreeFloatingBase(self, q, qdot, qddot_joints):
        return self.model.ForwardDynamicsFreeFloatingBase(q, qdot, qddot_joints)

    def ForwardDynamics(self, q, qdot, tau, fext=None, f_contacts=None):
        return self.model.ForwardDynamics(q, qdot, tau, fext, f_contacts)

    def ForwardDynamicsConstraintsDirect(self, *args):
        return self.model.ForwardDynamicsConstraintsDirect(*args)

    def InverseDynamics(self, q, qdot, qddot, f_ext=None, f_contacts=None):
        return self.model.InverseDynamics(q, qdot, qddot, f_ext, f_contacts)

    def NonLinearEffect(self, Q, QDot, f_ext=None, f_contacts=None):
        return self.model.NonLinearEffect(Q, QDot, f_ext, f_contacts)

    def ContactForcesFromForwardDynamicsConstraintsDirect(self, Q, QDot, Tau, f_ext=None):
        return self.model.ContactForcesFromForwardDynamicsConstraintsDirect(Q, QDot, Tau, f_ext)

    def bodyInertia(self, Q, updateKin=True):
        return self.model.bodyInertia(Q, updateKin)

    def ComputeConstraintImpulsesDirect(self, Q, QDotPre):
        return self.model.ComputeConstraintImpulsesDirect(Q, QDotPre)

    def checkGeneralizedDimensions(self, Q=None, Qdot=None, Qddot=None, torque=None):
        return self.model.checkGeneralizedDimensions(Q, Qdot, Qddot, torque)

    def stateSet(self):
        return self.model.stateSet()

    def activationDot(self, muscle_states):
        return self.model.activationDot(muscle_states)

    def muscularJointTorque(self, muscle_states, q, qdot):
        return self.model.muscularJointTorque(muscle_states, q, qdot)

    def getConstraints(self):
        return self.model.getConstraints()

    def markers(self, Q, updateKin=True):
        return self.model.markers(Q, updateKin)

    def nbRigidContacts(self):
        return self.model.nbRigidContacts()

    def path(self):
        return self.model.path()

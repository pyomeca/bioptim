from collections import OrderedDict
from PyQt5 import QtCore, QtGui, QtWidgets

# Avoid clash with module name
examples_ = OrderedDict([
    ('acados', OrderedDict([
        ('Static arm', 'static_arm.py'),
        ('Cube', 'cube.py'),
        ('Pendulum', 'pendulum.py'),
    ])),
    ('getting_started', OrderedDict([
        ('Custom Bounds', 'custom_bounds.py'),
        ('Custom constraint', 'custom_constraint.py'),
        ('Custom dynamics', 'custom_dynamics.py'),
        ('Custom initial guess', 'custom_initial_guess.py'),
        ('Custom objectives', 'custom_objectives.py'),
        ('Custom parameters', 'custom_parameters.py'),
        ('Custom phase transitions', 'custom_phase_transitions.py'),
        ('Custom plotting', 'custom_plotting.py'),
        ('Example cyclic movement', 'example_cyclic_movement.py'),
        ('Example external forces', 'example_external_forces.py'),
        ('Example inequality constraint', 'example_inequality_constraint.py'),
        ('Example mapping', 'example_mapping.py'),
        ('Example multiphase', 'example_multiphase.py'),
        ('Example optimal time', 'example_optimal_time.py'),
        ('Example save and load', 'example_save_and_load.py'),
        ('Example simulation', 'example_simulation.py'),
        ('Pendulum', 'pendulum.py'),
    ])),
    ('moving_horizon_estimation', OrderedDict([
        ('Cyclic nmpc', 'cyclic_nmpc.py'),
        ('Mhe', 'mhe.py'),
        ('Multi cyclic nmpc', 'multi_cyclic_nmpc.py'),
     ])),
    ('muscle_driven_ocp', OrderedDict([
        ('Muscle activations tracker', 'muscle_activations_tracker.py'),
        ('Muscle excitations tracker', 'muscle_excitations_tracker.py'),
        ('Static arm', 'static_arm.py'),
        ('Static arm with contact', 'static_arm_with_contact.py')
    ])),
    ('muscle_driven_with_contact', OrderedDict([
        ('Contact forces inequality constraint muscle', 'contact_forces_inequality_constraint_muscle.py'),
        ('Contact forces inequality constraint muscle excitations', 'contact_forces_inequality_constraint_muscle_excitations.py'),
        ('Muscle activations contacts tracker', 'muscle_activations_contacts_tracker.py')
    ])),
    ('optimal_time_ocp', OrderedDict([
        ('Multiphase time constraint', 'multiphase_time_constraint.py'),
        ('Pendulum min time Lagrange', 'pendulum_min_time_Lagrange.py'),
        ('Pendulum min time Mayer', 'pendulum_min_time_Mayer.py'),
        ('Time constraint', 'time_constraint.py'),
    ])),
    ('symmetrical_torque_driven_ocp', OrderedDict([
        ('Symmetry by constraint', 'symmetry_by_constraint.py'),
        ('Symmetry by mapping', 'symmetry_by_mapping.py'),
    ])),
    ('torque_driven_ocp', OrderedDict([
        ('Maximize predicted height CoM', 'maximize_predicted_height_CoM.py'),
        ('phase transition uneven variable number', 'phase_transition_uneven_variable_number.py'),
        ('spring load', 'spring_load.py'),
        ('Track markers 2D pendulum', 'track_markers_2D_pendulum.py'),
        ('Track markers with torque actuators', 'track_markers_with_torque_actuators.py'),
        ('Trampo quaternions', 'trampo_quaternions.py'),
    ])),
    ('track', OrderedDict([
        ('Track marker on segment', 'track_marker_on_segment.py'),
        ('Track segment on rt', 'track_segment_on_rt.py'),
    ])),
     ])


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(846, 552)
        self.gridLayout_2 = QtWidgets.QGridLayout(Form)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.splitter = QtWidgets.QSplitter(Form)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")
        self.layoutWidget = QtWidgets.QWidget(self.splitter)
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.qtLibCombo = QtWidgets.QComboBox(self.layoutWidget)
        self.loadBtn = QtWidgets.QPushButton(self.layoutWidget)
        self.loadBtn.setObjectName("loadBtn")
        self.gridLayout.addWidget(self.loadBtn, 4, 1, 1, 1)
        self.exampleTree = QtWidgets.QTreeWidget(self.layoutWidget)
        self.exampleTree.setObjectName("exampleTree")
        self.exampleTree.headerItem().setText(0, "1")
        self.exampleTree.header().setVisible(False)
        self.gridLayout.addWidget(self.exampleTree, 3, 0, 1, 2)
        self.label = QtWidgets.QLabel(self.layoutWidget)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 4, 0, 1, 1)
        self.exampleFilter = QtWidgets.QLineEdit(self.layoutWidget)
        self.exampleFilter.setObjectName("exampleFilter")
        self.gridLayout.addWidget(self.exampleFilter, 0, 0, 1, 2)
        self.searchFiles = QtWidgets.QComboBox(self.layoutWidget)
        self.searchFiles.setObjectName("searchFiles")
        self.searchFiles.addItem("")
        self.searchFiles.addItem("")
        self.gridLayout.addWidget(self.searchFiles, 1, 0, 1, 2)
        self.layoutWidget1 = QtWidgets.QWidget(self.splitter)
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget1)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.loadedFileLabel = QtWidgets.QLabel(self.layoutWidget1)
        font = QtGui.QFont()
        font.setBold(True)
        self.loadedFileLabel.setFont(font)
        self.loadedFileLabel.setText("")
        self.loadedFileLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.loadedFileLabel.setObjectName("loadedFileLabel")
        self.verticalLayout.addWidget(self.loadedFileLabel)
        self.codeView = QtWidgets.QPlainTextEdit(self.layoutWidget1)
        font = QtGui.QFont()
        font.setFamily("Courier New")
        self.codeView.setFont(font)
        self.codeView.setObjectName("codeView")
        self.verticalLayout.addWidget(self.codeView)
        self.gridLayout_2.addWidget(self.splitter, 1, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        self.loadBtn.setText(_translate("Form", "Run Example"))
        self.exampleFilter.setPlaceholderText(_translate("Form", "Type to filter..."))
        self.searchFiles.setItemText(0, _translate("Form", "Title Search"))
        self.searchFiles.setItemText(1, _translate("Form", "Content Search"))
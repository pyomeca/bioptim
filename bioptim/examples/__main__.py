# based on https://github.com/pyqtgraph/pyqtgraph/tree/master/pyqtgraph/examples

import keyword
import os
import re
import sys
import subprocess

import pyqtgraph as pg
from functools import lru_cache
from collections import OrderedDict
from PyQt5 import QtCore, QtGui, QtWidgets

try:
    import acados

    is_acados = True
except ModuleNotFoundError:
    is_acados = False


# Avoid clash with module name
examples_ = OrderedDict(
    [
        ("acados", OrderedDict([("Static arm", "static_arm.py"), ("Cube", "cube.py"), ("Pendulum", "pendulum.py")])),
        (
            "fatigue",
            OrderedDict(
                [
                    ("Pendulum with fatigue", "pendulum_with_fatigue.py"),
                    ("Static arm with fatigue", "static_arm_with_fatigue.py"),
                ]
            ),
        ),
        (
            "getting_started",
            OrderedDict(
                [
                    ("Custom Bounds", "custom_bounds.py"),
                    ("Custom constraint", "custom_constraint.py"),
                    ("Custom dynamics", "custom_dynamics.py"),
                    ("Custom initial guess", "custom_initial_guess.py"),
                    ("Custom non casadi dynamics", "custom_non_casadi_dynamics.py"),
                    ("Custom objectives", "custom_objectives.py"),
                    ("Custom parameters", "custom_parameters.py"),
                    ("Custom phase transitions", "custom_phase_transitions.py"),
                    ("Custom plotting", "custom_plotting.py"),
                    ("Example cyclic movement", "example_cyclic_movement.py"),
                    ("Example external forces", "example_external_forces.py"),
                    ("Example inequality constraint", "example_inequality_constraint.py"),
                    ("Example mapping", "example_mapping.py"),
                    ("Example multiphase", "example_multiphase.py"),
                    ("Example optimal time", "example_optimal_time.py"),
                    ("Example simulation", "example_simulation.py"),
                    ("Pendulum", "pendulum.py"),
                ]
            ),
        ),
        (
            "moving_horizon_estimation",
            OrderedDict(
                [("Cyclic nmpc", "cyclic_nmpc.py"), ("Mhe", "mhe.py"), ("Multi cyclic nmpc", "multi_cyclic_nmpc.py")]
            ),
        ),
        (
            "muscle_driven_ocp",
            OrderedDict(
                [
                    ("Muscle activations tracker", "muscle_activations_tracker.py"),
                    ("Muscle excitations tracker", "muscle_excitations_tracker.py"),
                    ("Static arm", "static_arm.py"),
                    ("Static arm with contact", "static_arm_with_contact.py"),
                ]
            ),
        ),
        (
            "muscle_driven_with_contact",
            OrderedDict(
                [
                    ("Contact forces inequality constraint muscle", "contact_forces_inequality_constraint_muscle.py"),
                    (
                        "Contact forces inequality constraint muscle excitations",
                        "contact_forces_inequality_constraint_muscle_excitations.py",
                    ),
                    ("Muscle activations contacts tracker", "muscle_activations_contacts_tracker.py"),
                ]
            ),
        ),
        (
            "optimal_time_ocp",
            OrderedDict(
                [
                    ("Multiphase time constraint", "multiphase_time_constraint.py"),
                    ("Pendulum min time Mayer", "pendulum_min_time_Mayer.py"),
                    ("Time constraint", "time_constraint.py"),
                ]
            ),
        ),
        (
            "symmetrical_torque_driven_ocp",
            OrderedDict(
                [
                    ("Symmetry by constraint", "symmetry_by_constraint.py"),
                    ("Symmetry by mapping", "symmetry_by_mapping.py"),
                ]
            ),
        ),
        (
            "torque_driven_ocp",
            OrderedDict(
                [
                    ("Maximize predicted height center_of_mass", "maximize_predicted_height_CoM.py"),
                    (
                        "phase transition uneven variable number by bounds",
                        "phase_transition_uneven_variable_number_by_bounds.py",
                    ),
                    (
                        "phase transition uneven variable number by mapping",
                        "phase_transition_uneven_variable_number_by_mapping.py",
                    ),
                    ("spring load", "spring_load.py"),
                    ("Track markers 2D pendulum", "track_markers_2D_pendulum.py"),
                    ("Track markers with torque actuators", "track_markers_with_torque_actuators.py"),
                    ("quaternions", "example_quaternions.py"),
                    ("Soft contact", "example_soft_contact.py"),
                ]
            ),
        ),
        (
            "track",
            OrderedDict(
                [
                    ("Track marker on segment", "track_marker_on_segment.py"),
                    ("Track segment on rt", "track_segment_on_rt.py"),
                ]
            ),
        ),
        (
            "deep_neural_network",
            OrderedDict(
                [
                    ("pytorch ocp", "pytorch_ocp.py"),
                ]
            ),
        ),
    ]
)


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


path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, path)
app = pg.mkQApp()


QRegularExpression = QtCore.QRegularExpression

QFont = QtGui.QFont
QColor = QtGui.QColor
QTextCharFormat = QtGui.QTextCharFormat
QSyntaxHighlighter = QtGui.QSyntaxHighlighter


def charFormat(color, style="", background=None):
    """
    Return a QTextCharFormat with the given attributes.
    """
    _color = QColor()
    if type(color) is not str:
        _color.setRgb(color[0], color[1], color[2])
    else:
        _color.setNamedColor(color)

    _format = QTextCharFormat()
    _format.setForeground(_color)
    if "bold" in style:
        _format.setFontWeight(QFont.Weight.Bold)
    if "italic" in style:
        _format.setFontItalic(True)
    if background is not None:
        _format.setBackground(pg.mkColor(background))

    return _format


class LightThemeColors:
    Red = "#B71C1C"
    Pink = "#FCE4EC"
    Purple = "#4A148C"
    DeepPurple = "#311B92"
    Indigo = "#1A237E"
    Blue = "#0D47A1"
    LightBlue = "#01579B"
    Cyan = "#006064"
    Teal = "#004D40"
    Green = "#1B5E20"
    LightGreen = "#33691E"
    Lime = "#827717"
    Yellow = "#F57F17"
    Amber = "#FF6F00"
    Orange = "#E65100"
    DeepOrange = "#BF360C"
    Brown = "#3E2723"
    Grey = "#212121"
    BlueGrey = "#263238"


class DarkThemeColors:
    Red = "#F44336"
    Pink = "#F48FB1"
    Purple = "#CE93D8"
    DeepPurple = "#B39DDB"
    Indigo = "#9FA8DA"
    Blue = "#90CAF9"
    LightBlue = "#81D4FA"
    Cyan = "#80DEEA"
    Teal = "#80CBC4"
    Green = "#A5D6A7"
    LightGreen = "#C5E1A5"
    Lime = "#E6EE9C"
    Yellow = "#FFF59D"
    Amber = "#FFE082"
    Orange = "#FFCC80"
    DeepOrange = "#FFAB91"
    Brown = "#BCAAA4"
    Grey = "#EEEEEE"
    BlueGrey = "#B0BEC5"


LIGHT_STYLES = {
    "keyword": charFormat(LightThemeColors.Blue, "bold"),
    "operator": charFormat(LightThemeColors.Red, "bold"),
    "brace": charFormat(LightThemeColors.Purple),
    "defclass": charFormat(LightThemeColors.Indigo, "bold"),
    "string": charFormat(LightThemeColors.Amber),
    "string2": charFormat(LightThemeColors.DeepPurple),
    "comment": charFormat(LightThemeColors.Green, "italic"),
    "self": charFormat(LightThemeColors.Blue, "bold"),
    "numbers": charFormat(LightThemeColors.Teal),
}

DARK_STYLES = {
    "keyword": charFormat(DarkThemeColors.Blue, "bold"),
    "operator": charFormat(DarkThemeColors.Red, "bold"),
    "brace": charFormat(DarkThemeColors.Purple),
    "defclass": charFormat(DarkThemeColors.Indigo, "bold"),
    "string": charFormat(DarkThemeColors.Amber),
    "string2": charFormat(DarkThemeColors.DeepPurple),
    "comment": charFormat(DarkThemeColors.Green, "italic"),
    "self": charFormat(DarkThemeColors.Blue, "bold"),
    "numbers": charFormat(DarkThemeColors.Teal),
}


class PythonHighlighter(QSyntaxHighlighter):
    """Syntax highlighter for the Python language."""

    # Python keywords
    keywords = keyword.kwlist

    # Python operators
    operators = [
        r"=",
        # Comparison
        r"==",
        r"!=",
        r"<",
        r"<=",
        r">",
        r">=",
        # Arithmetic
        r"\+",
        r"-",
        r"\*",
        r"/",
        r"//",
        r"%",
        r"\*\*",
        # In-place
        r"\+=",
        r"-=",
        r"\*=",
        r"/=",
        r"\%=",
        # Bitwise
        r"\^",
        r"\|",
        r"&",
        r"~",
        r">>",
        r"<<",
    ]

    # Python braces
    braces = [r"\{", r"\}", r"\(", r"\)", r"\[", r"\]"]

    def __init__(self, document):
        super().__init__(document)

        # Multi-line strings (expression, flag, style)
        self.tri_single = (QRegularExpression("'''"), 1, "string2")
        self.tri_double = (QRegularExpression('"""'), 2, "string2")

        rules = []

        # Keyword, operator, and brace rules
        rules += [(r"\b%s\b" % w, 0, "keyword") for w in PythonHighlighter.keywords]
        rules += [(o, 0, "operator") for o in PythonHighlighter.operators]
        rules += [(b, 0, "brace") for b in PythonHighlighter.braces]

        # All other rules
        rules += [
            # 'self'
            (r"\bself\b", 0, "self"),
            # 'def' followed by an identifier
            (r"\bdef\b\s*(\w+)", 1, "defclass"),
            # 'class' followed by an identifier
            (r"\bclass\b\s*(\w+)", 1, "defclass"),
            # Numeric literals
            (r"\b[+-]?[0-9]+[lL]?\b", 0, "numbers"),
            (r"\b[+-]?0[xX][0-9A-Fa-f]+[lL]?\b", 0, "numbers"),
            (r"\b[+-]?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\b", 0, "numbers"),
            # Double-quoted string, possibly containing escape sequences
            (r'"[^"\\]*(\\.[^"\\]*)*"', 0, "string"),
            # Single-quoted string, possibly containing escape sequences
            (r"'[^'\\]*(\\.[^'\\]*)*'", 0, "string"),
            # From '#' until a newline
            (r"#[^\n]*", 0, "comment"),
        ]
        self.rules = rules
        self.searchText = None

    @property
    def styles(self):
        app = QtWidgets.QApplication.instance()
        return DARK_STYLES if app.property("darkMode") else LIGHT_STYLES

    def highlightBlock(self, text):
        """Apply syntax highlighting to the given block of text."""
        # Do other syntax formatting
        rules = self.rules.copy()
        for expression, nth, format in rules:
            format = self.styles[format]

            for n, match in enumerate(re.finditer(expression, text)):
                if n < nth:
                    continue
                start = match.start()
                length = match.end() - start
                self.setFormat(start, length, format)

        self.applySearchHighlight(text)
        self.setCurrentBlockState(0)

        # Do multi-line strings
        in_multiline = self.match_multiline(text, *self.tri_single)
        if not in_multiline:
            in_multiline = self.match_multiline(text, *self.tri_double)

    def match_multiline(self, text, delimiter, in_state, style):
        """Do highlighting of multi-line strings.

        =========== ==========================================================
        delimiter   (QRegularExpression) for triple-single-quotes or
                    triple-double-quotes
        in_state    (int) to represent the corresponding state changes when
                    inside those strings. Returns True if we're still inside a
                    multi-line string when this function is finished.
        style       (str) representation of the kind of style to use
        =========== ==========================================================
        """
        # If inside triple-single quotes, start at 0
        if self.previousBlockState() == in_state:
            start = 0
            add = 0
        # Otherwise, look for the delimiter on this line
        else:
            match = delimiter.match(text)
            start = match.capturedStart()
            # Move past this match
            add = match.capturedLength()

        # As long as there's a delimiter match on this line...
        while start >= 0:
            # Look for the ending delimiter
            match = delimiter.match(text, start + add)
            end = match.capturedEnd()
            # Ending delimiter on this line?
            if end >= add:
                length = end - start + add + match.capturedLength()
                self.setCurrentBlockState(0)
            # No; multi-line string
            else:
                self.setCurrentBlockState(in_state)
                length = len(text) - start + add
            # Apply formatting
            self.setFormat(start, length, self.styles[style])
            # Highlighting sits on top of this formatting
            # Look for the next match
            match = delimiter.match(text, start + length)
            start = match.capturedStart()

        self.applySearchHighlight(text)

        # Return True if still inside a multi-line string, False otherwise
        if self.currentBlockState() == in_state:
            return True
        else:
            return False

    def applySearchHighlight(self, text):
        if not self.searchText:
            return
        expr = f"(?i){self.searchText}"
        palette: QtGui.QPalette = app.palette()
        color = palette.highlight().color()
        fgndColor = palette.color(palette.ColorGroup.Current, palette.ColorRole.Text).name()
        style = charFormat(fgndColor, background=color.name())
        for match in re.finditer(expr, text):
            start = match.start()
            length = match.end() - start
            self.setFormat(start, length, style)


def unnestedDict(exDict):
    """Converts a dict-of-dicts to a singly nested dict for non-recursive parsing"""
    out = {}
    for kk, vv in exDict.items():
        if isinstance(vv, dict):
            out.update(unnestedDict(vv))
        else:
            out[kk] = vv
    return out


class ExampleLoader(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.ui = Ui_Form()
        self.cw = QtWidgets.QWidget()
        self.setCentralWidget(self.cw)
        self.ui.setupUi(self.cw)
        self.setWindowTitle("PyQtGraph Examples")
        self.codeBtn = QtWidgets.QPushButton("Run Edited Code")
        self.codeLayout = QtWidgets.QGridLayout()
        self.ui.codeView.setLayout(self.codeLayout)
        self.hl = PythonHighlighter(self.ui.codeView.document())
        app = QtWidgets.QApplication.instance()
        app.paletteChanged.connect(self.updateTheme)
        policy = QtWidgets.QSizePolicy.Policy.Expanding
        self.codeLayout.addItem(QtWidgets.QSpacerItem(100, 100, policy, policy), 0, 0)
        self.codeLayout.addWidget(self.codeBtn, 1, 1)
        self.codeBtn.hide()

        textFil = self.ui.exampleFilter
        self.curListener = None
        self.ui.exampleFilter.setFocus()

        def onComboChanged(searchType):
            if self.curListener is not None:
                self.curListener.disconnect()
            self.curListener = textFil.textChanged
            if searchType == "Content Search":
                self.curListener.connect(self.filterByContent)
            else:
                self.hl.searchText = None
                self.curListener.connect(self.filterByTitle)
            # Fire on current text, too
            self.curListener.emit(textFil.text())

        self.ui.searchFiles.currentTextChanged.connect(onComboChanged)
        onComboChanged(self.ui.searchFiles.currentText())

        self.itemCache = []
        self.populateTree(self.ui.exampleTree.invisibleRootItem(), examples_)
        self.ui.exampleTree.expandAll()

        self.resize(1000, 500)
        self.show()
        self.ui.splitter.setSizes([250, 750])
        self.ui.loadBtn.clicked.connect(self.loadFile)
        self.ui.exampleTree.currentItemChanged.connect(self.showFile)
        self.ui.exampleTree.itemDoubleClicked.connect(self.loadFile)

        # textChanged fires when the highlighter is reassigned the same document. Prevent this
        # from showing "run edited code" by checking for actual content change
        oldText = self.ui.codeView.toPlainText()

        def onTextChange():
            nonlocal oldText
            newText = self.ui.codeView.toPlainText()
            if newText != oldText:
                oldText = newText
                self.codeEdited()

        self.ui.codeView.textChanged.connect(onTextChange)
        self.codeBtn.clicked.connect(self.runEditedCode)

    def filterByTitle(self, text):
        self.showExamplesByTitle(self.getMatchingTitles(text))
        self.hl.setDocument(self.ui.codeView.document())

    def filterByContent(self, text=None):
        # Don't filter very short strings
        self.hl.searchText = text
        # Need to reapply to current document
        self.hl.setDocument(self.ui.codeView.document())
        text = text.lower()
        titles = []
        for key, val in examples_.items():
            if isinstance(val, OrderedDict):
                root_dir = key
                checkDict = unnestedDict(val)
                for kk, vv in checkDict.items():
                    path = os.getcwd() + "/" + root_dir
                    filename = os.path.join(path, vv)
                    contents = self.getExampleContent(filename).lower()
                    if text in contents:
                        titles.append(kk)
            else:
                pass
            self.showExamplesByTitle(titles)

    def getMatchingTitles(self, text, exDict=None, acceptAll=False):
        if exDict is None:
            exDict = examples_
        text = text.lower()
        titles = []
        for kk, vv in exDict.items():
            matched = acceptAll or text in kk.lower()
            if isinstance(vv, dict):
                titles.extend(self.getMatchingTitles(text, vv, acceptAll=matched))
            elif matched:
                titles.append(kk)
        return titles

    def showExamplesByTitle(self, titles):
        QTWI = QtWidgets.QTreeWidgetItemIterator
        flag = QTWI.IteratorFlag.NoChildren
        treeIter = QTWI(self.ui.exampleTree, flag)
        item = treeIter.value()
        while item is not None:
            parent = item.parent()
            show = item.childCount() or item.text(0) in titles
            item.setHidden(not show)

            # If all children of a parent are gone, hide it
            if parent:
                hideParent = True
                for ii in range(parent.childCount()):
                    if not parent.child(ii).isHidden():
                        hideParent = False
                        break
                parent.setHidden(hideParent)

            treeIter += 1
            item = treeIter.value()

    def simulate_black_mode(self):
        """
        used to simulate MacOS "black mode" on other platforms
        intended for debug only, as it manage only the QPlainTextEdit
        """
        # first, a dark background
        c = QtGui.QColor("#171717")
        p = self.ui.codeView.palette()
        p.setColor(QtGui.QPalette.ColorGroup.Active, QtGui.QPalette.ColorRole.Base, c)
        p.setColor(QtGui.QPalette.ColorGroup.Inactive, QtGui.QPalette.ColorRole.Base, c)
        self.ui.codeView.setPalette(p)
        # then, a light font
        f = QtGui.QTextCharFormat()
        f.setForeground(QtGui.QColor("white"))
        self.ui.codeView.setCurrentCharFormat(f)
        # finally, override application automatic detection
        app = QtWidgets.QApplication.instance()
        app.setProperty("darkMode", True)

    def updateTheme(self):
        self.hl = PythonHighlighter(self.ui.codeView.document())

    def populateTree(self, root, examples, root_dir=None):
        for key, val in examples.items():
            if not is_acados and key == "acados":
                pass
            else:
                item = QtWidgets.QTreeWidgetItem([key.replace("_", " ").capitalize()])
                if isinstance(val, OrderedDict):
                    bold_font = item.font(0)
                    bold_font.setBold(True)
                    item.setFont(0, bold_font)
                self.itemCache.append(item)
                if root_dir:
                    val = root_dir + "/" + val
                if isinstance(val, OrderedDict):
                    self.populateTree(item, val, root_dir=key)
                else:
                    item.file = val
                root.addChild(item)

    def currentFile(self):
        item = self.ui.exampleTree.currentItem()
        if hasattr(item, "file"):
            return os.path.join(path, item.file)
        return None

    def loadFile(self, edited=False):
        # Change current directory to executable directory
        main_dir = os.getcwd()
        executable_dir = os.path.dirname(self.currentFile())
        if main_dir != executable_dir:
            os.chdir(executable_dir)

        env = dict(os.environ, PYQTGRAPH_QT_LIB=str(self.ui.qtLibCombo.currentText()))

        if edited:
            path = os.path.abspath(os.path.dirname(__file__))
            proc = subprocess.Popen([sys.executable, "-"], stdin=subprocess.PIPE, cwd=path, env=env)
            code = str(self.ui.codeView.toPlainText()).encode("UTF-8")
            proc.stdin.write(code)
            proc.stdin.close()
        else:
            fn = self.currentFile()
            if fn is None:
                return
            subprocess.Popen([sys.executable, fn], env=env)

        # Go back to the main directory
        os.chdir(main_dir)

    def showFile(self):
        fn = self.currentFile()
        text = self.getExampleContent(fn)
        self.ui.codeView.setPlainText(text)
        self.ui.loadedFileLabel.setText(fn)
        self.codeBtn.hide()

    @lru_cache(100)
    def getExampleContent(self, filename):
        if filename is None:
            self.ui.codeView.clear()
            return
        if os.path.isdir(filename):
            filename = os.path.join(filename, "__main__.py")
        with open(filename, "r") as currentFile:
            text = currentFile.read()
        return text

    def codeEdited(self):
        self.codeBtn.show()

    def runEditedCode(self):
        self.loadFile(edited=True)

    def keyPressEvent(self, event):
        ret = super().keyPressEvent(event)
        if not QtCore.Qt.KeyboardModifier.ControlModifier & event.modifiers():
            return ret
        key = event.key()
        Key = QtCore.Qt.Key

        # Allow quick navigate to search
        if key == Key.Key_F:
            self.ui.exampleFilter.setFocus()
            event.accept()
            return

        if key not in [Key.Key_Plus, Key.Key_Minus, Key.Key_Underscore, Key.Key_Equal, Key.Key_0]:
            return ret
        font = self.ui.codeView.font()
        oldSize = font.pointSize()
        if key == Key.Key_Plus or key == Key.Key_Equal:
            font.setPointSize(oldSize + max(oldSize * 0.15, 1))
        elif key == Key.Key_Minus or key == Key.Key_Underscore:
            newSize = oldSize - max(oldSize * 0.15, 1)
            font.setPointSize(max(newSize, 1))
        elif key == Key.Key_0:
            # Reset to original size
            font.setPointSize(10)
        self.ui.codeView.setFont(font)
        event.accept()


def main():
    app = pg.mkQApp()
    loader = ExampleLoader()
    loader.ui.exampleTree.setCurrentIndex(loader.ui.exampleTree.model().index(0, 0))
    app.exec()


if __name__ == "__main__":
    main()

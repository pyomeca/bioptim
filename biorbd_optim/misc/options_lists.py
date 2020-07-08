from .enums import Instant, InterpolationType
from ..dynamics.dynamics_type import DynamicsType
from ..limits.constraints import Constraint
from ..limits.continuity import StateTransition
from ..limits.objective_functions import Objective
from ..limits.path_conditions import InitialConditions, Bounds


class OptionList:
    def __init__(self):
        self.options = [
            [],
        ]

    def __len__(self):
        if self.options == [[]]:
            return 0
        else:
            return len(self.options)

    def __iter__(self):
        self._iter_idx = 0
        return self

    def __next__(self):
        self._iter_idx += 1
        if self._iter_idx > len(self):
            raise StopIteration
        return self.options[self._iter_idx - 1]

    def __getitem__(self, item):
        return self.options[item]

    def _add(self, phase=0, **extra_arguments):
        for i in range(len(self.options), phase + 1):
            self.options.append([])
        self.options[phase].append(extra_arguments)

    def print(self):
        # TODO: Print all elements in the console
        pass


class UniquePerPhaseOptionList(OptionList):
    def _add(self, phase=-1, **extra_arguments):
        if phase == -1:
            phase = len(self)
        super(UniquePerPhaseOptionList, self)._add(phase=phase, **extra_arguments)

    def __getitem__(self, item):
        return super(UniquePerPhaseOptionList, self).__getitem__(item)[0]

    def __next__(self):
        self._iter_idx += 1
        if self._iter_idx > len(self):
            raise StopIteration
        return self.options[self._iter_idx - 1][0]


class ConstraintList(OptionList):
    def add(self, type, instant, phase=0, **extra_arguments):
        if not isinstance(type, Constraint):
            extra_arguments["custom_function"] = type
            type = Constraint.CUSTOM
        super(ConstraintList, self)._add(type=type, instant=instant, phase=phase, quadratic=None, **extra_arguments)


class ObjectiveList(OptionList):
    def add(
        self, type, instant=Instant.DEFAULT, weight=1, phase=0, custom_type=None, quadratic=None, **extra_arguments
    ):
        if not isinstance(type, Objective.Lagrange) and not isinstance(type, Objective.Mayer):
            extra_arguments = {**extra_arguments, "custom_function": type}

            if custom_type is None:
                raise RuntimeError(
                    "Custom objective function detected, but custom_function is missing. "
                    "It should either be Objective.Mayer or Objective.Lagrange"
                )
            type = custom_type(custom_type.CUSTOM)
            if isinstance(type, Objective.Lagrange):
                pass
            elif isinstance(type, Objective.Mayer):
                pass
            elif isinstance(type, Objective.Parameter):
                pass
            else:
                raise RuntimeError(
                    "Custom objective function detected, but custom_function is invalid. "
                    "It should either be Objective.Mayer or Objective.Lagrange"
                )

        super(ObjectiveList, self)._add(
            type=type, instant=instant, weight=weight, phase=phase, quadratic=quadratic, **extra_arguments
        )


class DynamicsTypeList(UniquePerPhaseOptionList):
    def add(self, type, dynamic_function=None, phase=-1):
        extra_arguments = {}
        if not isinstance(type, DynamicsType):
            extra_arguments["configure"] = type
            type = DynamicsType.CUSTOM

        super(DynamicsTypeList, self)._add(type=type, phase=phase, dynamic_function=dynamic_function, **extra_arguments)


class BoundsList(UniquePerPhaseOptionList):
    def add(
        self, bound, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT, phase=-1, **extra_arguments
    ):
        if not isinstance(bound, Bounds):
            bound = Bounds(min_bound=bound[0], max_bound=bound[1], interpolation=interpolation, **extra_arguments)

        super(BoundsList, self)._add(bound=bound, phase=phase)

    def __getitem__(self, item):
        return super(BoundsList, self).__getitem__(item)["bound"]

    def __next__(self):
        return super(BoundsList, self).__next__()["bound"]


class InitialConditionsList(UniquePerPhaseOptionList):
    def add(self, initial_condition, interpolation=InterpolationType.CONSTANT, phase=-1, **extra_arguments):
        if not isinstance(initial_condition, InitialConditions):
            initial_condition = InitialConditions(initial_condition, interpolation=interpolation, **extra_arguments)

        super(InitialConditionsList, self)._add(initial_condition=initial_condition, phase=phase)

    def __getitem__(self, item):
        return super(InitialConditionsList, self).__getitem__(item)["initial_condition"]

    def __next__(self):
        return super(InitialConditionsList, self).__next__()["initial_condition"]


class StateTransitionList(UniquePerPhaseOptionList):
    def add(self, transition, phase=-1, **extra_arguments):
        if not isinstance(transition, StateTransition):
            extra_arguments["custom_function"] = transition
            transition = StateTransition.CUSTOM
        super(StateTransitionList, self)._add(type=transition, phase=phase, **extra_arguments)


class ParametersList(OptionList):
    def add(self, parameter_name, function, initial_guess, bounds, size, phase=0, penalty_list=None, **extra_arguments):
        super(ParametersList, self)._add(
            function=function,
            phase=phase,
            name=parameter_name,
            initial_guess=initial_guess,
            bounds=bounds,
            size=size,
            penalty_list=penalty_list,
            **extra_arguments
        )

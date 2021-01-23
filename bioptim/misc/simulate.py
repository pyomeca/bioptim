from copy import copy

import numpy as np
from .enums import ControlType
from ..limits.path_conditions import InitialGuess


class Simulate:
    """
    Interface that allows to integrate a solution

    Methods
    -------
    from_solve(ocp, sol: dict, single_shoot: bool = False) -> dict
        Simulate the states using a solution structure. If single_shoot is True, the simulation is done in one go.
        Otherwise, the simulation reset at each node
    from_data(ocp, data: list, single_shoot: bool = True) -> dict
        Simulate the states using a get_data structure. If single_shoot is True, the simulation is done in one go.
        Otherwise, the simulation reset at each node
    from_controls_and_initial_states(ocp, states: InitialGuess, controls: InitialGuess,
            single_shoot: bool = False) -> dict
        Simulate the states from an initial guess set. If single_shoot is True, the simulation is done in one go.
        Otherwise, the simulation reset at each node
    from_controls_and_initial_states(ocp, states: InitialGuess, controls: InitialGuess,
            single_shoot: bool = False) -> dict
        Simulate the states from an initial guess set. If single_shoot is True, the simulation is done in one go.
        Otherwise, the simulation reset at each node
    """

    @staticmethod
    def from_solve(ocp, sol: dict, single_shoot: bool = False) -> dict:
        """
        Simulate the states using a solution structure. If single_shoot is True, the simulation is done in one go.
        Otherwise, the simulation reset at each node

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        sol: dict
            A solution structure
        single_shoot: bool
            If the simulation should reset [False] or not [True] at each node

        Returns
        -------
        A solution structure integrated
        """

        v_input = np.array(sol["x"]).squeeze()
        v_output = copy(v_input)
        offset = 0
        for nlp in ocp.nlp:
            # TODO adds StateTransitionFunctions between phases
            for idx_nodes in range(nlp.ns):
                x0 = v_output[offset : offset + nlp.nx] if single_shoot else v_input[offset : offset + nlp.nx]

                if nlp.control_type == ControlType.CONSTANT:
                    p = v_input[offset + nlp.nx : offset + nlp.nx + nlp.nu]
                elif nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                    p = np.vstack(
                        (
                            v_input[offset + nlp.nx : offset + nlp.nx + nlp.nu],
                            v_input[offset + 2 * nlp.nx + nlp.nu : offset + 2 * nlp.nx + 2 * nlp.nu],
                        )
                    ).T
                else:
                    raise NotImplementedError(f"Plotting {nlp.control_type} is not implemented yet")

                v_output[offset + nlp.nx + nlp.nu : offset + 2 * nlp.nx + nlp.nu] = np.array(
                    nlp.dynamics[idx_nodes](x0=x0, p=p)["xf"]
                ).squeeze()

                offset += nlp.nx + nlp.nu
        sol["x"] = v_output
        return sol

    @staticmethod
    def from_data(ocp, data: list, single_shoot: bool = True) -> dict:
        """
        Simulate the states using a get_data structure. If single_shoot is True, the simulation is done in one go.
        Otherwise, the simulation reset at each node

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        data: list
            The solution return from get_data
        single_shoot: bool
            If the simulation should reset [False] or not [True] at each node

        Returns
        -------
        A solution structure integrated
        """

        def concat_variables(variables: dict, offset_phases: int, idx_nodes: int) -> np.ndarray:
            """
            Concatenate all the variables of a dict in one np.ndarray

            Parameters
            ----------
            variables: dict
                The states or controls dictionary data structure
            offset_phases: int
                The offset due to the phase
            idx_nodes
                The offset due to the nodes

            Returns
            -------
            The matrix of data concatenated
            """

            var = np.ndarray(0)
            for key in variables.keys():
                var = np.append(var, variables[key][:, offset_phases + idx_nodes])
            return var

        states = data[0]
        controls = data[1]
        v = np.ndarray(0)

        offset_phases = 0
        for nlp in ocp.nlp:
            offset = 0
            if nlp.control_type == ControlType.CONSTANT:
                v_phase = np.ndarray((nlp.ns + 1) * nlp.nx + nlp.ns * nlp.nu)
            elif nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                v_phase = np.ndarray((nlp.ns + 1) * (nlp.nx + nlp.nu))
            else:
                raise NotImplementedError(f"Plotting {nlp.control_type} is not implemented yet")

            v_phase[offset : offset + nlp.nx] = concat_variables(states, offset_phases, 0)
            for idx_nodes in range(nlp.ns):
                x0 = (
                    v_phase[offset : offset + nlp.nx]
                    if single_shoot
                    else concat_variables(states, offset_phases, idx_nodes)
                )

                if nlp.control_type == ControlType.CONSTANT:
                    p = concat_variables(controls, offset_phases, idx_nodes)
                elif nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                    p = np.vstack(
                        (
                            concat_variables(controls, offset_phases, idx_nodes),
                            concat_variables(controls, offset_phases, idx_nodes + 1),
                        )
                    ).T
                else:
                    raise NotImplementedError(f"Plotting {nlp.control_type} is not implemented yet")

                v_phase[offset + nlp.nx + nlp.nu : offset + 2 * nlp.nx + nlp.nu] = np.array(
                    nlp.dynamics[idx_nodes](
                        x0=x0,
                        p=p,
                    )["xf"]
                ).squeeze()

                offset += nlp.nx + nlp.nu
            v = np.append(v, v_phase)
            offset_phases += nlp.ns
        return {"x": v}

    @staticmethod
    def from_controls_and_initial_states(
        ocp, states: InitialGuess, controls: InitialGuess, single_shoot: bool = False
    ) -> dict:
        """
        Simulate the states from an initial guess set. If single_shoot is True, the simulation is done in one go.
        Otherwise, the simulation reset at each node

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        states: InitialGuess
            The initial guess for the states
        controls: InitialGuess
            The initial guess for the controls
        single_shoot: bool
            If the simulation should reset [False] or not [True] at each node

        Returns
        -------
        A solution structure integrated
        """

        # todo flag single/multiple here and in from_solve (copy states)
        states.check_and_adjust_dimensions(ocp.nlp[0].nx, ocp.nlp[0].ns)
        v = states.init.evaluate_at(0)

        if not isinstance(controls, (list, tuple)):
            controls = (controls,)

        for idx_phase, nlp in enumerate(ocp.nlp):
            controls[idx_phase].check_and_adjust_dimensions(nlp.nu, nlp.ns - 1)
            for idx_nodes in range(nlp.ns):
                v = np.append(v, controls[idx_phase].init.evaluate_at(shooting_point=idx_nodes))
                v = np.append(v, states.init.evaluate_at(0))

        return Simulate.from_solve(ocp, {"x": v}, single_shoot)

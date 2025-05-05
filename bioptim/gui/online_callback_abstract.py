from abc import ABC, abstractmethod

from casadi import Callback, nlpsol_out, nlpsol_n_out, Sparsity

from ..misc.parameters_types import AnyDictOptional, AnyIterable, Str, Int, Bool, AnyTuple, IntIterableOptional


class OnlineCallbackAbstract(Callback, ABC):
    """
    CasADi interface of Ipopt callbacks

    Attributes
    ----------
    ocp: OptimalControlProgram
        A reference to the ocp to show
    nx: int
        The number of optimization variables
    ng: int
        The number of constraints

    Methods
    -------
    get_n_in() -> int
        Get the number of variables in
    get_n_out() -> int
        Get the number of variables out
    get_name_in(i: int) -> int
        Get the name of a variable
    get_name_out(_) -> str
        Get the name of the output variable
    get_sparsity_in(self, i: int) -> tuple[int]
        Get the sparsity of a specific variable
    eval(self, arg: list | tuple, force: bool = False) -> list[int]
        Send the current data to the plotter
    """

    def __init__(self, ocp, opts: AnyDictOptional = None, **show_options):
        """
        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp to show
        opts: dict
            Option to AnimateCallback method of CasADi
        show_options: dict
            The options to pass to PlotOcp
        """
        if opts is None:
            opts = {}

        Callback.__init__(self)
        self.ocp = ocp
        self.nx = self.ocp.variables_vector.shape[0]

        # There must be an option to add an if here
        from ..interfaces.ipopt_interface import IpoptInterface

        interface = IpoptInterface(ocp)
        all_g, _ = interface.dispatch_bounds()
        self.ng = all_g.shape[0]

        self.construct("AnimateCallback", opts)

    @abstractmethod
    def close(self):
        """
        Close the callback
        """

    @staticmethod
    def get_n_in() -> Int:
        """
        Get the number of variables in

        Returns
        -------
        The number of variables in
        """

        return nlpsol_n_out()

    @staticmethod
    def get_n_out() -> Int:
        """
        Get the number of variables out

        Returns
        -------
        The number of variables out
        """

        return 1

    @staticmethod
    def get_name_in(i: Int) -> Str:
        """
        Get the name of a variable

        Parameters
        ----------
        i: int
            The index of the variable

        Returns
        -------
        The name of the variable
        """

        return nlpsol_out(i)

    @staticmethod
    def get_name_out(_) -> Str:
        """
        Get the name of the output variable

        Returns
        -------
        The name of the output variable
        """

        return "ret"

    def get_sparsity_in(self, i: Int) -> AnyTuple:
        """
        Get the sparsity of a specific variable

        Parameters
        ----------
        i: int
            The index of the variable

        Returns
        -------
        The sparsity of the variable
        """

        n = nlpsol_out(i)
        if n == "f":
            return Sparsity.scalar()
        elif n in ("x", "lam_x"):
            return Sparsity.dense(self.nx)
        elif n in ("g", "lam_g"):
            return Sparsity.dense(self.ng)
        else:
            return Sparsity(0, 0)

    @abstractmethod
    def eval(self, arg: AnyIterable, enforce: Bool = False) -> IntIterableOptional:
        """
        Send the current data to the plotter

        Parameters
        ----------
        arg: list | tuple
            The data to send

        enforce: bool
            If True, the client will block until the server is ready to receive new data. This is useful at the end of
            the optimization to make sure the data are plot (and not discarded)

        Returns
        -------
        A list of error index
        """

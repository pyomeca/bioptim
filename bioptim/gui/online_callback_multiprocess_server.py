from multiprocessing import Process

from .online_callback_server import PlottingServer


def _start_as_multiprocess_internal(*args, **kwargs):
    """
    Starts the server (necessary for multiprocessing), this method should not be called directly, apart from
    run_as_multiprocess

    Parameters
    ----------
    same as PlottingServer
    """
    PlottingServer(*args, **kwargs)


class PlottingMultiprocessServer(PlottingServer):
    def __init__(self, *args, **kwargs):
        """
        Starts the server in a new process

        Parameters
        ----------
        Same as PlottingServer
        """

        process = Process(target=_start_as_multiprocess_internal, args=args, kwargs=kwargs)
        process.start()

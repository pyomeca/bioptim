from math import inf

import numpy as np
from casadi import horzcat, MX

from .enums import Instant


class Goal:
    @staticmethod
    def _get_instant(nlp, constraint):
        if not isinstance(constraint["instant"], (list, tuple)):
            constraint["instant"] = (constraint["instant"],)
        t = []
        x = MX()
        u = MX()
        for node in constraint["instant"]:
            if isinstance(node, int):
                if node < 0 or node > nlp["ns"]:
                    raise RuntimeError(f"Invalid instant, {node} must be between 0 and {nlp['ns']}")
                t.append(node)
                x = horzcat(x, nlp["X"][node])
                u = horzcat(u, nlp["U"][node])

            elif node == Instant.START:
                t.append(0)
                x = horzcat(x, nlp["X"][0])
                u = horzcat(u, nlp["U"][0])

            elif node == Instant.MID:
                if nlp["ns"] % 2 == 1:
                    raise (ValueError("Number of shooting points must be even to use MID"))
                t.append(nlp["X"][nlp["ns"] // 2])
                x = horzcat(x, nlp["X"][nlp["ns"] // 2])
                u = horzcat(u, nlp["U"][nlp["ns"] // 2])

            elif node == Instant.INTERMEDIATES:
                for i in range(1, nlp["ns"] - 1):
                    t.append(i)
                    x = horzcat(x, nlp["X"][i])
                    u = horzcat(u, nlp["U"][i])

            elif node == Instant.END:
                t.append(nlp["X"][nlp["ns"]])
                x = horzcat(x, nlp["X"][nlp["ns"]])

            elif node == Instant.ALL:
                t.extend([i for i in range(nlp["ns"] + 1)])
                for i in range(nlp["ns"]):
                    x = horzcat(x, nlp["X"][i])
                    u = horzcat(u, nlp["U"][i])
                x = horzcat(x, nlp["X"][nlp["ns"]])

            else:
                raise RuntimeError(" is not a valid instant")
        return t, x, u

    @staticmethod
    def _check_and_fill_index(var_idx, target_size, var_name="var"):
        if var_idx == ():
            var_idx = range(target_size)
        else:
            if isinstance(var_idx, int):
                var_idx = [var_idx]
            if max(var_idx) > target_size:
                raise RuntimeError(f"{var_name} in minimize_states cannot be higher than nx ({target_size})")
        return var_idx

    @staticmethod
    def _check_and_fill_tracking_data_size(data_to_track, target_size):
        if data_to_track == ():
            data_to_track = np.zeros(target_size)
        else:
            if len(data_to_track.shape) != len(target_size):
                if target_size[1] == 1 and len(data_to_track.shape) == 1:
                    # If we have a vector it is still okay
                    data_to_track = data_to_track.reshape(data_to_track.shape[0], 1)
                else:
                    raise RuntimeError(
                        f"data_to_track {data_to_track.shape}don't correspond to expected minimum size {target_size}"
                    )
            for i in range(len(target_size)):
                if data_to_track.shape[i] < target_size[i]:
                    raise RuntimeError(
                        f"data_to_track {data_to_track.shape} don't correspond to expected minimum size {target_size}"
                    )
        return data_to_track

    @staticmethod
    def _check_idx(name, elements, max_bound=inf, min_bound=0):
        if not isinstance(elements, (list, tuple)):
            elements = (elements,)
        for element in elements:
            if not isinstance(element, int):
                raise RuntimeError(f"{element} is not a valid index for {name}, it must be an integer")
            if element < min_bound or element >= max_bound:
                raise RuntimeError(f"{element} is not a valid index for {name}, it must be between 0 and {max_bound - 1}.")

# from ..optimization.optimal_control_program import OptimalControlProgram


def animate_with_bioviz(
    ocp: "OptimalControlProgram",
    data_to_animate: "Solution",
    show_now: bool = True,
    tracked_markers: dict[int, list[str]] | list[str] = None,
    **kwargs,
):
    """
    Animate the solution(s) using bioviz

    Parameters
    ----------
    ocp: OptimalControlProgram
        The optimal control program
    data_to_animate: Solution| List[Solution]
        The solution(s) to animate
    show_now: bool
        If the animation should be shown immediately
    tracked_markers: dict[int, list[str]] | list[str]
        The markers to track
    kwargs
        Any other parameters to pass to bioviz
    """

    all_bioviz = []
    for i, data in enumerate(data_to_animate):
        all_bioviz.append(
            ocp.nlp[i].model.animate(
                ocp,
                solution=data,
                show_now=show_now,
                tracked_markers=tracked_markers,
                **kwargs,
            )
        )

    return all_bioviz


def animate_with_pyorerun(
    ocp: "OptimalControlProgram",
    data_to_animate: "Solution",
    show_now: bool = True,
    tracked_markers: dict[int, list[str]] | list[str] = None,
    **kwargs,
):
    """
    Animate the solution(s) using pyorerun

    Parameters
    ----------
    ocp: OptimalControlProgram
        The optimal control program
    data_to_animate: Solution| List[Solution]
        The solution(s) to animate
    show_now: bool
        If the animation should be shown immediately
    tracked_markers: dict[int, list[str]] | list[str]
        The markers to track
    kwargs
        Any other parameters to pass to bioviz
    """

    for i, data in enumerate(data_to_animate):
        ocp.nlp[i].model.animate_with_pyorerun(
            ocp,
            solution=data,
            show_now=show_now,
            tracked_markers=tracked_markers,
            **kwargs,
        )

    return None

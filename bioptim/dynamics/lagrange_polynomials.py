import matplotlib.pyplot as plt
import numpy as np
from casadi import MX, SX, Function
from casadi.tools import collocation_points


def lagrange_polynomial(
    j: int, time_control_interval: MX | SX, integration_time: list, polynomial_order: int
) -> MX | SX:
    """
    Compute the j-th Lagrange polynomial \(L_j(\tau)\) in symbolic form.

    This polynomial is defined so that:

    .. math::
        L_j(\tau_i) = \\delta_{ij} =
        \\begin{cases}
            1 & \\text{if } i = j, \\\\
            0 & \\text{if } i \\neq j
        \\end{cases}

    for each of the collocation (or interpolation) points \\(\\tau_i\\) given in
    ``integration_time``. In other words, \\(L_j(\\tau)\\) is a polynomial in
    \\(\\tau\\) that is 1 at \\(\\tau_j = \\text{integration_time}[j]\\) and 0
    at every other \\(\\tau_r\\) with \\(r \\neq j\\).

    Parameters
    ----------
    j : int
        Index (0-based) of the Lagrange polynomial to compute,
        corresponding to the j-th collocation point.
    time_control_interval : MX or SX
        The symbolic variable (CasADi `MX` or `SX`) representing
        the dimensionless time within the control interval.
    integration_time : list of float
        The collocation points (dimensionless time) at which the
        Lagrange polynomials are defined. Typically in [0, 1].
    polynomial_order : int
        The total number of collocation points (also the polynomial order).

    Returns
    -------
    MX or SX
        A CasADi symbolic expression representing the j-th Lagrange polynomial
        evaluated at ``time_control_interval``.

    Notes
    -----
    The Lagrange polynomial \\( L_j(\\tau) \\) is constructed by the product:

    .. math::
        L_j(\\tau)
        = \\prod_{\\substack{r=0 \\\\ r \\neq j}}^{\\text{polynomial_order}-1}
          \\frac{\\tau - \\tau_r}{\\tau_j - \\tau_r},

    where \\(\\tau_j = \\text{integration_time}[j]\\) and
    \\(\\tau_r = \\text{integration_time}[r]\\). This function thus
    returns the polynomial in a symbolic form that can be used for
    further manipulation or evaluation (e.g., computing derivatives).
    """
    _l = 1
    for r in range(polynomial_order):
        if r != j:
            _l *= (time_control_interval - integration_time[r]) / (integration_time[j] - integration_time[r])
    return _l


def partial_lagrange_polynomial(
    j: int, time_control_interval: MX | SX, integration_time: list, polynomial_order: int, i: int
) -> MX | SX:
    """Compute the partial product of the j-th Lagrange polynomial without the i-th term."""
    _l = 1
    for r in range(polynomial_order):
        if r != j and r != i:
            _l *= (time_control_interval - integration_time[r]) / (integration_time[j] - integration_time[r])
    return _l


def lagrange_polynomial_derivative(
    j: int, time_control_interval: MX | SX, integration_time: list, polynomial_order: int
) -> MX | SX:
    """
    Compute the derivative of the j-th Lagrange polynomial \\( L_j(\\tau) \\) in symbolic form.

    By factoring out \\( L_j(\\tau) \\) and removing one factor from the product,
    the derivative can be expressed as:

    .. math::
        \\frac{d}{d\\tau} L_j(\\tau)
        = L_j(\\tau)
          \\sum_{\\substack{k=0 \\\\ k \\neq j}}^{\\text{polynomial_order}-1}
            \\frac{1}{\\tau - \\tau_k}.

    Parameters
    ----------
    j : int
        Index (0-based) of the Lagrange polynomial whose derivative is computed.
    time_control_interval : MX or SX
        The symbolic variable (CasADi `MX` or `SX`) representing
        the dimensionless time within the control interval.
    integration_time : list of float
        The collocation points (dimensionless time) at which the
        Lagrange polynomials are defined. Typically in [0, 1].
    polynomial_order : int
        The total number of collocation points (also the polynomial order).

    Returns
    -------
    MX or SX
        A CasADi symbolic expression representing the derivative
        of the j-th Lagrange polynomial evaluated at ``time_control_interval``.

    Notes
    -----
    The full expansion can also be written as:

    .. math::
        \\frac{d}{d\\tau} L_j(\\tau)
        = \\sum_{k \\neq j} \\left[
            \\frac{1}{(\\tau_j - \\tau_k)}
            \\prod_{\\substack{r \\neq j, r \\neq k}}^{} \\frac{(\\tau - \\tau_r)}{(\\tau_j - \\tau_r)}
          \\right],

    which, when factored appropriately, yields the compact form above.
    """
    sum_term = 0
    for k in range(polynomial_order):
        if k == j:
            continue

        partial_Ljk = partial_lagrange_polynomial(j, time_control_interval, integration_time, polynomial_order, k)
        sum_term += 1.0 / (integration_time[j] - integration_time[k]) * partial_Ljk

    return sum_term


def main():
    # ------------------------------------------------------------
    # 1) Choose polynomial order and define interpolation points
    # ------------------------------------------------------------
    polynomial_order = 9
    # Use CasADi's collocation_points for a standard set of nodes in [0,1].
    # "radau" => Radau collocation points; you can try "legendre" or "chebyshev" for others.
    integration_time = collocation_points(polynomial_order, "radau")

    # We'll pick one Lagrange polynomial index to demonstrate
    j = 0  # for instance, the 3rd polynomial (0-based index)

    # ------------------------------------------------------------
    # 2) Create CasADi symbolic variable
    # ------------------------------------------------------------
    tau = MX.sym("tau", 1)  # dimensionless time in [0,1]

    # ------------------------------------------------------------
    # 3) Build the polynomial and its derivative symbolically
    # ------------------------------------------------------------
    Lj_expr = lagrange_polynomial(j, tau, integration_time, polynomial_order)
    dLj_expr = lagrange_polynomial_derivative(j, tau, integration_time, polynomial_order)

    # ------------------------------------------------------------
    # 4) Create CasADi Function objects to evaluate them numerically
    # ------------------------------------------------------------
    Lj_fn = Function("Lj_fn", [tau], [Lj_expr])
    dLj_fn = Function("dLj_fn", [tau], [dLj_expr])

    # ------------------------------------------------------------
    # 5) Print polynomial values at each node (sanity check)
    # ------------------------------------------------------------
    print(f"L_{j}(tau) at the integration nodes:")
    for i, tau_i in enumerate(integration_time):
        val = Lj_fn(tau_i)
        print(f"  L_{j}({tau_i:.4f}) = {val}")

    # We expect L_j(tau_j) ~ 1 and L_j(tau_r) ~ 0 for r != j
    # The same for derivative checks, just for demonstration:
    print(f"\nDerivative dL_{j}/dtau at the integration nodes:")
    for i, tau_i in enumerate(integration_time):
        val = dLj_fn(tau_i)
        print(f"  dL_{j}/dtau({tau_i:.4f}) = {val}")

    # ------------------------------------------------------------
    # 6) Plot the polynomial and derivative over [0,1]
    # ------------------------------------------------------------
    x_plot = np.linspace(0, 1, 300)

    # Evaluate polynomial L_j(x) and derivative on this grid
    Lj_values = np.array([Lj_fn(x_val) for x_val in x_plot]).squeeze()
    dLj_values = np.array([dLj_fn(x_val) for x_val in x_plot]).squeeze()

    # numerical derivative
    dLj_values_num = np.gradient(Lj_values, x_plot)

    fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # --- (A) L_j(tau) ---
    axs[0].plot(x_plot, Lj_values, label=f"L_{j}(tau)")
    # Mark the interpolation nodes
    axs[0].scatter(
        integration_time, [np.array(Lj_fn(ti)) for ti in integration_time], color="red", zorder=3, label="Nodes"
    )
    axs[0].set_ylabel("L_j(tau)")
    axs[0].set_title(f"Lagrange polynomial L_{j}(tau)")
    axs[0].grid(True)
    axs[0].legend()

    # --- (B) dL_j/dtau ---
    axs[1].plot(x_plot, dLj_values, color="orange", label=f"dL_{j}/dtau")
    axs[1].plot(x_plot, dLj_values_num, color="green", label=f"dL_{j}/dtau (numerical)")
    axs[1].set_xlabel("tau")
    axs[1].set_ylabel("dL_j/dtau")
    axs[1].set_title(f"Derivative of L_{j}(tau)")
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.show()


# Run main if this script is executed (instead of imported).
if __name__ == "__main__":
    main()

from casadi import Opti, MX, Function
import l4casadi as l4c
import torch


class NeuralNetworkModel(torch.nn.Module):
    def __init__(self, layer_node_count: tuple[int]):
        super(NeuralNetworkModel, self).__init__()
        layers = torch.nn.ModuleList()
        layers.append(torch.nn.Linear(layer_node_count[0], layer_node_count[-1]))
        self._forward_model = torch.nn.Sequential(*layers)
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_model(x)


def main():
    opti = Opti()
    nx = nu = 1
    torch_model = NeuralNetworkModel(layer_node_count=(nu, nx))

    # ---- decision variables ---------
    x = opti.variable(nx, 1)  # state
    u = opti.variable(nu, 1)  # control

    # ---- dynamic constraints --------
    x_sym = MX.sym("x", nx, 1)
    u_sym = MX.sym("u", nu, 1)
    forward_model = l4c.L4CasADi(torch_model, device="cpu")
    f = Function("xdot", [x_sym, u_sym], [x_sym - forward_model(u_sym)])
    opti.subject_to(f(x, u) == 0)  # Adding this line yields the error : jac_adj_i0_adj_o0 is not provided by L4CasADi.

    # ---- solve NLP  ------
    opti.solver("ipopt")
    opti.solve()


if __name__ == "__main__":
    main()

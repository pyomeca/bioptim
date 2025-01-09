from casadi import *
import l4casadi as l4c
import torch


class NeuralNetworkModel(torch.nn.Module):
    def __init__(self, layer_node_count: tuple[int]):
        super(NeuralNetworkModel, self).__init__()

        # Initialize the layers of the neural network
        self._size_in = layer_node_count[0]
        self._size_out = layer_node_count[-1]
        first_and_hidden_layers_node_count = layer_node_count[:-1]
        layers = torch.nn.ModuleList()
        for i in range(len(first_and_hidden_layers_node_count) - 1):
            layers.append(
                torch.nn.Linear(first_and_hidden_layers_node_count[i], first_and_hidden_layers_node_count[i + 1])
            )
        layers.append(torch.nn.Linear(first_and_hidden_layers_node_count[-1], layer_node_count[-1]))

        self._forward_model = torch.nn.Sequential(*layers)
        self._forward_model.to("cpu")

        # Put the model in evaluation mode
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = torch.Tensor(x.shape[0], self._forward_model[-1].out_features)
        for i, data in enumerate(x):
            output[i, :] = self._forward_model(data)
        return output.to("cpu")


def main():
    opti = Opti()  # Optimization problem
    nx = nu = 1
    hidden_layers = (10,)
    N = 100  # number of control intervals

    # ---- decision variables ---------
    X = opti.variable(nx, N + 1)  # state trajectory
    U = opti.variable(nu, N)  # control trajectory (throttle)

    # ---- dynamic constraints --------
    torch_model = NeuralNetworkModel(layer_node_count=(nx + nu, *hidden_layers, nx))
    dynamic_model = l4c.L4CasADi(torch_model, device="cpu")
    x_sym = MX.sym("x", nx, 1)
    u_sym = MX.sym("u", nu, 1)
    f = Function("qddot", [x_sym, u_sym], [dynamic_model(vertcat(x_sym, u_sym).T).T])

    for k in range(N):  # loop over control intervals
        # Runge-Kutta 1 integration
        k1 = f(X[:, k], U[:, k])
        opti.subject_to(X[:, k + 1] == k1)  # close the gaps

    # ---- boundary conditions --------
    opti.subject_to(X[0, 0] == 1)  # PROBLEM LIES HERE: PUTTING ANY VALUE BUT 0 WILL MAKE IPOPT FAILS

    # ---- solve NLP              ------
    opti.solver("ipopt")  # set numerical backend
    opti.solve()  # actual solve


if __name__ == "__main__":
    main()

"""
TODO: Explain what is this example about
TODO: All the documentation

This example is similar to the getting_started/pendulum.py example, but the dynamics are computed using a non-casadi 
based model. This is useful when the dynamics are computed using a different library (e.g. TensorFlow, PyTorch, etc.)
"""

import biorbd
from bioptim import OptimalControlProgram, DynamicsFcn, BoundsList, Dynamics
from bioptim.models.torch.torch_model import TorchModel
import numpy as np
import torch


class NeuralNetworkModel(torch.nn.Module):
    def __init__(
        self,
        layer_node_count: tuple[int],
        dropout_probability: float,
        use_batch_norm: bool,
    ):
        super(NeuralNetworkModel, self).__init__()
        activations = torch.nn.GELU()

        # Initialize the layers of the neural network
        self._size_in = layer_node_count[0]
        self._size_out = layer_node_count[-1]
        first_and_hidden_layers_node_count = layer_node_count[:-1]
        layers = torch.nn.ModuleList()
        for i in range(len(first_and_hidden_layers_node_count) - 1):
            layers.append(
                torch.nn.Linear(first_and_hidden_layers_node_count[i], first_and_hidden_layers_node_count[i + 1])
            )
            if use_batch_norm:
                torch.nn.BatchNorm1d(first_and_hidden_layers_node_count[i + 1])
            layers.append(activations)
            layers.append(torch.nn.Dropout(dropout_probability))
        layers.append(torch.nn.Linear(first_and_hidden_layers_node_count[-1], layer_node_count[-1]))

        self._forward_model = torch.nn.Sequential(*layers)
        self._forward_model.to(self.get_torch_device())

        self._optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self._loss_function = torch.nn.HuberLoss()

        # Put the model in evaluation mode
        self.eval()

    @property
    def size_in(self) -> int:
        return self._size_in

    @property
    def size_out(self) -> int:
        return self._size_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = torch.Tensor(x.shape[0], self._forward_model[-1].out_features)
        for i, data in enumerate(x):
            output[i, :] = self._forward_model(data)
        return output.to(self.get_torch_device())

    @staticmethod
    def get_torch_device() -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_me(self, training_data: list[torch.Tensor], validation_data: list[torch.Tensor]):
        # More details about scheduler in documentation
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._optimizer, mode="min", factor=0.1, patience=20, min_lr=1e-8
        )

        max_epochs = 10
        for _ in range(max_epochs):
            self._perform_epoch_training(targets=training_data)
            validation_loss = self._perform_epoch_training(targets=validation_data, only_compute=True)
            print(f"Validation loss: {validation_loss}")
            scheduler.step(validation_loss)  # Adjust/reduce learning rate

    def _perform_epoch_training(
        self,
        targets: list[torch.Tensor],
        only_compute: bool = False,
    ) -> tuple[float, float]:

        # Perform the predictions
        if only_compute:
            with torch.no_grad():
                all_predictions = self(targets[0])
                all_targets = targets[1]

        else:
            # Put the model in training mode
            self.train()

            # If it is training, we are updating the model with each prediction, we therefore need to do it in a loop
            all_predictions = torch.tensor([]).to(self.get_torch_device())
            all_targets = torch.tensor([]).to(self.get_torch_device())
            for input, target in zip(*targets):
                self._optimizer.zero_grad()

                # Get the predictions and targets
                output = self(input[None, :])

                # Do some machine learning shenanigans
                current_loss = self._loss_function.forward(output, target[None, :])
                current_loss.backward()  # Backpropagation
                self._optimizer.step()  # Updating weights

                # Populate the return values
                all_predictions = torch.cat((all_predictions, output))
                all_targets = torch.cat((all_targets, target[None, :]))

            # Put back the model in evaluation mode
            self.eval()

        # Calculation of mean distance and error %
        epoch_accuracy = (all_predictions - all_targets).abs().mean().item()
        return epoch_accuracy


def prepare_ocp(
    model: torch.nn.Module,
    final_time: float,
    n_shooting: int,
) -> OptimalControlProgram:

    # Dynamics
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN)
    torch_model = TorchModel(torch_model=model)

    # Path bounds
    x_bounds = BoundsList()
    x_bounds["q"] = [-3.14 * 1.5] * torch_model.nb_q, [3.14 * 1.5] * torch_model.nb_q
    x_bounds["q"][:, [0, -1]] = 0  # Start and end at 0...
    x_bounds["q"][1, -1] = 3.14  # ...but end with pendulum 180 degrees rotated
    x_bounds["qdot"] = [-3.14 * 10.0] * torch_model.nb_qdot, [3.14 * 10.0] * torch_model.nb_qdot
    x_bounds["qdot"][:, [0, -1]] = 0  # Start and end without any velocity

    # Define control path bounds
    u_bounds = BoundsList()
    u_bounds["tau"] = [-100] * torch_model.nb_tau, [100] * torch_model.nb_tau
    u_bounds["tau"][1, :] = 0  # ...but remove the capability to actively rotate

    return OptimalControlProgram(
        torch_model,
        dynamics,
        n_shooting,
        final_time,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        use_sx=True,
    )


def generate_dataset(biorbd_model: biorbd.Model, data_point_count: int) -> list[torch.Tensor]:
    q = torch.rand(data_point_count, biorbd_model.nbQ())
    qdot = torch.rand(data_point_count, biorbd_model.nbQdot())
    tau = torch.rand(data_point_count, biorbd_model.nbGeneralizedTorque())

    qddot = torch.zeros(data_point_count, biorbd_model.nbQddot())
    for i in range(data_point_count):
        qddot[i, :] = torch.tensor(
            biorbd_model.ForwardDynamics(np.array(q[i, :]), np.array(qdot[i, :]), np.array(tau[i, :])).to_array()
        )

    return [torch.cat((q, qdot, tau), dim=1), qddot]


def main():
    # --- Prepare a predictive model --- #
    biorbd_model = biorbd.Model("models/pendulum.bioMod")
    training_data = generate_dataset(biorbd_model, data_point_count=1000)
    validation_data = generate_dataset(biorbd_model, data_point_count=100)

    model = NeuralNetworkModel(layer_node_count=(6, 10, 10, 2), dropout_probability=0.2, use_batch_norm=True)
    model.train_me(training_data, validation_data)

    ocp = prepare_ocp(model=model, final_time=1, n_shooting=40)

    # --- Solve the ocp --- #
    sol = ocp.solve()

    # --- Show the results graph --- #
    # sol.print_cost()
    sol.graphs(show_bounds=True)


if __name__ == "__main__":
    main()

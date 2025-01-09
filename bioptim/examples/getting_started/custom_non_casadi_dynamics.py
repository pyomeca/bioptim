"""
TODO: Explain what is this example about
TODO: All the documentation

This example is similar to the getting_started/pendulum.py example, but the dynamics are computed using a non-casadi 
based model. This is useful when the dynamics are computed using a different library (e.g. TensorFlow, PyTorch, etc.)
"""

import os
from typing import Self

import biorbd
from bioptim import OptimalControlProgram, DynamicsFcn, BoundsList, Dynamics
from bioptim.models.torch.torch_model import TorchModel
import numpy as np
import torch


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


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
                raise NotImplementedError("Batch normalization is not yet implemented")
                layers.append(torch.nn.BatchNorm1d(first_and_hidden_layers_node_count[i + 1]))
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

    def train_me(
        self, training_data: list[torch.Tensor], validation_data: list[torch.Tensor], max_epochs: int = 5
    ) -> None:
        # More details about scheduler in documentation
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._optimizer, mode="min", factor=0.1, patience=20, min_lr=1e-8
        )

        early_stopper = EarlyStopper(patience=20, min_delta=1e-5)
        for i in range(max_epochs):
            self._perform_epoch_training(targets=training_data)
            validation_loss = self._perform_epoch_training(targets=validation_data, only_compute=True)
            scheduler.step(validation_loss)  # Adjust/reduce learning rate

            # Check if the training should stop
            print(f"Validation loss: {validation_loss} (epoch: {i})")
            if early_stopper.early_stop(validation_loss):
                print("Early stopping")
                break

    def save_me(self, path: str) -> None:
        layer_node_count = tuple(
            [model.in_features for model in self._forward_model if isinstance(model, torch.nn.Linear)]
            + [self._forward_model[-1].out_features]
        )

        dropout_probability = tuple([model.p for model in self._forward_model if isinstance(model, torch.nn.Dropout)])
        if len(dropout_probability) == 0:
            dropout_probability = 0
        elif len(dropout_probability) > 1:
            # make sure that the dropout probability is the same for all layers
            if not all(prob == dropout_probability[0] for prob in dropout_probability):
                raise ValueError("Different dropout probabilities for different layers")
            dropout_probability = dropout_probability[0]

        use_batch_norm = any(isinstance(model, torch.nn.BatchNorm1d) for model in self._forward_model)

        dico = {
            "layer_node_count": layer_node_count,
            "dropout_probability": dropout_probability,
            "use_batch_norm": use_batch_norm,
            "state_dict": self.state_dict(),
        }
        torch.save(dico, path)

    @classmethod
    def load_me(cls, path: str) -> Self:
        data = torch.load(path, weights_only=True)
        inputs = {
            "layer_node_count": data["layer_node_count"],
            "dropout_probability": data["dropout_probability"],
            "use_batch_norm": data["use_batch_norm"],
        }
        model = NeuralNetworkModel(**inputs)
        model.load_state_dict(data["state_dict"])
        return model

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

    q = np.array([0, 0])
    qdot = np.array([0, 0])
    tau = np.array([0, 0])
    qddot = torch_model.forward_dynamics()(q, qdot, tau, [], [])
    biorbd_model = biorbd.Model("models/pendulum.bioMod")
    qddot2 = biorbd_model.ForwardDynamics(q, qdot, tau).to_array()
    print(qddot - qddot2)

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
        use_sx=False,
    )


def generate_dataset(biorbd_model: biorbd.Model, data_point_count: int) -> list[torch.Tensor]:
    q_ranges = np.array(
        [[[q_range.min(), q_range.max()] for q_range in segment.QRanges()] for segment in biorbd_model.segments()]
    ).squeeze()
    qdot_ranges = np.array(
        [
            [[qdot_range.min(), qdot_range.max()] for qdot_range in segment.QdotRanges()]
            for segment in biorbd_model.segments()
        ]
    ).squeeze()
    tau_ranges = np.array([-100, 100] * biorbd_model.nbGeneralizedTorque()).reshape(-1, 2)

    q = torch.rand(data_point_count, biorbd_model.nbQ()) * (q_ranges[:, 1] - q_ranges[:, 0]) + q_ranges[:, 0]
    qdot = (
        torch.rand(data_point_count, biorbd_model.nbQdot()) * (qdot_ranges[:, 1] - qdot_ranges[:, 0])
        + qdot_ranges[:, 0]
    )
    tau = (
        torch.rand(data_point_count, biorbd_model.nbGeneralizedTorque()) * (tau_ranges[:, 1] - tau_ranges[:, 0])
        + tau_ranges[:, 0]
    )

    q = q.to(torch.float)
    qdot = qdot.to(torch.float)
    tau = tau.to(torch.float)

    qddot = torch.zeros(data_point_count, biorbd_model.nbQddot())
    for i in range(data_point_count):
        qddot[i, :] = torch.tensor(
            biorbd_model.ForwardDynamics(np.array(q[i, :]), np.array(qdot[i, :]), np.array(tau[i, :])).to_array()
        )

    return [torch.cat((q, qdot, tau), dim=1), qddot]


def main():
    # --- Prepare a predictive model --- #
    force_new_training = False
    biorbd_model = biorbd.Model("models/pendulum.bioMod")
    training_data = generate_dataset(biorbd_model, data_point_count=30000)
    validation_data = generate_dataset(biorbd_model, data_point_count=3000)

    if force_new_training or not os.path.isfile("models/trained_pendulum_model.pt"):
        model = NeuralNetworkModel(layer_node_count=(6, 512, 512, 2), dropout_probability=0.2, use_batch_norm=False)
        model.train_me(training_data, validation_data, max_epochs=300)
        model.save_me("models/trained_pendulum_model.pt")
    else:
        model = NeuralNetworkModel.load_me("models/trained_pendulum_model.pt")

    ocp = prepare_ocp(model=model, final_time=1, n_shooting=40)

    # --- Solve the ocp --- #
    sol = ocp.solve()

    # --- Show the results graph --- #
    # sol.print_cost()
    sol.graphs(show_bounds=True)


if __name__ == "__main__":
    main()

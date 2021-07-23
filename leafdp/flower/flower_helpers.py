import flwr as fl
from flwr.common.typing import Parameters
from flwr.server.strategy import FedAvg
from flwr.server.server import shutdown
from flwr.common import Weights, Parameters, Scalar
import os
from typing import Dict, List, Tuple, Optional, Callable
from collections import OrderedDict
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from opacus import PrivacyEngine
from opacus.privacy_engine import get_noise_multiplier
import numpy as np
import argparse
from tqdm import tqdm
from leafdp.utils.model_utils import get_client_data, make_data_loader, FemnistDataset, get_target_delta
from leafdp.femnist.cnn import FemnistModel

BASEPATH = os.environ["BASEPATH"]


def get_weights(model) -> List[np.ndarray]:
    """Return model parameters as a list of NumPy ndarrays.

    Parameters
    ----------
    model
        The model we want to get the weights from.

    Returns
    -------
    list object
        List of weights
    """
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model: torch.nn.Module, weights: List[np.ndarray]) -> None:
    """Set model parameters from a list of NumPy ndarrays.

    Parameters
    ----------
    model
        The model we want to set the weights.
    weights
        Weights we want to set
    """
    params_dict = zip(model.state_dict().keys(), weights)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def weighted_acc_avg(results: List[Tuple[int, float, Optional[float]]]) -> float:
    """Aggregate evaluation results obtained from multiple clients.

    Parameters
    ----------
    results
        List containing tuples of client and their associated results.
    Returns
    -------
    float object
        The weighted accuracy of all clients.
    """
    num_total_evaluation_examples = sum(num_examples for num_examples, _, _ in results)

    weighted_acc = [num_examples * acc for num_examples, _, acc in results]
    return sum(weighted_acc) / num_total_evaluation_examples


def train_fl(
    parameters: List[np.ndarray],
    return_dict,
    config,
    client_id: str,
    dataset: str,
    batch_size: int = 256,
    virtual_batch_size: int = 256,
    lr: float = 0.0001,
    diff_privacy: bool = False,
    nm: float = 0.9555,
    mgn: float = 0.8583,
    delta: float = 1e-6,
    epsilon: float = 0.0,
    device: str = "cpu",
) -> None:
    """Train weights of sherlock model with pytorch.

    Parameters
    ----------
    parameters
        The model's parameters
    return_dict
        The dict containing return values for multiprocessing
    dataset
        The dataset used.
    client_id
        Client id to get corresponding client data.
    drop_rest
        Whether the rest category for data is kept or not.
    batch_size
        Number of elements per batch
    lr
        Optimizer learning rate
    diff_privacy
        Wether or not differential privacy is used in training
    nm
        Noise multiplier
    mgn
        Max gradient norm
    epsilon
        Optional, target epsilon for privacy engine
    device
        Wether to train on gpu or cpu
    """

    # Instantiate variables
    model = None
    train_loss = 0.0
    train_acc = 0.0
    privacy_results = None
    if diff_privacy:
        # Make a smaller batch size for memory purposes
        n_acc_steps = int(virtual_batch_size / batch_size)
        assert virtual_batch_size % batch_size == 0
    # Get data
    client_data = get_client_data(
        dataset=dataset,
        client_id=client_id,
        cross_silo=config["cross_silo"],
        train=True,
    )
    client_dataset = FemnistDataset(client_data)
    train_loader = make_data_loader(
        client_dataset,
        batch_size=batch_size,
        dp=diff_privacy,
        v_batch_size=virtual_batch_size,
    )
    # Get model builder depending on dataset
    if dataset == "femnist":
        n_classes = 62
        model = FemnistModel(num_classes=n_classes)
    else:
        # Default takes Femnist model
        model = FemnistModel()
    # Set parameters
    if parameters is not None:
        set_weights(model, parameters)
    # Move the model to the device before creating the privacy engine
    model = model.to(device)
    # Create optimizer
    optimizer = Adam(model.parameters(), lr)
    # Define loss function
    criterion = CrossEntropyLoss()
    # Init privacy engine
    privacy_engine = None

    model.train()

    if diff_privacy:
        # Attach the privacy engine to the optimizer
        # sample size is the total nb of elem in the dataset = batch_size*len(train_loader)
        alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
        delta = get_target_delta(len(train_loader)*batch_size)
        sample_rate = batch_size / len(client_dataset)
        # Calculate the theoretical number of batches if we kept original batch_size
        virtual_len = int(len(client_dataset) / virtual_batch_size)
        # If it's the first time the client is sampled we define adaptive
        if not bool(config["times_sampled"]) and config["strategy"] == "hybrid":
            # Set adaptive noise or fixed noise depending on data_size
            adaptive_noise = get_noise_multiplier(
                epsilon,
                delta,
                sample_rate * n_acc_steps,
                # epochs=max(1, int(config["frac_fit"] * config["rounds"])),
                epochs=2,
                alphas=alphas,
            )
            adaptive = nm < adaptive_noise
            return_dict["adaptive"] = adaptive
            print(f"Selected adaptive noise: {adaptive}")
        else:
            adaptive = config["adaptive"]
        privacy_engine = (
            PrivacyEngine(
                model,
                sample_rate=sample_rate * n_acc_steps,
                alphas=alphas,
                noise_multiplier=None,
                max_grad_norm=mgn,
                target_delta=delta,
                target_epsilon=epsilon,
                epochs=(
                    (config["rounds"] + config["times_sampled"] + 1) - config["round"]
                ),
            )
            if adaptive
            else PrivacyEngine(
                model,
                sample_rate=sample_rate * n_acc_steps,
                alphas=alphas,
                noise_multiplier=nm,
                max_grad_norm=mgn,
                target_delta=delta,
            )
        )
        # Load the step so that the real privacy budget is computed
        # Have to multiply the number of steps per number of batches done
        # Careful to use the theoretical number of batches and not the actual one
        state_dict = {"steps": (config["times_sampled"]) * virtual_len}
        privacy_engine.load_state_dict(state_dict)
        privacy_engine.to(device)
        privacy_engine.attach(optimizer)

    for i, (data, label) in enumerate(tqdm(train_loader)):
        # Send the tensors to the correct device
        data = data.to(device)
        label = label.to(device)
        out = model(data)
        loss = criterion(out, label)
        _, pred_ids = out.max(1)
        acc = (pred_ids == label).sum().item() / batch_size
        loss.backward()
        # take a real optimizer step after n_virtual_steps
        if diff_privacy:
            if ((i + 1) % n_acc_steps == 0) or ((i + 1) == len(train_loader)):
                optimizer.step()  # real step
                optimizer.zero_grad()
            else:
                optimizer.virtual_step()  # take a virtual step
        else:
            optimizer.step()
            optimizer.zero_grad()
        # Important! Use the detached loss to get it to prevent storing whole computational graph
        train_loss += (loss.detach().item() - train_loss) / (i + 1)
        train_acc += (acc - train_acc) / (i + 1)

    else:
        print(
            f"Round Results:",
            f"Train Loss: {train_loss}",
            f"Train Accuracy: {train_acc}",
        )
        # print best alpha and epsilon
        if diff_privacy:
            epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(delta)
            print(f"(ε = {epsilon:.2f}, δ = {delta}) for α = {best_alpha}")
            privacy_results = [epsilon, best_alpha]

    # Get the new parameters
    new_parameters = get_weights(model)

    # Prepare return values
    return_dict["parameters"] = new_parameters
    return_dict["data_size"] = len(train_loader)
    return_dict["train_results"] = (train_loss, train_acc)
    return_dict["privacy_results"] = privacy_results


def test_fl(
    parameters: List[np.ndarray],
    dataset: str,
    client_id: str,
    return_dict,
    use_val: bool = False,
    batch_size: int = 256,
    device="cpu",
):
    """Test federated model with pytorch.

    Parameters
    dataset
        The dataset used.
    client_id
        Client id to get corresponding client data.
    parameters
        The model's parameters
    return_dict
        Dict containing return values for multiprocessing
    use_val
        Wether validation set or test set is used
    batch_size
        Number of elements per batch
    device
        Run on gpu or cpu
    """

    model = None
    test_loss = 0.0
    test_acc = 0.0

    # Get data
    client_data = get_client_data(
        dataset=dataset,
        client_id=client_id,
        train=False,
        use_val=use_val,
    )
    client_dataset = FemnistDataset(client_data)
    test_loader = make_data_loader(
        client_dataset,
        batch_size=batch_size,
        dp=False,
    )

    # Get model builder depending on dataset
    if dataset == "femnist":
        n_classes = 62
        model = FemnistModel(num_classes=n_classes)
    else:
        model = None
    # Set parameters
    if parameters is not None:
        set_weights(model, parameters)
    # Move the model to the device before creating the privacy engine
    model = model.to(device)
    # Define loss function
    criterion = CrossEntropyLoss()

    with torch.no_grad():

        model.eval()

        for i, (data, label) in enumerate(tqdm(test_loader)):

            # Send the tensors to the correct device
            data = data.to(device)
            label = label.to(device)
            out = model(data)
            loss = criterion(out, label)
            _, pred_ids = out.max(1)
            acc = (pred_ids == label).sum().item() / batch_size
            # Important! Use the detached loss to get it to prevent storing whole computational graph
            test_loss += (loss.detach().item() - test_loss) / (i + 1)
            test_acc += (acc - test_acc) / (i + 1)

    print(
        f"Test Loss: {test_loss}",
        f"Test Accuracy: {test_acc}",
    )

    # Prepare return values
    return_dict["data_size"] = len(test_loader)
    return_dict["test_results"] = (test_loss, test_acc)


class FedAvgDp(FedAvg):
    """This class implements the FedAvg strategy for Differential Privacy context."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        fraction_fit: float = 0.1,
        fraction_eval: float = 0.1,
        min_fit_clients: int = 2,
        min_eval_clients: int = 2,
        min_available_clients: int = 2,
        eval_fn: Optional[Callable[[Weights], Optional[Tuple[float, float]]]] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
    ) -> None:
        FedAvg.__init__(
            self,
            fraction_fit,
            fraction_eval,
            min_fit_clients,
            min_eval_clients,
            min_available_clients,
            eval_fn,
            on_fit_config_fn,
            on_evaluate_config_fn,
            accept_failures,
            initial_parameters,
        )
        # Keep track of the maximum possible privacy budget
        self.max_epsilon = 0.0
        # Measure the number of clients adopting the adaptive strategy
        self.nb_adaptive = 0

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Optional[fl.common.Weights]:
        """Get the privacy budget"""
        if not results:
            return None
        if "epsilon" in results[0][1].metrics.keys():
            # Get the privacy budget of each client
            accepted_results = []
            disconnect_clients = []
            epsilons = []
            examples = []
            for c, r in results:
                if "adaptive" in r.metrics.keys():
                    self.nb_adaptive += 1
                if r.metrics["accept"]:
                    accepted_results.append([c, r])
                    epsilons.append(r.metrics["epsilon"])
                    examples.append(r.num_examples)
                else:
                    disconnect_clients.append(c)
            # Disconnect clients if needed
            if disconnect_clients:
                shutdown(disconnect_clients)
            results = accepted_results
            if epsilons:
                self.max_epsilon = max(self.max_epsilon, max(epsilons))
            print(f"Privacy budget ε at round {rnd}: {self.max_epsilon}")
        # Call aggregate_evaluate from base class (FedAvg)
        return super().aggregate_fit(rnd, results, failures)

    def configure_evaluate(self, rnd, parameters, client_manager):
        """Configure the next round of evaluation. Returns None since evaluation is made server side.
        You could comment this method if you want to keep the same behaviour as FedAvg."""
        if client_manager.num_available() < self.min_fit_clients:
            print(
                f"Total number of adaptive clients at round {rnd}: {self.nb_adaptive}"
            )
            print(
                f"{client_manager.num_available()} client(s) available(s), waiting for {self.min_available_clients} availables to continue."
            )
        # rnd -1 is a special round for last evaluation when all rounds are over
        if rnd == -1 and self.nb_adaptive > 0:
            print(f"Total number of adaptive clients: {self.nb_adaptive}")
        return None

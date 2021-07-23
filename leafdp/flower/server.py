import flwr as fl
import os
import argparse

from datetime import datetime
from typing import Tuple, Optional
import torch
from torchvision.transforms import transforms
import numpy as np

from leafdp.utils import model_utils
from leafdp.femnist.cnn import FemnistModel
from leafdp.vanilla.train import test_model
from leafdp.flower import flower_helpers

# Needs this if we want to launch grpc client
if os.environ.get("https_proxy"):
    del os.environ["https_proxy"]
if os.environ.get("http_proxy"):
    del os.environ["http_proxy"]

BASEPATH = os.environ["BASEPATH"]
# pylint: disable=no-member
# DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE: str = "cpu"  # Server is always on CPU since it's only there for testing
# pylint: enable=no-member
# The batch_size
batch_size = None
# The current model
model = None
# The model name
model_name = ""
# Whether or not we already got results for this model
saved_once = False
# Global variable to keep track of the best model so far
best_model = {
    "loss": np.Infinity,
    "acc": 0,
}
# test loader for server evaluation
test_loader = None
# test loader for each silo
per_silo_test_loader = []
# rounds variable
rounds = None


def get_eval_fn(model: torch.nn.Module):
    """Get the evaluation function for server side.

    Parameters
    ----------
    model
        The model we want to evaluate.

    Returns
    -------
    evaluate
        The evaluation function
    """

    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
        """Evaluation function for server side.

        Parameters
        ----------
        weights
            Updated model weights to evaluate.

        Returns
        -------
        loss
            Loss on the test set.
        accuracy
            Accuracy on the test set.
        """

        # Load model
        flower_helpers.set_weights(model, weights)
        model.to(DEVICE)

        loss, accuracy, report = test_model(
            model,
            test_loader,
            batch_size=batch_size,
            device=DEVICE,
            report=True,
        )
        # Compute loss, accuracy and report for each silo
        silo_results = []
        for loader in per_silo_test_loader:
            silo_loss, silo_acc, silo_report = test_model(
                model,
                loader,
                batch_size=batch_size,
                device=DEVICE,
                report=True,
            )
            silo_results.append(
                [
                    [silo_loss],
                    [silo_acc],
                    [silo_report["macro avg"]["f1-score"]],
                    [silo_report["weighted avg"]["f1-score"]],
                    [silo_report["weighted avg"]["recall"]],
                    [silo_report["macro avg"]["recall"]],
                ]
            )
        save_path = f"{BASEPATH}leafdp/flower/models/server/{model_name}"
        if loss < best_model["loss"] and accuracy > best_model["acc"]:
            best_model["loss"] = loss
            best_model["acc"] = accuracy
            # Save model
            print(f"Saving model: {dt_string}, loss={loss}, acc={accuracy}")
            model_utils.save_model(model, f"{save_path}.pth")
        # Save results
        print(f"Saving results")
        results = torch.FloatTensor(
            [
                [loss],
                [accuracy],
                [report["macro avg"]["f1-score"]],
                [report["weighted avg"]["f1-score"]],
                [report["weighted avg"]["recall"]],
                [report["macro avg"]["recall"]],
            ]
        )
        silo_results = torch.FloatTensor(silo_results)
        global saved_once
        if not saved_once:
            # Saving global results
            torch.save(results, f"{save_path}.pt")
            # Saving per silo results
            torch.save(silo_results, f"{save_path}_per_silo_results.pt")
            saved_once = True
        else:
            # Loading and saving global results
            prev_results = torch.load(f"{save_path}.pt")
            new_results = torch.cat((prev_results, results), 1)
            torch.save(new_results, f"{save_path}.pt")
            # Loading and saving per silo results
            prev_silo_results = torch.load(f"{save_path}_per_silo_results.pt")
            new_silo_results = torch.cat((prev_silo_results, silo_results), 2)
            torch.save(new_silo_results, f"{save_path}_per_silo_results.pt")
        return float(loss), {"accuracy": float(accuracy)}

    return evaluate


def fit_config(rnd: int):
    """Return training configuration dict for each round.
    We want to get the round number to keep track of the privacy budget for each client.
    """
    return {
        "round": rnd,
        "rounds": rounds,
        "frac_fit": frac_fit,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", type=int, default=42, help="Seed for generating random ids."
    )
    parser.add_argument(
        "-r", type=int, default=3, help="Number of rounds for the federated training"
    )
    parser.add_argument(
        "-nbc",
        type=int,
        default=2,
        help="Number of clients to keep track of dataset share",
    )
    parser.add_argument(
        "-d",
        type=str,
        default="femnist",
        help="The dataset we want to train on.",
    )
    parser.add_argument("-cs", type=int, default=1, help="Cross silo dataset or not.")
    parser.add_argument("-b", type=int, default=256, help="Batch size")
    parser.add_argument(
        "-fc",
        type=int,
        default=2,
        help="Min fit clients, min number of clients to be sampled next round",
    )
    parser.add_argument(
        "-ac",
        type=int,
        default=2,
        help="Min available clients, min number of clients that need to connect to the server before training round can start",
    )
    parser.add_argument(
        "--centralized",
        type=int,
        default=1,
        help="Whether evaluation is made by server or the client",
    )
    parser.add_argument(
        "-dp",
        type=int,
        default=0,
        help="Whether Differential Privacy is used or not.",
    )
    parser.add_argument(
        "--tepsilon",
        type=float,
        default=0.0,
        help="Target epsilon for the privacy budget.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="vanilla",
        help="Strategy to get the desired epsilon. Either 'hybrid', 'fix' or 'adaptive'",
    )
    args = parser.parse_args()
    seed = int(args.s)
    rounds = int(args.r)
    nbc = int(args.nbc)
    dataset = str(args.d)
    cross_silo = bool(args.cs)
    batch_size = int(args.b)
    fc = int(args.fc)
    ac = int(args.ac)
    centralized = bool(args.centralized)
    dp = bool(args.dp)
    target_epsilon = float(args.tepsilon)
    strat = str(args.strategy)
    mec = 5 if batch_size == 256 else 3  # Max number of evaluated clients
    frac_fit = fc / nbc
    # Determine the fraction of clients we want to evaluate on
    frac_eval = fc if mec / fc > 1 else mec / fc
    model = None
    # Get the date and format it
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d-%H%M%S")
    # Set the model name
    model_name = f"server-{dataset}_dp-{dp}_{strat}_eps-{target_epsilon}_r-{rounds}_nbc-{nbc}_fc-{fc}_ac-{ac}_b-{batch_size}_{dt_string}"
    # Generate indices based on the seed.
    if cross_silo:
        indices = model_utils.generate_indices(nbc, total=201, seed=seed)
    else:
        indices = model_utils.generate_indices(nbc, total=3551, seed=seed)
        # Get list of all clients
        client_ids = model_utils.get_clients_ids(dataset)
    all_data = {
        "x": [],
        "y": [],
    }

    for i in indices:
        client_id = i if cross_silo else client_ids[i]
        data = model_utils.get_client_data(
            dataset=dataset,
            cross_silo=cross_silo,
            client_id=client_id,
            train=False,
        )
        all_data["x"].extend(data["x"])
        all_data["y"].extend(data["y"])
        silo_dataset = model_utils.FemnistDataset(data)
        per_silo_test_loader.append(
            model_utils.make_data_loader(
                silo_dataset,
                batch_size=batch_size,
                dp=False,
            )
        )
    # Create dataset
    test_dataset = model_utils.FemnistDataset(all_data)  # , transform=transform
    # Get data
    test_loader = model_utils.make_data_loader(
        test_dataset,
        batch_size=batch_size,
        dp=False,
    )
    if dataset == "femnist":
        input_dim = 28
        n_classes = 62
        model = FemnistModel(input_dim=28, num_classes=n_classes)
    else:
        model = None

    # Send model to device
    model.to(DEVICE)
    # Defines strategy for the flower server
    # server version:
    if centralized:
        eval_fn = get_eval_fn(model)
        init_weights = flower_helpers.get_weights(model)
        init_param = fl.common.weights_to_parameters(init_weights)
    else:  # client version
        eval_fn = None
        init_param = None

    # strategy = fl.server.strategy.FedAvg(
    strategy = flower_helpers.FedAvgDp(
        fraction_fit=frac_fit,
        fraction_eval=frac_eval,
        min_fit_clients=fc,
        min_eval_clients=mec,
        min_available_clients=ac,
        eval_fn=eval_fn,
        on_fit_config_fn=fit_config,
        initial_parameters=init_param,
    )
    print(f"Indices for clients: {indices}")
    fl.server.start_server(
        "[::]:8080", config={"num_rounds": rounds}, strategy=strategy
    )

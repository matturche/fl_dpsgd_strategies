from typing import Dict, List, Tuple

import torch
import torch.multiprocessing as mp

from leafdp.utils import model_utils
from leafdp.flower import flower_helpers

import argparse
from datetime import datetime
import numpy as np

import flwr as fl

import os

# Needs this if we want to launch grpc client
if os.environ.get("https_proxy"):
    del os.environ["https_proxy"]
if os.environ.get("http_proxy"):
    del os.environ["http_proxy"]

BASEPATH = os.environ["BASEPATH"]

DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LeafClient(fl.client.NumPyClient):
    """Flower client for Leaf using PyTorch."""

    def __init__(
        self,
        dataset: str,
        cross_silo: bool,
        client_id: str,
        batch_size: int = 256,
        virtual_batch_size: int = 256,
        lr: float = 0.0001,
        diff_privacy: bool = False,
        target_epsilon: float = 0.0,
        nm: float = 0.9555,
        mgn: float = 0.8583,
        centralized: bool = True,
        strategy: str = "hybrid",
    ) -> None:
        """Flower client immplementation.

        Args
        ----
        dataset
            Which dataset the client should load.
        cross_silo
            Whether we are in a cross silo setting or not.
        client_id
            Client id to get data.
        diff_privacy
            Wether Differential Privacy is applied or not.
        target_epsilon
            Determine the limit for the privacy budget.
        centralized
            Whether the model is evaluated on the server side or client's.
        adaptive
            Wether the noise multiplier is adaptive or fixed.
        """
        self.dataset = dataset
        self.cross_silo = cross_silo
        self.client_id = client_id
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size
        self.lr = lr
        self.diff_privacy = diff_privacy
        self.target_epsilon = target_epsilon
        self.nm = nm
        self.mgn = mgn
        self.centralized = centralized
        self.results = []
        self.parameters = None
        # Variable keeping track of how many times a client was sampled
        self.times_sampled = 0
        self.strategy = strategy
        self.adaptive = strategy == "adaptive"

        if not self.centralized:
            # Prepare multiprocess
            manager = mp.Manager()
            return_dict = manager.dict()
            # Create the multiprocess
            p = mp.Process(
                target=flower_helpers.test_fl,
                args=(
                    self.parameters,
                    self.dataset,
                    self.client_id,
                    return_dict,
                    False,
                    self.batch_size,
                    DEVICE,
                ),
            )
            # Start the process
            p.start()
            # Wait for it to end
            p.join()
            try:
                p.close()
            except ValueError as e:
                print(f"Couldn't close the testing process: {e}.")
            # Get the return values
            test_results = return_dict["test_results"]
            # del everything related to multiprocess to free memory
            del (manager, return_dict, p)
            self.loss = test_results[0]
            self.accuracy = test_results[1]
        else:
            self.loss = 0.0
            self.accuracy = 0.0

    def get_parameters(self) -> List[np.ndarray]:
        # Return model parameters as a list of NumPy ndarrays
        return self.parameters

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        self.parameters = parameters

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int]:
        """Set model parameters, train model, return updated model parameters.

        Parameters
        ----------
        parameters
            List of parameters to update the model.
        config
            Configuration for fitting.

        Returns
        -------
        List object
            Newly trained parameters.
        Int object
            Size of client's training sample.
        Dict object
            Optional results dict, for example including metrics.
        """
        config["cross_silo"] = self.cross_silo
        config["times_sampled"] = self.times_sampled
        config["strategy"] = self.strategy
        config["adaptive"] = self.adaptive
        self.set_parameters(parameters)
        # Prepare multiprocess
        manager = mp.Manager()
        return_dict = manager.dict()
        # Create the multiprocess
        p = mp.Process(
            target=flower_helpers.train_fl,
            args=(
                self.parameters,
                return_dict,
                config,
                self.client_id,
                self.dataset,
                self.batch_size,
                self.virtual_batch_size,
                self.lr,
                self.diff_privacy,
                self.nm,
                self.mgn,
                1e-6,
                self.target_epsilon,
                DEVICE,
            ),
        )
        # Start the process
        p.start()
        # Wait for it to end
        p.join()
        try:
            p.close()
        except ValueError as e:
            print(f"Couldn't close the training process: {e}.")
        # Get the return values
        new_parameters = return_dict["parameters"]
        data_size = return_dict["data_size"]
        train_results = return_dict["train_results"]
        # Init metrics
        metrics = {}
        if self.diff_privacy:
            epsilon = return_dict["privacy_results"][0]
            # Hybryd approach for adaptive noise and fix noise
            if not bool(self.times_sampled) and self.strategy == "hybrid":
                self.adaptive = return_dict["adaptive"]
                if self.adaptive:
                    metrics["adaptive"] = 1
            accept = True
            if epsilon > self.target_epsilon + 0.3:  # leaving some room
                accept = False
                print(
                    f"Epsilon over target value ({self.target_epsilon}), disconnecting client."
                )
                # Overrides the new parameters with the ones received
                new_parameters = parameters
            metrics.update(
                {
                    "epsilon": epsilon,
                    "alpha": return_dict["privacy_results"][1],
                    "accept": accept,
                }
            )
            privacy_results = [metrics["epsilon"], metrics["alpha"]]
        else:
            privacy_results = []
        # del everything related to multiprocess to free memory
        del (manager, return_dict, p)

        self.times_sampled += 1
        self.set_parameters(new_parameters)
        self.results.append([train_results, privacy_results])
        return (
            new_parameters,
            data_size,
            metrics,
        )

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[int, float, float]:
        """Set model parameters, evaluate model on local test dataset, return result.

        Parameters
        ----------
        parameters
            List of parameters to evaluate
        config
            Optional configuration

        Returns
        -------
        List object
            Newly trained parameters.
        Int object
            Size of client's training sample.
        Dict object
            Optional results dict, for example including metrics.
        """

        # # Skip final client evaluation if only evaluating on server side
        # if self.centralized:
        #     return 1.0, 1, {"accuracy": 1.0}
        self.set_parameters(parameters)
        # Prepare multiprocess
        manager = mp.Manager()
        return_dict = manager.dict()
        # Create the multiprocess
        p = mp.Process(
            target=flower_helpers.test_fl,
            args=(
                self.parameters,
                self.dataset,
                self.client_id,
                return_dict,
                False,
                self.batch_size,
                DEVICE,
            ),
        )
        # Start the process
        p.start()
        # Wait for it to end
        p.join()
        try:
            p.close()
        except ValueError as e:
            print(f"Couldn't close the testing process: {e}.")
        # Get the return values
        data_size = return_dict["data_size"]
        test_results = return_dict["test_results"]
        # del everything related to multiprocess to free memory
        del (manager, return_dict, p)
        # save model localy if it's better than the current model
        if test_results[0] < self.loss and test_results[1] > self.accuracy:
            self.loss = test_results[0]
            self.accuracy = test_results[1]
            self.save_model()
        return (
            float(test_results[0]),
            data_size,
            {"accuracy": float(test_results[1])},
        )

    def save_model(self):
        # Get the date and format it
        now = datetime.now()
        dt_string = now.strftime("%Y%m%d-%H%M%S")
        save_dir = f"{BASEPATH}leafdp/femnist/models/clients/{self.client_id}/"
        if not os.path.exists(save_dir):
            try:
                os.mkdir(save_dir)
            except OSError:
                print("Creation of the directory failed")
            else:
                print("Successfully created the directory")
        save_path = f"{save_dir}{self.strategy}_dp-{self.diff_privacy}_{dt_string}.pth"
        model_utils.save_model(self.model, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", type=int, default=0, help="Client number for determining id."
    )
    parser.add_argument(
        "-s", type=int, default=42, help="Seed for generating random ids."
    )
    parser.add_argument(
        "-nbc",
        type=int,
        default=2,
        help="Number of clients to generate ids",
    )
    parser.add_argument(
        "-d",
        type=str,
        default="femnist",
        help="The dataset we want to train on.",
    )
    parser.add_argument("-cs", type=int, default=1, help="Cross silo dataset or not.")
    parser.add_argument("-b", type=int, default=256, help="Batch size")
    parser.add_argument("-vb", type=int, default=256, help="Virtual batch size")
    parser.add_argument(
        "-lr", type=float, default=0.0001, help="Learning rate for the optimizer"
    )
    parser.add_argument(
        "-dp", type=int, default=0, help="Use Differential Privacy or not"
    )
    parser.add_argument(
        "-nm", type=float, default=1.2, help="Noise multiplier for Private Engine."
    )
    parser.add_argument(
        "-mgn", type=float, default=1.0, help="Max grad norm for Private Engine."
    )
    parser.add_argument(
        "--centralized",
        type=int,
        default=1,
        help="Whether evaluation is made by server or the client",
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
        default="hybrid",
        help="Strategy to get the desired epsilon. Either 'hybrid', 'fix' or 'adaptive'",
    )
    args = parser.parse_args()
    client_share = int(args.c)
    seed = int(args.s)
    nbc = int(args.nbc)
    dataset = str(args.d)
    cross_silo = bool(args.cs)
    batch_size = int(args.b)
    virtual_batch_size = int(args.vb)
    lr = float(args.lr)
    dp = bool(args.dp)
    nm = float(args.nm)
    mgn = float(args.mgn)
    centralized = bool(args.centralized)
    target_epsilon = float(args.tepsilon)
    strat = str(args.strategy)

    # Generate ids
    if cross_silo:
        indices = model_utils.generate_indices(nbc, total=201, seed=seed)
        client_id = indices[client_share]
    else:
        indices = model_utils.generate_indices(nbc, total=3551, seed=seed)
        # Get list of all clients
        client_ids = model_utils.get_clients_ids(dataset)
        # Get id
        client_id = client_ids[indices[client_share]]
    # Set explicitely the spawn method for Python under 3.8 compatibility
    mp.set_start_method("spawn")

    # Start client
    client = LeafClient(
        dataset,
        cross_silo,
        client_id,
        batch_size=batch_size,
        virtual_batch_size=virtual_batch_size,
        lr=lr,
        diff_privacy=dp,
        target_epsilon=target_epsilon,
        nm=nm,
        mgn=mgn,
        centralized=True,
        strategy=strat,
    )
    print(f"Indices for clients: {indices}")
    fl.client.start_numpy_client("[::]:8080", client=client)

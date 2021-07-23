import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch import optim
from torchvision.datasets import EMNIST, CIFAR10, KMNIST
from torchvision.transforms import transforms
from leafdp.utils import model_utils
from leafdp.femnist.cnn import FemnistModel
from leafdp.vanilla import train
from datetime import datetime
import os

# from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import argparse

BASEPATH = os.environ["BASEPATH"]

DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def find_hyperparameters(
    config, model_builder, train_loader, test_loader, checkpoint_dir=None
):
    """Function used by ray to find hyperparameters given config dict.

    Parameters
    ----------
    config : dict
        Config dict containing parameters for the model
    model_builder : func
        Function building the model
    train_loader : DataLoader
        Train loader
    test_loader : DataLoader
        Test loader
    checkpoint_dir : str, optional
        Checkpoint path for the model, by default None
    """
    # Create model
    dp_model = model_builder(
        input_dim=config["input_dim"], num_classes=config["n_classes"]
    )

    # Move the model to the device before creating the privacy engine
    dp_model = dp_model.to(DEVICE)

    # Define loss function
    criterion = CrossEntropyLoss()
    # Get optimizer
    # optimizer = optim.SGD(dp_model.parameters(), config["lr"])
    optimizer = optim.Adam(dp_model.parameters(), config["lr"])
    # opt_name = str(optimizer).split(" ")[0]
    train.train_model(
        dp_model,
        optimizer,
        train_loader,
        test_loader,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        criterion=criterion,
        diff_privacy=True,
        dataset_len=config["train_len"],
        virtual_batch_size=config["virtual_batch_size"],
        delta=config["delta"],
        nm=config["nm"],
        mgn=config["mgn"],
        test=True,
        device=DEVICE,
        tune_model=True,
    )

    print("Finished Training!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        type=str,
        default="emnist",
        help="The dataset we want to train on.",
    )
    parser.add_argument(
        "-dp",
        type=int,
        default=1,
        help="Whether differential privacy is used or not.",
    )
    parser.add_argument("-b", type=int, default=64, help="Batch size")
    parser.add_argument(
        "-vb",
        type=int,
        default=256,
        help="Virtual batch size for differential privacy.",
    )
    parser.add_argument(
        "-cpu",
        type=float,
        default=1.0,
        help="Number of cpus per trials, defaults to 1.",
    )
    parser.add_argument(
        "-gpu",
        type=float,
        default=0.0,
        help="Number of gpus per trials, defaults to 0.",
    )
    parser.add_argument(
        "-e",
        type=int,
        default=50,
        help="Max number of epochs per trials, defaults to 50.",
    )
    parser.add_argument(
        "-s",
        type=int,
        default=50,
        help="Number of samples, defaults to 50.",
    )
    args = parser.parse_args()
    dataset = str(args.d)
    dp = bool(args.dp)
    batch_size = int(args.b)
    virtual_batch_size = int(args.vb)
    cpus = float(args.cpu)
    gpus = float(args.gpu)
    num_samples = int(args.s)
    epochs = int(args.e)

    if dataset == "emnist":
        # Set transforms
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        # Get dataloaders
        train_set = EMNIST(
            ".", split="balanced", train=True, download=True, transform=transform
        )
        train_len = len(train_set)
        print(f"len dataset : {train_len}")
        test_val_set = EMNIST(
            ".", split="balanced", train=False, download=True, transform=transform
        )
        test_share = int(0.8 * len(test_val_set))
        val_share = len(test_val_set) - test_share
        test_set, val_set = torch.utils.data.random_split(
            test_val_set, [test_share, val_share]
        )
        train_loader = model_utils.make_data_loader(
            train_set, batch_size, dp, virtual_batch_size
        )
        test_loader = DataLoader(test_set, batch_size=virtual_batch_size)
        input_dim = 28
        n_classes = 62
        model_builder = FemnistModel
    if dataset == "mnist":
        # Set transforms
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        # Get dataloaders
        train_set = EMNIST(
            ".", split="mnist", train=True, download=True, transform=transform
        )
        train_len = len(train_set)
        print(f"len dataset : {train_len}")
        test_val_set = EMNIST(
            ".", split="mnist", train=False, download=True, transform=transform
        )
        test_share = int(0.8 * len(test_val_set))
        val_share = len(test_val_set) - test_share
        test_set, val_set = torch.utils.data.random_split(
            test_val_set, [test_share, val_share]
        )
        train_loader = model_utils.make_data_loader(
            train_set, batch_size, dp, virtual_batch_size
        )
        test_loader = DataLoader(test_set, batch_size=virtual_batch_size)
        input_dim = 28
        n_classes = 10
        model_builder = FemnistModel
    if dataset == "cifar10":
        # Set transforms
        transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ]
        )
        # Get dataloaders
        train_set = CIFAR10(".", train=True, download=True, transform=transform)
        train_len = len(train_set)
        print(f"len dataset : {train_len}")
        test_val_set = CIFAR10(".", train=False, download=True, transform=transform)
        test_share = int(0.8 * len(test_val_set))
        val_share = len(test_val_set) - test_share
        test_set, val_set = torch.utils.data.random_split(
            test_val_set, [test_share, val_share]
        )
        train_loader = model_utils.make_data_loader(
            train_set, batch_size, dp, virtual_batch_size
        )
        test_loader = DataLoader(test_set, batch_size=virtual_batch_size)
        input_dim = 32
        n_classes = 10
        model_builder = FemnistModel
    if dataset == "kmnist":
        # Set transforms
        transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ]
        )
        # Get dataloaders
        train_set = KMNIST(".", train=True, download=True, transform=transform)
        train_len = len(train_set)
        print(f"len dataset : {train_len}")
        test_val_set = KMNIST(".", train=False, download=True, transform=transform)
        test_share = int(0.8 * len(test_val_set))
        val_share = len(test_val_set) - test_share
        test_set, val_set = torch.utils.data.random_split(
            test_val_set, [test_share, val_share]
        )
        train_loader = model_utils.make_data_loader(
            train_set, batch_size, dp, virtual_batch_size
        )
        test_loader = DataLoader(test_set, batch_size=virtual_batch_size)
        input_dim = 32
        n_classes = 10
        model_builder = FemnistModel
    else:
        exit

    # Get the date and format it
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d-%H%M%S")

    checkpoint_dir = f"{BASEPATH}models_dp/vanilla/hyperparam_finding/"

    config = {
        "dp": dp,
        "input_dim": input_dim,
        "n_classes": n_classes,
        "train_len": train_len,
        "batch_size": batch_size,
        "epochs": epochs,
        "virtual_batch_size": virtual_batch_size,
        "delta": model_utils.get_target_delta(train_len),
        "lr": tune.loguniform(1e-4, 1e-2),
        "nm": tune.loguniform(0.7, 1.3),
        "mgn": tune.loguniform(0.8, 1.5),
    }

    scheduler = ASHAScheduler(
        metric="loss", mode="min", max_t=50, grace_period=1, reduction_factor=2
    )

    reporter = CLIReporter(metric_columns=["loss", "accuracy", "training_iteration"])

    result = tune.run(
        tune.with_parameters(
            find_hyperparameters,
            model_builder=model_builder,
            train_loader=train_loader,
            test_loader=test_loader,
        ),
        resources_per_trial={"cpu": cpus, "gpu": gpus},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        checkpoint_at_end=False,
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final test loss: {}".format(best_trial.last_result["loss"]))
    print(
        "Best trial final test accuracy: {}".format(best_trial.last_result["accuracy"])
    )

    best_trained_model = model_builder(input_dim=input_dim, num_classes=n_classes)
    best_trained_model.to(DEVICE)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(
        os.path.join(best_checkpoint_dir, "checkpoint")
    )
    best_trained_model.load_state_dict(model_state)

    # Get model name
    model_name = f"{dataset}-model_dp-True_b-{best_trial.config['batch_size']}_e-50_Adam-lr-{best_trial.config['lr']}_nm-{best_trial.config['nm']}_mgn-{best_trial.config['mgn']}_{dt_string}"
    # Set savepath
    save_path = f"{BASEPATH}leafdp/vanilla/hyperparam_finding/{model_name}.pth"
    # Save model
    checkpoint = {
        "num_classes": n_classes,
        "input_dim": input_dim,
        "state_dict": model_state,
    }
    model_utils.save_model(best_trained_model, save_path, checkpoint=checkpoint)
    # Load validation dataset
    val_loader = DataLoader(val_set, batch_size=best_trial.config["batch_size"])
    # Validate model
    print("Validation of best model found:")
    train.test_model(
        best_trained_model,
        val_loader,
        batch_size=best_trial.config["batch_size"],
        device=DEVICE,
    )

from tqdm import tqdm
import argparse
import json
import numpy as np
from sklearn.metrics import classification_report
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch import optim
from torchvision.datasets import EMNIST, CIFAR10, KMNIST
from torchvision.transforms import transforms
from opacus import PrivacyEngine
from leafdp.utils import model_utils
from leafdp.femnist.cnn import FemnistModel
from ray import tune

from datetime import datetime
import os

BASEPATH = os.environ["BASEPATH"]


def train_model(
    model: torch.nn.Module,
    optimizer,
    train_loader: DataLoader,
    test_loader: DataLoader = None,
    epochs: int = 100,
    batch_size: int = 256,
    criterion=CrossEntropyLoss(),
    diff_privacy: bool = False,
    dataset_len: int = 0,
    virtual_batch_size: int = 0,
    delta: float = 1e-6,
    nm: float = 1.0,
    mgn: float = 1.0,
    target_epsilon: float = None,
    test: bool = True,
    savepath: str = None,
    checkpoint: dict = None,
    device: str = "cpu",
    tune_model: bool = False,
):
    """Train weights of sherlock model with pytorch.

    Parameters
    ----------
    model
        The model to train
    train_loader
        Data generator for training
    test_loader
        Data generator for testing
    optimizer
        Optimizer whith which to train the model
    epochs
        Number of times entire dataset is used
    batch_size
        Number of elements per batch
    criterion
        The loss function
    diff_privacy
        Wether or not differential privacy is used in training
    virtual_batch_size
        True batch size when using dp.
    delta
        Privacy parameter
    nm
        Noise multiplier
    mgn
        Max Grad Norm
    target_epsilon
        The value of epsilon privacy we want to attain
    test
        Wether or not to test the model after every epoch
    savepath
        Path to save the model to after training
    checkpoint
        Checkpoint to save model
    device
        Wether to train on gpu or cpu
    Returns
    -------
    train_losses, train_accs
        Tuple containing list of train losses and train accuracies
    test_losses, test_accs
        Same as training but for testing
    """

    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    epsilons = []
    best_alphas = []

    privacy_engine = None
    if diff_privacy:
        n_acc_steps = int(virtual_batch_size / batch_size)
        print(f"Number of accumulated steps: {n_acc_steps}")
        alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
        # delta = get_target_delta(len(train_loader)*batch_size)
        sample_rate = batch_size / dataset_len
        print(f"Sample_rate : {sample_rate}")
        # Calculate the theoretical number of batches if we kept original batch_size
        # virtual_len = int(dataset_len / virtual_batch_size)
        print(
            "Values used for the privacy Engine:",
            f"sample_rate: {sample_rate} |",
            f"nm: {nm} |",
            f"mgn: {mgn} |",
            f"delta: {delta} |",
            f"target_epsilon: {target_epsilon} |",
            f"epochs: {epochs} |",
        )
        privacy_engine = (
            PrivacyEngine(
                model,
                sample_rate=sample_rate * n_acc_steps,
                alphas=alphas,
                noise_multiplier=None,
                max_grad_norm=mgn,
                target_delta=delta,
                target_epsilon=target_epsilon,
                epochs=epochs,
            )
            if target_epsilon is not None
            else PrivacyEngine(
                model,
                sample_rate=sample_rate * n_acc_steps,
                alphas=alphas,
                noise_multiplier=nm,
                max_grad_norm=mgn,
                target_delta=delta,
                epochs=epochs,
            )
        )
        privacy_engine.to(device)
        privacy_engine.attach(optimizer)

    for e in range(epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        for i, (data, label) in enumerate(tqdm(train_loader)):

            # # If the batch size contains only one element, uncomment if the
            # # drop_last param in the dataloader was set to False
            # if label.shape[0] == 1:
            #     data = {key: torch.cat((data[key], data[key]), 0) for key in data}
            #     label = torch.cat((label, label), 0)
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
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            print(
                f"Epoch: {e+1}/{epochs}",
                f"Train Loss: {train_loss}",
                f"Train Accuracy: {train_acc}",
            )
            if test:
                test_loss, test_acc, report = test_model(
                    model,
                    test_loader,
                    virtual_batch_size,
                    criterion,
                    device=device,
                    report=True,
                )
                test_losses.append(test_loss)
                test_accs.append(test_acc)
            # print best alpha and epsilon
            if diff_privacy:
                epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(delta)
                print(f"(ε = {epsilon:.2f}, δ = {delta}) for α = {best_alpha}")
                epsilons.append(epsilon)
                best_alphas.append(best_alpha)
            if tune_model:
                with tune.checkpoint_dir(e) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save((model.state_dict(), optimizer.state_dict()), path)

                tune.report(loss=test_loss, accuracy=test_acc)

    if savepath is not None:
        model_utils.save_model(model, savepath, checkpoint)
    # return (train_losses, train_accs, epsilons, best_alphas), (test_losses, test_accs, report)
    return {
        "train_losses": train_losses,
        "train_accs": train_accs,
        "test_losses": test_losses,
        "test_accs": test_accs,
        "report": report,
        "epsilons": epsilons,
        "alphas": best_alphas,
    }


def test_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    batch_size: int = 256,
    criterion=CrossEntropyLoss(),
    device="cpu",
    report=False,
):
    """Test sherlock model with pytorch.

    Parameters
    ----------
    model
        The model to test
    test_loader
        Data generator for testing
    batch_size
        Number of elements per batch
    criterion
        The loss function
    device
        Run on gpu or cpu
    report
        Wether or not we want a classification report
    Returns
    -------
    total_loss
        Average loss on the entire test dataset
    test_acc
        Average accuracy on the entire test dataset
    class_report
        Dictionnary containing different metrics
    """

    test_loss = 0.0
    test_acc = 0.0
    y_pred, y_true = [], []

    with torch.no_grad():

        model.eval()
        for i, (data, label) in enumerate(tqdm(test_loader)):
            # Send the tensors to the correct device
            # Tensor approach
            data = data.to(device)
            label = label.to(device)
            # Prediction
            out = model(data)
            y_pred.extend(out.cpu().numpy())
            y_true.extend(label.cpu().numpy())
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

    # Compute report
    if report:
        y_pred = np.argmax(y_pred, axis=1)
        class_report = classification_report(
            y_true, y_pred, output_dict=True, labels=np.unique(y_pred)
        )
        print(f"Macro avg: {class_report['macro avg']}")
        print(f"Weighted avg: {class_report['weighted avg']}")
        return test_loss, test_acc, class_report

    return test_loss, test_acc


if __name__ == "__main__":
    DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BASEPATH = os.environ["BASEPATH"]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        type=str,
        default="emnist",
        help="Which dataset to use for training.",
    )
    parser.add_argument(
        "-dp", type=int, default=0, help="Use Differential Privacy or not"
    )
    parser.add_argument("-b", type=int, default=256, help="Batch size")
    parser.add_argument(
        "-vb",
        type=int,
        default=256,
        help="Virtual batch size for differential privacy.",
    )
    parser.add_argument(
        "-e", type=int, default=100, help="Number of epochs for training."
    )
    parser.add_argument(
        "-lr", type=float, default=0.0001, help="Learning rate for the optimizer"
    )
    parser.add_argument(
        "-nm", type=float, default=1.2, help="Noise multiplier for Private Engine."
    )
    parser.add_argument(
        "-mgn", type=float, default=1.0, help="Max grad norm for Private Engine."
    )
    args = parser.parse_args()
    dataset = str(args.d)
    diff_privacy = bool(args.dp)
    batch_size = int(args.b)
    virtual_batch_size = int(args.vb) if diff_privacy else batch_size
    epochs = int(args.e)
    lr = float(args.lr)
    nm = float(args.nm)
    mgn = float(args.mgn)

    model = None
    # Set transforms
    # transform = None
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            # transforms.Normalize((0.5), (0.5)),
        ]
    )
    if dataset == "emnist":
        # Get dataloaders
        train_set = EMNIST(
            ".", split="balanced", train=True, download=True, transform=transform
        )

        test_val_set = EMNIST(
            ".", split="balanced", train=False, download=True, transform=transform
        )
        input_dim = 28
        n_classes = 47
        model = FemnistModel(input_dim=input_dim, num_classes=n_classes)
    elif dataset == "mnist":
        # Get dataloaders
        train_set = EMNIST(
            ".", split="mnist", train=True, download=True, transform=transform
        )
        test_val_set = EMNIST(
            ".", split="mnist", train=False, download=True, transform=transform
        )
        input_dim = 28
        n_classes = 10
        model = FemnistModel(input_dim=input_dim, num_classes=n_classes)
    elif dataset == "cifar10":
        # Get dataloaders
        train_set = CIFAR10(".", train=True, download=True, transform=transform)
        test_val_set = CIFAR10(".", train=False, download=True, transform=transform)
        input_dim = 32
        n_classes = 10
        model = FemnistModel(input_dim=32, num_classes=n_classes)
    elif dataset == "kmnist":
        # Get dataloaders
        train_set = KMNIST(".", train=True, download=True, transform=transform)
        test_val_set = KMNIST(".", train=False, download=True, transform=transform)
        input_dim = 28
        n_classes = 10
        model = FemnistModel(input_dim=32, num_classes=n_classes)
    else:
        exit
    train_len = len(train_set)
    print(f"len dataset : {train_len}")
    test_share = int(0.8 * len(test_val_set))
    val_share = len(test_val_set) - test_share
    test_set, val_set = torch.utils.data.random_split(
        test_val_set, [test_share, val_share]
    )
    train_loader = model_utils.make_data_loader(
        train_set, batch_size, diff_privacy, virtual_batch_size
    )
    test_loader = DataLoader(test_set, batch_size=virtual_batch_size)

    # Get the date and format it
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d-%H%M%S")
    save_dir = f"{BASEPATH}leafdp/vanilla/models/"
    # Move the model to the device before creating the privacy engine
    model = model.to(DEVICE)
    # Define loss function
    criterion = CrossEntropyLoss()
    # Get optimizer
    optimizer = optim.Adam(model.parameters(), lr)
    # Get target delta
    delta = model_utils.get_target_delta(train_len)
    results = train_model(
        model,
        optimizer,
        train_loader,
        test_loader=test_loader,
        epochs=epochs,
        batch_size=batch_size,
        criterion=criterion,
        diff_privacy=diff_privacy,
        dataset_len=train_len,
        virtual_batch_size=virtual_batch_size,
        delta=delta,
        nm=nm,
        mgn=mgn,
        test=True,
        device=DEVICE,
    )
    model_name = f"{dataset}-model_dp-True_b-{virtual_batch_size}_e-{epochs}_lr-{lr}_nm-{nm}_mgn-{mgn}_{dt_string}"
    save_path = save_dir + model_name
    # Save model
    print(f"Save model.")
    checkpoint = {
        "num_classes": n_classes,
        "input_dim": input_dim,
        "state_dict": model.state_dict(),
    }
    model_utils.save_model(model, f"{save_path}.pth", checkpoint=checkpoint)
    # Load validation dataset
    val_loader = DataLoader(val_set, batch_size=virtual_batch_size)
    # Validate model
    print("Validation the model:")
    test_model(
        model,
        val_loader,
        batch_size=virtual_batch_size,
        device=DEVICE,
    )
    # Save results
    save_results = f"{save_path}_results.json"
    with open(save_results, "w") as outfile:
        json.dump(results, outfile)
        print(f"Saved results.")

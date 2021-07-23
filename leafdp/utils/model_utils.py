import json
import numpy as np
import os
from collections import defaultdict
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
import torch
from torch._C import dtype
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import argparse
from leafdp.femnist.cnn import FemnistModel

BASEPATH = os.environ["BASEPATH"]

# Function to count the number of parameters in a model
def count_parameters(model) -> int:
    """Count parameters in a torch model.

    Parameters
    ----------
    model : torch.nn.module
        The model from which you want to count parameters.

    Returns
    -------
    int
        Total number of parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_model(filepath: str, dataset="femnist") -> torch.nn.Module:
    """Load pytorch model.

    Parameters
    ----------
    filepath
        Filepath to the model
    Returns
    -------
    model
        Desired model
    """
    try:
        checkpoint = torch.load(filepath)
        if dataset == "femnist":
            model = FemnistModel(
                input_dim=28, num_classes=62
            )
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k,v in checkpoint["state_dict"].items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        else:
            model = None
            print("Couldn't create model with given dataset.")
        return model
    except OSError as err:
        print("OS error: {0}".format(err))
    except KeyError:
        print("Checkpoint missing key arguments")


def save_model(
    model: torch.nn.Module,
    filepath: str,
    checkpoint: dict = None,
):

    """Save pytorch model.

    Parameters
    ----------
    model
        The model to train
    checkpoint
        Dict containing the state of the model and other parameter for reconstruction
    filepath
        Path to save the model
    """

    if checkpoint is None:
        checkpoint = {
            "num_classes": 62,
            "input_dim": 28,
            "state_dict": model.state_dict(),
        }
    try:
        torch.save(checkpoint, filepath)
    except OSError as err:
        print("OS error: {0}".format(err))


def remake_dataset(dataset: str, train: bool = True, use_val: bool = False):
    """Remake a Leaf dataset so it can be used with Flower, transforms
    big JSONs in one per user.

    Parameters
    ----------
    dataset : str
        The name of the dataset to remake.
    train : bool, optional
        Train dataset transformed, by default True
    use_val : bool, optional
        Val dataset transformed, by default False
    """
    data_dir = _get_data_dir(dataset, train=train, use_val=use_val)
    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith(".json")]
    for f in files:
        save_dir = os.path.join(data_dir, str(f).split(".json")[0])
        if not os.path.exists(save_dir):
            try:
                os.mkdir(save_dir)
            except OSError:
                print("Creation of the directory failed")
            else:
                print("Successfully created the directory")
        file_path = os.path.join(data_dir, f)
        with open(file_path, "r") as inf:
            cdata = json.load(inf)
        for user in cdata["users"]:
            user_data = cdata["user_data"][user]
            save_file = os.path.join(save_dir, user + ".json")
            with open(save_file, "w") as outfile:
                json.dump(user_data, outfile)
                print(
                    f"User {user}'s data has been successfully writen at {save_file}."
                )
    print("Successfully created all user data.")

def _add_client_to_silo(silos, client, i: int):
    cur_indice = i % len(silos)
    if len(silos[cur_indice]["users"]) < silos[cur_indice]["max_clients"]:
        silos[cur_indice]["users"].append(client["user_id"])
        silos[cur_indice]["users_data"]["x"].extend(client["user_data"]["x"])
        silos[cur_indice]["users_data"]["y"].extend(client["user_data"]["y"])
    else:
        _add_client_to_silo(silos, client, cur_indice+1)
    
            
def make_cross_silo_dataset(dataset: str, nb_elements_min: int = 350):
    """Transform a Leaf dataset in a cross-silo dataset.

    Parameters
    ----------
    dataset : str
        The name of the dataset we want to transform.
    nb_elements_min : int, optional
        The number of elements min a client should have to start a silo, by default 350
    """
    silo_clients = []
    silo_infos = {
        "users": [],
        "len_data_users": [],
    }
    cross_device_clients = []
    # Train directory
    data_dir = _get_data_dir(dataset, train=True, use_val=False)
    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith(".json")]
    save_dir = os.path.join(data_dir, "cross_silo_clients")
    if not os.path.exists(save_dir):
        try:
            os.mkdir(save_dir)
        except OSError:
            print("Creation of the directory failed")
        else:
            print("Successfully created the directory")
    # Load every data and build silos clients and keep other data
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, "r") as inf:
            cdata = json.load(inf)
        for user in cdata["users"]:
            user_data = cdata["user_data"][user]
            if len(user_data["y"]) > nb_elements_min:
                silo_clients.append(
                    {
                        "users": [user],
                        "users_data": {"x": user_data["x"], "y": user_data["y"]},
                    }
                )
            else:
                cross_device_clients.append(
                    {
                        "user_id": user,
                        "user_data": user_data,
                    }
                )
    if silo_clients:
        avg_clients = round((len(cross_device_clients)+len(silo_clients))/len(silo_clients))
        # Shuffle remaining clients
        rng = np.random.default_rng(42)
        rng.shuffle(cross_device_clients)
        # Determines max nb of clients for some silos
        for i, silo in enumerate(silo_clients):
            silo["max_clients"] = avg_clients - 3 if i%2 == 0 else avg_clients + 3
        for i, user in enumerate(cross_device_clients):
            _add_client_to_silo(silo_clients, user, i)
        for silo in silo_clients:
            silo_infos["users"].append(silo["users"])
            silo_infos["len_data_users"].append(len(silo["users_data"]["y"]))
        silo_infos["mean"] = float(np.mean(silo_infos["len_data_users"]))
        silo_infos["std"] = float(np.std(silo_infos["len_data_users"]))
        silo_infos["total_data"] = int(np.sum(silo_infos["len_data_users"]))
        silo_infos["nb_silos"] = len(silo_clients)
        silo_infos["avg_clients"] = avg_clients

    # Save data
    for j, client in enumerate(silo_clients):
        save_data_file = os.path.join(save_dir, f"silo_{j}_data.json")
        with open(save_data_file, "w") as outfile:
            json.dump(client["users_data"], outfile)
            print(f"Silo {j}'s data has been successfully written at {save_data_file}.")
    # Save general infos
    save_infos_file = os.path.join(save_dir, f"silo_{j}_infos.json")
    with open(save_infos_file, "w") as outfile:
        json.dump(silo_infos, outfile)
        print(f"Silo infos have been successfully written at {save_infos_file}.")
    # Test directory
    test_silos = [[] for _ in silo_clients]
    data_dir = _get_data_dir(dataset, train=False, use_val=False)
    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith(".json")]
    save_dir = os.path.join(data_dir, "cross_silo_clients")
    if not os.path.exists(save_dir):
        try:
            os.mkdir(save_dir)
        except OSError:
            print("Creation of the directory failed")
        else:
            print("Successfully created the directory")
    # Load test data and check if in given silo
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, "r") as inf:
            cdata = json.load(inf)
        for user in cdata["users"]:
            user_data = cdata["user_data"][user]
            for i in range(len(silo_clients)):
                if user in silo_clients[i]["users"]:
                    test_silos[i].append(
                        {
                            "user_id": user,
                            "x": user_data["x"],
                            "y": user_data["y"],
                        }
                    )
    # Reorder test data so that it aligns with train data
    ordered_test_silos = [{"x": [], "y": []} for _ in silo_clients]
    for i in range(len(silo_clients)):
        for user in silo_clients[i]["users"]:
            for test_client in test_silos[i]:
                if test_client["user_id"] == user:
                    ordered_test_silos[i]["x"].extend(test_client["x"])
                    ordered_test_silos[i]["y"].extend(test_client["y"])
    # Save data
    for j, test_silo in enumerate(ordered_test_silos):
        save_data_file = os.path.join(save_dir, f"silo_{j}_data.json")
        with open(save_data_file, "w") as outfile:
            json.dump(test_silo, outfile)
            print(f"Silo {j}'s data has been successfully written at {save_data_file}.")

    print(f"Successfully created cross-silo dataset with min elems {nb_elements_min}.")
    print(f"There are {len(silo_clients)} silos.")
    print(f"Total number of samples: {silo_infos['total_data']}")
    print(f"Mean: {silo_infos['mean']}")
    print(f"Std: {silo_infos['std']}")
    print(f"Average nb of clients per silo: {silo_infos['avg_clients']}")


class FemnistDataset(Dataset):
    def __init__(self, data: dict, transform=None, target_transform=None):
        # self.xs = np.array([np.array(x).reshape(28, 28) for x in data["x"]]).astype("float32")
        self.xs = torch.tensor(np.array(data["x"]).astype("float32"))
        # self.ys = torch.Tensor(np.array([_one_hot(y, 62) for y in data["y"]]).astype("long"))
        self.ys = torch.tensor(np.array(data["y"]), dtype=torch.long)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, idx):
        x = self.xs[idx]
        y = self.ys[idx]
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)
        return x, y


def _get_data_dir(dataset, train: bool = True, use_val: bool = False):
    # Determines the data directory
    if train:
        return os.path.join(BASEPATH, "data", dataset, "data", "train")
    eval_set = "eval" if use_val else "test"
    return os.path.join(BASEPATH, "data", dataset, "data", eval_set)


def read_dir(
    dataset: str, data: bool = False, train: bool = True, use_val: bool = False
):
    """Parses directory to get clients, groups and data.

    Assumes the data in the input directories are .json files with
    keys 'users' and 'user_data'

    Parameters
    ----------
    data_dir : str
        The directory we want to read data from
    data : bool, optional
        Whether or not data is returned, by default False
    train : bool, optional
        Whether the train set or test set is returned, by default True
    use_val : bool, optional
        Replace test set with eval set if True, by default False

    Returns
    -------
    clients : list
        Client ids
    groups : list
        Groups ids
    data : dict
        The data associated with each client
    """
    clients = []
    groups = []
    data = defaultdict(lambda: None)

    data_dir = _get_data_dir(dataset, train=train, use_val=use_val)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith(".json")]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, "r") as inf:
            cdata = json.load(inf)
        clients.extend(cdata["users"])
        if "hierarchies" in cdata:
            groups.extend(cdata["hierarchies"])
        data.update(cdata["user_data"])

    clients = list(sorted(data.keys()))
    if data:
        return clients, groups, data
    else:
        return clients, groups, None


def get_clients_ids(dataset: str):
    """Get all ids from a given dataset.

    Parameters
    ----------
    dataset : str
        The dataset from which we get ids.

    Returns
    -------
    list[int]
        The list of ids.
    """
    data_dir = _get_data_dir(dataset, train=True)
    folders = os.listdir(data_dir)
    folders = [f for f in folders if not f.endswith(".json")]
    clients_ids = []
    for f in folders:
        for i in os.listdir(os.path.join(data_dir, f)):
            clients_ids.append(i)
    return clients_ids


def generate_indices(
    nbc: int,
    total: int = 3551,
    seed: int = 42,
):
    # Generate indices based on the seed and the number of clients
    np.random.seed(seed)
    indices = []
    while len(indices) < nbc:
        i = np.random.randint(0, total, size=1).item()
        if i not in indices:
            indices.append(i)
    return indices


def get_client_data(
    dataset: str,
    client_id: str,
    cross_silo: bool = True,
    train: bool = True,
    use_val: bool = False,
):
    """Get data from target client.

    Parameters
    ----------
    dataset : str
        The dataset from which we want data.
    client_id : str
        The id of the client we want to get the data.
    cross_silo : bool, optional
        Whether the dataset is cross-silo or not, by default True
    train : bool, optional
        Get train data or not, by default True
    use_val : bool, optional
        Get val or test data, by default False

    Returns
    -------
    dic
        The client's data
    """
    dir_dir = _get_data_dir(dataset, train=train, use_val=use_val)
    if cross_silo:
        dir_dir = dir_dir + "/cross_silo_clients/"
    dir_files = os.listdir(dir_dir)
    if cross_silo:
        for f in dir_files:
            if f"silo_{client_id}_" in f:
                file_path = os.path.join(dir_dir, f)
                with open(file_path, "r") as inf:
                    cdata = json.load(inf)
                return cdata
    data_dirs = [f.split(".json")[0] for f in dir_files if f.endswith(".json")]
    for d in data_dirs:
        files = os.listdir(os.path.join(dir_dir, d))
        files = [f for f in files if f.endswith(".json")]
        for f in files:
            if client_id in f:
                file_path = os.path.join(dir_dir, d, f)
                with open(file_path, "r") as inf:
                    cdata = json.load(inf)
                return cdata
    print("Couldn't find requested client id.")
    return None


def make_data_loader(dataset, batch_size: int, dp: bool, v_batch_size: int = 128):
    """Make a dataloader given a dataset object.

    Parameters
    ----------
    dataset : Dataset
        The dataset object we want to get a dataloader from
    batch_size : int
        The batch size
    dp : bool
        Whether differential privacy is used or not
    v_batch_size : int, optional
        Virtual batch size to adapt the steps, by default 128

    Returns
    -------
    DataLoader
        The dataloader object
    """
    if not dp:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    assert v_batch_size % batch_size == 0
    data_len = len(dataset)
    sample_rate = batch_size / data_len
    sampler = UniformWithReplacementSampler(
        num_samples=data_len,
        sample_rate=sample_rate,
    )
    return DataLoader(dataset, batch_sampler=sampler)


def get_target_delta(data_size: int) -> float:
    """Generate target delta given the size of a dataset. Delta should be
    less than the inverse of the datasize.

    Parameters
    ----------
    data_size : int
        The size of the dataset.

    Returns
    -------
    float
        The target delta value.
    """
    den = 1
    while data_size // den >= 1:
        den *= 10
    return 1 / den


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", type=str, default=None, help="Dataset you wish to remake."
    )
    parser.add_argument("-t", type=int, default=1, help="Train dataset or not.")
    parser.add_argument("-v", type=int, default=0, help="Val dataset or not.")
    parser.add_argument("-cs", type=int, default=1, help="Cross silo dataset or not.")
    parser.add_argument(
        "-nb",
        type=int,
        default=350,
        help="Min nb of elements to create a sillo from a client.",
    )
    args = parser.parse_args()
    dataset = str(args.d)
    train = bool(args.t)
    use_val = bool(args.v)
    cross_silo = bool(args.cs)
    nb_elem_min = int(args.nb)
    if dataset is not None:
        if cross_silo:
            make_cross_silo_dataset(dataset=dataset, nb_elements_min=nb_elem_min)
        else:
            remake_dataset(dataset=dataset, train=train, use_val=use_val)

# DPSGD Strategies for Cross-Silo Federated Learning

## Presentation
This project aims to show different uses of Federated Learning and Differential Privacy. Three methods are proposed to obtain the desired privacy budget: fixed, adaptive, and hybrid. All these methods refer to the way we obtain the noise multiplier applied to the gradient during the DPSGD algorithm. For Federated Learning, [Flower](https://flower.dev/) is used and for Differential Privacy, [Opacus](https://opacus.ai/) was chosen. 

We chose to use the [LEAF](https://leaf.cmu.edu/) benchmark to show the results of our method as it is one of the only benchmarks available at the time for Federated settings.
This repository is made to act as an extension of the LEAF repository. Follow the installation instructions on the [LEAF github](https://github.com/TalwalkarLab/leaf/tree/master/data/femnist) before continuing. Once it's done download this repository and paste it inside the root directory of LEAF.
To construct the FEMNIST dataset as we did you should run the `preprocess.sh` file with the full-sized dataset command, as stated in their GitHub.

## Installation
You can create the virtual environment and install dependencies using [Poetry](https://python-poetry.org/). If it was not already installed on your computer you should restart it so Poetry commands can work on your terminal. Move to the root of this project ("leaf-fl-dp" directory) and run `poetry install`. Then run `poetry shell` to instantiate the virtual environment.

## Usage
### Vanilla 
For hyperparameters and training vanilla models (with or without DP) all happens in the `vanilla/` folder. Before using any python file on its own, you should run the following command: `export BASEPATH="your_path_to_leaf_repo/"`. So that scripts execute correctly.

Hyperparameters search is done by running `find_hyperparameters.py`, with the following possible arguments:

```python
-d: str = Which dataset to use for training. (emnist, mnist, cifar10).
-dp: int(bool) = Whether Differential Privacy is used or not. Defaults to 1.
-b: int = Number of elements per batch, defaults to 32.
-vb: int = Virtual batch size for differential privacy. Defaults to 256.
-cpu: float = Number of cpus per trials, defaults to 1.0.
-gpu: float = Number of gpus per trials, defaults to 0.0.
-s: int = Number of samples, defaults to 50.
-e: int = Max number of epochs per trials, defaults to 50.
```
The best model found will be saved in the `hyperparam_finding/` directory.

For training models, you should launch `train.py`. You can add the following arguments to the script:

```python
-d: str = Which dataset to use for training. (femnist, mnist, cifar10).
-dp: int(bool) = Whether Differential Privacy is used or not. Defaults to 0.
-b: int = Number of elements per batch. Defaults to 256.
-vb: int = Virtual batch size for differential privacy. Defaults to 256.
-e: int = Number of epochs.
-lr: float = Learning rate for the optimizer.
-nm: float = Noise multiplier for the Privacy Engine.
-mgn: float = Max Grad Norm for the Privacy Engine.
```

Trained models and results can be found in the `models/` directory.


### Federated 

For running federated models, you first have to remake the FEMNIST dataset, as it normally is composed of 32 JSON files containing the clients' data. Except that in Flower's case, we are simulating distant clients so we can't afford to load each JSON file N times in search of our dataset. To this effect, you should run `model_utils.py` in `/leafdp/utils/` so that it creates one JSON file per client:

```shell
python model_utils.py -d femnist -t 1 -cs 1 -nb 350
python model_utils.py -d femnist -t 0 -cs 1 -nb 350
```

Options are as follows:
```python
-d: str = Which dataset to remake (femnist).
-t: int(bool) = Whether train or test set is remade, defaults to 1.
-cs: int(bool) = Whether we make cross-silo or cross-device setting, defaults to 1.
-nb: int = The minimum amount of data a client has to own to be considered a silo, defaults to 350.
```

Next, modify the `run.sh` file in `/leafdp/flower/` by replacing the BASEPATH variable with your own and run the file in your terminal. It will launch by default two random clients (depending on the chosen seed) for 3 rounds. You can use the following command and arguments:

```shell
./run.sh NBCLIENTS NBMINCLIENTS NBFITCLIENTS CENTRALIZED BATCHSIZE VBATCHSIZE NBROUNDS LR DP NM MGN EPS STRAT SEED
```

With:
```python
-- NBCLIENTS: int = The number of clients to launch. 2 by default.
-- NBMINCLIENTS: int = The number of clients to wait before launching federated training. 2 by default.
-- NBFITCLIENTS: int = The number of clients sampled each round. 2 by default.
-- CENTRALIZED: int(bool) = Whether the model is evaluated on the server or the client-side. Defaults to 1.
-- BATCHSIZE: int = Number of elements per batch. Defaults to 256.
-- VBATCHSIZE: int = Number of elements per virtual batch, used for differential privacy. Defaults to 256.
-- NBROUNDS: int = Number of rounds. Defaults to 3.
-- LR: float = Learning rate for the optimizer. Defaults to 0.0001.
-- DP: int(bool) = Whether Differential Privacy is used or not. Defaults to 0.
-- NM: float = Noise multiplier for the Privacy Engine. Defaults to 1.2.
-- MGN: float = Max Grad Norm for the Privacy Engine. Defaults to 1.0.
-- EPS: float = Target epsilon for the privacy budget. Defaults to 0.0.
-- STRAT: str = Strategy used to respect target epsilon["fix", "adaptive", "hybrid"]. Defaults to "vanilla".
-- SEED: int = Seed for generating indexes to get clients. Defaults to 42.
```
Models and results are located in the `server/` folder. 

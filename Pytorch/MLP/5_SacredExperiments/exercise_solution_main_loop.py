# We need to use the dataset_to_use variable for some stuff. Because we are running this script from
# exercise_solution_run_experiment.py, then we can use __main__ as a reference to exercise_solution_run_experiment.py
from __main__ import dataset_to_use
from exercise_solution_get_results import get_results
import os
import json
import torch
import torch.nn as nn
from torchvision import datasets
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sacred import Experiment


# Different name for each dataset
ex = Experiment('exercise_{}_mlp'.format(dataset_to_use))


# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
@ex.config
def my_config():
    random_seed = 42
    lr = 1e-3
    neurons_per_layer = (100, 200, 100)
    n_epochs = 2
    batch_size = 512
    dropout = 0.2
    # 2. Add an option on main_loop.py so that the best model of each run is saved, instead of saving only the best
    # model out of all the runs. The best place to save it is on the folders with the ids for each run.
    save_each = True
    # 3. Add an option to load a model from a run and continue training after changing some of the hyper-parameters,
    # like the learning rate, the optimizer or the number of epochs.
    continue_training_run = None


# %% ----------------------------------- Helper Functions --------------------------------------------------------------
def acc(model, x, y, return_labels=False):
    with torch.no_grad():
        logits = model(x)
        pred_labels = np.argmax(logits.cpu().numpy(), axis=1)
    if return_labels:
        return pred_labels
    else:
        return 100*accuracy_score(y.cpu().numpy(), pred_labels)


# %% -------------------------------------- MLP Class ------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, neurons_per_layer, dropout):
        super(MLP, self).__init__()
        dims = (784, *neurons_per_layer)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU(),
                nn.BatchNorm1d(dims[i+1]),
                nn.Dropout(dropout)
            ) for i in range(len(dims)-1)
        ])
        self.layers.extend(nn.ModuleList([nn.Linear(neurons_per_layer[-1], 10)]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


@ex.automain
def my_main(random_seed, lr, neurons_per_layer, n_epochs, batch_size, dropout, save_each, continue_training_run):

    # %% --------------------------------------- Set-Up ----------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # %% -------------------------------------- Data Prep --------------------------------------------------------------
    if dataset_to_use == "mnist":
        data_train = datasets.MNIST(root='.', train=True, download=True)
    else:
        data_train = datasets.FashionMNIST(root='.', train=True, download=True)
    x_train, y_train = data_train.data.view(len(data_train), -1).float().to(device), data_train.targets.to(device)
    x_train.requires_grad = True
    if dataset_to_use == "mnist":
        data_test = datasets.MNIST(root='.', train=False, download=True)
    else:
        data_test = datasets.FashionMNIST(root='.', train=False, download=True)
    x_test, y_test = data_test.data.view(len(data_test), -1).float().to(device), data_test.targets.to(device)

    # %% -------------------------------------- Training Prep ----------------------------------------------------------
    model = MLP(neurons_per_layer, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # %% -------------------------------------- Training Loop ----------------------------------------------------------
    if continue_training_run:
        print("\n ------------ Continuing run number {} on run number {} on {} ---------------".format(
            continue_training_run, ex.current_run._id, dataset_to_use))
        with open(os.getcwd() + "/exercise_{}_mlp_runs/{}/config.json".format(dataset_to_use, continue_training_run),
                  "r") as s:
            config_old = json.load(s)
        print("The old config was:", config_old)
        with open(os.getcwd() + "/exercise_{}_mlp_runs/{}/config.json".format(dataset_to_use, ex.current_run._id),
                  "r") as s:
            config_new = json.load(s)
        print("The new config is:", config_new)
        # Updates all the variables with names equal to the keys of the config_new dict with the corresponding values
        locals().update(config_new)
        try:
            results_so_far = pd.read_excel(os.getcwd() + "/exercise_{}_experiments.xlsx".format(dataset_to_use))
            loss_test_best = results_so_far[results_so_far["experiment_id"] == int(continue_training_run)]["test loss"][0]
        except:
            print("There was an error getting the best result from previous run ({})."
                  " Will save the best model out of this run".format(continue_training_run))
            loss_test_best = 1000
        model.load_state_dict(torch.load(os.getcwd() + "/exercise_{}_mlp_runs/{}/exercise_mlp_{}.pt".format(
            dataset_to_use, continue_training_run, dataset_to_use)))
    else:
        print("\n ------------ Doing run number {} on {} with configuration ---------------".format(ex.current_run._id,
                                                                                                    dataset_to_use))
        print(ex.current_run.config)
        if not save_each:
            try:
                get_results(dataset_to_use)
                results_so_far = pd.read_excel(os.getcwd() + "/exercise_solution_{}_experiments.xlsx".format(dataset_to_use))
                loss_test_best = min(results_so_far["test loss"].values)
            except:
                loss_test_best = 1000
                print("No results so far, will save the best model out of this run")
        else:
            loss_test_best = 1000
    best_epoch, loss_best, acc_best = 0, 1000, 0

    print("Starting training loop...")
    for epoch in range(n_epochs):

        loss_train = 0
        model.train()
        for batch in range(len(x_train) // batch_size + 1):
            inds = slice(batch * batch_size, (batch + 1) * batch_size)
            optimizer.zero_grad()
            logits = model(x_train[inds])
            loss = criterion(logits, y_train[inds])
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        model.eval()
        with torch.no_grad():
            y_test_pred = model(x_test)
            loss = criterion(y_test_pred, y_test)
            loss_test = loss.item()

        acc_train, acc_test = acc(model, x_train, y_train), acc(model, x_test, y_test)
        print("Epoch {} | Train Loss {:.5f}, Train Acc {:.2f} - Test Loss {:.5f}, Test Acc {:.2f}".format(
            epoch, loss_train/batch_size, acc_train, loss_test, acc_test))

        if loss_test < loss_test_best:
            if save_each:
                torch.save(model.state_dict(), "exercise_{}_mlp_runs/{}/exercise_mlp_{}.pt".format(
                    dataset_to_use, ex.current_run._id, dataset_to_use))
            else:
                torch.save(model.state_dict(), "exercise_mlp_{}.pt".format(dataset_to_use))
            print("A new model has been saved!")
            loss_test_best = loss_test
        if loss_test < loss_best:
            best_epoch, loss_best, acc_best = epoch, loss_test, acc_test

        ex.log_scalar("training loss", loss_train/batch_size, epoch)
        ex.log_scalar("training acc", acc_train, epoch)
        ex.log_scalar("testing loss", loss_test, epoch)
        ex.log_scalar("testing acc", acc_test, epoch)
        ex.info["epoch"], ex.info["test loss"], ex.info["test acc"] = best_epoch, loss_best, acc_best

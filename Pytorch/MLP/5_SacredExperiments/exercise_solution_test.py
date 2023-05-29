# %% -------------------------------------------------------------------------------------------------------------------

# 4. Create a training file that takes as inputs the dataset and the experiment id to use, loads the corresponding model
# and config, and tests this model on the test set (this makes more sense when we have a held-out set)

# %% --------------------------------------- Imports -------------------------------------------------------------------
import os
import json
import torch
import torch.nn as nn
from torchvision import datasets
import numpy as np
from sklearn.metrics import accuracy_score


DATASET = "fashionmnist"
RUN = 4
with open(os.getcwd() + "/exercise_{}_mlp_runs/{}/config.json".format(DATASET, RUN),
          "r") as s:
    config = json.load(s)


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


# %% ----------------------------------------- Set-Up ------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(config["random_seed"])
np.random.seed(config["random_seed"])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# %% ----------------------------------------- Data Prep ---------------------------------------------------------------
if DATASET == "mnist":
    data_train = datasets.MNIST(root='.', train=False, download=True)
else:
    data_train = datasets.FashionMNIST(root='.', train=False, download=True)
x_test, y_test = data_train.data.view(len(data_train), -1).float().to(device), data_train.targets.to(device)

# %% ------------------------------------------------ Testing ----------------------------------------------------------
model = MLP(config["neurons_per_layer"], config["dropout"]).to(device)
model.load_state_dict(torch.load(os.getcwd() + "/exercise_{}_mlp_runs/{}/exercise_mlp_{}.pt".format(
            DATASET, RUN, DATASET)))
model.eval()
print("The test accuracy is", acc(model, x_test, y_test))

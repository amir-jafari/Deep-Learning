# %% -------------------------------------------------------------------------------------------------------------------
# Fit a MLP to the FashionMNIST dataset: https://github.com/zalandoresearch/fashion-mnist
# %% --------------------------------------- Imports -------------------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
# 2. Try using more/less layers and different hidden sizes to get a good better fit. Also play with the dropout.
# Try different batch sizes and get a feeling of how they can influence convergence and speed
LR = 1e-3
N_NEURONS = (100, 200, 100)
N_EPOCHS = 30
BATCH_SIZE = 512
DROPOUT = 0.2
# 4. Add an option to only test the model, by loading the model you saved on the training phase
TRAIN = True  # Set TRAIN = False to load a trained model and predict on the test set
# 3. Add an option to save the model on each epoch, and stop saving them when the validation
# loss begins to increase (early stopping) - https://pytorch.org/tutorials/beginner/saving_loading_models.html
SAVE_MODEL = True


# %% ----------------------------------- Helper Functions --------------------------------------------------------------
def acc(x, y, return_labels=False):
    with torch.no_grad():
        logits = model(x)
        pred_labels = np.argmax(logits.cpu().numpy(), axis=1)
    if return_labels:
        return pred_labels
    else:
        return 100*accuracy_score(y.cpu().numpy(), pred_labels)


# 6. Define a function to show some images that were incorrectly classified
def show_mistakes(x, y, idx, label_names):
    y_pred = acc(x, y, return_labels=True)
    idx_mistakes = np.argwhere((y.cpu().numpy() == y_pred) == 0).flatten()
    plt.title("MLP prediction: {} - True label: {}".format(label_names[y_pred[idx_mistakes[idx]]],
                                                           label_names[y[idx_mistakes[idx]]]))
    plt.imshow(x[idx_mistakes[idx]].reshape(28, 28).cpu().numpy())
    plt.show()


# %% -------------------------------------- MLP Class ------------------------------------------------------------------
class MLPModuleList(nn.Module):
    def __init__(self, neurons_per_layer, dropout=DROPOUT):
        super(MLPModuleList, self).__init__()
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


# %% -------------------------------------- Data Prep ------------------------------------------------------------------
# 1. Download the data using datasets.FashionMNIST
if TRAIN:
    data_train = datasets.FashionMNIST(root='.', train=True, download=True)
    x_train, y_train = data_train.data.view(len(data_train), -1).float().to(device), data_train.targets.to(device)
    x_train.requires_grad = True
data_test = datasets.FashionMNIST(root='.', train=False, download=True)
x_test, y_test = data_test.data.view(len(data_test), -1).float().to(device), data_test.targets.to(device)

# %% -------------------------------------- Training Prep ----------------------------------------------------------
model = MLPModuleList(N_NEURONS).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# %% -------------------------------------- Training Loop ----------------------------------------------------------
if TRAIN:
    loss_test_best = 1000  # The model will be saved whenever the test loss gets smaller
    print("Starting training loop...")
    for epoch in range(N_EPOCHS):

        loss_train = 0
        model.train()
        for batch in range(len(x_train)//BATCH_SIZE + 1):
            inds = slice(batch*BATCH_SIZE, (batch+1)*BATCH_SIZE)
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

        print("Epoch {} | Train Loss {:.5f}, Train Acc {:.2f} - Test Loss {:.5f}, Test Acc {:.2f}".format(
            epoch, loss_train, acc(x_train, y_train), loss_test, acc(x_test, y_test)))

        # 3. Add an option to save the model on each epoch, and stop saving them when the validation
        # loss begins to increase (early stopping) - https://pytorch.org/tutorials/beginner/saving_loading_models.html
        if loss_test < loss_test_best and SAVE_MODEL:
            torch.save(model.state_dict(), "mlp_fashionmnist.pt")
            print("The model has been saved!")
            loss_test_best = loss_test

# %% ------------------------------------------ Final test -------------------------------------------------------------
# 4. Add an option to only test the model, by loading the model you saved on the training phase
# We need to have the exact same class to load our model. If you want to load it in another script,
model.load_state_dict(torch.load("mlp_fashionmnist.pt"))  # you need to import or define the class again
model.eval()
y_test_pred = acc(x_test, y_test, return_labels=True)
print("The accuracy on the test set is", 100*accuracy_score(y_test.cpu().numpy(), y_test_pred), "%")
# 5. Print out the confusion matrix
print("The confusion matrix is")
print(confusion_matrix(y_test.cpu().numpy(), y_test_pred))

show_mistakes(x_test, y_test, 0, data_test.classes)

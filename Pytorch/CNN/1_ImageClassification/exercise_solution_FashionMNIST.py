# %% --------------------------------------- Imports -------------------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from sklearn.metrics import accuracy_score


# %% --------------------------------------- Set-Up --------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 1e-3
N_EPOCHS = 30
BATCH_SIZE = 128


# %% ----------------------------------- Helper Functions --------------------------------------------------------------
def acc(x, y, return_labels=False):
    with torch.no_grad():
        logits = torch.empty(len(x), 10)
        for batch in range(len(x) // BATCH_SIZE + 1):
            inds = slice(batch * BATCH_SIZE, (batch + 1) * BATCH_SIZE)
            logits[inds] = model(x[inds])
        pred_labels = np.argmax(logits.cpu().numpy(), axis=1)
    if return_labels:
        return pred_labels
    else:
        return 100*accuracy_score(y.cpu().numpy(), pred_labels)


# %% -------------------------------------- CNN Class ------------------------------------------------------------------
# 3. Try modifying the architecture and the hyper-parameters to get a better performance.
# Try including more conv layers and more kernels in each layer. This can allow for less MLP layers at the end.
# To do so, you will need to play around with zero-padding and maybe stack two conv layers together without any pooling.
# You can also remove the pooling and the MLP, and replace it with a final Global Average Pooling layer.
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, (3, 3))
        self.convnorm1 = nn.BatchNorm2d(16)
        self.pad1 = nn.ZeroPad2d(2)

        self.conv2 = nn.Conv2d(16, 32, (3, 3))
        self.convnorm2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d((2, 2))

        self.conv3 = nn.Conv2d(32, 64, (3, 3))
        self.convnorm3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, (3, 3))  # output (n_examples, 128, whatever, whatever)
        # Converts the output of self.conv4 to shape (n_examples, 128, 1, 1) by averaging the 128 feature maps
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(128, 10)  # The global avg pooling means taking
        self.act = torch.relu  # the global average of all the feature maps, so we end up with 128 "averages"

    def forward(self, x):
        x = self.pad1(self.convnorm1(self.act(self.conv1(x))))
        x = self.pool2(self.convnorm2(self.act(self.conv2(x))))
        x = self.act(self.conv4(self.convnorm3(self.act(self.conv3(x)))))
        return self.linear(self.global_avg_pool(x).view(-1, 128))
        # The above line of code is equivalent to:
        # return self.linear1(torch.mean(x.view(x.size(0), x.size(1), -1), dim=2))


# %% -------------------------------------- Data Prep ------------------------------------------------------------------
data_train = datasets.FashionMNIST(root='.', train=True, download=True)
# Reshaped to (n_examples, n_channels, height_pixels, width_pixels)
x_train, y_train = data_train.data.view(len(data_train), 1, 28, 28).float().to(device), data_train.targets.to(device)
x_train.requires_grad = True
data_test = datasets.FashionMNIST(root='.', train=False, download=True)
x_test, y_test = data_test.data.view(len(data_test), 1, 28, 28).float().to(device), data_test.targets.to(device)

# %% -------------------------------------- Training Prep ----------------------------------------------------------
model = CNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# %% -------------------------------------- Training Loop ----------------------------------------------------------
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
        epoch, loss_train/BATCH_SIZE, acc(x_train, y_train), loss_test, acc(x_test, y_test)))

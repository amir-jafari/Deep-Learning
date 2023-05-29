# %% --------------------------------------- Imports -------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score


# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 0.05
N_NEURONS = 60
N_EPOCHS = 6000
PRINT_LOSS_EVERY = 1000


# %% ----------------------------------- Helper Functions --------------------------------------------------------------
# 1. Define a function to generate the y-points for a circle, taking as input the x-points and the radius r.
def circle(x, r, neg=False):
    if neg:
        return -np.sqrt(r ** 2 - x ** 2)
    else:
        return np.sqrt(r**2 - x**2)


# %% -------------------------------------- MLP Class ------------------------------------------------------------------
# 3. Choose the right number of input and output neurons, define and train a MLP to classify this data.
class MLP(nn.Module):
    def __init__(self, hidden_dim):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(2, hidden_dim)
        self.act1 = torch.tanh
        self.linear2 = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        return self.linear2(self.act1(self.linear1(x)))


# %% -------------------------------------- Data Prep ------------------------------------------------------------------
# 2. Use this function to generate the data to train the network. Label points with r=2 as 0 and points with r=4 as 1.
# Note that for each value on the x-axis there should be two values on the y-axis, and vice versa.
x1 = np.arange(-2, 2, 0.1)
y1_pos = circle(np.arange(-2, 2, 0.1), 2, neg=False)
y1_neg = circle(np.arange(-2, 2, 0.1), 2, neg=True)
p1 = np.vstack((np.hstack((x1, x1)), np.hstack((y1_pos, y1_neg)))).T
t1 = np.zeros((int(len(p1))))

x2 = np.arange(-4, 4, 0.2)
y2_pos = circle(np.arange(-4, 4, 0.2), 4, neg=False)
y2_neg = circle(np.arange(-4, 4, 0.2), 4, neg=True)
p2 = np.vstack((np.hstack((x2, x2)), np.hstack((y2_pos, y2_neg)))).T
t2 = np.ones((int(len(p2))))

p = torch.FloatTensor(np.vstack((p1, p2)))
t = torch.LongTensor(np.hstack((t1, t2)))

# %% -------------------------------------- Training Prep --------------------------------------------------------------
model = MLP(N_NEURONS)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# %% -------------------------------------- Training Loop --------------------------------------------------------------
for epoch in range(N_EPOCHS):
    optimizer.zero_grad()
    logits = model(p)
    loss = criterion(logits, t)
    loss.backward()
    optimizer.step()
    if epoch % PRINT_LOSS_EVERY == 0:
        print("Epoch {} | Loss {:.5f}".format(epoch, loss.item()))

# %% -------------------------------------- Check Approx ---------------------------------------------------------------
print(accuracy_score(t.numpy(), np.argmax(logits.detach().numpy(), axis=1))*100)

# 4. Make a contour plot of the MLP as a function of the x and y axis. You can follow
# https://scikit-learn.org/stable/auto_examples/semi_supervised/plot_label_propagation_versus_svm_iris.html
p_np = p.detach().cpu().numpy()
h = .02
x_min, x_max = p[:, 0].min() - 1, p[:, 0].max() + 1
y_min, y_max = p[:, 1].min() - 1, p[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = np.argmax(model(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])).detach().numpy(), axis=1)
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.6)
plt.scatter(p_np[:, 0], p_np[:, 1], c=t.numpy(), cmap=plt.cm.coolwarm)
plt.show()

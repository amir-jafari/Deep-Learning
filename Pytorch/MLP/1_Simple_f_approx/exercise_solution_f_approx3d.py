# %% -------------------------------------------------------------------------------
# Fit a MLP to the function y = x_1^2 + x_2^2. A 3D plot can be found at:
# http://www.livephysics.com/tools/mathematical-tools/online-3-d-function-grapher/
# %% --------------------------------------- Imports -------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed even if it not used explicitly
import torch
import torch.nn as nn


# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 2.5e-4
N_NEURONS1, N_NEURONS2 = 5, 3
N_EPOCHS = 10000
PRINT_LOSS_EVERY = 500
VIZ_FIT_EVERY = 3000


# %% ----------------------------------- Helper Functions --------------------------------------------------------------
# Always define helper functions at the beginning of the script, right after the imports and the hyper-parameters

# 1. Define the function y = x1**2 - x2**2 you will train the MLP to approximate.
def f_to_approx(x1, x2):
    return x1**2 - x2**2


# 2. Define a helper function to plot the real function and the MLP approximation. Hint:
# from mpl_toolkits.mplot3d import Axes3D, use ax.contour3D on 3 inputs with shapes (sqrt(n_examples), sqrt(n_examples))
# You may do 4. first to get the data and figure out why the shapes are like this
def compare_with_real(x1, x2, y_pred, mse):
    ax = plt.axes(projection='3d')
    ax.set_title("MLP (green/blue) fit to $y = x_1^2 - x_2^2$ (grey) | MSE: {:.5f}".format(mse))
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$'); ax.set_zlabel('y')
    ax.contourf3D(x1, x2, f_to_approx(x1, x2), 100, cmap='binary')
    ax.contour3D(x1, x2, y_pred, 50, cmap='viridis')
    plt.show()


# %% -------------------------------------- MLP Class ------------------------------------------------------------------
# 3. Define a MLP class to approximate this function using some data you will generate in 4.
# Play with the number of layers, number of neurons and types of hidden activation functions (tanh, ReLu, etc.)
# You may do 4. first to get the data and figure out the right input and output shapes on the MLP
class MLP(nn.Module):
    def __init__(self, hidden_dim1, hidden_dim2):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(2, hidden_dim1)
        self.linear2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.linear3 = nn.Linear(hidden_dim2, 1)
        self.act = torch.tanh

    def forward(self, x):
        return self.linear3(self.act(self.linear2(self.act(self.linear1(x)))))


# %% -------------------------------------- Data Prep ------------------------------------------------------------------
# 4. Generate the data to train the network. Hint:
# Use np.meshgrid() and then reshape the input to (n_examples, 2) and the target to (n_examples, 1)
p1, p2 = np.linspace(-2, 2, 100), np.linspace(-2, 2, 100)
p1, p2 = np.meshgrid(p1, p2)
t = f_to_approx(p1, p2)
p = torch.Tensor(np.hstack((p1.reshape(-1, 1), p2.reshape(-1, 1))))
p.requires_grad = True
t = torch.Tensor(t.reshape(-1, 1))

# %% -------------------------------------- Training Prep --------------------------------------------------------------
# 5. Use Adam or another optimizer and train the network. Find an appropriate learning rate and number of epochs.
model = MLP(N_NEURONS1, N_NEURONS2)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# %% -------------------------------------- Training Loop --------------------------------------------------------------
for epoch in range(N_EPOCHS):
    optimizer.zero_grad()
    t_pred = model(p)
    loss = criterion(t, t_pred)
    loss.backward()
    optimizer.step()
    if epoch % PRINT_LOSS_EVERY == 0:
        print("Epoch {} | Loss {:.10f}".format(epoch, loss.item()))
    if epoch % VIZ_FIT_EVERY == 0:
        compare_with_real(p1, p2, t_pred.detach().numpy().reshape(-1, len(p1)), loss.item())

# %% -------------------------------------- Check Approx ---------------------------------------------------------------
# 6. Use the function you defined in 2. to visualize how well your MLP fits the original function
compare_with_real(p1, p2, t_pred.detach().numpy().reshape(-1, len(p1)), loss.item())

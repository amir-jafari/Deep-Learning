# %% -------------------------------------------------------------------------------------------------------------------
# Fit a MLP to the function y = x_1^2 + x_2^2. A 3D plot can be found at:
# http://www.livephysics.com/tools/mathematical-tools/online-3-d-function-grapher/
# %% --------------------------------------- Imports -------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed even if it not used explicitly
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam


# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 2.5e-4
N_NEURONS1, N_NEURONS2 = 5, 3
N_EPOCHS = 15000


# %% ----------------------------------- Helper Functions --------------------------------------------------------------
# Always define helper functions at the beginning of the script, right after the imports and the hyper-parameters

# 1. Define the function y = x1**2 - x2**2 you will train the MLP to approximate.
def f_to_approx(x1, x2):
    return x1**2 - x2**2


# 2. Define a helper function to plot the real function and the MLP approximation. Hint:
# from mpl_toolkits.mplot3d import Axes3D, use ax.contour3D on 3 inputs with shapes (sqrt(n_examples), sqrt(n_examples))
# You may do 3. first to get the data and figure out why the shapes are like this
def compare_with_real(x1, x2, y_pred, mse):
    ax = plt.axes(projection='3d')
    ax.set_title("MLP (green/blue) fit to $y = x_1^2 - x_2^2$ (grey) | MSE: {:.5f}".format(mse))
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$'); ax.set_zlabel('y')
    ax.contourf3D(x1, x2, f_to_approx(x1, x2), 100, cmap='binary')
    ax.contour3D(x1, x2, y_pred, 50, cmap='viridis')
    plt.show()


# %% -------------------------------------- Data Prep ------------------------------------------------------------------
# 3. Generate the data to train the network using the function you defined in 1. Hint:
# Use np.meshgrid() and then reshape the input to (n_examples, 2) and the target to (n_examples, 1)
p1, p2 = np.linspace(-2, 2, 100), np.linspace(-2, 2, 100)
p1, p2 = np.meshgrid(p1, p2)
p = np.hstack((p1.reshape(-1, 1), p2.reshape(-1, 1)))
t = f_to_approx(p1, p2).reshape(-1)

# %% -------------------------------------- Training Prep --------------------------------------------------------------
# 4. Define a MLP to approximate this function using the data you just generated.
# Play with the number of layers, neurons and hidden activation functions (tanh, ReLu, etc.)
model = Sequential([
    Dense(N_NEURONS1, input_dim=2),
    Activation("tanh"),
    Dense(N_NEURONS2),
    Activation("tanh"),
    Dense(1)
])
# 5. Use Adam or another optimizer and train the network. Find an appropriate learning rate and number of epochs.
model.compile(optimizer=Adam(lr=LR), loss="mean_squared_error")

# %% -------------------------------------- Training Loop --------------------------------------------------------------
train_hist = model.fit(p, t, epochs=N_EPOCHS, batch_size=len(p))

# %% -------------------------------------- Check Approx ---------------------------------------------------------------
# 6. Use the function you defined in 2. to visualize how well your MLP fits the original function
compare_with_real(p1, p2, model.predict(p).reshape(-1, len(p1)), train_hist.history["loss"][-1])

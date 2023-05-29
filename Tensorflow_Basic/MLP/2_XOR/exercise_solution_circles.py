# %% --------------------------------------- Imports -------------------------------------------------------------------
# Learn a circular Decision Boundary using  MLP
# %% --------------------------------------- Imports -------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical


# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 0.2
N_NEURONS = 10
N_EPOCHS = 2000


# %% ----------------------------------- Helper Functions --------------------------------------------------------------
# 1. Define a function to generate the y-points for a circle, taking as input the x-points and the radius r.
def circle(x, r, neg=False):
    if neg:
        return -np.sqrt(r ** 2 - x ** 2)
    else:
        return np.sqrt(r**2 - x**2)


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

p = np.vstack((p1, p2))
t = np.hstack((t1, t2))
t = to_categorical(t, num_classes=2)

# %% -------------------------------------- Training Prep --------------------------------------------------------------
# 3. Choose the right number of input and output neurons, define and train a MLP to classify this data.
model = Sequential([
    Dense(N_NEURONS, input_dim=2),
    Activation("sigmoid"),
    Dense(2),
    Activation("softmax")
])
model.compile(optimizer=SGD(lr=LR), loss="categorical_crossentropy", metrics=["accuracy"])

# %% -------------------------------------- Training Loop --------------------------------------------------------------
train_hist = model.fit(p, t, batch_size=len(p), epochs=N_EPOCHS)

# %% -------------------------------------- Check Approx ---------------------------------------------------------------
# 4. Use model.evaluate to get the final accuracy on the whole set and print it out
print("The MLP accuracy is", 100*model.evaluate(p, t)[1], "%")

# 5. Make a contour plot of the MLP as a function of the x and y axis. You can follow
# https://scikit-learn.org/stable/auto_examples/semi_supervised/plot_label_propagation_versus_svm_iris.html
h = .02
x_min, x_max = p[:, 0].min() - 1, p[:, 0].max() + 1
y_min, y_max = p[:, 1].min() - 1, p[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = np.argmax(model.predict(np.c_[xx.ravel(), yy.ravel()]), axis=1)
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.6)

plt.scatter(p[:, 0], p[:, 1], c=np.argmax(t, axis=1), cmap=plt.cm.coolwarm)

plt.show()

plt.savefig("exercise_expected_solution.png")

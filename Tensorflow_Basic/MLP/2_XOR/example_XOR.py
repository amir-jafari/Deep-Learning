# %% --------------------------------------- Imports -------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 0.2
N_NEURONS = 2
N_EPOCHS = 3000


# %% ----------------------------------- Helper Functions --------------------------------------------------------------
def acc(x, y):
    """ Simple function to get the accuracy. The label with the highest prob is chosen """
    # model.evaluate can also be used, this is just to illustrate the argmax operation
    probs = model.predict(x)  # (n_examples, n_labels) --> Need to operate on columns, so axis=1 on argmax
    pred_labels = np.argmax(probs, axis=1)
    return 100*accuracy_score(np.argmax(y, axis=1), pred_labels)  # Uses argmax again to come back from one-hot-encoded


# %% -------------------------------------- Data Prep ------------------------------------------------------------------
p = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
t = np.array([0, 0, 1, 1])
# One-hot-encodes the targets to use the softmax and categorical cross-entropy performance index
t = tf.keras.utils.to_categorical(t, num_classes=2)

# %% -------------------------------------- Training Prep --------------------------------------------------------------
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(N_NEURONS, input_dim=2, kernel_initializer=tf.keras.initializers.glorot_uniform(42)),
    tf.keras.layers.Activation("sigmoid"),
    tf.keras.layers.Dense(2, kernel_initializer=tf.keras.initializers.glorot_uniform(42)),  # Output layer with softmax to map to the two classes
    tf.keras.layers.Activation("softmax")
])
# Compiles using categorical cross-entropy performance index and tracks the accuracy during training
model.compile(optimizer=tf.keras.optimizers.Adam(lr=LR), loss="categorical_crossentropy", metrics=["accuracy"])

# %% -------------------------------------- Training Loop --------------------------------------------------------------
model.fit(p, t, batch_size=len(p), epochs=N_EPOCHS)

# %% -------------------------------------- Check Approx ---------------------------------------------------------------
print("The MLP accuracy is", acc(p, t), "%")
# Gets the weights and biases from the first layer
w, b = model.layers[0].get_weights()
# Gets some points on the interval [0, 1] and the two decision boundaries
a = np.arange(0, 1.1, 0.1)
db1 = -w[0, 0]*a/w[1, 0] - b[0]/w[1, 0]  # Note than on Keras the shape of the weight matrix
db2 = -w[0, 1]*a/w[1, 1] - b[1]/w[1, 1]  # is RxS (n_inputs x n_neurons)
# Plots the points and the decision boundaries
plt.scatter(p[:2, 0], p[:2, 1], marker="*", c="b")
plt.scatter(p[2:, 0], p[2:, 1], marker=".", c="b")
plt.plot(a, db1, c="b")
plt.plot(a, db2, c="b")

plt.show()

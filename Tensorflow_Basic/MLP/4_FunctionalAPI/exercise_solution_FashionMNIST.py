# %% -------------------------------------------------------------------------------------------------------------------
# Fit a MLP to the FashionMNIST dataset: https://github.com/zalandoresearch/fashion-mnist
# %% --------------------------------------- Imports -------------------------------------------------------------------
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 1e-3
N_NEURONS = (100, 100)
N_EPOCHS = 30
BATCH_SIZE = 512
DROPOUT = 0.2

# %% ----------------------------------- Helper Functions --------------------------------------------------------------
# 2. Retrieve the output of the Add layer and the output of the BatchNorm layer and plot them
# Hint: https://github.com/keras-team/keras/issues/2664
def plot_intermidiate(x, idx, net):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle("Left: Intermidiate output + Input \n Right: Intermidiate output + Input after BatchNorm")
    # Defines a new model until the layer we want the output of. The weights will be the same.
    sub_model = Model(inputs=net.layers[0].input, outputs=net.get_layer("Add").output)
    mmm = sub_model.predict(x[idx].reshape(1, -1))
    ax1.imshow(mmm.reshape(28, 28))
    model = Model(inputs=net.layers[0].input, outputs=net.get_layer("Batch").output)
    mmm = model.predict(x_test[0].reshape(1, -1))
    ax2.imshow(mmm.reshape(28, 28))
    plt.show()


# %% -------------------------------------- Data Prep ------------------------------------------------------------------
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train.reshape(len(x_train), -1), x_test.reshape(len(x_test), -1)
y_train, y_test = to_categorical(y_train, num_classes=10), to_categorical(y_test, num_classes=10)

# %% -------------------------------------- Training Prep ----------------------------------------------------------
inputs = Input(shape=(784,))
x = Dense(N_NEURONS[0], activation='relu')(inputs)
x = Dense(784, activation='relu', name="Intermidiate")(x)
x = Dropout(DROPOUT)(x)
# 1. Use the functional API to add the input to the model to the output
# of a Dense Layer with 768 neurons, and then Use BatchNorm on this
add_out = Add(name="Add")([x, inputs])
bn_out = BatchNormalization(name="Batch")(add_out)
x = Dense(N_NEURONS[1], activation='relu')(bn_out)
x = Dropout(DROPOUT)(x)
probs = Dense(10, activation="softmax")(x)
model = Model(inputs=inputs, outputs=probs)
model.compile(optimizer=Adam(lr=LR), loss="categorical_crossentropy", metrics=["accuracy"])

# %% -------------------------------------- Training Loop ----------------------------------------------------------
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS, validation_data=(x_test, y_test))

# %% ------------------------------------------ Final test -------------------------------------------------------------
y_test_pred = np.argmax(model.predict(x_test), axis=1)
print("The accuracy on the test set is", 100*accuracy_score(np.argmax(y_test, axis=1), y_test_pred), "%")
plot_intermidiate(x_test, 0, model)

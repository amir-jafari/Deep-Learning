# %% -------------------------------------------------------------------------------------------------------------------
# Fit a MLP to the FashionMNIST dataset: https://github.com/zalandoresearch/fashion-mnist
# %% --------------------------------------- Imports -------------------------------------------------------------------
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(42)

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


# %% ----------------------------------- Helper Functions --------------------------------------------------------------
# 6. Define a function to show some images that were incorrectly classified
def show_mistakes(x, y, idx):
    label_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    y_pred = np.argmax(model.predict(x), axis=1)
    y = np.argmax(y, axis=1)
    idx_mistakes = np.argwhere((y == y_pred) == 0).flatten()
    plt.title("MLP prediction: {} - True label: {}".format(label_names[y_pred[idx_mistakes[idx]]],
                                                           label_names[y[idx_mistakes[idx]]]))
    plt.imshow(x[idx_mistakes[idx]].reshape(28, 28))
    plt.show()


# %% -------------------------------------- Data Prep ------------------------------------------------------------------
# 1. Download the data using keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train.reshape(len(x_train), -1), x_test.reshape(len(x_test), -1)
y_train, y_test = to_categorical(y_train, num_classes=10), to_categorical(y_test, num_classes=10)

# %% -------------------------------------- Training Prep ----------------------------------------------------------
model = Sequential([
    Dense(N_NEURONS[0], input_dim=784),
    Activation("relu"),
    Dropout(DROPOUT, seed=SEED),
    BatchNormalization()
])
for n_neurons in N_NEURONS[1:]:
    model.add(Dense(n_neurons, activation="relu"))
    model.add(Dropout(DROPOUT))
    model.add(BatchNormalization())
model.add(Dense(10, activation="softmax"))
model.compile(optimizer=Adam(lr=LR), loss="categorical_crossentropy", metrics=["accuracy"])

# %% -------------------------------------- Training Loop ----------------------------------------------------------
if TRAIN:
    # 3. Add an option to save the model on each epoch, and stop saving it when the validation
    # loss begins to increase (early stopping) - https://keras.io/callbacks/: ModelCheckpoint
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS, validation_data=(x_test, y_test),
              callbacks=[ModelCheckpoint("mlp_fashionmnist.hdf5", monitor="val_loss", save_best_only=True)])

# %% ------------------------------------------ Final test -------------------------------------------------------------
# 4. Add an option to only test the model, by loading the model you saved on the training phase
model = load_model('mlp_fashionmnist.hdf5')
y_test_pred = np.argmax(model.predict(x_test), axis=1)
print("The accuracy on the test set is", 100*accuracy_score(np.argmax(y_test, axis=1), y_test_pred), "%")
# 5. Print out the confusion matrix
print("The confusion matrix is")
print(confusion_matrix(np.argmax(y_test, axis=1), y_test_pred))

show_mistakes(x_test, y_test, 1)

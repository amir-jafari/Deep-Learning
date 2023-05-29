# %% -------------------------------------------------------------------------------------------------------------------
# Fit a MLP to the FashionMNIST dataset: https://github.com/zalandoresearch/fashion-mnist
# %% --------------------------------------- Imports -------------------------------------------------------------------
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
tf.compat.v1.disable_eager_execution()

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 1e-3
N_NEURONS = (100, 100)
N_EPOCHS = 1
BATCH_SIZE = 512
DROPOUT = 0.2

# %% -------------------------------------- Data Prep ------------------------------------------------------------------
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train.reshape(len(x_train), -1), x_test.reshape(len(x_test), -1)
y_train, y_test = to_categorical(y_train, num_classes=10), to_categorical(y_test, num_classes=10)


# %% -------------------------------------- Training Prep ----------------------------------------------------------
# 1. Use model subclassing to mimic 4_FunctionalAPI/exercise_FashionMNIST.py
class MLP(Model):
    def __init__(self, return_intermidiate=False):
        super(MLP, self).__init__()
        self.return_intermidiate = return_intermidiate
        self.dense1 = Dense(N_NEURONS[0], activation='relu')
        self.dense2 = Dense(784, activation='relu')
        self.drop = Dropout(DROPOUT)
        self.bn = BatchNormalization()
        self.dense3 = Dense(N_NEURONS[1], activation="relu")
        self.out = Dense(10, activation="softmax")

    def call(self, x):
        xx = self.drop(self.dense2(self.dense1(x)))
        xx = self.bn(xx + tf.cast(x, tf.float32))
        if self.return_intermidiate:
            return xx
        return self.out(self.drop(self.dense3(xx)))


model = MLP()
model.compile(optimizer=Adam(lr=LR), loss="categorical_crossentropy", metrics=["accuracy"])

# %% -------------------------------------- Training Loop ----------------------------------------------------------
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS, validation_data=(x_test, y_test))

# %% ------------------------------------------ Final test -------------------------------------------------------------
y_test_pred = np.argmax(model.predict(x_test), axis=1)
print("The accuracy on the test set is", 100*accuracy_score(np.argmax(y_test, axis=1), y_test_pred), "%")

# 2. Retrieve the output of the BatchNorm layer and plot it
model.return_intermidiate = True  # Sets this to True so that xx will be return when calling model.call
mmm = model.call(tf.cast(tf.reshape(x_test[0], (1, -1)), tf.float32))
# Not able to get the actual data from here...!




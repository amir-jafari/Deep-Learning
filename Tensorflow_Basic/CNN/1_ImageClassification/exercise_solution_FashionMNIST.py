# %% --------------------------------------- Imports -------------------------------------------------------------------
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, ZeroPadding2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
import tensorflow


# %% --------------------------------------- Set-Up --------------------------------------------------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 1e-3
N_EPOCHS = 30
BATCH_SIZE = 128
num_classes = 10
img_rows, img_cols = 28, 28
# %% -------------------------------------- Data Prep ------------------------------------------------------------------
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# Reshapes to (n_examples, n_channels, height_pixels, width_pixels)
if tf.keras.backend.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
# x_train, x_test = x_train.reshape(len(x_train), 1, 28, 28), x_test.reshape(len(x_test), 1, 28, 28)
# y_train, y_test = to_categorical(y_train, num_classes=10), to_categorical(y_test, num_classes=10)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)
# %% -------------------------------------- Training Prep ----------------------------------------------------------
# 3. Try modifying the architecture and the hyper-parameters to get a better performance.
# Try including more conv layers and more kernels in each layer. This can allow for less MLP layers at the end.
# To do so, you will need to play around with zero-padding and maybe stack two conv layers together without any pooling.
# You can also remove the pooling and the MLP, and replace it with a final Global Average Pooling layer.
model = Sequential([
    Conv2D(16, 3, activation="relu"),
    BatchNormalization(),
    ZeroPadding2D(2),
    Conv2D(32, 3, activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2),
    Conv2D(64, 3, activation="relu"),
    BatchNormalization(),
    Conv2D(128, 3, activation="relu"),
    GlobalAveragePooling2D(),
    Dense(10, activation="softmax"),
])

model.compile(optimizer=Adam(lr=LR), loss="categorical_crossentropy", metrics=["accuracy"])

# %% -------------------------------------- Training Loop ----------------------------------------------------------
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS, validation_data=(x_test, y_test),
          callbacks=[ModelCheckpoint("cnn_fashionmnist.hdf5", monitor="val_loss", save_best_only=True)])

# %% ------------------------------------------ Final test -------------------------------------------------------------
print("Final accuracy on validations set:", 100*model.evaluate(x_test, y_test)[1], "%")


# %% --------------------------------------- Imports -------------------------------------------------------------------
import os
import random
import numpy as np
import torch
import torchaudio
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
tf.random.set_seed(SEED)
DATA_PATH = os.getcwd()# This should be updated to the location where GTZAN is stored

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 1e-3
N_EPOCHS = 50
BATCH_SIZE = 16
DROPOUT = 0.5
SEQ_LEN = 10  # seconds
HIDDEN_SIZES = [256, 128]
SAMPLE_RATE = 22050


# %% ----------------------------------- Helper Functions --------------------------------------------------------------
def load_data():
    dataset = torchaudio.datasets.GTZAN(root=DATA_PATH, download=True)
    x, y = [], []
    for waveform, label in dataset:
        waveform = waveform.mean(dim=0)  # Convert to mono
        waveform = waveform.numpy()
        length = waveform.shape[0]
        target_length = SAMPLE_RATE * SEQ_LEN

        if length > target_length:
            waveform = waveform[:target_length]  # Trim
        else:
            waveform = np.pad(waveform, (0, target_length - length))  # Pad

        x.append(waveform)
        y.append(label)

    return np.array(x), np.array(y)


# %% -------------------------------------- Data Prep ------------------------------------------------------------------
x, y = load_data()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=SEED, stratify=y)
x_train, x_test = x_train.reshape(len(x_train), -1, 1), x_test.reshape(len(x_test), -1, 1)
y_train, y_test = tf.keras.utils.to_categorical(y_train, num_classes=10), tf.keras.utils.to_categorical(y_test,
                                                                                                        num_classes=10)

# %% -------------------------------------- Training Prep ----------------------------------------------------------
model1 = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(8, 3, activation="relu"),
    tf.keras.layers.MaxPooling1D(4),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv1D(16, 3, activation="relu"),
    tf.keras.layers.MaxPooling1D(4),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv1D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling1D(4),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv1D(64, 3, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(4),
    tf.keras.layers.Conv1D(128, 3, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(4),
    tf.keras.layers.Conv1D(256, 3, activation="relu"),
    tf.keras.layers.MaxPooling1D(4),
    tf.keras.layers.BatchNormalization(),
])
HIDDEN_SIZES[0] = 256
for hidden_size in HIDDEN_SIZES[:-1]:
    model1.add(
        tf.keras.layers.LSTM(units=hidden_size, dropout=DROPOUT, recurrent_dropout=DROPOUT, return_sequences=True))
model1.add(tf.keras.layers.LSTM(units=HIDDEN_SIZES[-1], dropout=DROPOUT, recurrent_dropout=DROPOUT))
model1.add(tf.keras.layers.Dense(10, activation="softmax"))
model1.compile(optimizer=tf.keras.optimizers.RMSprop(lr=LR), loss="categorical_crossentropy", metrics=["accuracy"])

# %% -------------------------------------- Training Loop ----------------------------------------------------------
print("Starting training loop...")
model1.fit(tf.cast(x_train, tf.float32), tf.cast(y_train, tf.float32), batch_size=BATCH_SIZE, epochs=N_EPOCHS,
           validation_data=(tf.cast(x_test, tf.float32), tf.cast(y_test, tf.float32)),
           callbacks=[tf.keras.callbacks.ModelCheckpoint("example_cnn_lstm_music_genre_classifier.hdf5",
                                                         monitor="val_accuracy", save_best_only=True)])

# %% ------------------------------------------ Final Test -------------------------------------------------------------
model = tf.keras.models.load_model('example_cnn_lstm_music_genre_classifier.hdf5')
print("Final accuracy on validation set:", 100 * model.evaluate(x_test, y_test)[1], "%")
print("Confusion Matrix:")
print(confusion_matrix(np.argmax(y_test, axis=1), np.argmax(model.predict(x_test), axis=1)))

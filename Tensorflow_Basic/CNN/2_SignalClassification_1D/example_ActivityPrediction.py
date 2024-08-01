# %% --------------------------------------- Imports -------------------------------------------------------------------
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix

# Check if dataset is already downloaded and extract it
if "WISDM_ar_v1.1" not in os.listdir(os.getcwd()):
    try:
        os.system("wget http://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz")
        os.system("tar -xvzf WISDM_ar_latest.tar.gz")
    except:
        print("There was a problem downloading the data!!")
        raise
    if "WISDM_ar_v1.1" not in os.listdir(os.getcwd()):
        print("There was a problem downloading the data!!")
        import sys
        sys.exit()

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 1e-4
N_EPOCHS = 15
BATCH_SIZE = 128
DROPOUT = 0.5
DATA_PATH = os.path.join(os.getcwd(), "WISDM_ar_v1.1", "WISDM_ar_v1.1_raw.txt")

# %% -------------------------------------- Data Prep ------------------------------------------------------------------
data_raw = pd.read_csv(DATA_PATH, on_bad_lines='skip', header=None)
data_raw.dropna(inplace=True)  # There is only one NaN
data_raw[5] = data_raw[5].apply(lambda f: float(f[:-1]))  # Fixes last column (z-axis)
groups = data_raw.groupby(1)  # Groups by type of activity (classes)

# Loops over the data to get signals per class
x, y, min_signal_length = [], [], 10000000
for label, group in groups:
    for user in np.unique(group[0]):
        x.append(group[group[0] == user][[3, 4, 5]].values)
        y.append(label)
        if min_signal_length > len(x[-1]):
            min_signal_length = len(x[-1])

# Uses the min_signal_length to get fixed length signals, but this means losing some data
x_prep, y_prep = [], []
for signal, label in zip(x, y):
    for i in range(len(signal)//min_signal_length):  # Losing some data here
        x_prep.append(signal[min_signal_length*i:min_signal_length*(i+1)])
        y_prep.append(label)

x, y = np.array(x_prep), np.array(y_prep)
del x_prep, y_prep
print(np.unique(y, return_counts=True))

# Final usual pre-processing
le = LabelEncoder()
y = le.fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=SEED, test_size=0.2, stratify=y)
sd = StandardScaler()
x_train = sd.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
x_test = sd.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)
y_train, y_test = to_categorical(y_train, num_classes=6), to_categorical(y_test, num_classes=6)

# %% -------------------------------------- Training Prep ----------------------------------------------------------
model = Sequential([
    Conv1D(16, 136, activation="relu", input_shape=(x_train.shape[1], x_train.shape[2])),  # Adjust input shape
    BatchNormalization(),
    Conv1D(32, 136, activation="relu"),
    BatchNormalization(),
    Conv1D(64, 136, activation="relu"),
    BatchNormalization(),
    Conv1D(128, 136, activation="relu"),
    BatchNormalization(),
    Flatten(),
    Dense(136, activation="relu"),
    Dropout(DROPOUT),
    BatchNormalization(),
    Dense(6, activation="softmax")
])
model.compile(optimizer=Adam(learning_rate=LR), loss="categorical_crossentropy", metrics=["accuracy"])

# %% -------------------------------------- Training Loop ----------------------------------------------------------
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS, validation_data=(x_test, y_test))

# %% ------------------------------------------ Final test -------------------------------------------------------------
print("Final accuracy on validation set:", 100*model.evaluate(x_test, y_test)[1], "%")
y_test_pred = np.argmax(model.predict(x_test), axis=1)
print(le.inverse_transform(np.arange(6)))
print(confusion_matrix(np.argmax(y_test, axis=1), y_test_pred))

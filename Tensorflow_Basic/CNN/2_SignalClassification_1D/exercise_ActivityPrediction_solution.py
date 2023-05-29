# %% --------------------------------------- Imports -------------------------------------------------------------------
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix

if "WISDM_ar_v1.1" not in os.listdir(os.getcwd()):
    try:
        os.system("wget http://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz")
        os.system("tar -xvzf WISDM_ar_latest.tar.gz")
    except:
        print("There was a problem downloading the data!!")
        raise
        # Go to http://www.cis.fordham.edu/wisdm/dataset.php and click Download Latest version, and untar to current dir
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
DATA_PATH = os.getcwd() + "/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt"
# 1. The previous pre-processing was very naive because the model was trained on very long sequences, even when it was
# the shortest sequence length in our dataset. A good model should be able to recognize human activity right after the
# user starts doing such activity. WISDM_ar_v1.1_raw_about.txt states that the time step is 50 milliseconds.
# If we want the model to tell us what the user has started to do after say 3 seconds, it should work for input signals
# of 3/0.05 = 60 time steps.
# Define a hyper-parameter called MIN_SIGNAL_DUR that is equal to this number of seconds and pre-process accordingly.
MIN_SIGNAL_DUR = 3

# %% -------------------------------------- Data Prep ------------------------------------------------------------------
data_raw = pd.read_csv(DATA_PATH, error_bad_lines=False, header=None)
data_raw.dropna(inplace=True)
data_raw[5] = data_raw[5].apply(lambda f: float(f[:-1]))
groups = data_raw.groupby(1)
x, y = [], []
for label, group in groups:
    for user in np.unique(group[0]):
        x.append(group[group[0] == user][[3, 4, 5]].values)
        y.append(label)
x_prep, y_prep, min_signal_length = [], [], int(MIN_SIGNAL_DUR/0.05)
for signal, label in zip(x, y):
    for i in range(len(signal)//min_signal_length):
        x_prep.append(signal[min_signal_length*i:min_signal_length*(i+1)])
        y_prep.append(label)
x, y = np.array(x_prep), np.array(y_prep)
del x_prep, y_prep
print(np.unique(y, return_counts=True))
le = LabelEncoder()
y = le.fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=SEED, test_size=0.2, stratify=y)
sd = StandardScaler()
y_train, y_test = to_categorical(y_train, num_classes=6), to_categorical(y_test, num_classes=6)

# %% -------------------------------------- Training Prep ----------------------------------------------------------
# 2. The previous model will not work with this new input sequence length. It could also be argued that the convolutions
# had kernel sizes that were too big, compared to the usual kernel size in the literature. Modify the CNN & Dense layers
# to work with the new input shape and train the model. Follow the same pattern we used on the example and compare.
model = Sequential([
    Conv1D(16, 15, activation="relu"),
    BatchNormalization(),
    Conv1D(32, 15, activation="relu"),
    BatchNormalization(),
    Conv1D(64, 15, activation="relu"),
    BatchNormalization(),
    Conv1D(128, 15, activation="relu"),
    BatchNormalization(),
    Flatten(),
    Dense(15, activation="relu"),
    Dropout(DROPOUT),
    BatchNormalization(),
    Dense(6, activation="softmax")
])

# 3. Even the kernel size from 2. could be considered too big. Modify the model from 2. to use kernels size of 3. Due to
# this, we would need some pooling in-between CNN layers in order not to have a very huge model. Also, replace the
# penultimate dense layer and the flatten operation with a Global Average Pooling layer.
# Refer to 1_ImageClassification/exercise_solution_FashionMNIST for an example. Train the model and compare with 2.
model1 = Sequential([
    Conv1D(16, 3, activation="relu"),
    BatchNormalization(),
    Conv1D(32, 3, activation="relu"),
    BatchNormalization(),
    MaxPooling1D(2),
    Conv1D(64, 3, activation="relu"),
    BatchNormalization(),
    Conv1D(128, 3, activation="relu"),
    BatchNormalization(),
    MaxPooling1D(2),
    Conv1D(256, 3, activation="relu"),
    GlobalAveragePooling1D(),
    Dense(6, activation="softmax")
])

model.compile(optimizer=Adam(lr=LR), loss="categorical_crossentropy", metrics=["accuracy"])
model1.compile(optimizer=Adam(lr=LR), loss="categorical_crossentropy", metrics=["accuracy"])

# %% -------------------------------------- Training Loop ----------------------------------------------------------
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS, validation_data=(x_test, y_test))
model1.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS, validation_data=(x_test, y_test))

# %% ------------------------------------------ Final test -------------------------------------------------------------
print("Final accuracy on validations set:", 100*model.evaluate(x_test, y_test)[1], "%")
y_test_pred = np.argmax(model.predict(x_test), axis=1)
print(le.inverse_transform(np.array([0, 1, 2, 3, 4, 5])))
print(confusion_matrix(np.argmax(y_test, axis=1), y_test_pred))

print("Final accuracy on validations set:", 100*model1.evaluate(x_test, y_test)[1], "%")
y_test_pred = np.argmax(model1.predict(x_test), axis=1)
print(le.inverse_transform(np.array([0, 1, 2, 3, 4, 5])))
print(confusion_matrix(np.argmax(y_test, axis=1), y_test_pred))

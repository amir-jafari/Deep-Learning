# %% -------------------------------------------------------------------------------------------------------------------

# -------------------------------------
# Use a GRU to forecast a Chirp signal
# -------------------------------------

# %% --------------------------------------- Imports -------------------------------------------------------------------
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam
from scipy.signal import chirp
import matplotlib.pyplot as plt

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
PLOT_SIGNAL, PLOT_RESULT = False, True

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 1e-3
N_EPOCHS = 5
BATCH_SIZE = 20
DROPOUT = 0
SEQ_LEN = 50
HIDDEN_SIZE = 16
STATEFUL = True
LOSS_WHOLE_SEQ = True

# %% -------------------------------------- Data Prep ------------------------------------------------------------------
time_steps = np.linspace(0, 10, 5000)
x = chirp(time_steps, f0=1, f1=0.1, t1=10, method='linear')
if PLOT_SIGNAL:
    plt.plot(time_steps, x)
    plt.show()
x_train, x_test = x[:int(0.75*len(x))], x[int(0.75*len(x)):]
x_train_prep = np.empty((len(x_train)-SEQ_LEN, SEQ_LEN, 1))
if LOSS_WHOLE_SEQ:
    y_train_prep = np.empty((len(x_train)-SEQ_LEN, SEQ_LEN, 1))
else:
    y_train_prep = np.empty((len(x_train)-SEQ_LEN, 1))
for idx in range(len(x_train)-SEQ_LEN):
    x_train_prep[idx, :, :] = x_train[idx:SEQ_LEN+idx].reshape(-1, 1)
    if LOSS_WHOLE_SEQ:
        y_train_prep[idx, :] = x_train[idx+1:SEQ_LEN+idx+1].reshape(-1, 1)
    else:
        y_train_prep[idx, :] = x_train[SEQ_LEN+idx].reshape(1, 1)
x_test_prep = np.empty((len(x_test)-SEQ_LEN, SEQ_LEN, 1))
if LOSS_WHOLE_SEQ:
    y_test_prep = np.empty((len(x_test)-SEQ_LEN, SEQ_LEN, 1))
else:
    y_test_prep = np.empty((len(x_test) - SEQ_LEN, 1))
for idx in range(len(x_test)-SEQ_LEN):
    x_test_prep[idx, :, :] = x_test[idx:SEQ_LEN+idx].reshape(-1, 1)
    if LOSS_WHOLE_SEQ:
        y_test_prep[idx, :] = x_test[idx+1:SEQ_LEN+idx+1].reshape(-1, 1)
    else:
        y_test_prep[idx, :] = x_test[SEQ_LEN+idx].reshape(1, 1)
x_train, y_train, x_test, y_test = x_train_prep, y_train_prep, x_test_prep, y_test_prep
del x_train_prep, y_train_prep, x_test_prep, y_test_prep

# %% -------------------------------------- Training Prep ----------------------------------------------------------
if HIDDEN_SIZE is None:
    model = Sequential([GRU(units=1, dropout=DROPOUT, stateful=STATEFUL,
                            batch_input_shape=(BATCH_SIZE, SEQ_LEN, 1) if STATEFUL else None,
                            return_sequences=True if LOSS_WHOLE_SEQ else False)])
else:
    model = Sequential([GRU(units=HIDDEN_SIZE, dropout=DROPOUT, stateful=STATEFUL,
                            batch_input_shape=(BATCH_SIZE, SEQ_LEN, 1) if STATEFUL else None,
                            return_sequences=True if LOSS_WHOLE_SEQ else False),
                        Dense(1, activation="linear")])
model.compile(optimizer=Adam(lr=LR), loss="mean_squared_error")

# %% -------------------------------------- Training Loop ----------------------------------------------------------
print("Starting training loop...")
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS, validation_data=(x_test, y_test))

# %% ------------------------------------------ Final Test -------------------------------------------------------------
if PLOT_RESULT:
    pred_train = model.predict(x_train, batch_size=BATCH_SIZE)
    pred_test = model.predict(x_test, batch_size=BATCH_SIZE)
    if LOSS_WHOLE_SEQ:
        predictions_train, predictions_test = [], []
        for idx in range(len(pred_train)):
            predictions_train.append(pred_train[idx, -1].reshape(-1))
        for idx in range(len(pred_test)):
            predictions_test.append(pred_test[idx, -1].reshape(-1))
        pred_train, pred_test = np.array(predictions_train), np.array(predictions_test)
    plt.plot(time_steps, x, label="Real Time Series", linewidth=2)
    plt.plot(time_steps[SEQ_LEN:len(pred_train)+SEQ_LEN],
             pred_train, linestyle='dashed', label="Train Prediction")
    plt.scatter(time_steps[len(pred_train)+2*SEQ_LEN:],
                pred_test, color="y", label="Test Prediction")
    plt.title("Chirp function with 1 Hz frequency at t=0 and 0.1 Hz freq at t=10\n"
              "GRU predictions using the previous {} time steps".format(SEQ_LEN))
    plt.xlabel("Time"); plt.ylabel("Signal Intensity")
    plt.legend()
    plt.show()

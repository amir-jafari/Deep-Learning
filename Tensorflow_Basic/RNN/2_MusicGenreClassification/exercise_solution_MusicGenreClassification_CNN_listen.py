# %% --------------------------------------- Imports -------------------------------------------------------------------
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv1D, BatchNormalization, LSTM, Dense, MaxPooling1D, Input
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from scipy.io import wavfile
from sklearn.metrics import confusion_matrix

if "genres" not in os.listdir(os.getcwd()):
    try:
        os.system("wget http://opihi.cs.uvic.ca/sound/genres.tar.gz")
        os.system("tar -xvzf genres.tar.gz")
    except:
        print("There was an error trying to download the data!")
        # Go to http://marsyas.info/downloads/datasets.html and Download the GTZAN genre collection (Approximately 1.2GB)
    if "genres" not in os.listdir(os.getcwd()):
        print("There was an error trying to download the data!")
        import sys
        sys.exit()

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
DATA_PATH = "/genres"

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 1e-3
N_EPOCHS = 50
BATCH_SIZE = 16
DROPOUT = 0.5
SEQ_LEN = 10  # seconds
NUM_KERNELS, KERNEL_SIZE, POOL_SIZE = [8, 16, 32, 64, 128, 256], 3, 4
HIDDEN_SIZES = [256, 128]
# 4. Add an option to only load a saved model, do so and play around with the function defined in 3.
ONLY_GET_INTERM_OUT = False

# %% ----------------------------------- Helper Functions --------------------------------------------------------------
def get_max_length():
    max_length = 0
    for subdir in [f for f in os.listdir(os.getcwd() + DATA_PATH) if os.path.isdir(os.getcwd() + DATA_PATH + "/" + f)]:
        for file in os.listdir(os.getcwd() + DATA_PATH + "/" + subdir):
            _, example = wavfile.read(os.getcwd() + DATA_PATH + "/" + subdir + "/" + file)
            if len(example) > max_length:
                max_length = len(example)
    return max_length

def load_data():
    x, y, label, label_dict = [], [], 0, {}
    for subdir in [f for f in os.listdir(os.getcwd() + DATA_PATH) if os.path.isdir(os.getcwd() + DATA_PATH + "/" + f)]:
        label_dict[subdir] = label
        for file in os.listdir(os.getcwd() + DATA_PATH + "/" + subdir):
            _, example = wavfile.read(os.getcwd() + DATA_PATH + "/" + subdir + "/" + file)
            if len(example) > 22050*SEQ_LEN:  # Trims from the beginning to max_seq_length
                example = example[:22050*SEQ_LEN]  # 22050 is the sampling frequency (22050 samples per second)
            else:  # Pads up to the max_seq_length
                example = np.hstack((example, np.zeros((22050*SEQ_LEN - len(example)))))
            x.append(example)
            y.append(label)
        label += 1
    return np.array(x), np.array(y), label_dict

# 3. Define a function that takes as input an example, the model, the layer name, the path to save the file,
# the channel id and an option to average all the channels of the intermediate output, and that saves the output of
# such layer to a .wav file. You can use scipy.io.wavfile.write
def get_intermediate(example, net, layer_name, path, channel_idx=0, avg_channels=False):
    sub_model = Model(inputs=model.layers[0].input, outputs=net.get_layer(layer_name).output)
    mmm = sub_model.predict(example.reshape(1, -1, 1))
    if avg_channels:
        mmm = np.mean(mmm[0, :], axis=1)
    else:
        mmm = mmm[0, :, channel_idx]
    wavfile.write(path, 22050, mmm)

# %% -------------------------------------- Data Prep ------------------------------------------------------------------
if SEQ_LEN == "get_max_from_data":
    SEQ_LEN = get_max_length()//22050  # In seconds
x, y, labels = load_data()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=SEED, stratify=y)
x_train, x_test = x_train.reshape(len(x_train), -1, 1), x_test.reshape(len(x_test), -1, 1)
y_train, y_test = to_categorical(y_train, num_classes=10), to_categorical(y_test, num_classes=10)
del x, y

if not ONLY_GET_INTERM_OUT:
    # %% -------------------------------------- Training Prep ----------------------------------------------------------
    # 1. Re-implement the network from the example, but this time using Model (see MLP/4_FunctionalAPI), and give each
    # layer a distinct name. You can use a for loop to write all the CNN architecture in a couple of lines
    inputs = Input((None, 1))

    cnn = Conv1D(NUM_KERNELS[0], KERNEL_SIZE, activation="relu", name="cnn1")(inputs)
    pool = MaxPooling1D(POOL_SIZE, name="pool1")(cnn)
    bn = BatchNormalization(name="bn1")(pool)
    for idx, n_kernels in enumerate(NUM_KERNELS[1:]):
        cnn = Conv1D(n_kernels, KERNEL_SIZE, activation="relu", name="cnn{}".format(idx+2))(bn)
        pool = MaxPooling1D(POOL_SIZE, name="pool{}".format(idx+2))(cnn)
        bn = BatchNormalization(name="bn{}".format(idx+2))(pool)

    HIDDEN_SIZES[0], lstm = NUM_KERNELS[-1], bn
    for hidden_size in HIDDEN_SIZES[:-1]:
        lstm = LSTM(units=hidden_size, dropout=DROPOUT, return_sequences=True)(lstm)
    lstm = LSTM(units=HIDDEN_SIZES[-1], dropout=DROPOUT)(lstm)
    probs = Dense(10, activation="softmax")(lstm)

    model = Model(inputs=inputs, outputs=probs)
    model.compile(optimizer=RMSprop(lr=LR), loss="categorical_crossentropy", metrics=["accuracy"])

    # %% -------------------------------------- Training Loop ----------------------------------------------------------
    print("Starting training loop...")
    # 2. Train the model and save it
    model.fit(tf.cast(x_train, tf.float32), tf.cast(y_train, tf.float32), batch_size=BATCH_SIZE, epochs=N_EPOCHS,
               validation_data=(tf.cast(x_test, tf.float32), tf.cast(y_test, tf.float32)),
               callbacks=[ModelCheckpoint("exercise_cnn_lstm_music_genre_classifier.hdf5", monitor="val_accuracy",
                                          save_best_only=True)])

# %% ------------------------------------------ Final Test -------------------------------------------------------------
# 4. Add an option to only load a saved model, do so and play around with the function defined in 3.
model = load_model('exercise_cnn_lstm_music_genre_classifier.hdf5')
print("Final accuracy on validations set:", 100*model.evaluate(x_test, y_test)[1], "%")
print(labels)
print("The confusion matrix is:")
print(confusion_matrix(np.argmax(y_test, axis=1), np.argmax(model.predict(x_test), axis=1)))

# %% ------------------------------------------ Output CNN -------------------------------------------------------------
get_intermediate(x_test[0], model, "cnn1", "0_1.wav")
get_intermediate(x_test[0], model, "cnn2", "0_2.wav")
get_intermediate(x_test[0], model, "cnn1", "0_1_avg.wav", avg_channels=True)
get_intermediate(x_train[40], model, "cnn1", "40_1.wav")

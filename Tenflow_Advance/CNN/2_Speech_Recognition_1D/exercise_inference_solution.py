# %% -------------------------------------------------------------------------------------------------------------------

# % -------------------------------------------------------------------------------------
# Use the model from the example to make inference on self-recorded spoken digits
# % -------------------------------------------------------------------------------------

# %% --------------------------------------- Imports -------------------------------------------------------------------
import numpy as np
import tensorflow as tf
from scipy.io import wavfile


# %% --------------------------------------- Set-Up --------------------------------------------------------------------
SAVE_MODEL_PATH = "example_saved_model/"


# %% --------------------------------------- Helper Functions ----------------------------------------------------------
# 1. Define a function to load the "zero.wav", "one.wav", etc. files and preprocess it to input into the model
def load_data(file_path, max_seq_length=1):
    _, example = wavfile.read(file_path)
    if len(example) > 8000*max_seq_length:
        example = example[:8000*max_seq_length]
    else:
        example = np.hstack((example, np.zeros((8000*max_seq_length - len(example)))))
    return tf.dtypes.cast(tf.convert_to_tensor([example.reshape(-1, 1)]), tf.float32)


# 2. Define a function to take as input the pre-processed tensor and return the predicted label and probabilities
def predict(file_path, return_probs=False):
    x = load_data(file_path)
    if return_probs:
        return tf.argmax(model(x), axis=1).numpy().item(), np.round(tf.nn.softmax(model(x)).numpy(), 3)
    else:
        return tf.argmax(model(x), axis=1).numpy().item()


# %% -------------------------------------- CNN Class ------------------------------------------------------------------
# 3. Re-define the model class from the example
class CNN(tf.keras.Model):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = tf.keras.layers.Conv1D(16, 30)
        self.convnorm1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv1D(32, 30)
        self.convnorm2 = tf.keras.layers.BatchNormalization()
        self.pool2 = tf.keras.layers.MaxPool1D(2)

        self.conv3 = tf.keras.layers.Conv1D(64, 30)
        self.convnorm3 = tf.keras.layers.BatchNormalization()

        self.conv4 = tf.keras.layers.Conv1D(128, 30)
        self.convnorm4 = tf.keras.layers.BatchNormalization()
        self.pool4 = tf.keras.layers.MaxPool1D(2)

        self.conv5 = tf.keras.layers.Conv1D(256, 30)
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.linear = tf.keras.layers.Dense(10)

        self.act = tf.nn.relu
        self.training = True

    def call(self, x):
        x = self.convnorm1(self.act(self.conv1(x)), training=self.training)
        x = self.pool2(self.convnorm2(self.act(self.conv2(x)), training=self.training))
        x = self.convnorm3(self.act(self.conv3(x)), training=self.training)
        x = self.pool4(self.convnorm4(self.act(self.conv4(x)), training=self.training))
        return self.linear(self.global_avg_pool(self.act(self.conv5(x))))


# %% -------------------------------------- Inference Prep  ------------------------------------------------------------
# 4. Load the model and set it up for inference
model = CNN()
model_path = SAVE_MODEL_PATH + "cnn_spoken_digit_recognizer"
model.load_weights(model_path)
model.training = False

# %% ----------------------------------------- Inference ---------------------------------------------------------------
# 5. Print out the real labels, and the predicted labels and predicted probabilities of each label by the model
for sound in ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]:
    print("\nreal label:", sound)
    pred, probs = predict(sound + ".wav", return_probs=True)
    print("pred label:", pred)
    print("label probs:", probs)

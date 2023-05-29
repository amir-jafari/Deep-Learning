# %% -------------------------------------------------------------------------------------------------------------------

# 4. Create a training file that takes as inputs the dataset and the experiment id to use, loads the corresponding model
# and config, and tests this model on the test set (this makes more sense when we have a held-out set)

# %% --------------------------------------- Imports -------------------------------------------------------------------
import os
import json
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score


DATASET = "fashionmnist"
RUN = 4
with open(os.getcwd() + "/exercise_{}_mlp_runs/{}/config.json".format(DATASET, RUN),
          "r") as s:
    config = json.load(s)


# %% ----------------------------------------- Set-Up ------------------------------------------------------------------
tf.random.set_seed(config["random_seed"])
np.random.seed(config["random_seed"])

# %% ----------------------------------------- Data Prep ---------------------------------------------------------------
if DATASET == "mnist":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
else:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_test, y_test = tf.reshape(x_test, (len(x_test), -1)), tf.convert_to_tensor(y_test)

# %% ------------------------------------------------ Testing ----------------------------------------------------------
model = tf.saved_model.load(os.getcwd() + "//exercise_{}_mlp_runs/{}/exercise_mlp_{}/".format(
            DATASET, RUN, DATASET))
model.training, model.drop = False, 0
print("The test accuracy is", accuracy_score(y_test, tf.argmax(model(x_test), axis=1).numpy())*100)

# %% -------------------------------------------------------------------------------------------------------------------

# % -----------------------------------------------------------------------------------------------------
# Modify the data prep in order to have a generator instead of loading the whole data into memory at once
# % -----------------------------------------------------------------------------------------------------

# %% --------------------------------------- Imports -------------------------------------------------------------------
import os
from shutil import copyfile
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
from PIL import Image

if "jpg" and "imagelabels.mat" in os.listdir(os.getcwd()):
    pass
else:
    try:
        os.system("wget http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz")
        os.system("wget http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat")
        os.system("tar -xvzf 102flowers.tgz")
    except:
        print("There was an error downloading the data!")
        raise
        # Go to http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html and click "Dataset images" and "The image labels"
        # Unzip 102flowers.tgz on the current working directory and move imagelabels.mat also in there

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
DATA_DIR = "/jpg/"
TRAIN_TEST_SPLIT = False

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 1e-2
N_NEURONS = (100, 200, 100)
N_EPOCHS = 100
BATCH_SIZE = 512
DROPOUT = 0.5

# %% -------------------------------------- MLP Class ------------------------------------------------------------------
class MLP(tf.keras.Model):
    """ MLP with len(neurons_per_layer) hidden layers """

    def __init__(self, neurons_per_layer, dropout=DROPOUT):
        super(MLP, self).__init__()
        self.model_layers = [tf.keras.Sequential([tf.keras.layers.Dense(neurons_per_layer[0], input_shape=(784,))])]
        self.model_layers[0].add(tf.keras.layers.ReLU())
        self.model_layers[0].add(tf.keras.layers.BatchNormalization())
        for neurons in neurons_per_layer[1:]:
            self.model_layers.append(
                tf.keras.Sequential([
                    tf.keras.layers.Dense(neurons),
                    tf.keras.layers.ReLU(),
                    tf.keras.layers.BatchNormalization()])
            )
        self.model_layers.append(tf.keras.layers.Dense(102))
        self.drop = dropout
        self.training = True

    def call(self, x):
        for layer in self.model_layers[:-1]:
            x = tf.nn.dropout(layer(x, training=self.training), self.drop)
        return self.model_layers[-1](x)

# %% -------------------------------------- Data Prep ------------------------------------------------------------------
if TRAIN_TEST_SPLIT:
    if not os.path.isdir("data"):
        os.mkdir("data")
        os.mkdir("data/train");
        os.mkdir("data/test")
        x = np.array([i for i in range(1, 8190)])  # Creates example vector with ids corresponding to the file names
        y = loadmat("imagelabels.mat")["labels"].reshape(-1)
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=SEED, test_size=0.3, stratify=y)
        for example, label in zip(x_train, y_train):
            id_path = "0" * (5 - len(str(example))) + str(example)  # For each example, gets the actual id of the file
            copyfile(os.getcwd() + DATA_DIR + "image_{}.jpg".format(id_path),
                     os.getcwd() + "/data/train/" + "image_{}.jpg".format(id_path))
            with open(os.getcwd() + "/data/train/" + "image_{}.txt".format(id_path), "w") as s:
                s.write(str(label - 1))  # labels start at 1 --> 1 to 0, 2 to 1, etc.
        for example, label in zip(x_test, y_test):
            id_path = "0" * (5 - len(str(example))) + str(example)
            copyfile(os.getcwd() + DATA_DIR + "image_{}.jpg".format(id_path),
                     os.getcwd() + "/data/test/" + "image_{}.jpg".format(id_path))
            with open(os.getcwd() + "/data/test/" + "image_{}.txt".format(id_path), "w") as s:
                s.write(str(label - 1))
    else:
        print("The data split dir is not empty! Do you want to overwrite it?")

# 1. Define a generator function that loads each image and label file and returns the preprocessed tensors.
# Use PIL.Image or other option to read directly from its path and do not depend on TensorFlow path type
def load_both(image_paths, label_paths):
    for img_path, label_path in zip(image_paths, label_paths):
        img = np.array(Image.open(img_path))
        img = tf.image.rgb_to_grayscale(img)
        with open(label_path, "r") as f:
            flower = int(f.read())
        yield tf.reshape(tf.image.resize(img, [28, 28]), (784,)), tf.convert_to_tensor([flower])

# 2. Use tf.data.Dataset.from_generator to get the training and testing datasets from these functions

train_paths_x = [os.getcwd() + "/data/train/" + path for path in os.listdir("data/train") if path[-4:] == ".jpg"]
train_paths_y = [os.getcwd() + "/data/train/" + path for path in os.listdir("data/train") if path[-4:] == ".txt"]
ds_train = tf.data.Dataset.from_generator(load_both, output_types=(tf.float32, tf.int32), args=[train_paths_x, train_paths_y])
ds_train = ds_train.shuffle(buffer_size=len(train_paths_x))
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

test_paths_x = [os.getcwd() + "/data/test/" + path for path in os.listdir("data/test") if path[-4:] == ".jpg"]
test_paths_y = [os.getcwd() + "/data/test/" + path for path in os.listdir("data/test") if path[-4:] == ".txt"]
ds_test = tf.data.Dataset.from_generator(load_both, output_types=(tf.float32, tf.int32), args=[test_paths_x, test_paths_y])
ds_test = ds_test.batch(len(test_paths_x))

# %% ---------------------------------------- Training Prep ------------------------------------------------------------

model = MLP(N_NEURONS)
optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
@tf.function
def train(x, y):
    model.training = True
    model.drop = DROPOUT
    with tf.GradientTape() as tape:
        logits = model(x)
        loss = criterion(y, logits)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(y, logits)

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
@tf.function
def eval(x, y):
    model.training = False
    model.drop = 0
    logits = model(x)
    loss = criterion(y, logits)
    test_loss(loss)
    test_accuracy(y, logits)


# %% ---------------------------------------- Training Loop ------------------------------------------------------------
print("Starting training loop...")
for epoch in range(N_EPOCHS):

    for image_batch, label_batch in ds_train:
        train(image_batch, label_batch)

    for image_batch, label_batch in ds_test:
        eval(image_batch, label_batch)

    print("Epoch {} | Train Loss {:.5f}, Train Acc {:.2f} - Test Loss {:.5f}, Test Acc {:.2f}".format(
        epoch, train_loss.result(), train_accuracy.result() * 100, test_loss.result(), test_accuracy.result() * 100))
    train_loss.reset_states(); train_accuracy.reset_states(); test_loss.reset_states(); test_accuracy.reset_states()

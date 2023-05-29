# %% -------------------------------------------------------------------------------------------------------------------
# Fit a MLP to the FashionMNIST dataset: https://github.com/zalandoresearch/fashion-mnist
# %% --------------------------------------- Imports -------------------------------------------------------------------
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
tf.random.set_seed(42)
np.random.seed(42)

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
# 2. Try using more/less layers and different hidden sizes to get a good better fit. Also play with the dropout.
# Try different batch sizes and get a feeling of how they can influence convergence and speed
LR = 1e-3
N_NEURONS = (100, 200, 100)
N_EPOCHS = 30
BATCH_SIZE = 512
DROPOUT = 0.2
# 4. Add an option to only test the model, by loading the model you saved on the training phase
TRAIN = True  # Set TRAIN = False to load a trained model and predict on the test set
# 3. Add an option to save the model on each epoch, and stop saving them when the validation
# loss begins to increase (early stopping) - https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/saved_model
SAVE_MODEL = True


# %% ----------------------------------- Helper Functions --------------------------------------------------------------
# 6. Define a function to show some images that were incorrectly classified
def show_mistakes(x, y, idx):
    label_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    y_pred = tf.argmax(model(x), axis=1).numpy()
    idx_mistakes = np.argwhere((y.numpy() == y_pred) == 0).flatten()
    plt.title("MLP prediction: {} - True label: {}".format(label_names[y_pred[idx_mistakes[idx]]],
                                                           label_names[y[idx_mistakes[idx]]]))
    plt.imshow(x.numpy()[idx_mistakes[idx]].reshape(28, 28))
    plt.show()


# %% -------------------------------------- MLP Class ------------------------------------------------------------------
class MLPList(tf.keras.Model):
    """ MLP with len(neurons_per_layer) hidden layers """
    def __init__(self, neurons_per_layer, dropout=DROPOUT):
        super(MLPList, self).__init__()
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
        self.model_layers.append(tf.keras.layers.Dense(10))
        self.drop = dropout
        self.training = True

    def call(self, x):
        for layer in self.model_layers[:-1]:
            x = tf.nn.dropout(layer(x, training=self.training), self.drop)
        return self.model_layers[-1](x)


# %% -------------------------------------- Data Prep ------------------------------------------------------------------
# 1. Download the data using tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train, x_test = tf.reshape(x_train, (len(x_train), -1)), tf.reshape(x_test, (len(x_test), -1))
x_train, x_test = tf.dtypes.cast(x_train, tf.float32), tf.dtypes.cast(x_test, tf.float32)
y_train, y_test = tf.convert_to_tensor(y_train), tf.convert_to_tensor(y_test)

# %% -------------------------------------- Training Prep ----------------------------------------------------------
if TRAIN:
    model = MLPList(N_NEURONS)
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


# %% -------------------------------------- Training Loop ----------------------------------------------------------
if TRAIN:
    loss_test_best = 1000  # The model will be saved whenever the test loss gets smaller
    print("Starting training loop...")
    for epoch in range(N_EPOCHS):

        for batch in range(len(x_train) // BATCH_SIZE + 1):
            inds = slice(batch * BATCH_SIZE, (batch + 1) * BATCH_SIZE)
            train(x_train[inds], y_train[inds])

        eval(x_test, y_test)

        print("Epoch {} | Train Loss {:.5f}, Train Acc {:.2f} - Test Loss {:.5f}, Test Acc {:.2f}".format(
            epoch, train_loss.result(), train_accuracy.result()*100, test_loss.result(), test_accuracy.result()*100))
        # 3. Add an option to save the model on each epoch, and stop saving them when the validation
        # loss begins to increase (early stopping) -
        # https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/saved_model
        if test_loss.result().numpy() < loss_test_best and SAVE_MODEL:
            tf.saved_model.save(model, os.getcwd() + '/mlp_fashionmnist/')
            print("The model has been saved!")
            loss_test_best = test_loss.result().numpy()
        train_loss.reset_states(); train_accuracy.reset_states(); test_loss.reset_states(); test_accuracy.reset_states()

# %% ------------------------------------------ Final test -------------------------------------------------------------
# 4. Add an option to only test the model, by loading the model you saved on the training phase
model = tf.saved_model.load(os.getcwd() + "/mlp_fashionmnist/")
model.training, model.drop = False, 0
y_test_pred = tf.argmax(model(x_test), axis=1).numpy()
print("The accuracy on the test set is", 100*accuracy_score(y_test.numpy(), y_test_pred), "%")
# 5. Print out the confusion matrix
print("The confusion matrix is")
print(confusion_matrix(y_test.numpy(), y_test_pred))

show_mistakes(x_test, y_test, 0)

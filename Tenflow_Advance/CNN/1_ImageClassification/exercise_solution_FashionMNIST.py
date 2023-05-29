# %% --------------------------------------- Imports -------------------------------------------------------------------
import numpy as np
import tensorflow as tf


# %% --------------------------------------- Set-Up --------------------------------------------------------------------
tf.random.set_seed(42)
np.random.seed(42)

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 1e-3
N_EPOCHS = 10
BATCH_SIZE = 128


# %% -------------------------------------- CNN Class ------------------------------------------------------------------
# 3. Try modifying the architecture and the hyper-parameters to get a better performance.
# Try including more conv layers and more kernels in each layer. This can allow for less MLP layers at the end.
# To do so, you will need to play around with zero-padding and maybe stack two conv layers together without any pooling.
# You can also remove the pooling and the MLP, and replace it with a final Global Average Pooling layer.
class CNN(tf.keras.Model):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(16, 3)
        self.convnorm1 = tf.keras.layers.BatchNormalization()
        self.pad1 = tf.keras.layers.ZeroPadding2D(2)

        self.conv2 = tf.keras.layers.Conv2D(32, 3)
        self.convnorm2 = tf.keras.layers.BatchNormalization()
        self.pool2 = tf.keras.layers.MaxPool2D(2)

        self.conv3 = tf.keras.layers.Conv2D(64, 3)
        self.convnorm3 = tf.keras.layers.BatchNormalization()

        self.conv4 = tf.keras.layers.Conv2D(128, 3)  # output (n_examples, whatever, whatever, 128)
        # Converts the output of self.conv4 to shape (n_examples, 128) by averaging the 128 feature maps
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()  # The global avg pooling means taking
        # the global average of all the feature maps, so we end up with 128 "averages"
        self.linear = tf.keras.layers.Dense(10)

        self.act = tf.nn.relu
        self.training = True

    def call(self, x):
        x = self.pad1(self.convnorm1(self.act(self.conv1(x)), training=self.training))
        x = self.pool2(self.convnorm2(self.act(self.conv2(x)), training=self.training))
        x = self.act(self.conv4(self.convnorm3(self.act(self.conv3(x)), training=self.training)))
        return self.linear(self.global_avg_pool(x))
        # The above line of code is equivalent to:
        # return self.linear(tf.math.reduce_mean(tf.reshape(x, (x.shape[0], -1, x.shape[3])), axis=1))


# %% -------------------------------------- Data Prep ------------------------------------------------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train, x_test = tf.reshape(x_train, (len(x_train), 28, 28, 1), ), tf.reshape(x_test, (len(x_test), 28, 28, 1))
x_train, x_test = tf.dtypes.cast(x_train, tf.float32), tf.dtypes.cast(x_test, tf.float32)
y_train, y_test = tf.convert_to_tensor(y_train), tf.convert_to_tensor(y_test)

# %% -------------------------------------- Training Prep ----------------------------------------------------------
model = CNN()
optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
@tf.function
def train(x, y):
    model.training = True
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
    logits = model(x)
    loss = criterion(y, logits)
    test_loss(loss)
    test_accuracy(y, logits)

# %% -------------------------------------- Training Loop ----------------------------------------------------------
print("Starting training loop...")
for epoch in range(N_EPOCHS):

    for batch in range(len(x_train)//BATCH_SIZE + 1):  # Loops over the number of batches
        inds = slice(batch*BATCH_SIZE, (batch+1)*BATCH_SIZE)  # Gets a slice to index the data
        train(x_train[inds], y_train[inds])

    eval(x_test, y_test)

    print("Epoch {} | Train Loss {:.5f}, Train Acc {:.2f} - Test Loss {:.5f}, Test Acc {:.2f}".format(
        epoch, train_loss.result(), train_accuracy.result()*100, test_loss.result(), test_accuracy.result()*100))
    train_loss.reset_states(); train_accuracy.reset_states(); test_loss.reset_states(); test_accuracy.reset_states()

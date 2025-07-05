import os
import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, Reshape, Dense, LeakyReLU, Flatten, BatchNormalization, Dropout
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets.fashion_mnist import load_data

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
weight_init = glorot_normal(seed=SEED)

def change_image_shape(images):
    shape_tuple = images.shape
    if len(shape_tuple) == 3:
        images = images.reshape(-1, shape_tuple[-1], shape_tuple[-1], 1)
    elif shape_tuple == 4 and shape_tuple[-1] > 3:
        images = images.reshape(-1, shape_tuple[-1], shape_tuple[-1], shape_tuple[1])
    return images

(x_train, y_train), (x_test, y_test) = load_data()
x_train, x_test = change_image_shape(x_train), change_image_shape(x_test)
y_train, y_test = y_train.reshape(-1), y_test.reshape(-1)

channel = x_train.shape[-1]
x_train = (x_train.astype('float32') - 127.5) / 127.5
x_test = (x_test.astype('float32') - 127.5) / 127.5
img_size = x_train[0].shape
n_classes = len(np.unique(y_train))

discriminator_optimizer = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
generator_optimizer = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
latent_dim = 100  # Increased latent dimension for better representation

def generator_fc(generator_optimizer):
    noise = Input(shape=(latent_dim,))
    x = Dense(256, kernel_initializer=weight_init)(noise)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dense(512, kernel_initializer=weight_init)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dense(1024, kernel_initializer=weight_init)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dense(np.prod(img_size), activation='tanh', kernel_initializer=weight_init)(x)
    out = Reshape(img_size)(x)
    model = Model(inputs=noise, outputs=out)
    return model

def discriminator_fc(discriminator_optimizer):
    img = Input(shape=img_size)
    x = Flatten()(img)
    x = Dense(1024, kernel_initializer=weight_init)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)
    x = Dense(512, kernel_initializer=weight_init)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)
    x = Dense(256, kernel_initializer=weight_init)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)
    out = Dense(1, activation='sigmoid', kernel_initializer=weight_init)(x)
    model = Model(inputs=img, outputs=out)
    model.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def generator_trainer(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(optimizer=generator_optimizer, loss='binary_crossentropy')
    return model

class GAN:
    def __init__(self, g_model, d_model):
        self.img_size = img_size
        self.z = latent_dim
        self.generator = g_model
        self.discriminator = d_model
        self.train_gen = generator_trainer(self.generator, self.discriminator)
        self.loss_D, self.loss_G = [], []
        self.acc_D = []

    def train(self, imgs, steps_per_epoch=50, batch_size=128):
        bs_half = batch_size // 2

        for epoch in range(steps_per_epoch):
            # Train Discriminator
            idx = np.random.randint(0, imgs.shape[0], bs_half)
            real_img = imgs[idx]

            noise = np.random.normal(0, 1, size=(bs_half, latent_dim))
            fake_img = self.generator.predict(noise, verbose=0)

            # Label smoothing for better training stability
            real_labels = np.ones((bs_half, 1)) * 0.9  # Smooth real labels
            fake_labels = np.zeros((bs_half, 1)) + 0.1  # Smooth fake labels

            loss_real = self.discriminator.train_on_batch(real_img, real_labels)
            loss_fake = self.discriminator.train_on_batch(fake_img, fake_labels)

            # Store discriminator loss and accuracy properly
            d_loss = 0.5 * (loss_fake[0] + loss_real[0])
            d_acc = 0.5 * (loss_fake[1] + loss_real[1])
            self.loss_D.append(d_loss)
            self.acc_D.append(d_acc)

            # Train Generator
            noise = np.random.normal(0, 1, size=(batch_size, latent_dim))
            # Generator wants discriminator to classify fake images as real
            valid_labels = np.ones((batch_size, 1))
            loss_gen = self.train_gen.train_on_batch(noise, valid_labels)
            self.loss_G.append(loss_gen)

            if (epoch + 1) % (steps_per_epoch // 10) == 0:
                print('Steps (%d / %d): [Loss_D_real: %f, Loss_D_fake: %f, acc: %.2f%%] [Loss_G: %f]' %
                      (epoch + 1, steps_per_epoch, loss_real[0], loss_fake[0], 100 * d_acc, loss_gen))

d_model = discriminator_fc(discriminator_optimizer)
g_model = generator_fc(generator_optimizer)
gan = GAN(g_model=g_model, d_model=d_model)

def plt_img(generator):
    """
    Ultra-clean plotting function that completely avoids axis object access.
    Uses only plt.subplot() and basic plt functions to eliminate all converter warnings.
    """
    np.random.seed(42)
    n = n_classes
    noise = np.random.normal(size=(n * n, latent_dim))
    decoded_imgs = generator.predict(noise)
    decoded_imgs = decoded_imgs * 0.5 + 0.5
    x_real = x_test * 0.5 + 0.5

    # Create figure using only basic plt functions - completely avoids axis object access
    plt.figure(figsize=(2 * n, 2 * (n + 1)))

    # Display real images in the first row
    for i in range(n):
        plt.subplot(n + 1, n, i + 1)
        if channel == 3:
            plt.imshow(x_real[y_test == i][0].reshape(img_size[0], img_size[1], img_size[2]))
        else:
            plt.imshow(x_real[y_test == i][0].reshape(img_size[0], img_size[1]), cmap='gray')
        plt.xticks([])  # Ultra-clean approach - no axis object access
        plt.yticks([])  # Ultra-clean approach - no axis object access

    # Display generated images in subsequent rows
    for i in range(n):
        for j in range(n):
            subplot_idx = (i + 1) * n + j + 1
            plt.subplot(n + 1, n, subplot_idx)
            if channel == 3:
                plt.imshow(decoded_imgs[i * n + j].reshape(img_size[0], img_size[1], img_size[2]))
            else:
                plt.imshow(decoded_imgs[i * n + j].reshape(img_size[0], img_size[1]), cmap='gray')
            plt.xticks([])  # Ultra-clean approach - no axis object access
            plt.yticks([])  # Ultra-clean approach - no axis object access

    plt.tight_layout()
    plt.show()

EPOCHS = 100
for epoch in range(EPOCHS):
    print('EPOCH # ', epoch + 1, '-' * 50)
    gan.train(x_train, steps_per_epoch=50, batch_size=128)
    if (epoch + 1) % 1 == 0:
        plt_img(gan.generator)

# Plot training losses with ultra-clean formatting - no axis object access
plt.figure(figsize=(10, 5))
x_values = list(range(len(gan.loss_D)))
plt.plot(x_values, gan.loss_D, label='Discriminator Loss', color='red', linewidth=2)
plt.plot(x_values, gan.loss_G, label='Generator Loss', color='blue', linewidth=2)
plt.title('GAN Training Losses', fontsize=14)
plt.xlabel('Training Steps', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()

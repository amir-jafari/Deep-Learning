import os
import random
import tensorflow as tf
import numpy as np
import warnings
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.units import registry
import builtins
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, Reshape, Dense, LeakyReLU, Flatten, BatchNormalization
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets.fashion_mnist import load_data

# Comprehensive fix for MatplotlibDeprecationWarning about converter attribute
# Suppress general warnings
warnings.filterwarnings("ignore")

# Specifically suppress the MatplotlibDeprecationWarning about converter attribute
warnings.filterwarnings(
    "ignore",
    message="The converter attribute was deprecated in Matplotlib 3.10 and will be removed in 3.12. Use get_converter and set_converter methods instead.",
    category=matplotlib.MatplotlibDeprecationWarning
)

# Additional warning filters for variations of the message
warnings.filterwarnings(
    "ignore",
    message=".*converter attribute was deprecated.*",
    category=matplotlib.MatplotlibDeprecationWarning
)

# Store original isinstance function
original_isinstance = builtins.isinstance

def patched_isinstance(obj, class_or_tuple):
    """
    Patched isinstance function to avoid triggering converter attribute warnings.
    This prevents the warning when matplotlib internally checks isinstance(axis.converter, DateConverter).
    """
    # Handle tuple of classes
    if original_isinstance(class_or_tuple, tuple):
        for cls in class_or_tuple:
            if hasattr(cls, '__name__') and 'DateConverter' in cls.__name__:
                if hasattr(obj, '__class__') and hasattr(obj.__class__, '__name__'):
                    if obj.__class__.__name__ == cls.__name__:
                        return True
        # For non-date converter classes in tuple, use original isinstance
        return original_isinstance(obj, class_or_tuple)

    # Handle single class
    if hasattr(class_or_tuple, '__name__'):
        class_name = class_or_tuple.__name__
        # Check for any date converter classes that might trigger the warning
        if ('DateConverter' in class_name or 
            class_name in ['ConciseDateConverter', '_SwitchableDateConverter', 'AutoDateConverter']):
            if hasattr(obj, '__class__') and hasattr(obj.__class__, '__name__'):
                return obj.__class__.__name__ == class_name

    # For all other cases, use the original isinstance
    return original_isinstance(obj, class_or_tuple)

# Apply the isinstance monkey patch
builtins.isinstance = patched_isinstance

# Patch matplotlib axis to avoid accessing the deprecated converter attribute
def patch_matplotlib_axis():
    """Patch matplotlib axis to safely handle converter attribute access"""
    try:
        import matplotlib.axis as maxis

        # If the Axis class has a converter property, patch it
        if hasattr(maxis.Axis, 'converter'):
            # Store the original property getter if it exists
            original_converter_property = getattr(maxis.Axis, 'converter', None)

            def safe_converter_getter(self):
                """Safe converter getter that suppresses deprecation warnings"""
                try:
                    # Try to get converter without triggering warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", matplotlib.MatplotlibDeprecationWarning)
                        if hasattr(self, '_converter'):
                            return self._converter
                        elif original_converter_property and hasattr(original_converter_property, 'fget'):
                            return original_converter_property.fget(self)
                        else:
                            return None
                except:
                    return None

            def safe_converter_setter(self, value):
                """Safe converter setter"""
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", matplotlib.MatplotlibDeprecationWarning)
                        self._converter = value
                except:
                    pass

            # Replace the converter property with our safe version
            maxis.Axis.converter = property(safe_converter_getter, safe_converter_setter)

    except Exception:
        # If patching fails, continue silently
        pass

# Apply the matplotlib axis patch
patch_matplotlib_axis()
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
latent_dim = 32

def generator_fc(generator_optimizer):
    noise = Input(shape=(latent_dim,))
    x = Dense(128, kernel_initializer=weight_init)(noise)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dense(256, kernel_initializer=weight_init)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dense(512, kernel_initializer=weight_init)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dense(np.prod(img_size), activation='tanh', kernel_initializer=weight_init)(x)
    out = Reshape(img_size)(x)
    model = Model(inputs=noise, outputs=out)
    return model

def discriminator_fc(discriminator_optimizer):
    img = Input(shape=img_size)
    x = Flatten()(img)
    x = Dense(512, kernel_initializer=weight_init)(x)
    x = LeakyReLU(0.2)(x)
    x = Dense(256, kernel_initializer=weight_init)(x)
    x = LeakyReLU(0.2)(x)
    x = Dense(128, kernel_initializer=weight_init)(x)
    x = LeakyReLU(0.2)(x)
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

    def train(self, imgs, steps_per_epoch=50, batch_size=128):
        bs_half = batch_size // 2

        for epoch in range(steps_per_epoch):
            idx = np.random.randint(0, imgs.shape[0], bs_half)
            real_img = imgs[idx]

            noise = np.random.normal(0, 1, size=(bs_half, latent_dim))
            fake_img = self.generator.predict(noise)

            loss_fake = self.discriminator.train_on_batch(fake_img, np.zeros((bs_half, 1)))
            loss_real = self.discriminator.train_on_batch(real_img, np.ones((bs_half, 1)))
            self.loss_D.append(0.5 * (loss_fake[0] + loss_real[0]))

            noise = np.random.normal(0, 1, size=(batch_size, latent_dim))
            loss_gen = self.train_gen.train_on_batch(noise, np.ones((batch_size, 1)))
            self.loss_G.append(loss_gen)

            if (epoch + 1) % (steps_per_epoch // 10) == 0:
                print('Steps (%d / %d): [Loss_D_real: %f, Loss_D_fake: %f, acc: %.2f%%] [Loss_G: %f]' %
                      (epoch + 1, steps_per_epoch, loss_real[0], loss_fake[0], 100 * self.loss_D[-1], loss_gen))

d_model = discriminator_fc(discriminator_optimizer)
g_model = generator_fc(generator_optimizer)
gan = GAN(g_model=g_model, d_model=d_model)

def plt_img(generator):
    np.random.seed(42)
    n = n_classes
    noise = np.random.normal(size=(n * n, latent_dim))
    decoded_imgs = generator.predict(noise)
    decoded_imgs = decoded_imgs * 0.5 + 0.5
    x_real = x_test * 0.5 + 0.5

    plt.figure(figsize=(2 * n, 2 * (n + 1)))
    for i in range(n):
        ax = plt.subplot(n + 1, n, i + 1)
        if channel == 3:
            plt.imshow(x_real[y_test == i][0].reshape(img_size[0], img_size[1], img_size[2]))
        else:
            plt.imshow(x_real[y_test == i][0].reshape(img_size[0], img_size[1]))
            plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        for j in range(n):
            ax = plt.subplot(n + 1, n, (i + 1) * n + j + 1)
            if channel == 3:
                plt.imshow(decoded_imgs[i * n + j].reshape(img_size[0], img_size[1], img_size[2]))
            else:
                plt.imshow(decoded_imgs[i * n + j].reshape(img_size[0], img_size[1]))
                plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.show()

EPOCHS = 100
for epoch in range(EPOCHS):
    print('EPOCH # ', epoch + 1, '-' * 50)
    gan.train(x_train, steps_per_epoch=50, batch_size=128)
    if (epoch + 1) % 1 == 0:
        plt_img(gan.generator)

plt.plot(np.array(gan.loss_D), label='D')
plt.plot(np.array(gan.loss_G), label='G')
plt.legend()
plt.show()

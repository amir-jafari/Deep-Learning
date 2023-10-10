# # %%---------------------------------------------------------------------------------------------
import tensorflow as tf

import numpy as np
x = np.linspace(-4,4,500)
y = np.sin(x)

inputs = tf.keras.layers.Input(shape=(1,))
x1 =tf.keras.layers.Dense(10,activation='relu')(inputs)
ouputs = tf.keras.layers.Dense(1, activation='relu')(x1)

model = tf.keras.models.Model(inputs, ouputs)
model.compile(loss='mse',optimizer='sgd',metrics=['accuracy'])
model.fit(x, y, epochs=10)

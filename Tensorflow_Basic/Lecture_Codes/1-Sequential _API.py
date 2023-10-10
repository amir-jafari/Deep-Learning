# # %%---------------------------------------------------------------------------------------------
import tensorflow as tf
import numpy as np
x = np.linspace(-4,4,500)
y = np.sin(x)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(10, activation='relu' , input_shape=(1,)))
model.add(tf.keras.layers.Dense(1, activation='linear' , input_shape=(10,)))

model.compile(loss='mse',optimizer='sgd',metrics=['accuracy'])
model.fit(x, y, epochs=10)

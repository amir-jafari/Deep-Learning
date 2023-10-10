# %%---------------------------------------------------------------------------------------------
import tensorflow as tf
import numpy as np
x = np.linspace(-4,4,500).reshape(-1,1)
y = np.sin(x)

class Mymodel(tf.keras.models.Model):
    def __init__(self):
        super(Mymodel,self).__init__()
        self.dense1 = tf.keras.layers.Dense(1, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='relu')
        self.dense3 = tf.keras.layers.Dense(1, activation='linear')
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        output = self.dense3(x)
        return output

model = Mymodel()

model.compile(loss='mse',optimizer='sgd',metrics=['accuracy'])
model.fit(x, y, epochs=10)


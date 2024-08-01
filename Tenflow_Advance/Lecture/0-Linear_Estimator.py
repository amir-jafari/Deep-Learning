import tensorflow as tf


features = [[1., 1.5], [2., 2.5], [3., 3.5]]
labels = [[0.3], [0.5], [0.7]]
eval_features = [[4., 4.5], [5., 5.5], [6., 6.5]]
eval_labels = [[0.8], [0.9], [1.]]

dataset = tf.data.Dataset.from_tensor_slices((features, labels)).batch(1)
eval_dataset = tf.data.Dataset.from_tensor_slices(
      (eval_features, eval_labels)).batch(1)

model = tf.keras.models.Sequential([tf.keras.layers.Dense(1)])
optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.05)

model.compile(optimizer=optimizer, loss="mse")

model.fit(dataset)


model.evaluate(eval_dataset, return_dict=True)
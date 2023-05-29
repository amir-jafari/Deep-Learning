import numpy as np
import jax.numpy as jnp
size = 10
index = 0
value = 23

# In NumPy arrays are mutable
x = np.arange(size)
print(x)
x[index] = value
print(x)

x = jnp.arange(size)
print(x)
# x[index] = value
x = x.at[index].set(value)
print(x)
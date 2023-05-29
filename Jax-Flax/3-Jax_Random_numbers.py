from jax import random
import jax.numpy as jnp
import numpy as np
from jax import device_put

seed = 0
key = random.PRNGKey(seed)
x = random.normal(key, (10, ))  # you need to explicitly pass the key i.e. PRNG state
print(type(x), x)  # notice the DeviceArray type - that leads us to the next cell!

size = 3000

# Data is automagically pushed to the AI accelerator! (DeviceArray structure)
# No more need for ".to(device)" (PyTorch syntax)
x_jnp = random.normal(key, (size, size), dtype=jnp.float32)
x_np = np.random.normal(size=(size, size)).astype(np.float32)  # some diff in API exists!

jnp.dot(x_jnp, x_jnp.T).block_until_ready()  # 1) on GPU - fast
jnp.dot(x_np, x_np.T)  # 2) on CPU - slow (NumPy only works with CPUs)
jnp.dot(x_np, x_np.T).block_until_ready()  # 3) on GPU with transfer overhead

x_np_device = device_put(x_np)  # push NumPy explicitly to GPU
jnp.dot(x_np_device, x_np_device.T).block_until_ready()  # same as 1)

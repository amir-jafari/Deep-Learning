import jax
print(jax.device_count())

from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)
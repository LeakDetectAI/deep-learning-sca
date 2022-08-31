import tensorflow as tf
from tensorflow.python.client import device_lib

import deepscapy

# import keras
# import csrank as cs
print(device_lib.list_local_devices())
print(tf.test.gpu_device_name())

# print('version', keras.__version__)
# print('path', keras.__path__)
print('tf version', tf.__version__)
print('tf path', tf.__path__)
print('tf path', deepscapy.__path__)
devices = [x.name for x in device_lib.list_local_devices()]
print('devices', devices)

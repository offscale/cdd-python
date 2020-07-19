try:
    import tensorflow as tf
    import numpy as np
except ImportError:
    from collections import namedtuple

    tf = namedtuple('TensorFlow', ('data',))(namedtuple('data', ('Dataset',)))
    np = namedtuple('numpy', ('ndarray',))

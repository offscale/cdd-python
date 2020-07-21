"""
Mocks for methods
"""

from typing import Optional, Union, Tuple

try:
    import tensorflow as tf
    import numpy as np
except ImportError:
    from collections import namedtuple

    tf = namedtuple('TensorFlow', ('data',))(namedtuple('data', ('Dataset',)))
    np = namedtuple('numpy', ('ndarray',))


class C(object):
    """ C class (mocked!) """

    def method(self, dataset_name, tfds_dir, K, as_numpy, data_loader_kwargs):
        """
        Acquire from the official tensorflow_datasets model zoo, or the ophthalmology focussed ml-prepare library

        :param dataset_name: name of dataset.
        :type dataset_name: ```str```

        :param tfds_dir: directory to look for models in.
        :type tfds_dir: ```Optional[str]```

        :param K: backend engine, e.g., `np` or `tf`.
        :type K: ```Union[np, tf]```

        :param as_numpy: Convert to numpy ndarrays
        :type as_numpy: ```Optional[bool]```

        :param data_loader_kwargs: pass this as arguments to data_loader function
        :type data_loader_kwargs: ```**data_loader_kwargs```

        :return: Train and tests dataset splits.
        :rtype: ```Union[Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[np.ndarray, np.ndarray]]```
        """
        return np.empty(0), np.empty(0)

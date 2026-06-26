import os
import random
import numpy as np
import tensorflow as tf


def set_global_seed(seed: int) -> None:
    """Set seeds for numpy, random, and tensorflow (best-effort determinism)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

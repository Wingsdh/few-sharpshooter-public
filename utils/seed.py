# -*- encoding: utf-8 -*-
"""
-------------------------------------------------
   File Name：    seed.py
   Description :
   Author :       Wings DH
   Time：         6/20/21 6:18 PM
-------------------------------------------------
   Change Activity:
                   6/20/21: Create
-------------------------------------------------
"""
import random
import os
import numpy as np
import tensorflow as tf


def set_seed(seed=21):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

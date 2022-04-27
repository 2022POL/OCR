from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping

import numRec as nr
from matplotlib import pyplot as plt
import sys
import numpy as np
from PIL import Image
import tensorflow as tf
import os
from matplotlib import cm


nr()

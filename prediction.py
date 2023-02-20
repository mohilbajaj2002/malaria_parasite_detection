import os
import math
import glob
import cv2
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import tensorflow_datasets as tfds
from PIL import Image as im
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD, Adam, RMSprop

import config
import utilities as utils
import architectures as arch
import data_generation as datagen

model_to_be_used = config.best_model
arch_type = model_to_be_used.split('_')[2]
datatype = model_to_be_used.split('_')[4]
new_shape = utils.get_image_shape(arch_type)
factor = int(datatype[0]) if len(datatype) > 0 else 0       #int(datatype[0])

# Preprocessing image
img= Image.open(config.prediction_image_path)
input_image = np.array(img)
input_image = datagen.preprocess_image_input_prediction(input_image, factor, new_shape)

# Loading model
model_path = f'{config.saved_model_root_path}/{config.best_model}'
model = tf.keras.models.load_model(model_path)

# Making prediction
pred = model.predict(input_image)
pred_class = 'uninfected' if pred > 0.5 else 'parasitisized'          # binary
print(f'The blood scan is {pred_class} (prediction probability: {round(pred[0][0], 3)})')

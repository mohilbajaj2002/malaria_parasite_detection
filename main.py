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

# Loading Data
train, test = tfds.load('malaria', split=['train[:80%]', 'train[80%:]'])
df = tfds.as_dataframe(train)
df_test = tfds.as_dataframe(test)

# Creating Data folders
for factor_list in config.factor_list_of_list:
    print(f'Creating New Data: {factor_list}...')
    factor_list_str = [str(i) for i in factor_list]
    factor_list_name = '_' + ''.join(factor_list_str)
    main = [config.training_folder + factor_list_name, config.test_folder + factor_list_name, config.aug_test_folder + factor_list_name]
    datagen.create_image_folder(df, main[0], factor_list)
    datagen.create_image_folder(df_test, main[1], factor_list)
    print(f'Data Check - Images in {config.classes[0]} folder: ', len(os.listdir(f'{config.data_root_path}/{main[0]}/{config.classes[0]}')))  #parasitisized

    aug_base = os.path.join(config.data_root_path, main[2])
    os.makedirs(aug_base, exist_ok=True)

    # Generating train and validation data
    for arch_name in config.architecture_list:
        resnet_toggle = utils.get_resnet_toggle(arch_name)

        # Starting model training
        for batch_size in config.batch_size_list:
            train_generator, validation_generator = datagen.create_data_gen(config.img_shape, config.batch_size, main[0], main[1], main[2], resnet_toggle)
            for epochs in config.epoch_list:
                for opt_name in config.optimizer_list:
                    optimizer = utils.get_optimizer(opt_name)
                    no_of_classes = config.no_of_classes
                    img_shape = config.img_shape
                    model_no = sum(os.path.isdir(i) for i in os.listdir(config.saved_model_root_path)) + 1
                    datatype = factor_list_name
                    classifier_details = utils.get_classifier_details(arch_name)
                    callback_details = '3_callbacks'
                    model_name =  f'Attempt_{model_no}_{arch_name}_datatype_{datatype}_{classifier_details}_{opt_name}_{callback_details}_epochs_{epochs}'
                    model_path = f'{config.saved_model_root_path}/{model_name}'
                    history_path = f'{config.saved_history_root_path}/{model_name}' + '.npy'
                    img_shape = utils.get_image_shape(arch_name)
                    model, history = arch.build_model(utils.get_arch_fn(arch_name), config.no_of_classes, img_shape, optimizer, batch_size, epochs, train_generator, validation_generator, model_path, history_path)

                    # Model Performance
                    utils.plot_history(history, 'accuracy', model_name)
                    utils.display_performance_metrics(model, validation_generator, model_name)

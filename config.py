import os
import sys

# File to enter basic data about the project

project_name = 'malaria'

classes = ['parasitisized', 'uninfected']

# Folders & Paths
training_folder = 'training'
test_folder = 'test'
aug_test_folder = 'aug_test'
saved_model_folder = 'saved_model'
saved_history_folder = 'saved_history'

root_path = os.path.dirname(os.path.realpath(sys.argv[0]))
data_root_path = f'{root_path}/{project_name}'
saved_model_root_path = f'{root_path}/{saved_model_folder}'
saved_history_root_path = f'{root_path}/{saved_history_folder}'

# For data aumentation and pre-processing
factor_list_of_list = [[0], [0, 2], [0,3]] #[0], [2], [3], [0, 2], [0,3], [2, 3], [0, 2, 3]

# Model parameters & other training options
no_of_classes = len(classes)
input_shape = (1, 224, 224, 3)
img_shape = [224, 224]
hhna_img_shape = [150, 150, 3]
lr_scheduler_epoch_factor = 0.2
early_stopping_epoch_factor = 0.75
architecture_list = ['VGG16', 'InceptionV3'] #'VGG16', 'InceptionV3', 'InceptionResNetV2', 'Resnet50', 'HHNA'
batch_size_list = [32] # 32, 64, 128
epoch_list = [1] # 50, 100, 200
optimizer_list = ['SGD', 'RMSProp', 'Adam']

# for prediction
best_model = ''
prediction_image_path = ''

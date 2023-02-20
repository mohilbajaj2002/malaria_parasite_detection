
import os
import sys
import config
import numpy as np
import tensorflow as tf

def get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))

root_path = f'{get_script_path()}/{config.project_name}'


def get_lab_name(lab):
  if(int(lab) == 0):
    return config.classes[0]
  elif(int(lab) == 1):
    return config.classes[1]


def preprocess_image_input(input_image, factor):
  factor_contrast = factor
  factor_brightness = factor
  input_image = Image.fromarray((input_image * 255).astype(np.uint8)) # PIL.Image.fromarray(np.uint8(input_image))

  enhancer_contrast = ImageEnhance.Contrast(input_image)
  eq_contrast = enhancer_contrast.enhance(factor_contrast)

  enhancer_brightness = ImageEnhance.Brightness(eq_contrast)
  eq_brightness = enhancer_brightness.enhance(factor_brightness)

  enhanced_img = np.array(eq_brightness).astype(np.float32)
  return enhanced_img


def preprocess_image_input_prediction(input_image, factor, new_shape):
  factor_contrast = factor
  factor_brightness = factor
  input_image = Image.fromarray((input_image * 255).astype(np.uint8)) # PIL.Image.fromarray(np.uint8(input_image))

  enhancer_contrast = ImageEnhance.Contrast(input_image)
  eq_contrast = enhancer_contrast.enhance(factor_contrast)

  enhancer_brightness = ImageEnhance.Brightness(eq_contrast)
  eq_brightness = enhancer_brightness.enhance(factor_brightness)

  enhanced_img = np.array(eq_brightness).astype(np.float32)
  enhanced_img = enhanced_img/255.0
  enhanced_img = np.resize(enhanced_img, new_shape)
  enhanced_img = np.expand_dims(enhanced_img, axis=0)
  return enhanced_img


def create_image(image_array, name_prefix, index, final_path):
  data = Image.fromarray(np.uint8(image_array))
  name = f'{name_prefix}_{index}.jpg'
  path = f'{final_path}/{name}'
  data.save(path)


def create_image_folder(df, folder_type, factor_list):
  print(f'Creating {folder_type} folder...')
  base = os.path.join(root_path, folder_type)
  os.makedirs(base, exist_ok=True)
  uni_labels = df.label.unique()
  for lab in uni_labels:
    lab_name = get_lab_name(lab)
    final_path = os.path.join(base, lab_name)
    os.makedirs(final_path, exist_ok=True)
    df2 = df[df['label'] == lab]
    df2 = df2.reset_index(drop=True)
    for index, row in df2.iterrows():
      img_array = np.array(row['image'])
      for factor in factor_list:
          name_prefix = f'preprocess_factor_{factor}'
          if(factor != 0):
              img_array = preprocess_image_input(img_array, factor)
          create_image(img_array, name_prefix, index, final_path)


def preprocess_image_input_resnet(input_image):
  enhanced_img = input_image.astype(np.float32)
  enhanced_img = tf.keras.applications.resnet50.preprocess_input(enhanced_img)
  return enhanced_img


def get_class_mode(no_of_classes):
    if(no_of_classes == 2):
        return 'binary'
    else:
        return 'categorical'


def create_data_gen(img_shape, no_of_classes, batch_size, training_dir, validation_dir, aug_validation_dir, resnet_toggle):

  r, c = img_shape[0], img_shape[1]
  train_name = f'{root_path}/{training_dir}'
  valid_name = f'{root_path}/{validation_dir}'
  aug_valid_name = f'{root_path}/{aug_validation_dir}'

  if(resnet_toggle):
    train_datagen = ImageDataGenerator(
                                    rescale=1./255,
                                    rotation_range=40,        # 40, 80, 120
                                    width_shift_range=0.2,    # 0.2, 0.4
                                    height_shift_range=0.2,   # 0.2, 0.4
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    vertical_flip=True,    # newly added
                                    fill_mode='nearest',
                                    preprocessing_function=preprocess_image_input_resnet)

    validation_datagen = ImageDataGenerator(rescale=1/255,
                                            preprocessing_function=preprocess_image_input_resnet)
  else:
    train_datagen = ImageDataGenerator(
                                    rescale=1./255,
                                    rotation_range=40,        # 40, 80, 120
                                    width_shift_range=0.2,    # 0.2, 0.4
                                    height_shift_range=0.2,   # 0.2, 0.4
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    vertical_flip=True,    # newly added
                                    fill_mode='nearest')

    validation_datagen = ImageDataGenerator(rescale=1/255)

  # Flow training images in batches of 128 using train_datagen generator
  train_generator = train_datagen.flow_from_directory(
        train_name,  # This is the source directory for training images
        target_size=(r, c),  # All images will be resized to this size
        batch_size=batch_size,
        # binary, categorical, sparse
        class_mode=get_class_mode(no_of_classes))

  # Flow training images in batches of 32 using validation_datagen generator
  validation_generator = validation_datagen.flow_from_directory(
        valid_name,  # This is the source directory for training images
        target_size=(r, c),  # All images will be resized to this size
        batch_size=batch_size,
        # binary, categorical, sparse
        class_mode=get_class_mode(no_of_classes),
        shuffle=False,
        save_to_dir=aug_valid_name,
        save_prefix='aug',
        save_format='jpg')

  return train_generator, validation_generator

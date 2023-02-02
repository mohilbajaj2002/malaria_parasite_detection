import numpy as np
import pandas as pd
import tensorflow as tf

def scheduler(epoch, lr):
  if epoch < scheduler_epoch:
    return lr
  else:
    return lr * tf.math.exp(-0.1)

class MyCallback(tf.keras.callbacks.Callback):
  def __init__(self, count=0, epoch = 10):
    self.count = count
    self.epoch = epoch
  def on_epoch_end(self, epoch, logs=None):
    if(epoch >= early_stopping_epoch and (logs['val_loss'] - logs['loss'])/logs['val_loss'] >= 0.05):
      net_epoch = epoch - self.epoch
      self.epoch = epoch
      self.count = self.count + 1
      if(self.count >= 9 and net_epoch == 1):
        self.model.stop_training = True
        print ('\n Model starting to overfit....stopping now! \n')
      elif(net_epoch != 1):
        self.count= 0

def build_model_vgg16(no_of_classes, img_shape, optimizer):
  # load the base VGG-16 model
  base_model = tf.keras.applications.vgg16.VGG16(input_shape=(224, 224, 3),
                                                 include_top=False,
                                                 weights='imagenet')

  # add a GAP layer
  output = layers.GlobalAveragePooling2D()(base_model.output)

  # output layer
  if(no_of_classes == 2):
      no_of_classes = no_of_classes - 1
      activation_fn = 'sigmoid'
      loss_fn = 'binary_crossentropy'
  else:
      activation_fn = 'softmax'
      loss_fn = 'categorical_crossentropy'

  output = layers.Dense(no_of_classes, activation=activation_fn)(output)

  # set the inputs and outputs of the model
  model = Model(base_model.input, output)

  # freeze the earlier layers
  for layer in base_model.layers[:-4]:
      layer.trainable=False

  # choose the optimizer
  #optimizer = tf.keras.optimizers.RMSprop(0.001)

  # configure the model for training
  model.compile(optimizer=optimizer,
                loss=loss_fn,
                metrics = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

  # display the summary
  model.summary()

  return model


  def build_model_inception_v3(no_of_classes, img_shape, optimizer):
    # load the base Inception-V3 model
    base_model = tf.keras.applications.inception_v3.InceptionV3(input_shape=(224, 224, 3),
                                                                include_top=False,
                                                                weights='imagenet')

    # add a GAP layer
    output = layers.GlobalAveragePooling2D()(base_model.output)

    # output layer
    if(no_of_classes == 2):
        no_of_classes = no_of_classes - 1
        activation_fn = 'sigmoid'
        loss_fn = 'binary_crossentropy'
    else:
        activation_fn = 'softmax'
        loss_fn = 'categorical_crossentropy'

    output = layers.Dense(no_of_classes, activation=activation_fn)(output)

    # set the inputs and outputs of the model
    model = Model(base_model.input, output)

    # freeze the earlier layers
    for layer in base_model.layers[:-4]:
        layer.trainable=False

    # choose the optimizer
    #optimizer = tf.keras.optimizers.RMSprop(0.001)

    # configure the model for training
    model.compile(optimizer=optimizer,
                  loss=loss_fn,
                  metrics = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    # display the summary
    model.summary()

    return model


def build_model_inception_resnet_v2(no_of_classes, img_shape, optimizer):
  # load the base Inception-Resnet-V2 model
  base_model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(input_shape=(224, 224, 3),
                                                              include_top=False,
                                                              weights='imagenet')

  # add a GAP layer
  output = layers.GlobalAveragePooling2D()(base_model.output)

  # output layer
  if(no_of_classes == 2):
      no_of_classes = no_of_classes - 1
      activation_fn = 'sigmoid'
      loss_fn = 'binary_crossentropy'
  else:
      activation_fn = 'softmax'
      loss_fn = 'categorical_crossentropy'

  output = layers.Dense(no_of_classes, activation=activation_fn)(output)

  # set the inputs and outputs of the model
  model = Model(base_model.input, output)

  # freeze the earlier layers
  for layer in base_model.layers[:-4]:
      layer.trainable=False

  # choose the optimizer
  #optimizer = tf.keras.optimizers.RMSprop(0.001)

  # configure the model for training
  model.compile(optimizer=optimizer,
                loss=loss_fn,
                metrics = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

  # display the summary
  model.summary()

  return model


  def build_model_resnet50(no_of_classes, img_shape, optimizer):
    # load the base Resnet50 model
    base_model = tf.keras.applications.resnet50.ResNet50(input_shape=(224, 224, 3),
                                                                include_top=False,
                                                                weights='imagenet')

    # add a GAP layer
    output = layers.GlobalAveragePooling2D()(base_model.output)

    # output layer
    if(no_of_classes == 2):
        no_of_classes = no_of_classes - 1
        activation_fn = 'sigmoid'
        loss_fn = 'binary_crossentropy'
    else:
        activation_fn = 'softmax'
        loss_fn = 'categorical_crossentropy'

    output = layers.Dense(no_of_classes, activation=activation_fn)(output)

    # set the inputs and outputs of the model
    model = Model(base_model.input, output)

    # freeze the earlier layers
    for layer in base_model.layers[:-4]:
        layer.trainable=False

    # choose the optimizer
    #optimizer = tf.keras.optimizers.RMSprop(0.001)

    # configure the model for training
    model.compile(optimizer=optimizer,
                  loss=loss_fn,
                  metrics = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    # display the summary
    model.summary()

    return model


def block_without_pooling(input, filters):
  conv1 = tf.keras.layers.Conv2D(filters, 3, activation='relu', padding="same", input_shape=input_shape[1:])(input)
  conv2 = tf.keras.layers.Conv2D(filters, 5, activation='relu', padding="same", input_shape=input_shape[1:])(input)
  add = tf.keras.layers.Add()([conv1, conv2])
  output = tf.keras.layers.BatchNormalization()(add)
  return output

def block_with_pooling(input, filters):
  conv1 = tf.keras.layers.Conv2D(filters, 3, activation='relu', padding="same", input_shape=input_shape[1:])(input)
  conv2 = tf.keras.layers.Conv2D(filters, 5, activation='relu', padding="same", input_shape=input_shape[1:])(input)
  add = tf.keras.layers.Add()([conv1, conv2])
  bn = tf.keras.layers.BatchNormalization()(add)
  output = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding="valid")(bn)
  return bn, output

def sfa_module(out1, out2, out3, out4):
  concat1 = tf.keras.layers.Concatenate()([out1, out2])
  sfa1 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding="valid")(concat1)
  concat2 = tf.keras.layers.Concatenate()([out3, out4])
  concat3 = tf.keras.layers.Concatenate()([sfa1, concat2])
  output = tf.keras.layers.AveragePooling2D(pool_size=(8, 8), padding="valid")(concat3)
  return output

def build_model_hhna(no_of_classes, img_shape, optimizer=tf.keras.optimizers.Adam(0.001)):
  r = img_shape[0]
  c = img_shape[1]

  input = tf.keras.Input(shape=(r, c, 3))
  block1_output = block_without_pooling(input, 16)
  bn2, block2_output = block_with_pooling(block1_output, 32)
  block3_output = block_without_pooling(block2_output, 64)
  bn4, block4_output = block_with_pooling(block3_output, 128)
  block5_output = block_without_pooling(block4_output, 128)
  bn6, block6_output = block_with_pooling(block5_output, 256)
  block7_output = block_without_pooling(block6_output, 256)
  bn8, block8_output = block_with_pooling(block7_output, 512)
  block9_output = block_without_pooling(block8_output, 1024)
  sfa_output = sfa_module(block1_output, bn2, block3_output, bn4)
  vfa_sfa_concat = tf.keras.layers.Concatenate()([block9_output, sfa_output])
  gap = tf.keras.layers.GlobalAveragePooling2D()(vfa_sfa_concat)
  dropout = tf.keras.layers.Dropout(0.2)(gap)
  if(no_of_classes == 2):
    no_of_classes = no_of_classes - 1
    activation_fn = 'sigmoid'
    loss_fn = 'binary_crossentropy'
  else:
    activation_fn = 'softmax'
    loss_fn = 'categorical_crossentropy'
  classifier = tf.keras.layers.Dense(no_of_classes, activation=activation_fn)(dropout)

  model = Model(input, classifier)

  # choose the optimizer
  #optimizer = tf.keras.optimizers.Adam(0.001)

  # configure the model for training
  model.compile(optimizer=optimizer,
                loss=loss_fn,
                metrics = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

  # display the summary
  model.summary()

  return model


def train_model(model, batch_size, epochs, train_generator, validation_generator, model_path, history_path):
  scheduler_epoch = int(config.lr_scheduler_epoch_factor*epochs)
  early_stopping_epoch = int(config.early_stopping_epoch_factor*epochs)

  checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=model_path,
                                                  monitor='val_loss',
                                                  save_best_only=True)

  reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)

  # Training model
  hist = model.fit(train_generator, steps_per_epoch=train_generator.samples //
                           batch_size, epochs=epochs, validation_data=validation_generator, validation_steps=validation_generator.samples //
                           batch_size, callbacks=[checkpoint, reduce_lr, MyCallback()])

  # Save history for training, validation curves
  np.save(history_path, hist.history)

  return model, hist


def build_model(model_fn, no_of_classes, img_shape, optimizer, batch_size, epochs, train_generator, validation_generator, model_path, history_path):
  model = model_fn(no_of_classes, img_shape, optimizer)
  model, hist = train_model(model, batch_size, epochs, train_generator, validation_generator, model_path, history_path)
  return model, hist

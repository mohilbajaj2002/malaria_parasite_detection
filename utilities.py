import config
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def plot_history(history, key, model_name):
    #history = history.tolist()          # when reading from file
    fig = plt.figure()
    fig.suptitle(f'Results for {model_name}', fontsize=15)
    fig.subplots_adjust(hspace=0.4, top=0.85)
    s1 = fig.add_subplot(1,2,1)
    s1.title.set_text("Loss vs Val_loss")
    plt.plot(history.history['loss'][3:])
    plt.plot(history.history['val_loss'][3:])
    #plt.plot(history['loss'][3:])        # when reading from file
    #plt.plot(history['val_loss'][3:])    # when reading from file
    plt.xlabel("Epochs")
    plt.ylabel('loss')
    plt.legend(['loss', 'val_loss'])
    s2 = fig.add_subplot(1,2,2)
    s2.title.set_text(f"{key} vs Val_{key}")
    plt.plot(history.history[key])
    plt.plot(history.history['val_'+ key])
    #plt.plot(history[key])                # when reading from file
    #plt.plot(history['val_'+ key])        # when reading from file
    plt.xlabel("Epochs")
    plt.ylabel(key)
    plt.legend([key, 'val_'+key])
    plt.show()


def display_performance_metrics(model, validation_generator, model_name):
    labels = validation_generator.labels
    preds = model.predict(validation_generator)
    predictions = [np.round(x[0]) for x in preds]
    print(f'Results for Model: {model_name}')
    print(classification_report(labels, predictions))
    print(confusion_matrix(labels, predictions))
    print('')


def get_optimizer(opt_name):
    if(opt_name == 'SGD'):
        return 'SGD'
    elif(opt_name == 'RMSProp'):
        return tf.keras.optimizers.RMSprop(0.001)
    elif(opt_name == 'Adam'):
        return tf.keras.optimizers.Adam(0.001)


# ['VGG16', 'InceptionV3', 'InceptionResNetV2', 'Resnet50', 'HHNA']
def get_arch_fn(arch_name):
    if(arch_name == 'VGG16'):
        return build_model_vgg16
    elif(arch_name == 'InceptionV3'):
        return build_model_inception_v3
    elif(arch_name == 'InceptionResNetV2'):
        return build_model_inception_resnet_v2
    elif(arch_name == 'Resnet50'):
        return build_model_resnet50
    elif(arch_name == 'HHNA'):
        return build_model_hhna


def get_resnet_toggle(name):
    if(name == 'Resnet50'):
        return True
    else:
        return False

def get_classifier_details(arch_name):
    if(arch_name == 'HHNA'):
        return '3_layer_classifer'
    else:
        return '2_layer_classifer'

def get_image_shape(arch_name):
    if(arch_name == 'HHNA'):
        return config.hhna_img_shape
    else:
        return config.img_shape

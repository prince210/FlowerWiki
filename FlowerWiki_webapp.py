###################### using Flask  ##############################
from flask import Flask, redirect, url_for, request, render_template

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras.models import model_from_json
from keras import backend as K

from scipy.misc import imread, imresize
import os
import sys
import numpy as np
import keras.models
# import pickle
import re
import tensorflow as tf


IMG_SIZE = 128
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)


app = Flask(__name__)


def loadmodel():
    K.clear_session()
    json_file = open('model.json', 'r')
    loaded_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_json)
    loaded_model.load_weights('my_new_model_vgg16.h5')
    loaded_model.compile(optimizer='sgd',
                         loss='categorical_crossentropy',
                         metrics=['accuracy'])
    return loaded_model

    print("loaded model")



config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))


def predict(image_path, model):
    img = image.load_img(image_path, target_size=(128, 128))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')
    preds = model.predict(x)
    return preds

flower_name = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'tulip']
label_dict = {}
for i in range(len(flower_name)):
    label_dict[i] = flower_name[i]

print("Yes! we are in..")


@app.route('/', methods=['GET'])
def home():
    return render_template(r'indexx.html')

@app.route('/out.html', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, r'webapp_photo/')
        print(file_path)

        if not os.path.isdir(file_path):
            os.mkdir(file_path)

        for file in request.files.getlist("inpFile"):
            print(file)
            print("in for")
            filename = file.filename
            destination = "/".join([file_path, filename])
            print(destination)
            file.save(destination)
            model = loadmodel()
        preds = predict(destination, model)

        if np.argmax(preds) == 0:
            word_label = label_dict.get(np.argmax(preds))
        elif np.argmax(preds) == 1:
            word_label = label_dict.get(np.argmax(preds))
        elif np.argmax(preds) == 2:
            word_label = label_dict.get(np.argmax(preds))
        elif np.argmax(preds) == 3:
            word_label = label_dict.get(np.argmax(preds))
        elif np.argmax(preds) == 4:
            word_label = label_dict.get(np.argmax(preds))

        return word_label
    return render_template(r'out.html')


if __name__ == '__main__':
    app.run(debug=True)

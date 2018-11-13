import logging
import numpy as np

from keras import models
from keras.preprocessing import image
from keras.utils import get_file

from deepseenet import eyesnet_risk_factor
from deepseenet.utils import crop2square
import cv2

LEFT_RIGHT_PATH = 'https://github.com/yfpeng/EyesNet/releases/download/v0.1/leftright_model_20180702.h5'
LEFT_RIGHT_MD5 = '365e03195e6d79a8b13ed279f1c8dd1a'


def preprocess_image(image_path):
    """
    Loads an image into a Numpy array

    Args:
        image_path: Path or file object.

    Returns:
        Numpy array
    """
    logging.debug('Processing: %s', image_path)
    img = image.load_img(image_path)
    img = crop2square(img)
    img = img.resize((512, 512))
    x = image.img_to_array(img)
    x = cv2.addWeighted(x, 4, cv2.GaussianBlur(x, (0, 0), 1000 / 30), -4, 128)
    x = np.expand_dims(x, axis=0)
    x = eyesnet_risk_factor.preprocess_input(x)
    return x
    # logging.debug('Processing: %s', image_path)
    # img = image.load_img(image_path).resize((224, 224))
    # x = image.img_to_array(img)
    # x = np.expand_dims(x, axis=0)
    # # x = eyesnet_risk_factor.preprocess_input(x)
    # return x


def EyesNetLeftRight(model='areds1'):
    """
    Instantiates the EyesNet field2 left or right architecture.

    Args:
        model: One of 'areds1' (pre-training on AREDS1), or the path to the model file to be loaded.

    Returns:
        A Keras model instance. 0 for right and 1 for left
    """
    if model == 'areds1':
        model = get_file(
            'leftright_model.h5',
            LEFT_RIGHT_PATH,
            cache_dir='models',
            file_hash=LEFT_RIGHT_MD5
        )
    logging.info('Loading the model: %s', model)
    return models.load_model(model)


def get_left_right(score):
    y = np.argmax(score, axis=1)
    if y == 0:
        return 'R'
    elif y == 1:
        return 'L'
    else:
        raise ValueError('It must be a binary classification')

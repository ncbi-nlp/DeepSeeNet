import logging

import numpy as np
from keras import models
from keras.preprocessing import image

from deepseenet import eyesnet_risk_factor


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
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = eyesnet_risk_factor.preprocess_input(x)
    return x


def EyesNetDrarwi(model):
    """
    Instantiates the EyesNet drusen architecture.

    Args:
        model: One of 'areds1' (pre-training on AREDS1),
              or the path to the model file to be loaded.

    Returns:
        A Keras model instance.
    """
    logging.info('Loading the model: %s', model)
    return models.load_model(model)


def get_drarwi(score):
    y = np.argmax(score, axis=1)
    return y
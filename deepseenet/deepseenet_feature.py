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


def EyesNetFeature(model_file, layer_name: str):
    """
    Instantiates the EyesNet drusen architecture.

    Args:
        model_file: One of 'areds1' (pre-training on AREDS1), or the path to the model file to be loaded.
        layer_name: name of layer

    Returns:
        A Keras model instance.
    """
    logging.info('Loading the model: %s', model_file)
    model = models.load_model(model_file)
    layer_output = model.get_layer(layer_name).output
    model = models.Model(inputs=model.input, outputs=layer_output)
    return model

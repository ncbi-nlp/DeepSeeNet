import logging

import numpy as np
from keras import models
from keras.preprocessing import image
from keras.utils import get_file

from deepseenet import deepseenet_risk_factor
from deepseenet.utils import crop2square

DRUSEN_PATH = 'https://github.com/ncbi-nlp/DeepSeeNet/releases/download/0.1/drusen_model.h5'
DRUSEN_MD5 = '997a8229f972482e127e8a32d1967549'


def preprocess_image(image_path):
    """
    Loads an image into a Numpy array

    Args:
        image_path: Path or file object.

    Returns:
        Numpy array
    """
    logging.debug('Processing: %s', image_path)
    img = crop2square(image.load_img(image_path)).resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = deepseenet_risk_factor.preprocess_input(x)
    return x


def DeepSeeNetDrusen(model='areds'):
    """
    Instantiates the EyesNet drusen architecture.

    Args:
        model: One of 'areds1' (pre-training on AREDS1),
              or the path to the model file to be loaded.

    Returns:
        A Keras model instance.
    """
    if model == 'areds':
        model = get_file(
            'drusen_model.h5',
            DRUSEN_PATH,
            cache_dir='models',
            file_hash=DRUSEN_MD5
        )
    logging.info('Loading the model: %s', model)
    return models.load_model(model)


def get_drusen_size(score):
    y = np.argmax(score, axis=1)
    if y == 0:
        return 'small/none'
    elif y == 1:
        return 'intermediate'
    elif y == 2:
        return 'large'
    else:
        raise KeyError
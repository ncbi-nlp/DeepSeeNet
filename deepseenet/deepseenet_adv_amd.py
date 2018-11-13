import logging
import numpy as np

from keras import models
from keras.preprocessing import image
from keras.utils import get_file

from deepseenet import eyesnet_risk_factor
from deepseenet.utils import crop2square

ADVANCED_AMD_PATH = 'https://github.com/yfpeng/EyesNet/releases/download/v0.1/adv_amd_model.h5'
ADVANCED_AMD_MD5 = '0adbf448491ead63ac384da671c4f7ee'


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
    x = eyesnet_risk_factor.preprocess_input(x)
    return x


def EyesNetAdvancedAMD(model='areds1'):
    """
    Instantiates the EyesNet advanced AMD architecture.

    Args:
        model: One of 'areds1' (pre-training on AREDS1),
              or the path to the model file to be loaded.

    Returns:
        A Keras model instance.
    """
    if model == 'areds1':
        model = get_file(
            'advanced_amd_model.h5',
            ADVANCED_AMD_PATH,
            cache_dir='models',
            file_hash=ADVANCED_AMD_MD5
        )
    logging.info('Loading the model: %s', model)
    return models.load_model(model)

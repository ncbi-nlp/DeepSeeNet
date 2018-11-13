import logging
import numpy as np
import cv2
from keras import models
from keras.preprocessing import image
from keras.utils import get_file

from deepseenet import eyesnet_risk_factor
from deepseenet.utils import crop2square

FIELD2_PATH = 'https://github.com/yfpeng/EyesNet/releases/download/v0.1/field2_model_20180703.h5'
FIELD2_MD5 = ''


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
    # img = crop2square(image.load_img(image_path)).resize((512, 512))
    # x = image.img_to_array(img)
    # x = np.expand_dims(x, axis=0)
    # x = eyesnet_risk_factor.preprocess_input(x)
    # return x


def EyesNetField2(model='areds1'):
    """
    Instantiates the EyesNet field2 architecture.

    Args:
        model: One of 'areds1' (pre-training on AREDS1), or the path to the model file to be loaded.

    Returns:
        A Keras model instance. 0 for field2 and 1 for others
    """
    if model == 'areds1':
        model = get_file(
            'field2_model.h5',
            FIELD2_PATH,
            cache_dir='models',
            file_hash=FIELD2_MD5
        )
    logging.info('Loading the model: %s', model)
    return models.load_model(model)


def get_field(score):
    y = np.argmax(score, axis=1)
    if y == 0:
        return 'F2'
    elif y == 1:
        return 'Others'
    else:
        raise ValueError('It must be a binary classification')

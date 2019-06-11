import logging
import numpy as np

from keras import models
from keras.preprocessing import image
from keras.utils import get_file

from deepseenet import deepseenet_risk_factor
from deepseenet.utils import crop2square

GA_PATH = 'https://github.com/ncbi-nlp/DeepSeeNet/releases/download/0.2/ga_model.h5'
GA_MD5 = '59350371c73eaaff397d477b702c456a'


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


def DeepSeeNetGA(model='areds'):
    """
    Instantiates the EyesNet pigmentary abnormality architecture.

    Args:
        model: One of 'areds1' (pre-training on AREDS1),
              or the path to the model file to be loaded.

    Returns:
        A Keras model instance.
    """
    if model == 'areds':
        model = get_file(
            'ga_model.h5',
            GA_PATH,
            cache_dir='models',
            file_hash=GA_MD5
        )
    logging.info('Loading the model: %s', model)
    return models.load_model(model)

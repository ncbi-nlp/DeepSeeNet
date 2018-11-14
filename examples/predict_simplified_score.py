"""
Grades color fundus photographs using the AREDS Simplified Severity Scale.

Usage:
    predict_simplified_score [options] <left_eye_image> <right_eye_image>

Options:
    -d <str>    Drusen model
    -p <str>    Pigment model
    -a <str>    Advanced AMD model
"""
import logging
import sys

import docopt
import numpy as np
from keras.preprocessing import image

from deepseenet import eyesnet_simplified
from deepseenet.utils import crop2square, pick_device


def preprocess_image(image_path, target_size=(224, 224)):
    """
    Loads an image into a Numpy array

    Args:
        image_path: Path or file object.
        model_name: only 'inceptionv3' supported
        target_size: 224x224 by default

    Returns:
        Numpy array
    """
    logging.info('Processing: %s', image_path)
    img = crop2square(image.load_img(image_path)).resize(target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x


if __name__ == '__main__':
    argv = docopt.docopt(__doc__, argv=sys.argv[1:])
    logging.basicConfig(level=logging.DEBUG)
    logging.debug(argv)

    drusen_model = argv['-d'] if '-d' in argv else None
    pigment_model = argv['-p'] if '-p' in argv else None
    advanced_amd_model = argv['-a'] if '-a' in argv else None

    pick_device()
    clf = eyesnet_simplified.EyesNetSimplifiedScore(drusen_model, pigment_model, advanced_amd_model)
    score = clf.predict(argv['<left_eye_image>'], argv['<right_eye_image>'], verbose=1)
    print('The simplified score:', score)

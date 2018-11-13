"""
Predict the drusen size of color fundus photographs.

Usage:
    predict_drusen [options] <eye_image>

Options:
    -d <str>    Drusen model. [default: areds1]
"""
import logging
import sys

import docopt

from deepseenet.eyesnet_drusen import EyesNetDrusen, preprocess_image, get_drusen_size
from deepseenet.utils import pick_device

if __name__ == '__main__':
    argv = docopt.docopt(__doc__, argv=sys.argv[1:])
    logging.basicConfig(level=logging.DEBUG)
    logging.debug(argv)

    pick_device()
    clf = EyesNetDrusen(argv['-d'])
    x = preprocess_image(argv['<eye_image>'])
    score = clf.predict(x, verbose=1)
    print('The drusen score:', score)
    print('The drusen size:', get_drusen_size(score))

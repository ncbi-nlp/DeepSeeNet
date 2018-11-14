"""
Predict if the fundus photographs is field2.

Usage:
    predict field2 [options] --output <file> [<file> ...]

Options:
    --output=<file>     Specify the output file name.
    --model=<str>       Field2 model. [default: areds1]
    --verbose           Print more information about progress.
    --file-list=<file>  File list.
"""
import csv
import logging

import numpy as np
import tqdm
from PIL import Image, ImageChops

from examples.utils import parse_args
from deepseenet.eyesnet_f2 import preprocess_image, EyesNetField2, get_field
from deepseenet.utils import pick_device


def is_greyscale(image_path):
    """
    Check if image is monochrome (1 channel or 3 identical channels)
    """
    im = Image.open(image_path)
    if im.mode not in ("L", "RGB"):
        raise ValueError("Unsupported image mode")

    if im.mode == "RGB":
        rgb = im.split()
        if ImageChops.difference(rgb[0], rgb[1]).getextrema()[1] != 0:
            return False
        if ImageChops.difference(rgb[0], rgb[2]).getextrema()[1] != 0:
            return False
    return True


if __name__ == '__main__':
    argv = parse_args(__doc__)

    pick_device()

    clf = EyesNetField2(argv['--model'])

    if argv['--file-list'] is not None:
        with open(argv['--file-list']) as fp:
            files = [line.strip() for line in fp]
    else:
        files = argv['<file>']

    with open(argv['--output'], 'w') as fp:
        writer = csv.writer(fp, lineterminator='\n')
        writer.writerow(['file', 'field', '0', '1'])
        for file in tqdm.tqdm(files):
            logging.debug('Process %s', file)
            try:
                if is_greyscale(file):
                    score = np.asarray([[0, 1]])
                else:
                    x = preprocess_image(file)
                    score = clf.predict(x, verbose=1 if argv['--verbose'] else 0)
                writer.writerow([file, get_field(score)] + [str(a) for a in np.nditer(score)])
            except:
                logging.exception('Cannot process %s', file)

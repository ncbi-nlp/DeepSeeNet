"""
Predict if the fundus photographs is left or right eye of field2.

Usage:
    predict left_right [options] --output <file> [<file> ...]

Options:
    --output=<file>     Specify the output file name.
    --model=<str>       Left/Right model. [default: areds1]
    --verbose           Print more information about progress.
    --file-list=<file>  File list.
"""
import csv
import logging

import numpy as np
import tqdm

from examples.utils import parse_args
from deepseenet.eyesnet_left_right import preprocess_image, EyesNetLeftRight, get_left_right
from deepseenet.utils import pick_device


if __name__ == '__main__':
    argv = parse_args(__doc__)

    pick_device()

    clf = EyesNetLeftRight(argv['--model'])

    if argv['--file-list'] is not None:
        with open(argv['--file-list']) as fp:
            files = [line.strip() for line in fp]
    else:
        files = argv['<file>']

    with open(argv['--output'], 'w') as fp:
        writer = csv.writer(fp, lineterminator='\n')
        writer.writerow(['file', 'eye', '0', '1'])
        for file in tqdm.tqdm(files):
            logging.debug('Process %s', file)
            try:
                x = preprocess_image(file)
                score = clf.predict(x, verbose=1 if argv['--verbose'] else 0)
                writer.writerow([file, get_left_right(score)] + [str(a) for a in np.nditer(score)])
            except:
                logging.exception('Cannot process %s', file)

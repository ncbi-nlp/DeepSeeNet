"""
Predict risk factor.

Usage:
    predict risk_factor [options] --classes=<int> --model <file> --output <file> --input <file>

Options:
    --image-dir=<str>   Image directory
    --output=<file>     Specify the output file name.
    --model=<file>      Keras model.
    --verbose           Print more information about progress.
    --input=<file>      CSV file contains "filename".
    --classes=<int>     Number of classes
"""
import csv
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm

from examples.utils import parse_args
from deepseenet.eyesnet_drarwi import preprocess_image, EyesNetDrarwi, get_drarwi
from deepseenet.utils import pick_device


if __name__ == '__main__':
    argv = parse_args(__doc__)

    pick_device()

    clf = EyesNetDrarwi(argv['--model'])
    input_df = pd.read_csv(argv['--input'])

    predictions = []
    with open(argv['--output'], 'w') as fp:
        writer = csv.writer(fp, lineterminator='\n')
        heading = list(input_df.columns.values)
        heading += ['pred'] + [str(i) for i in range(int(argv['--classes']))]
        writer.writerow(heading)

        for i, row in tqdm.tqdm(input_df.iterrows(), total=len(input_df)):
            file = row.filename
            if argv['--image-dir']:
                file = Path(argv['--image-dir']) / file
            logging.debug('Process %s', file)
            try:
                x = preprocess_image(file)
                score = clf.predict(x, verbose=1 if argv['--verbose'] else 0)
                row_o = [str(row[i]) for i in range(input_df.shape[1])]
                row_o += [str(get_drarwi(score)[0])] + [str(a) for a in np.nditer(score)]
                writer.writerow(row_o)
            except:
                logging.exception('Cannot process %s', file)

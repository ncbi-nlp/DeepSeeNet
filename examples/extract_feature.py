"""
Extract features from images.

Usage:
    extract extract_features [options] --model <file> --layer-name <str> --output <file> --input <file>

Options:
    --image-dir=<str>   Image directory
    --output=<file>     Specify the output file name.
    --model=<file>      Keras model.
    --verbose           Print more information about progress.
    --input=<file>      CSV file contains "filename".
    --layer-name=<str>  Name of layer

Possible layer name:
    global_dense1
    geoawi_dense1
"""
import csv
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm

from examples.utils import parse_args
from deepseenet.eyesnet_feature import EyesNetFeature, preprocess_image
from deepseenet.utils import pick_device


if __name__ == '__main__':
    argv = parse_args(__doc__)

    pick_device()

    clf = EyesNetFeature(argv['--model'], argv['--layer-name'])
    input_df = pd.read_csv(argv['--input'])

    predictions = []
    with open(argv['--output'], 'w') as fp:
        writer = csv.writer(fp, lineterminator='\n')
        heading = list(input_df.columns.values)
        heading += [str(i) for i in range(clf.output.get_shape()[1])]
        writer.writerow(heading)

        for i, row in tqdm.tqdm(input_df.iterrows(), total=len(input_df)):
            file = row.filename
            if argv['--image-dir']:
                file = Path(argv['--image-dir']) / file
            logging.debug('Process %s', file)
            try:
                x = preprocess_image(file)
                features = clf.predict(x, verbose=1 if argv['--verbose'] else 0)
                features = features.flatten()
                row_o = [str(row[i]) for i in range(input_df.shape[1])]
                row_o += [str(a) for a in np.nditer(features)]
                writer.writerow(row_o)
            except:
                logging.exception('Cannot process %s', file)

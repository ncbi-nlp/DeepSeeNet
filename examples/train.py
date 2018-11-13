"""
Train an individual risk factor model.

Usage:
    train [options] <dataset> <best_model>

Options:
    --n_classes=<int>   Number of classes [default: 2]
    --prefix=<str>      Data directory [default: .]
"""
import logging
import multiprocessing
import os
import sys

import docopt
import numpy as np
import pandas as pd
from keras import backend as K
from keras import callbacks
from keras.optimizers import Adam
from keras.preprocessing import image

from deepseenet import eyesnet_risk_factor
from deepseenet.data_generator import DataGenerator
from deepseenet.utils import pick_device, crop2square


def preprocess_image(image_path):
    img = crop2square(image.load_img(image_path)).resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = eyesnet_risk_factor.preprocess_input(x)
    return x


def train(model, train_data, valid_data, best_model, batch_size=32, n_classes=2):
    early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=K.epsilon(), patience=2, verbose=1)
    best_model_cp = callbacks.ModelCheckpoint(best_model, save_best_only=True, monitor='val_acc', verbose=1)

    optimizer = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    if n_classes == 2:
        model.compile(optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    cpu_count = multiprocessing.cpu_count()
    workers = max(int(cpu_count / 3), 1)

    train_generator = DataGenerator(train_data, preprocess_image=preprocess_image,
                                    batch_size=batch_size, n_classes=n_classes, shuffle=True)
    valid_generator = DataGenerator(valid_data, preprocess_image=preprocess_image,
                                    batch_size=batch_size, n_classes=n_classes, shuffle=False)
    train_chunck_number = train_generator.get_epoch_num()
    model.fit_generator(
        train_generator,
        class_weight=train_generator.class_weights(),
        use_multiprocessing=True,
        workers=workers,
        steps_per_epoch=train_chunck_number,
        callbacks=[early_stop, best_model_cp],
        epochs=100,
        validation_data=valid_generator,
        validation_steps=valid_generator.get_epoch_num(),
        verbose=1)


def prep_instances(train_file, val_size=0.8, shuffle=True, parent='.'):
    """
    Read the dataset.

    Args:
        train_file(str): the file path of training dataset
        val_size(float): should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the
            validation split.
        shuffle(bool): Whether or not to shuffle the data before splitting.
        parent(str): data directory
    Returns:
        list: List containing train-validation split of inputs.
    """
    df = pd.read_csv(train_file)
    rows = df.values.tolist()

    for i in range(len(rows)):
        rows[i][0] = os.path.join(parent, rows[i][0])

    if shuffle:
        np.random.shuffle(rows)
    split_size = int(len(rows) * val_size)
    return rows[:split_size], rows[split_size:]


if __name__ == '__main__':
    argv = docopt.docopt(__doc__, argv=sys.argv[1:])
    logging.basicConfig(level=logging.DEBUG)
    logging.debug(argv)

    pick_device()
    n_classes = int(argv['--n_classes'])

    train_data, valid_data = prep_instances(argv['<dataset>'], parent=argv['--prefix'])
    logging.info('Training instances: %s', len(train_data))
    logging.info('Validation instances: %s', len(valid_data))
    model = eyesnet_risk_factor.RiskFactorModel(n_classes=n_classes)
    train(model, train_data, valid_data, argv['<best_model>'], n_classes=n_classes)


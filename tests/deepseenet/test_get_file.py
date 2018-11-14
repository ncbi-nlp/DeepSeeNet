import os

from ..context import deepseenet
from keras.utils import get_file


def test_get_file():
    drusen_model = get_file(
        'drusen_model.h5',
        deepseenet.deepseenet_drusen.DRUSEN_PATH,
        cache_dir='models',
        md5_hash=deepseenet.deepseenet_drusen.DRUSEN_MD5
    )
    print(drusen_model)
    assert os.path.exists(drusen_model)

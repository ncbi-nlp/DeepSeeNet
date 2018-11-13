from ..context import deepseenet
from keras.utils import get_file


def test_get_file():
    drusen_model = get_file(
        'drusen_model.h5',
        deepseenet.model.DRUSEN_PATH,
        cache_dir='models',
        file_hash=''
    )
    print(drusen_model)




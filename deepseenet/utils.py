import logging
import os

import GPUtil


def pick_device():
    try:
        GPUtil.showUtilization()
        # Get the first available GPU
        DEVICE_ID_LIST = GPUtil.getFirstAvailable()
        DEVICE_ID = DEVICE_ID_LIST[0]  # grab first element from list
        # Set CUDA_VISIBLE_DEVICES to mask out all other GPUs than the first available device id
        os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
        logging.debug('Device ID (unmasked): ' + str(DEVICE_ID))
    except:
        logging.exception('Cannot detect GPUs')


def crop2square(img):
    """
    Crop the image to a square based on the short edge.

    Args:
        img: PIL Image instance.

    Returns:
        A PIL Image instance.
    """
    short_side = min(img.size)
    x0 = (img.size[0] - short_side) / 2
    y0 = (img.size[1] - short_side) / 2
    x1 = img.size[0] - x0
    y1 = img.size[1] - y0
    return img.crop((x0, y0, x1, y1))


def cal_chunk_number(total_size, batch_size):
    if total_size % batch_size == 0:
        return total_size // batch_size
    else:
        return (total_size // batch_size) + 1

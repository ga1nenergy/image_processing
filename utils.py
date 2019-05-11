import numpy as np
from PIL import Image


def conv(kernel, matrix):
    return np.sum(np.multiply(kernel, matrix))


def channels_to_image(channels):
    channels_image = (Image.fromarray(channels[2]),
                      Image.fromarray(channels[1]),
                      Image.fromarray(channels[0]))

    new_image = Image.merge("RGB", channels_image)
    return new_image

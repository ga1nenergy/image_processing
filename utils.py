import numpy as np
import cv2 as cv
import utils
from PIL import Image

PROCESSING_TYPE = np.float32
FINAL_TYPE = np.uint8


def conv(kernel, matrix):
    return np.sum(np.multiply(kernel, matrix))


def channels_to_image(channels):
    channels_image = (Image.fromarray(channels[0].astype(np.uint8)),
                      Image.fromarray(channels[1].astype(np.uint8)),
                      Image.fromarray(channels[2].astype(np.uint8)))

    new_image = Image.merge("RGB", channels_image)
    return new_image


def opencv_equalizer(channels):
    equalized_channels = []

    for channel in channels:
        buf = np.matrix(channel.astype(np.float) / 255.0, dtype=np.uint8)
        equalized_channel = cv.equalizeHist(np.matrix(channel.astype(np.float) / 255.0, dtype=np.uint8))
        equalized_channels.append(np.matrix(equalized_channel*255, dtype=np.uint8))

    return equalized_channels


def calc_entropy(channels):
    distrib = np.zeros((3, 256), dtype=float)

    H, W = channels[0].shape
    for y in range(H):
        for x in range(W):
            distrib[0, channels[0][y, x]] += 1
            distrib[1, channels[1][y, x]] += 1
            distrib[2, channels[2][y, x]] += 1

    distrib = np.divide(distrib, H * W)

    idx_i, idx_j = np.nonzero(distrib)
    idx = tuple(zip(idx_i, idx_j))

    entropy = np.zeros(3)
    for i in idx:
        entropy[i[0]] += -distrib[i] * np.log(distrib[i]) / np.log(256)

    return entropy


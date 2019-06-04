import numpy as np
import cv2 as cv
from PIL import Image
import jpeg_codec

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


def calc_entropy_rle(channels):
    channels_distribution = []

    for channel in channels:
        image = np.matrix(channel.astype(float))
        H, W = image.shape

        if H % 8 != 0 or W % 8 != 0:
            # image = np.resize(image, (H + 8 - H % 8, W + 8 - W % 8))
            image = np.vstack((image, np.zeros((8 - H % 8, W))))
            image = np.hstack((image, np.zeros((image.shape[0], 8 - W % 8))))
            H, W = image.shape

        channel_distribution = dict()

        for y in range(0, H - 8 + 1, 8):
            for x in range(0, W - 8 + 1, 8):
                block = image[y:y+8, x:x+8]
                block_dct = cv.dct(block)
                snake_array = jpeg_codec.snake(block_dct)
                rle_array = jpeg_codec.rle(snake_array)

                for pair in rle_array:
                    if pair != "*":
                        if pair in channel_distribution:
                            channel_distribution[pair] += 1
                        else:
                            channel_distribution[pair] = 1

        channels_distribution.append(channel_distribution)

    entropy = np.zeros(3)
    for i in range(3):
        probs = list(channels_distribution[i].values())
        count = np.sum(np.array(probs))
        probs = [float(prob) / count for prob in probs]

        for prob in probs:
            entropy[i] += -prob * np.log2(prob)

    return entropy
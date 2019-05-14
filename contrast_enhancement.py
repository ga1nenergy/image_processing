import numpy as np
from PIL import Image


def histogram_equalizer(channels):
    [H, W] = channels[0].shape
    disc_rate = 4096

    contrast_channels = []

    for channel in channels:
        contrast_channel = np.matrix((channel/255.0)*(disc_rate-1))
        cdf = np.array(np.histogram(
            contrast_channel, bins=range(disc_rate))[0]) / (H*W)
        for i in range(1, cdf.shape[0]):
            cdf[i] = cdf[i-1] + cdf[i]
        for y in range(H):
            for x in range(W):
                contrast_channel[y, x] = 255.0 * cdf[int(np.floor(contrast_channel[y, x]))]

        contrast_channels.append(contrast_channel)

    return contrast_channels


def non_linear_exponential_transform(channels):
    pass
import numpy as np
from PIL import Image


def histogram_equalizer(channels):
    [H, W] = channels[0].shape

    contrast_channels = []

    for channel in channels:
        contrast_channel = np.matrix(channel)
        min_var = np.amin(channel)
        max_var = np.amax(channel)
        cdf = np.array(Image.fromarray(channel).histogram()) / (H*W)
        for i in range(1, cdf.shape[0]):
            cdf[i] = cdf[i-1] + cdf[i]
        for y in range(H):
            for x in range(W):
                contrast_channel[y, x] = 255 * cdf[contrast_channel[y, x]]

        contrast_channels.append(contrast_channel)

    return contrast_channels


def non_linear_exponential_transform(channels):

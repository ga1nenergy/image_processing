import numpy as np


def gamma_correction(channels):
    gamma = 1.8
    A = 25/255.0
    corrected_channels = []

    for channel in channels:
        corrected_channels.append(
            np.matrix(A*np.power(channel.astype(np.float), gamma), dtype=np.uint8))

    return corrected_channels

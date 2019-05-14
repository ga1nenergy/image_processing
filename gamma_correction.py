import numpy as np


def gamma_correction(channels):
    gamma = 1/2.2
    A = 50.0
    corrected_channels = []

    for channel in channels:
        corrected_channels.append(
            np.matrix(A*np.power(channel, gamma), dtype=channel.dtype))

    return corrected_channels

import numpy as np
from utils import PROCESSING_TYPE


def gamma_correction(channels):
    gamma = 1/2.2
    A = 30.0
    corrected_channels = []

    for channel in channels:
        corrected_channels.append(
            np.matrix(A*np.power(channel, gamma), dtype=PROCESSING_TYPE))

    return corrected_channels

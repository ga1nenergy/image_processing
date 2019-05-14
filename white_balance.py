import numpy as np


def gray_world(channels):
    [H, W] = channels[0].shape
    R_avg = np.sum(channels[0]) / (H * W)
    G_avg = np.sum(channels[1]) / (H * W)
    B_avg = np.sum(channels[2]) / (H * W)

    R_energy = (np.power(channels[0], 2)).mean()
    G_energy = (np.power(channels[1], 2)).mean()
    B_energy = (np.power(channels[2], 2)).mean()

    energies = (R_energy, G_energy, B_energy)
    channel_avgs = (R_avg, G_avg, B_avg)

    main_channel_avg = channel_avgs[energies.index(np.max(energies))]
    alpha = main_channel_avg / R_avg
    beta = main_channel_avg / G_avg
    gamma = main_channel_avg / B_avg

    return [alpha * channels[0], beta * channels[1], gamma * channels[2]]


def white_patch(channels):
    R_max = np.max(channels[0])
    G_max = np.max(channels[0])
    B_max = np.max(channels[0])

    R_energy = (np.power(channels[0], 2)).mean()
    G_energy = (np.power(channels[1], 2)).mean()
    B_energy = (np.power(channels[2], 2)).mean()

    energies = (R_energy, G_energy, B_energy)

    main_channel_max = np.max(energies)
    alpha = main_channel_max / R_max
    beta = main_channel_max / G_max
    gamma = main_channel_max / B_max

    return [alpha * channels[0], beta * channels[1], gamma * channels[2]]
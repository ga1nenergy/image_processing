import numpy as np


def median_filter(channels):
    [H, W] = channels[0].shape

    filtered_channels = []

    for channel in channels:
        filtered_channel = np.zeros(channel.shape, dtype=np.float32)
        for y in range(1, H - 1):
            for x in range(1, W - 1):
                # print(np.median(np.array(channel[y-1:y+2, x-1:x+2])).astype(np.uint8))
                filtered_channel[y, x] = np.median(np.array(channel[y-1:y+2, x-1:x+2])).astype(np.float32)
        filtered_channels.append(filtered_channel)

    return filtered_channels


def gaussian_filter(channels):
    [H, W] = channels[0].shape
    R = 1

    filtered_channels = []

    for channel in channels:
        filtered_channel = np.zeros(channel.shape, dtype=np.uint8)
        for y in range(1, H - 1):
            for x in range(1, W - 1):
                window = channel[y-R:y+R+1, x-R:x+R+1]
                var = channel[y-R:y+R+1, x-R:x+R+1].var()
                z = 0

                for i in range(y-R, y+R):
                    for j in range(x-R, x+R):
                        w = np.exp(-((x-j)**2+(y-i)**2)/(2*var))
                        filtered_channel[y, x] += w * channel[i, j]
                        z += w

                filtered_channel[y, x] /= z
        filtered_channels.append(filtered_channel)

    return filtered_channels


def bilateral_filter(channels):
    [H, W] = channels[0].shape
    R = 1

    filtered_channels = []

    for channel in channels:
        filtered_channel = np.zeros(channel.shape, dtype=np.uint8)
        for y in range(1, H - 1):
            for x in range(1, W - 1):
                window = channel[y-R:y+R+1, x-R:x+R+1]
                var = channel[y-R:y+R+1, x-R:x+R+1].var()
                z = 0

                for i in range(y-R, y+R):
                    for j in range(x-R, x+R):
                        w = np.exp(-((x-j)**2+(y-i)**2)/(2*var))*np.exp(-((channel[i, j]-channel[y, x])**2)/(2*var))
                        filtered_channel[y, x] += w * channel[i, j]
                        z += w

                filtered_channel[y, x] /= z
        filtered_channels.append(filtered_channel)

    return filtered_channels
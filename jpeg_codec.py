import cv2
import numpy as np
import json
import utils


EOS = "*"


def snake(matrix):
    H, W = matrix.shape

    array = []

    flag = 1
    for s in range(0, H):
        for i in range(0, s + 1):
            if flag == 1:
                array.append(matrix[s - i, i])
            else:
                array.append(matrix[i, s - i])
        flag *= -1

    for s in range(H - 2, -1, -1):
        for i in range(0, s + 1):
            if flag == 1:
                array.append(matrix[H - 1 - i, H - 1 - (s - i)])
            else:
                array.append(matrix[H - 1 - (s - i), H - 1 - i])
        flag *= -1

    return array


def de_snake(array):
    H = np.sqrt(len(array)).astype(int)
    matrix = np.zeros((H, H), dtype=utils.PROCESSING_TYPE)

    flag = 1
    pos = 0
    for s in range(0, H):
        for i in range(0, s + 1):
            if flag == 1:
                matrix[s - i, i] = array[pos]
            else:
                matrix[i, s - i] = array[pos]
            pos += 1
        flag *= -1

    for s in range(H - 2, -1, -1):
        for i in range(0, s + 1):
            if flag == 1:
                matrix[H - 1 - i, H - 1 - (s - i)] = array[pos]
            else:
                matrix[H - 1 - (s - i), H - 1 - i] = array[pos]
            pos += 1
        flag *= -1

    return matrix


def rle(snake_arr):
    idx = np.nonzero(snake_arr)[0]

    if idx.size != 0:
        rle_array = [(int(idx[0]), int(snake_arr[idx[0]]))]
        rle_array.extend([(int(idx[i] - idx[i-1] - 1), int(snake_arr[idx[i]])) for i in range(1, idx.size)])
    else:
        rle_array = []
    rle_array.extend(EOS)

    return rle_array


def de_rle(rle_array):
    snake_arr = np.array([], dtype=int)

    count = 0
    for pair in rle_array[:-1]:
        snake_arr = np.append(snake_arr, np.zeros((1, pair[0])))
        snake_arr = np.append(snake_arr, pair[1])
        count += pair[0] + 1

    snake_arr = np.append(snake_arr, np.zeros((1, 64 - count)))

    return snake_arr


def quantization_matrix(quality):
    Q = np.ones((8, 8))
    for i in range(8):
        for j in range(8):
            Q[i, j] += (i + j)*quality
    return Q


def image2jpeg(channels, filename, quality=0):
    jpeg_channels = []

    for channel in channels:
        image = np.matrix(channel.astype(float))
        H, W = image.shape

        if H % 8 != 0 or W % 8 != 0:
            # image = np.resize(image, (H + 8 - H % 8, W + 8 - W % 8))
            image = np.vstack((image, np.zeros((8 - H % 8, W))))
            image = np.hstack((image, np.zeros((image.shape[0], 8 - W % 8))))
            H, W = image.shape

        Q = quantization_matrix(quality)

        jpeg_channel = []

        channel_distribution = np.zeros((1,256))

        for y in range(0, H - 8 + 1, 8):
            for x in range(0, W - 8 + 1, 8):
                block = image[y:y+8, x:x+8]
                block_dct = cv2.dct(block)
                dct_quantized = np.matrix(np.round(np.divide(block_dct, Q)), dtype=int)
                snake_array = snake(dct_quantized)
                # snake_array = list(map(lambda e: e.astype(int), snake_array))
                rle_array = rle(snake_array)
                jpeg_channel.append(rle_array)

        jpeg_channels.append(jpeg_channel)

    res = [channels[0].shape[0], channels[0].shape[1], quality]
    res.append(jpeg_channels)

    with open(filename, "w") as file:
        json.dump(res, file)


def jpeg2image(filename):
    with open(filename, "r") as file:
        true_H, true_W, quality, jpeg_channels = json.load(file)
    # true_H = big_array[0]
    # true_W = big_array[1]
    # quality = big_array[2]
    # jpeg_channels = big_array[3]

    H = true_H
    W = true_W

    if (true_W % 8 != 0) or (true_H % 8 != 0):
        H = true_H + 8 - true_H % 8
        W = true_W + 8 - true_W % 8

    channels = []
    Q = quantization_matrix(quality)

    for jpeg_channel in jpeg_channels:
        x = 0
        y = 0
        channel = np.zeros((H, W), dtype=int)
        for jpeg_rle_array in jpeg_channel:
            jpeg_snake = de_rle(jpeg_rle_array)
            quantized_dct_block = de_snake(jpeg_snake)
            dct_block = np.multiply(quantized_dct_block, Q)
            block = np.clip(cv2.dct(dct_block, flags=cv2.DCT_INVERSE), 0, 255)
            channel[y:y + 8, x:x + 8] = block.astype(int)

            x += 8
            if x == W:
                x = 0
                y += 8

        channels.append(channel[:true_H, :true_W])

    return channels

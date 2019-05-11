import numpy as np
import sys
from utils import conv

np.set_printoptions(threshold=sys.maxsize)

def debayer(image):
    [H, W] = image.shape
    [H_mask, W_mask] = image.shape

    if H % 2 != 0:
        image = np.vstack((image, np.zeros((1, W_mask), dtype=int)))
        H_mask = image.shape[0]
    if W % 2 != 0:
        image = np.hstack((image, np.zeros((H_mask, 1), dtype=int)))
        W_mask = image.shape[1]

    mask_R_cell = np.matrix([[0xFFFF, 0], [0, 0]])
    mask_G_cell = np.matrix([[0, 0xFFFF], [0xFFFF, 0]])
    mask_B_cell = np.matrix([[0, 0], [0, 0xFFFF]])

    mask_R = np.tile(mask_R_cell, int(W_mask/2))
    mask_R = np.tile(mask_R.T, int(H_mask/2)).T

    mask_G = np.tile(mask_G_cell, int(W_mask/2))
    mask_G = np.tile(mask_G.T, int(H_mask/2)).T

    mask_B = np.tile(mask_B_cell, int(W_mask/2))
    mask_B = np.tile(mask_B.T, int(H_mask/2)).T

    R_layer = np.bitwise_and(image, mask_R)
    # print(R_layer[:, -2])
    R_layer = np.matrix(np.resize(R_layer, (H, W)))
    R_layer = np.hstack((R_layer[:, 0], R_layer))
    R_layer = np.hstack((R_layer, R_layer[:, -1]))
    R_layer = np.vstack((R_layer[0:1, :], R_layer))
    R_layer = np.vstack((R_layer, R_layer[-1, :]))

    G_layer = np.bitwise_and(image, mask_G)
    G_layer = np.matrix(np.resize(G_layer, (H, W)))
    G_layer = np.hstack((G_layer[:, 0], G_layer))
    G_layer = np.hstack((G_layer, G_layer[:, -1]))
    G_layer = np.vstack((G_layer[0, :], G_layer))
    G_layer = np.vstack((G_layer, G_layer[-1, :]))

    B_layer = np.bitwise_and(image, mask_B)
    B_layer = np.matrix(np.resize(B_layer, (H, W)))
    B_layer = np.hstack((B_layer[:, 0], B_layer))
    B_layer = np.hstack((B_layer, B_layer[:, -1]))
    B_layer = np.vstack((B_layer[0, :], B_layer))
    B_layer = np.vstack((B_layer, B_layer[-1, :]))

    kernel_cross = 0.25 * np.matrix([[1, 0, 1],
                                     [0, 0, 0],
                                     [1, 0, 1]])
    kernel_plus = 0.25 * np.matrix([[0, 1, 0],
                                    [1, 0, 1],
                                    [0, 1, 0]])

    R_layer_final = np.matrix(R_layer[1:-1-1, 1:-1-1])
    G_layer_final = np.matrix(G_layer[1:-1-1, 1:-1-1])
    B_layer_final = np.matrix(B_layer[1:-1-1, 1:-1-1])

    for y in range(1, H-1):
        for x in range(1, W-1):
            if G_layer[y, x] == 0:
                G_layer_final[y, x] = conv(kernel_plus, G_layer[y-1:y+2, x-1:x+2])

            if R_layer[y, x] == 0:
                if (R_layer[y, x-1] != 0 and R_layer[y, x+1] != 0):
                    R_layer_final[y-1, x-1] = (R_layer[y, x-1] + R_layer[y, x+1]) / 2
                elif (R_layer[y-1, x] != 0 and R_layer[y+1, x] != 0):
                    R_layer_final[y-1, x-1] = (R_layer[y-1, x] + R_layer[y+1, x]) / 2
                else:
                    R_layer_final[y-1, x-1] = conv(kernel_cross, R_layer[y-1:y+2, x-1:x+2])

            if B_layer[y, x] == 0:
                if (B_layer[y, x-1] != 0 and B_layer[y, x+1] != 0):
                    B_layer_final[y-1, x-1] = (B_layer[y, x-1] + B_layer[y, x+1]) / 2
                elif (B_layer[y-1, x] != 0 and B_layer[y+1, x] != 0):
                    B_layer_final[y-1, x-1] = (B_layer[y-1, x] + B_layer[y+1, x]) / 2
                else:
                    B_layer_final[y-1, x-1] = conv(kernel_cross, B_layer[y-1:y+2, x-1:x+2])

    # print(R_layer[50+1:50+1+10, 50+1:50+1+10])
    # print(R_layer_final[50+1:50+1+10, 50:50+10])

    R_layer_final = np.matrix((255 * R_layer_final / 0xFFFF).astype(np.uint8))
    G_layer_final = np.matrix((255 * G_layer_final / 0xFFFF).astype(np.uint8))
    B_layer_final = np.matrix((255 * B_layer_final / 0xFFFF).astype(np.uint8))

    return [R_layer_final, G_layer_final, B_layer_final]
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from debayering import debayer
from noise_reduction import median_filter, gaussian_filter
from white_balance import gray_world, white_patch
from contrast_enhancement import histogram_equalizer
from gamma_correction import gamma_correction
import utils

pil_image = Image.open('example_very_tiny.tiff')
bayer_image = np.array(pil_image)

def main():
    # Debayering
    channels = debayer(bayer_image)
    name = "debayer"

    image = utils.channels_to_image(channels)
    # debayered_image.show()
    image.save("debayer.png", "PNG")

    channels = median_filter(channels)
    name += "_median"
    image = utils.channels_to_image(channels)
    # image.show()
    image.save(name+".png", "PNG")

    channels = gray_world(channels)
    name += "_gray"
    image = utils.channels_to_image(channels)
    # image.show()
    image.save(name+".png", "PNG")

    channels = gamma_correction(channels)
    name += "_correction"
    image = utils.channels_to_image(channels)
    # image.show()
    image.save(name+".png", "PNG")

    # before equalizer
    plt.bar(range(256), Image.fromarray(channels[1]).histogram())
    plt.show()

    equalized_channels = histogram_equalizer(channels)

    # after equalizer
    plt.bar(range(256), Image.fromarray(channels[1]).histogram())
    plt.show()

    name += "_equalizer"
    image = utils.channels_to_image(channels)
    image.show()

main()
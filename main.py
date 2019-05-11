import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from debayering import debayer
from noise_reduction import median_filter, gaussian_filter
from white_balance import gray_world, white_patch
from contrast_enhancement import histogram_equalizer
import utils

pil_image = Image.open('example_very_tiny.tiff')
image = np.array(pil_image)

def main():
    # Debayering
    channels = debayer(image)
    name = "debayer"

    # debayered_image = utils.channels_to_image(channels)
    # debayered_image.show()
    # debayered_image.save("debayer.png", "PNG")

    filtered_channels = median_filter(channels)
    name += "_median"
    filtered_image = utils.channels_to_image(filtered_channels)
    filtered_image.show()

    # before equalizer
    plt.bar(range(256), Image.fromarray(filtered_channels[1]).histogram())
    plt.show()

    equalized_channels = histogram_equalizer(filtered_channels)

    # after equalizer
    plt.bar(range(256), Image.fromarray(equalized_channels[1]).histogram())
    plt.show()

    name += "_equalizer"
    equalized_image = utils.channels_to_image(equalized_channels)
    equalized_image.show()

    balanced_channels = gray_world(filtered_channels)
    name += "_gray"
    balanced_image = utils.channels_to_image(balanced_channels)
    balanced_image.show()
    balanced_image.save(name+".png", "PNG")

main()
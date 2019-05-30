import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from debayering import debayer
from noise_reduction import median_filter, gaussian_filter
from white_balance import gray_world, white_patch
from contrast_enhancement import histogram_equalizer
from gamma_correction import gamma_correction
import utils
import jpeg_codec

pil_image = Image.open('example_cut.tiff')
bayer_image = np.array(pil_image)


def test_sample():
    image = np.array(Image.open('cut.png'))
    channels = [image[:, :, 0],
                image[:, :, 1],
                image[:, :, 2]]
    # utils.channels_to_image(channels).show()
    jpeg = jpeg_codec.image2jpeg(channels, "foo", quality=10)
    de_jpeg = jpeg_codec.jpeg2image("foo")
    jpeg_image = utils.channels_to_image(de_jpeg)
    jpeg_image.save("jpeg_cut.png", "PNG")


def snake_tester():
    m = np.random.randint(0, 9, (8, 8))
    print(m)
    s = jpeg_codec.snake(m)
    de_s = jpeg_codec.de_snake(s)
    print(de_s)
    r = jpeg_codec.rle(s)
    de_r = jpeg_codec.de_rle(r)
    pass


def main():
    # Debayering
    channels = debayer(bayer_image)
    name = "debayer"

    image = utils.channels_to_image(channels)
    # debayered_image.show()
    image.save("debayer.png", "PNG")

    # opencv_channels = utils.opencv_equalizer(channels)

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

    channels = histogram_equalizer(channels)

    # after equalizer
    plt.bar(range(256), Image.fromarray(channels[1]).histogram())
    plt.show()

    name += "_equalizer"
    image = utils.channels_to_image(channels)
    image.save(name+".png", "PNG")
    image.show()

    jpeg_codec.image2jpeg(channels, "jpeg_file.json", quality=5)
    de_jpeg = jpeg_codec.jpeg2image("jpeg_file.json")
    name += "_jpeg"
    image = utils.channels_to_image(de_jpeg)
    image.show()

    image.save(name+".png", "PNG")

# snake_tester()
main()

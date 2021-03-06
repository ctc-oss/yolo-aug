from imgaug import augmenters as iaa
from imgaug import parameters as iap


# https://imgaug.readthedocs.io/en/latest/source/examples_basics.html#a-simple-and-common-augmentation-sequence
def default():
    return iaa.Sequential([
        iaa.Fliplr(0.5),  # horizontal flips
        iaa.Crop(percent=(0, 0.1)),  # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
        # Strengthen or weaken the contrast in each image.
        iaa.ContrastNormalization((0.75, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-3, 3),
            shear=(-8, 8)
        )
    ], random_order=True)


# https://imgaug.readthedocs.io/en/latest/source/parameters.html#introduction
def stochastic():
    return iaa.Sequential([
        iaa.GaussianBlur(
            sigma=iap.Uniform(0.0, 1.0)
        ),
        iaa.ContrastNormalization(
            iap.Choice(
                [1.0, 1.5, 3.0],
                p=[0.5, 0.3, 0.2]
            )
        ),
        iaa.Affine(
            rotate=iap.Normal(0.0, 30),
            translate_px=iap.RandomSign(iap.Poisson(3))
        ),
        iaa.AddElementwise(
            iap.Discretize(
                (iap.Beta(0.5, 0.5) * 2 - 1.0) * 64
            )
        ),
        iaa.Multiply(
            iap.Positive(iap.Normal(0.0, 0.1)) + 1.0
        )
    ])


def blur():
    return iaa.Sequential([
        iaa.MotionBlur()
    ])

from multiprocessing import Pool
from typing import Tuple, Optional, List, Union

import imgaug.augmenters as iaa
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.util import random_noise
from itertools import repeat


def augment_batch_mp(
    X: np.ndarray,
    n_workers: int,
    shear_and_stretch: bool = True,
    norm_mode: str = "linear",
) -> np.ndarray:
    """
    Augment a batch of images using multiprocessing.

    Args:
        X: A numpy array of images to be augmented.
        n_workers: The number of worker processes to use.
        shear_and_stretch: If true, apply shear and stretch transformations.
        norm_mode: Normalization mode, can be 'linear', 'squared' or 'log'.

    Returns:
        A numpy array of augmented images.
    """
    resizer = iaa.Resize(X.shape[-2:])
    with Pool(processes=n_workers) as p:
        new_X = p.map(wrapper_augment_single, [(el, shear_and_stretch) for el in X])
    return np.array(new_X)


def wrapper_augment_single(args: Tuple) -> np.ndarray:
    """
    Wrapper function for augment_single to enable multiprocessing.

    Args:
        args: Tuple of arguments for augment_single.

    Returns:
        An augmented image.
    """
    return augment_single(*args)


def augment_single(
    img: np.ndarray, shear_and_stretch: bool, norm_mode: str = "linear"
) -> np.ndarray:
    """
    Apply augmentation to a single image.

    Args:
        img: A numpy array representing the image.
        shear_and_stretch: If true, apply shear and stretch transformations.
        norm_mode: Normalization mode, can be 'linear', 'squared' or 'log'.

    Returns:
        An augmented image.
    """
    resizer = iaa.Resize(img.shape[-2:])
    augmentation_params = sample_simple_augmentation_factors()

    i1 = process(img[0], *augmentation_params, shear_and_stretch=shear_and_stretch)
    i2 = process(img[1], *augmentation_params, shear_and_stretch=shear_and_stretch)

    img = np.array([i1, i2])
    img = random_crop(img)
    img = resizer.augment_image(image=img.swapaxes(-1, 0))
    img = normalise(img, mode=norm_mode).swapaxes(-1, 0)
    return img


def augment_batch(X: np.ndarray, norm_mode: str = "linear") -> np.ndarray:
    """
    Apply augmentation to a batch of images without multiprocessing.

    Args:
        X: A numpy array of images to be augmented.
        norm_mode: Normalization mode, can be 'linear', 'squared' or 'log'.

    Returns:
        A numpy array of augmented images.
    """
    resizer = iaa.Resize(X.shape[-2:])
    new_X = []
    for img in X:
        augmentation_params = sample_simple_augmentation_factors()
        i1 = process(img[0], *augmentation_params)
        i2 = process(img[1], *augmentation_params)
        img = np.array([i1, i2])
        img = random_crop(img)
        img = resizer.augment_image(image=img.swapaxes(-1, 0))
        img = normalise(img, mode=norm_mode)
        new_X.append(img.swapaxes(-1, 0))
    return np.array(new_X)


def adjust_brightness(img: np.ndarray, brightness: float) -> np.ndarray:
    """
    Adjust the brightness of an image.

    Args:
        img: A numpy array representing the image.
        brightness: The amount to adjust the brightness.

    Returns:
        The image with adjusted brightness.
    """
    return np.clip(img + brightness, 0, 1)


def adjust_contrast(img: np.ndarray, contrast: float) -> np.ndarray:
    """
    Adjust the contrast of an image.

    Args:
        img: A numpy array representing the image.
        contrast: The amount to adjust the contrast.

    Returns:
        The image with adjusted contrast.
    """
    mean = np.mean(img)
    img = (img - mean) * contrast + mean
    return np.clip(img, 0, 1)


def random_crop(
    image: np.ndarray, output_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Randomly crop an image.

    Args:
        image: A numpy array representing the image.
        output_size: The desired output size of the image.

    Returns:
        The cropped image.
    """
    h, w = image.shape[-2:]
    if output_size == None:
        new_h, new_w = (
            int(3 * h / 4),
            int(3 * w / 4),
        )
    else:
        new_h, new_w = output_size

    top = np.random.randint(0, h - new_h)
    left = np.random.randint(0, w - new_w)

    image = image[:, top : top + new_h, left : left + new_w]

    return image


def normalise(
    img: np.ndarray,
    maximum: Optional[float] = None,
    minimum: Optional[float] = None,
    mode: str = "linear",
) -> np.ndarray:
    """
    Normalise an image.

    Args:
        img: A numpy array representing the image.
        maximum: The maximum value for normalization.
        minimum: The minimum value for normalization.
        mode: Normalization mode, can be 'linear', 'squared' or 'log'.

    Returns:
        The normalised image.
    """
    img = np.asarray(img)
    if maximum == None:
        maximum = np.max(img)
    if minimum == None:
        minimum = np.min(img)
    if mode == "linear":
        return (img - minimum) / (maximum - minimum)
    elif mode == "squared":
        img_new = (img - minimum) / (maximum - minimum)
        return 1 - (img_new - 1) ** 2
    elif mode == "log":
        gain = 100
        img_new = (img - minimum) / (maximum - minimum)
        return np.log(gain * img_new + 1) / math.log(gain + 1)
    else:
        print("Unkown mode: ", mode)


def sample_simple_augmentation_factors(
    correct_bias: bool = False,
    flip_chance: float = 0.5,
    max_shearing_angle_horizontal: int = 20,
    max_shearing_angle_vertical: int = 20,
    max_stretching_ratio_x: float = 0.2,
    max_stretching_ratio_y: float = 0.2,
    brightness_range: List[float] = [-0.2, 0.2],
    contrast_range: List[float] = [0.8, 1.2],
    max_noise_level: float = 0.003,
    cropping_length: int = 50,
    visualise: bool = False,
) -> Tuple:
    """
    Sample augmentation factors.

    Args:
        correct_bias: If True, apply correction for bias.
        flip_chance: The chance of flipping the image.
        max_shearing_angle_horizontal: The maximum horizontal shearing angle.
        max_shearing_angle_vertical: The maximum vertical shearing angle.
        max_stretching_ratio_x: The maximum stretching ratio along x-axis.
        max_stretching_ratio_y: The maximum stretching ratio along y-axis.
        brightness_range: The range of brightness adjustment.
        contrast_range: The range of contrast adjustment.
        max_noise_level: The maximum noise level.
        cropping_length: The length to crop the image.
        visualise: If True, display the image.

    Returns:
        A tuple of augmentation factors.
    """

    shearing_angle_h = (np.random.uniform() * 2 - 1) * max_shearing_angle_horizontal
    shearing_angle_v = (np.random.uniform() * 2 - 1) * max_shearing_angle_vertical
    stretching_ratio_x = (np.random.uniform() * 2 - 1) * max_stretching_ratio_x
    stretching_ratio_y = (np.random.uniform() * 2 - 1) * max_stretching_ratio_y

    if visualise:
        print(
            "shear h, shear v, stretch x, stretch y",
            shearing_angle_h,
            shearing_angle_v,
            stretching_ratio_x,
            stretching_ratio_y,
        )

    flipped = np.random.uniform() < flip_chance

    brightness = np.random.uniform() * (
        np.max(brightness_range) - np.min(brightness_range)
    ) + np.min(brightness_range)
    contrast = np.random.uniform() * (
        np.max(contrast_range) - np.min(contrast_range)
    ) + np.min(contrast_range)
    noise = np.random.uniform() * np.max(max_noise_level)

    return (
        shearing_angle_h,
        shearing_angle_v,
        stretching_ratio_x,
        stretching_ratio_y,
        flipped,
        brightness,
        contrast,
        noise,
    )


def process(
    img: np.ndarray,
    shearing_angle_h: float,
    shearing_angle_v: float,
    stretching_ratio_x: float,
    stretching_ratio_y: float,
    flipped: bool,
    brightness: float,
    contrast: float,
    noise: float,
    correct_bias: bool = False,
    cropping_side_length: int = 50,
    target_side_length: int = 100,
    shear_and_stretch: bool = True,
    visualise: bool = False,
) -> np.ndarray:
    """
    Process an image according to the given parameters.

    Args:
        img: A numpy array representing the image.
        shearing_angle_h: The horizontal shearing angle.
        shearing_angle_v: The vertical shearing angle.
        stretching_ratio_x: The stretching ratio along x-axis.
        stretching_ratio_y: The stretching ratio along y-axis.
        flipped: If true, flip the image.
        brightness: The brightness adjustment.
        contrast: The contrast adjustment.
        noise: The noise level.
        correct_bias: If True, apply correction for bias.
        cropping_side_length: The side length for cropping.
        target_side_length: The target side length for resizing.
        shear_and_stretch: If true, apply shear and stretch transformations.
        visualise: If True, display the image.

    Returns:
        The processed image.
    """
    if correct_bias:
        img = img.T * -1
        img = img[::-1, ::-1].T

    if visualise:
        plt.imshow(img)
        plt.show()

    if shear_and_stretch:
        img = iaa.geometric.Affine(
            scale={"x": 1 + stretching_ratio_x, "y": 1 + stretching_ratio_y}
        ).augment_image(img)

        if visualise:
            plt.imshow(img)
            plt.show()
        # shear horizontally
        img = iaa.geometric.Affine(shear=shearing_angle_h, order=5).augment_image(img)

        if visualise:
            plt.imshow(img)
            plt.show()
        # shear vertically
        img = np.rot90(img)
        img = iaa.geometric.Affine(shear=shearing_angle_v, order=5).augment_image(img)
        img = np.rot90(img, k=3)

        if visualise:
            plt.imshow(img)
            plt.show()
    # maybe flip
    if flipped:
        img = img[::-1, ::-1].T
    img = iaa.Resize(
        {"height": target_side_length, "width": target_side_length}
    ).augment_image(img)
    if visualise:
        plt.imshow(img)
        plt.show()

    img = adjust_brightness(img, brightness)
    img = adjust_contrast(img, contrast)
    img = random_noise(img, var=noise)

    return img

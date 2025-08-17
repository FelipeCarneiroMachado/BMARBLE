"""
Laplacian of Gaussian pre-processing for color independent representation

Created by Felipe Carneiro Machado
06/08/2025
"""
from tecnica.config import config_dict

import numpy as np
from scipy import ndimage


def laplacianOfGaussian(left, right, config = config_dict):
    """
    Computes the LoG for the images, creating a color independent representation
    :param left: left channel
    :param right: right channel
    :param config: configuration dictionary, accepts default
    :return: tuple(numpy matrix, numpy matrix): LoG processed pair
    """

    sigma = config['sigma']

    size = int(2*(np.ceil(3*sigma))+1)

    # Creates the kernel to convolve with channels
    x, y = np.meshgrid(np.arange(-size / 2 + 1, size / 2 + 1),
                       np.arange(-size / 2 + 1, size / 2 + 1))
    kernel = ((x**2 + y**2 - (2.0*sigma**2)) / sigma**4) * np.exp(-(x**2+y**2) / (2.0*sigma**2))

    # Makes sure the kernel is zero-sum
    kernel = kernel - np.mean(kernel)
    kern_size = kernel.shape[0]


    # applying filter
    log_left = ndimage.convolve(left.astype(np.float64), kernel, mode='constant')
    log_right = ndimage.convolve(right.astype(np.float64), kernel, mode='constant')

    return log_left, log_right



"""
General utilities

Created by Felipe Carneiro Machado
06/08/2025
"""
from tecnica.config import config_dict

import cv2 as cv
import numpy as np



def resize_anaglyph(anaglyph, config = config_dict):
    """
    Resizes anaglyph image to divisible by block size dimensions
    :param anaglyph: input anaglyph image
    :param config: config dictionary, supports default
    :return: resized anaglyph image
    """
    f = lambda x : (config["block_size"] - (x % config["block_size"])
                    if x % config["block_size"] != 0
                    else 0)
    #Resizing the channels
    resized = cv.copyMakeBorder(anaglyph
                                  , 0,
                                  f(anaglyph.shape[0]),
                                  0,
                                  f(anaglyph.shape[1]),
                                  cv.BORDER_CONSTANT,
                                  value=[0, 0, 0])
    return resized


def split_channels(anaglyph : np.ndarray, config : dict = config_dict):
    """
    Splits channels, with color channel spreading to empty channels
    :param anaglyph:
    :param config:
    :return:
    """
    # Follows red-cyan pattern
    l = anaglyph.copy()
    l[:, :, 1] = l[:, :, 0]
    l[:, :, 2] = l[:, :, 0]
    r = anaglyph.copy()
    r[:, :, 0] = r[:, :, 1]
    return l, r



def valid_coordinate(y: int, x: int, dimensions: tuple[int, int, int]):
    """
    Returns true if coordinates are valid (inside image bounds)
    :param y:
    :param x:
    :param dimensions:
    :return:
    """
    if 0 <= x < (dimensions[1]):
        if 0 <= y < (dimensions[0]):
            return True
        else:
            return False
    else:
        return False



def valid_block(y, x, dimensions, config = config_dict):
    """
    Returns true if all coordinates inside block are valid (inside image bounds)
    Block coordinates given by top right corner position
    :param y:
    :param x:
    :param dimensions:
    :param config:
    :return:
    """
    if 0 <= x <= (dimensions[1] - config["block_size"]):
        if 0 <= y <= (dimensions[0] - config["block_size"]):
            return True
        else:
            return False
    else:
        return False
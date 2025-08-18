"""
General utilities

Created by Felipe Carneiro Machado
06/08/2025
"""
from bmarble.config import config_dict

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

# Return the channels to the original dimensions
def return_dimensions(result_left:np.ndarray, result_right:np.ndarray, dimensions:np.shape, config:dict = config_dict):
    l = result_left[:dimensions[0], :dimensions[1]]
    r = result_right[:dimensions[0], :dimensions[1]]
    return l, r

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

def get_anaglyph_channels(anaglyph, avoid_image_show=False, config = config_dict):
    """
    Maps anaglyph to Left and Right channels

    This code was adapted from https://github.com/andreldc/pixradio, all rights reserved.
    Parameters:
    anaglyph (uint8 np.array): Anaglyph image
    right_left_factor (float): Factor for merging Right channel image

    Returns:
    l_channel, r_channel, r1_channel, r2_channel (uint8 cv2 image np.array):
    l_channel is the left channel
    r_channel is the merged right channel
    r1_channel and r2_channel are the right channel

    Requirements:
    """
    blue, green, red = cv.split(anaglyph)

    l_channel = red
    r1_channel = green
    r2_channel = blue
    r_channel = np.rint(r1_channel * 0.95 +
                        r2_channel * (1 - 0.95)
                        ).astype("uint8")

    return (l_channel, r_channel, r1_channel, r2_channel)


def change_range(vector, old_min, old_max, new_min, new_max):
    """
    Scale np.array from range old_min, old_max be within new_min and new_max
    This code was adapted from https://github.com/andreldc/pixradio, all rights reserved.
    """

    if vector.min() != vector.max() and old_max != old_min:
        new_vector = ((vector-old_min)/(old_max-old_min))*(new_max-new_min) + new_min
    else:
        new_vector = vector

    return new_vector

def normalize(vector, new_min=0, new_max=255):
    """
    Scale vector to be within min_val and max_val relative to its own min and max values
    This code was adapted from https://github.com/andreldc/pixradio, all rights reserved.
    """

    old_min = vector.min()
    old_max = vector.max()

    return change_range(vector, old_min, old_max, new_min, new_max)

def convert_to_image(vector):
    """
    Converts vector to image (0-255 uint)
    This code was adapted from https://github.com/andreldc/pixradio, all rights reserved.
    """
    return np.rint(normalize(vector, 0, 255)).astype("uint8")

"""
Functions for refining disparity maps, for better results

This code was adapted from https://github.com/andreldc/pixradio, all rights reserved.
"""
import numpy as np
import cv2 as cv

import bmarble.utils as utils

from bmarble.reciprocity import get_reciprocity


def get_refinement(l_valid_disparity, r_valid_disparity, l_reciprocity, r_reciprocity):
    """
    Refines the initial disparity with a closing morphological operator
    """

    k_size = 35

    # Defines the Kernel used for closing operation - O(1)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k_size, k_size))

    # Perform Closing Operation - O(1)
    l_closed_disparity = cv.morphologyEx(l_valid_disparity.astype("uint8"), cv.MORPH_CLOSE, kernel)
    r_closed_disparity = cv.morphologyEx(r_valid_disparity.astype("uint8"), cv.MORPH_CLOSE, kernel)

    # Substitute 0 with values found on closing operation - O(1)
    l_both_disparity = np.where(l_reciprocity == 0, l_closed_disparity, l_valid_disparity)
    r_both_disparity = np.where(r_reciprocity == 0, r_closed_disparity, r_valid_disparity)

    cv.imwrite("./l_both_disparity.jpg", utils.convert_to_image(l_both_disparity))
    cv.imwrite("./r_both_disparity.jpg", utils.convert_to_image(r_both_disparity))

    # Aggregated Reciprocity Mask - O(N)
    l_both_disparity_valid, r_both_disparity_valid, l_both_reciprocity, r_both_reciprocity = \
        get_reciprocity(l_both_disparity, r_both_disparity, prevent_result_override=True)

    return l_both_disparity_valid, r_both_disparity_valid, l_both_reciprocity, r_both_reciprocity

"""
Block Matching functions for finding correspondences

Created by Felipe Carneiro Machado,
10/08/2025
"""
import numpy as np
from bmarble.config import config_dict
import bmarble.utils as utils



def sad(leftY, leftX, rightY, rightX, log_left, log_right, config=config_dict):
    """
    Computes SAD between two blocks
    :param leftY:
    :param leftX:
    :param rightY:
    :param rightX:
    :param log_left:
    :param log_right:
    :param config:
    :return:
    """
    # Alias to shorten the expression
    with config["block_size"] as bs:
        return np.sum(np.abs(log_left[leftY:leftY+bs, leftX:leftX+bs] - log_right[rightY:rightY+bs, rightX:rightX+bs]))


# noinspection DuplicatedCode
def minimize_sad_l2r(x, y, log_left, log_right, hor_window, config = config_dict):
    """
    Finds the correspondent block through SAD minimization for the left channel
    :param x: original x coordinate
    :param y: original y coordinate
    :param log_left:
    :param log_right:
    :param hor_window: horizontal search window size
    :param config:
    :return: correspondent block coordinates
    """

    # Initializaton
    best_sad = np.inf
    best_coord = (None, None)

    # Limits of the search window, considering the asymmetry and lack of vertical disparity
    xmin = x - hor_window
    xmax = x if config["one_sided_search"] else x + hor_window + 1

    ymin, ymax = (y, y) if config["no_vertical_search"] else (y - config["vertical_window"], y + config["vertical_window"] + 1)

    # Iteration through search window
    for iterX in range(xmin, xmax):
        for iterY in range(ymin, ymax):
            if utils.valid_block(iterY, iterX, log_left.shape):
                current_sad = sad(y, x, iterY, iterX, log_left, log_right)
                if current_sad < best_sad:
                    best_sad = current_sad
                    best_coord = (iterY, iterX)

    return best_coord


# noinspection DuplicatedCode
def minimize_sad_r2l(x, y, log_left, log_right, hor_window, config = config_dict):
    """
    Finds the correspondent block through SAD minimization for the right channel
    :param x: original x coordinate
    :param y: original y coordinate
    :param log_left:
    :param log_right:
    :param hor_window: horizontal search window size
    :param config:
    :return: correspondent block coordinates
    """

    # Initialization
    best_sad = np.inf
    best_coord = (None, None)

    # Limits of the search window, considering the asymmetry and lack of vertical disparity
    xmin = x if config["one_sided_search"] else x - hor_window
    xmax = x + hor_window + 1

    ymin, ymax = (y, y) if config["no_vertical_search"] else (y - config["vertical_window"], y + config["vertical_window"] + 1)

    # Iteration through search window
    for iterX in range(xmin, xmax):
        for iterY in range(ymin, ymax):
            if utils.valid_block(iterY, iterX, log_left.shape):
                current_sad = sad(iterY, iterX, y, x, log_left, log_right)
                if current_sad < best_sad:
                    best_sad = current_sad
                    best_coord = (iterY, iterX)

    return best_coord

# Convenience function to update disparity maps
def update_dmap(org, found, dmap, bs):
    dmap[org[0]:org[0] + bs, org[1]:org[1] + bs] = np.ones((bs, bs)) * found[1] - org[1]

def get_full_correspondences(log_left, log_right, config = config_dict):
    """
    Computes the full disparity map with a large initial window
    :param log_left: Laplacian of Gaussian preprocessed left channel
    :param log_right: Laplacian of Gaussian preprocessed right channel
    :param config: config dictionary, supports default
    :return: (left, right) disparity map
    """

    # Initialization
    dimensions = log_left.shape
    dmap_left = np.zeros_like(log_left)
    dmap_right = np.zeros_like(log_right)



    # Iterating over image blocks
    for y in range(0, dimensions[0], config["block_size"]):
        for x in range(0, dimensions[1], config["block_size"]):
            # Finding correspondences
            left_match = minimize_sad_l2r(x, y, log_left, log_right, config["max_window"])
            right_match = minimize_sad_r2l(x, y, log_left, log_right, config["max_window"])
            # Writing to disparity map
            update_dmap((y, x), left_match, dmap_left, config["block_size"])
            update_dmap((y, x), right_match, dmap_right, config["block_size"])

    return dmap_left, dmap_right

def calculate_window(dmap_left, dmap_right, config = config_dict):
    """
    Calculates the new search window based on the disparity map
    :param dmap_left:
    :param dmap_right:
    :param config:
    :return:
    """
    # Computes the histogram
    values =  np.concatenate([np.abs(dmap_left.flatten()), np.abs(dmap_right.flatten())])
    histogram = np.bincount(values, minlength=config["max_window"]+1)
    histogram = histogram / np.sum(histogram)
    # Iterates backwards over the histogram until the first element above threshold
    for i in range(len(histogram), -1, -1):
        if histogram[i] >= config["dw_threshold"]:
            return round(i * config["dw_extension"]) # Extends the window by a small multiplicative factor


def rematch_invalid_correspondences(dmap_left, dmap_right, log_left, log_right, new_window, config = config_dict):
    """
    Uses the calculated window to get correspondences to the invalid blocks
    :param dmap_left:
    :param dmap_right:
    :param log_left:
    :param log_right:
    :param new_window:
    :param config:
    :return: tuple of updated dmaps
    """
    # Get invalid matches by their top right coordinate
    invalid_left_coords = filter(lambda c: c[0] % config["block_size"] == 0 and c[1] % config["block_size"] == 0,
                                 zip(*np.where(np.abs(dmap_left) > new_window)))
    invalid_right_coords = filter(lambda c: c[0] % config["block_size"] == 0 and c[1] % config["block_size"] == 0,
                                 zip(*np.where(np.abs(dmap_right) > new_window)))

    # Rematch at the invalid coordinates
    for y, x in invalid_left_coords:
        match = minimize_sad_l2r(x, y, log_left, log_right, new_window)
        update_dmap((y, x), match, dmap_left, config["block_size"])
    for y, x in invalid_right_coords:
        match = minimize_sad_r2l(x, y, log_left, log_right, new_window)
        update_dmap((y, x), match, dmap_right, config["block_size"])

    return dmap_left, dmap_right

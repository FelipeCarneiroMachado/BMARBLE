"""
Anaglyphical reversion entry point

Created by Felipe Carneiro Machado
06/08/2025
"""
import cv2

import bmarble.utils as utils
import bmarble.correspondence as correspondence
import bmarble.colorize as colorize

from bmarble.config import config_dict
from preprocessing import laplacianOfGaussian
from bmarble.reciprocity import get_reciprocity
from bmarble.refining import get_refinement


def reverse(anaglyph, config = config_dict):
    """
        Extracts a stereo pair from a red-cyan anaglyph

        The Main entry point for the package, executes the algorithm as described in the associated
        paper

        Args:
            anaglyph (numpy matrix): red-cyan anaglyph
            config (dict): configuration dictionary, accepts default

        Returns:
            tuple[numpy matrix, numpy matrix]: stereo pair

    """

    # Resizes anaglyph to divisible by block size dimensions
    original_dimensions = anaglyph.shape
    anaglyph = utils.resize_anaglyph(anaglyph)
    dimensions = anaglyph.shape

    # Splitting channels, with color channel spreading
    left, right = utils.split_channels(anaglyph)

    # Computes LoG of the channels
    log_left, log_right = laplacianOfGaussian(left, right)

    # First round of Block Matching, with large window
    dmap_left, dmap_right = correspondence.get_full_correspondences(log_left, log_right)

    # Calculates the best window based on disparity maps
    new_window = correspondence.calculate_window(dmap_left, dmap_right)

    # Second round of Block Matching over invalid correspondences
    final_dmap_left, final_dmap_right = correspondence.rematch_invalid_correspondences(dmap_left, dmap_right, log_left, log_right, new_window)

    # Determines valid correspondences through reciprocity
    valid_dmap_left, valid_dmap_right, reciprocity_map_left, reciprocity_map_right = get_reciprocity(final_dmap_left, final_dmap_right)

    # Refines the disparity/reciprocity maps
    refined_valid_dmap_left, refined_valid_dmap_right, refined_reciprocity_map_left, refined_reciprocity_map_right = get_refinement(
        valid_dmap_left, valid_dmap_right,
        reciprocity_map_left, reciprocity_map_right
    )

    # Direct color transfer on valid correspondences
    partial_colorized_left, partial_colorized_right = colorize.recover(
        cv2.cvtColor(anaglyph, cv2.COLOR_RGB2BGR),  # Conversion to BGR is needed for compatibility with adapted code
        refined_valid_dmap_left, refined_valid_dmap_right
    )

    # Colorization on occluded regions
    colorized_left, colorized_right = colorize.colorize(
        cv2.cvtColor(anaglyph, cv2.COLOR_RGB2BGR),
        partial_colorized_left, partial_colorized_right,
        refined_reciprocity_map_left, refined_reciprocity_map_right
    )

    # Returns to RGB (also compatibility related)
    colorized_left = cv2.cvtColor(colorized_left, cv2.COLOR_BGR2RGB)
    colorized_right = cv2.cvtColor(colorized_right, cv2.COLOR_BGR2RGB)

    # Removes padding
    result_left, result_right = utils.return_dimensions(colorized_left, colorized_right, dimensions)

    return result_left, result_right

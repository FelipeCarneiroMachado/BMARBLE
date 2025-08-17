"""
Anaglyphical reversion entry point

Created by Felipe Carneiro Machado
06/08/2025
"""
from tecnica.config import config_dict
import tecnica.utils as utils
from preprocessing import laplacianOfGaussian
import tecnica.correspondence as correspondence


def reverse(anaglyph, config = config_dict):
    """
        Extracts a stereo pair from a red-cyan anaglyph

        The Main entry point for the package, executes the algorithm as described in the associated
        paper

        Args:
            anaglyph (numpy matrix): red-cyan anaglyph
            config (dict): configuration dictionary, accepts default

        Returns:
            tuple(numpy matrix, numpy matrix): stereo pair

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

    final_dmap_left, final_dmap_right = correspondence.rematch_invalid_correspondences(dmap_left, dmap_right, log_left, log_right, new_window)

import copy
import random

import numpy as np
import stimupy

INTENSITY_VALUES = np.array([5, 10, 17, 27, 42, 57, 75, 96, 118, 137, 152, 178, 202])
# INTENSITY_VALUES = np.array([5, 10, 17, 27, 41, 57, 74, 92, 124, 150, 176, 200])


def matching_field(
    variegated_array,
    ppd,
    field_size,
    field_intensity=0.5,
    check_visual_size=(1, 1),
    field_position=None,
):
    """Produce matching field stimulus: patch on variegated checkerboard

    Parameters
    ----------
    variegated_array : numpy.ndarray
        array of intensity values to use for checkerboard, one per check
    ppd : Sequence[Number, Number], Number, or None (default)
        pixels per degree (vertical, horizontal)
    field_size : Sequence[Number, Number], Number, or None (default)
        visual size of matching patch (height, width) in degrees visual angle
    field_intensity : Number
        intensity value of matching patch, by default .5
    check_visual_size : Sequence[Number, Number], Number, or None (default)
        visual size of a single check (height, width) in degrees visual angle, by default (1, 1)
    field_position : Number, Sequence[Number, Number], or None (default)
        position of the patch (relative to checkerboard), in degrees visual angle.
        If None, patch will be placed in center of image.

    Returns
    -------
    dict[str: Any]
        dict with the stimulus (key: "img"),
        mask with integer index for the shape (key: "field_mask"),
        and additional keys containing stimulus parameters
    """
    board_shape = variegated_array.shape

    # Generate checkerboard
    checkerboard = stimupy.checkerboards.checkerboard(
        board_shape=board_shape, check_visual_size=check_visual_size, ppd=ppd
    )

    # Apply variegation
    checkerboard["img"] = stimupy.components.draw_regions(
        checkerboard["checker_mask"], intensities=variegated_array.flatten() / 255.0
    )

    # Overlay matching field
    field = stimupy.components.shapes.rectangle(
        visual_size=checkerboard["visual_size"],
        ppd=ppd,
        rectangle_size=field_size,
        intensity_rectangle=field_intensity,
        rectangle_position=field_position,
    )
    combined = copy.deepcopy(checkerboard)
    combined["field_mask"] = field["rectangle_mask"]
    combined["img"] = np.where(combined["field_mask"], field["img"], checkerboard["img"])
    combined["variegated_array"] = copy.deepcopy(variegated_array)

    return combined


def perturb_array(variegated_array):
    """Randomly flip/rotate a variegated array

    "Randomizes" a variegated array, keeping the variegation intact.

    Parameters
    ----------
    variegated_array : numpy.ndarray
        array to perturb

    Returns
    -------
    numpy.ndarray
        perturbed copy of input array
    """

    perturbed_array = variegated_array.copy()

    # Flip
    if random.choice((True, False)):
        perturbed_array = np.fliplr(perturbed_array)
    if random.choice((True, False)):
        perturbed_array = np.flipud(perturbed_array)

    # Rotate
    perturbed_array = np.rot90(perturbed_array, k=random.choice((1, 2, 3)))

    return perturbed_array

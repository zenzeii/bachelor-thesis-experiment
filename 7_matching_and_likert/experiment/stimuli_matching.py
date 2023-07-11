import copy
from copy import deepcopy
import numpy as np
import stimupy

resolution = {
    "visual_size": (10, 20),
    "ppd": 32,
}
target_size = resolution["visual_size"][1] / 10
intensity_background = 0.3

stimuli_names = ["sbc",
                 "bullseye_low_freq",
                 "bullseye_high_freq",
                 "sbc_separate",
                 "sbc_separate_small",
                 "bullseye_low_separate",
                 "whites",
                 "whites_high_freq",
                 "whites_high_freq_equal_aspect",
                 "whites_narrow",
                 "whites_separate",
                 "strip"
                ]


def stims(stim, intensity_target, target_side):
    stims = {}

    if target_side == "Left":
        left_target, right_target = 1, 0
        intensities = intensity_target, 1.0
        intensity_target_bullseye_left = intensity_target
        intensity_target_bullseye_right = 0
        intensity_strip = intensity_target, 0.0
        checkerboard_target = ((2.0, 2.0), (2.0, 2.0))
        checkerboard_narrow_target = ((1.0, 2.0), (1.0, 2.0))

    elif target_side == "Right":
        left_target, right_target = 0, 1
        intensities = 0.0, intensity_target
        intensity_target_bullseye_left = 1
        intensity_target_bullseye_right = intensity_target
        intensity_strip = 1.0, intensity_target
        checkerboard_target = ((2.0, 7.0), (2.0, 7.0))
        checkerboard_narrow_target = ((1.0, 7.0), (1.0, 7.0))


    radii = np.array([0.5, 1.5, 2.5]) * target_size
    left = stimupy.stimuli.rings.rectangular_generalized(
        ppd=resolution["ppd"],
        visual_size=(resolution["visual_size"][0], resolution["visual_size"][1] / 2),
        radii=radii,
        target_indices=left_target,
        intensity_frames=(1, 1),
        intensity_target=intensity_target
    )
    right = stimupy.stimuli.rings.rectangular_generalized(
        ppd=resolution["ppd"],
        visual_size=(resolution["visual_size"][0], resolution["visual_size"][1] / 2),
        radii=radii,
        target_indices=right_target,
        intensity_frames=(0, 0),
        intensity_target=intensity_target
    )
    stims["sbc"] = stimupy.utils.stack_dicts(left, right, direction="horizontal")

    left = stimupy.stimuli.rings.rectangular_generalized(
        ppd=resolution["ppd"],
        visual_size=(resolution["visual_size"][0], resolution["visual_size"][1] / 2),
        radii=radii,
        target_indices=left_target,
        intensity_frames=(1, 0),
        intensity_target=intensity_target
    )
    right = stimupy.stimuli.rings.rectangular_generalized(
        ppd=resolution["ppd"],
        visual_size=(resolution["visual_size"][0], resolution["visual_size"][1] / 2),
        radii=radii,
        target_indices=right_target,
        intensity_frames=(0, 1),
        intensity_target=intensity_target
    )
    stims["bullseye_low_freq"] = stimupy.utils.stack_dicts(left, right, direction="horizontal")

    #TODO
    left = stimupy.stimuli.rings.rectangular(
        ppd=resolution["ppd"],
        visual_size=(resolution["visual_size"][0], resolution["visual_size"][1] / 2),
        target_indices=0,
        intensity_frames=(1.0, 0.0),
        n_frames=5,
        intensity_target=intensity_target_bullseye_left
    )
    right = stimupy.stimuli.rings.rectangular(
        ppd=resolution["ppd"],
        visual_size=(resolution["visual_size"][0], resolution["visual_size"][1] / 2),
        target_indices=0,
        intensity_frames=(0.0, 1.0),
        n_frames=5,
        intensity_target=intensity_target_bullseye_right
    )
    bullseye_hfe = stimupy.utils.stack_dicts(left, right, direction="horizontal")
    stims["bullseye_high_freq"] = bullseye_hfe

    # Mask separation frame
    separate_mask = np.where(bullseye_hfe["grating_mask"] == 2, 1, 0)
    separate_mask = np.where(bullseye_hfe["grating_mask"] == 3, 1, separate_mask)
    separate_mask = np.where(bullseye_hfe["grating_mask"] == 7, 1, separate_mask)
    separate_mask = np.where(bullseye_hfe["grating_mask"] == 8, 1, separate_mask)
    separate_mask = np.where(bullseye_hfe["target_mask"], 1, separate_mask)

    # Mask inner frame
    frame_mask = np.where(bullseye_hfe["grating_mask"] == 2, 1, 0)
    frame_mask = np.where(bullseye_hfe["grating_mask"] == 7, 1, frame_mask)
    frame_mask = np.where(bullseye_hfe["target_mask"], 1, frame_mask)

    # sbc_separated
    sbc_separate = deepcopy(stims["bullseye_low_freq"])
    sbc_separate["img"] = np.where(separate_mask, sbc_separate["img"], intensity_background)
    stims["sbc_separate"] = sbc_separate

    # sbc_separated_smallest
    sbc_smallest = deepcopy(stims["bullseye_high_freq"])
    sbc_smallest["img"] = np.where(frame_mask, sbc_smallest["img"], intensity_background)
    stims["sbc_separate_small"] = sbc_smallest

    # bullseye_low_separate
    stims["bullseye_low_separate"] = deepcopy(stims["bullseye_high_freq"])
    stims["bullseye_low_separate"]["img"] = np.where(
        separate_mask, stims["bullseye_low_separate"]["img"], intensity_background
    )

    whites = {
        "whites": stimupy.stimuli.whites.white(
            **resolution,
            bar_width=target_size,
            target_indices=(2, -3),
            target_heights=target_size,
            intensity_bars=(0, 1),
            intensity_target=intensities
        ),
        "whites_high_freq": stimupy.stimuli.whites.white(
            **resolution,
            bar_width=target_size / 2,
            target_indices=(4, -5),
            target_heights=target_size,
            intensity_bars=(0, 1),
            intensity_target=intensities
        ),
        "whites_high_freq_equal_aspect": stimupy.stimuli.whites.white(
            **resolution,
            bar_width=target_size / 2,
            target_indices=(4, -5),
            target_heights=target_size / 2,
            intensity_bars=(0, 1),
            intensity_target=intensities
        ),
        "whites_narrow": stimupy.stimuli.whites.white(
            ppd=resolution["ppd"],
            visual_size=(6, resolution["visual_size"][1]),
            bar_width=target_size,
            target_indices=(2, -3),
            target_heights=target_size,
            intensity_bars=(0, 1),
            intensity_target=intensities
        ),
    }
    whites["whites_narrow"] = stimupy.utils.pad_dict_to_visual_size(
        dct=whites["whites_narrow"], **resolution, pad_value=intensity_background
    )

    whites["whites_separate"] = deepcopy(whites["whites_narrow"])
    whites["whites_separate"]["img"] = np.where(
        separate_mask, whites["whites_separate"]["img"], intensity_background
    )

    stims = {**stims, **whites}

    # TODO why flipped?
    stims["strip"] = stimupy.stimuli.whites.white(
        ppd=resolution["ppd"],
        visual_size=(target_size, resolution["visual_size"][1]),
        bar_width=target_size,
        target_indices=(2, 7),
        target_heights=2,
        intensity_bars=(1, 0),
        intensity_target=intensity_strip
    )
    stims["strip"] = stimupy.utils.pad_dict_to_visual_size(
        dct=stims["strip"], **resolution, pad_value=intensity_background
    )

    checkerboards = {
        "checkerboard": stimupy.stimuli.checkerboards.checkerboard(
            **resolution,
            check_visual_size=target_size,
            target_indices=checkerboard_target,
            intensity_checks=(1, 0),
        ),
        "checkerboard_narrow": stimupy.stimuli.checkerboards.checkerboard(
            ppd=resolution["ppd"],
            visual_size=(6, resolution["visual_size"][1]),
            check_visual_size=target_size,
            target_indices=checkerboard_narrow_target,
        ),
    }
    checkerboards["checkerboard_narrow"] = stimupy.utils.pad_dict_to_visual_size(
        dct=checkerboards["checkerboard_narrow"], **resolution, pad_value=intensity_background
    )

    checkerboards["checkerboard_separate"] = deepcopy(checkerboards["checkerboard_narrow"])
    checkerboards["checkerboard_separate"]["img"] = np.where(
        separate_mask, checkerboards["checkerboard_separate"]["img"], intensity_background
    )

    stims = {**stims, **checkerboards}


    return stims[stim]

# TODO Shuffle/randomize trial
# TODO make not buggy
# TODO fix break screen
# TODO fix deprication from library C:\Users\Hafidz\StudioProjects\bachelor-thesis-experiment\venv\lib\site-packages\hrl\graphics\graphics.py:160: DeprecationWarning: tostring() is deprecated. Use tobytes() instead.
#   byts = channelsToInt(self.greyToChannels(grys[::-1,])).tostring()

# %% MATCHING FIELD
def matching_field(intensity_match):
    # Generate checkerboard
    checkerboard = stimupy.checkerboards.checkerboard(
        board_shape=(5, 5), check_visual_size=(0.5, 0.5), ppd=resolution["ppd"]
    )

    # TODO: apply variegation

    # Overlay matching field
    field = stimupy.components.shapes.rectangle(
        visual_size=checkerboard["visual_size"],
        ppd=resolution["ppd"],
        rectangle_size=(1, 1),
        intensity_rectangle=intensity_match,
    )
    combined = copy.deepcopy(checkerboard)
    combined["field_mask"] = field["rectangle_mask"]
    combined["img"] = np.where(combined["field_mask"], field["img"], checkerboard["img"])
    return combined

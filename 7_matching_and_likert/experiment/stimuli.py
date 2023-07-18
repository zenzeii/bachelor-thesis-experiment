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
intensity_target = 0.5
radii = np.array([0.5, 1.5, 2.5]) * target_size

__all__ = [
    "sbc",
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
     "strip",
     "checkerboard",
     "checkerboard_narrow",
     "checkerboard_separate",
]

stimuli_names = [
     "sbc",
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
     "strip",
     "checkerboard",
     "checkerboard_narrow",
     "checkerboard_separate",
]

catch_trials = [
    "catch_trial_1",
    "catch_trial_2",
    "catch_trial_3",
    "catch_trial_4",
    "catch_trial_5"
]

# TODO add catch trial, add text display explaining task, correct keyboard when reaching maximum

def check_target_side(target_side):
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

    elif target_side == "Both":
        left_target, right_target = 1, 1
        intensities = intensity_target, intensity_target
        intensity_target_bullseye_left = intensity_target
        intensity_target_bullseye_right = intensity_target
        intensity_strip = intensity_target, intensity_target
        checkerboard_target = ((2.0, 2.0), (2.0, 7.0))
        checkerboard_narrow_target = ((1.0, 2.0), (1.0, 7.0))

    return left_target, right_target, intensities, intensity_target_bullseye_left, intensity_target_bullseye_right, intensity_strip, intensity_target, checkerboard_target, checkerboard_narrow_target

def sbc(target_side):
    left_target, right_target, intensities, intensity_target_bullseye_left, intensity_target_bullseye_right, intensity_strip, intensity_target, checkerboard_target, checkerboard_narrow_target = check_target_side(target_side)

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
    return stimupy.utils.stack_dicts(left, right, direction="horizontal")

def bullseye_low_freq(target_side):
    left_target, right_target, intensities, intensity_target_bullseye_left, intensity_target_bullseye_right, intensity_strip, intensity_target, checkerboard_target, checkerboard_narrow_target = check_target_side(target_side)


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
    return stimupy.utils.stack_dicts(left, right, direction="horizontal")

def bullseye_high_freq(target_side):
    left_target, right_target, intensities, intensity_target_bullseye_left, intensity_target_bullseye_right, intensity_strip, intensity_target, checkerboard_target, checkerboard_narrow_target = check_target_side(target_side)

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
    return stimupy.utils.stack_dicts(left, right, direction="horizontal")

def sbc_separate(target_side):
    bullseye_hfe = bullseye_high_freq(target_side)

    # Mask separation frame
    separate_mask = np.where(bullseye_hfe["grating_mask"] == 2, 1, 0)
    separate_mask = np.where(bullseye_hfe["grating_mask"] == 3, 1, separate_mask)
    separate_mask = np.where(bullseye_hfe["grating_mask"] == 7, 1, separate_mask)
    separate_mask = np.where(bullseye_hfe["grating_mask"] == 8, 1, separate_mask)
    separate_mask = np.where(bullseye_hfe["target_mask"], 1, separate_mask)

    # sbc_separated
    bullseye_lfe = bullseye_low_freq(target_side)
    sbc_separate = deepcopy(bullseye_lfe)
    sbc_separate["img"] = np.where(separate_mask, sbc_separate["img"], intensity_background)
    return sbc_separate

def sbc_separate_small(target_side):
    bullseye_hfe = bullseye_high_freq(target_side)

    # Mask inner frame
    frame_mask = np.where(bullseye_hfe["grating_mask"] == 2, 1, 0)
    frame_mask = np.where(bullseye_hfe["grating_mask"] == 7, 1, frame_mask)
    frame_mask = np.where(bullseye_hfe["target_mask"], 1, frame_mask)

    sbc_smallest = deepcopy(bullseye_hfe)
    sbc_smallest["img"] = np.where(frame_mask, sbc_smallest["img"], intensity_background)
    return sbc_smallest

def bullseye_low_separate(target_side):
    bullseye_hfe = bullseye_high_freq(target_side)

    # Mask separation frame
    separate_mask = np.where(bullseye_hfe["grating_mask"] == 2, 1, 0)
    separate_mask = np.where(bullseye_hfe["grating_mask"] == 3, 1, separate_mask)
    separate_mask = np.where(bullseye_hfe["grating_mask"] == 7, 1, separate_mask)
    separate_mask = np.where(bullseye_hfe["grating_mask"] == 8, 1, separate_mask)
    separate_mask = np.where(bullseye_hfe["target_mask"], 1, separate_mask)

    bullseye_ls = deepcopy(bullseye_hfe)
    bullseye_ls["img"] = np.where(
        separate_mask, bullseye_ls["img"], intensity_background
    )
    return bullseye_ls

def whites(target_side):
    left_target, right_target, intensities, intensity_target_bullseye_left, intensity_target_bullseye_right, intensity_strip, intensity_target, checkerboard_target, checkerboard_narrow_target = check_target_side(
        target_side)
    return stimupy.stimuli.whites.white(
            **resolution,
            bar_width=target_size,
            target_indices=(2, -3),
            target_heights=target_size,
            intensity_bars=(0, 1),
            intensity_target=intensities
        )
def whites_high_freq(target_side):
    left_target, right_target, intensities, intensity_target_bullseye_left, intensity_target_bullseye_right, intensity_strip, intensity_target, checkerboard_target, checkerboard_narrow_target = check_target_side(
        target_side)
    return stimupy.stimuli.whites.white(
            **resolution,
            bar_width=target_size / 2,
            target_indices=(4, -5),
            target_heights=target_size,
            intensity_bars=(0, 1),
            intensity_target=intensities
        )

def whites_high_freq_equal_aspect(target_side):
    left_target, right_target, intensities, intensity_target_bullseye_left, intensity_target_bullseye_right, intensity_strip, intensity_target, checkerboard_target, checkerboard_narrow_target = check_target_side(
        target_side)
    return stimupy.stimuli.whites.white(
            **resolution,
            bar_width=target_size / 2,
            target_indices=(4, -5),
            target_heights=target_size,
            intensity_bars=(0, 1),
            intensity_target=intensities
        )
def whites_narrow(target_side):
    left_target, right_target, intensities, intensity_target_bullseye_left, intensity_target_bullseye_right, intensity_strip, intensity_target, checkerboard_target, checkerboard_narrow_target = check_target_side(
        target_side)
    return stimupy.stimuli.whites.white(
            ppd=resolution["ppd"],
            visual_size=(6, resolution["visual_size"][1]),
            bar_width=target_size,
            target_indices=(2, -3),
            target_heights=target_size,
            intensity_bars=(0, 1),
            intensity_target=intensities
        )

def whites_separate(target_side):
    bullseye_hfe = bullseye_high_freq(target_side)

    # Mask separation frame
    separate_mask = np.where(bullseye_hfe["grating_mask"] == 2, 1, 0)
    separate_mask = np.where(bullseye_hfe["grating_mask"] == 3, 1, separate_mask)
    separate_mask = np.where(bullseye_hfe["grating_mask"] == 7, 1, separate_mask)
    separate_mask = np.where(bullseye_hfe["grating_mask"] == 8, 1, separate_mask)
    separate_mask = np.where(bullseye_hfe["target_mask"], 1, separate_mask)

    whites_s = deepcopy(whites_narrow(target_side))
    whites_s["img"] = np.where(
        separate_mask, whites_s["img"], intensity_background
    )

    return whites_s

def strip(target_side):
    left_target, right_target, intensities, intensity_target_bullseye_left, intensity_target_bullseye_right, intensity_strip, intensity_target, checkerboard_target, checkerboard_narrow_target = check_target_side(
        target_side)

    strip_stim = stimupy.stimuli.whites.white(
        ppd=resolution["ppd"],
        visual_size=(target_size, resolution["visual_size"][1]),
        bar_width=target_size,
        target_indices=(2, 7),
        target_heights=2,
        intensity_bars=(1, 0),
        intensity_target=intensity_strip
    ),
    strip_stim = stimupy.utils.pad_dict_to_visual_size(
        dct=strip_stim, **resolution, pad_value=intensity_background
    )

    return strip_stim


def checkerboard(target_side):
    left_target, right_target, intensities, intensity_target_bullseye_left, intensity_target_bullseye_right, intensity_strip, intensity_target, checkerboard_target, checkerboard_narrow_target = check_target_side(
        target_side)

    return stimupy.stimuli.checkerboards.checkerboard(
        **resolution,
        check_visual_size=target_size,
        target_indices=checkerboard_target,
        intensity_checks=(1, 0),
    )

def checkerboard_narrow(target_side):
    left_target, right_target, intensities, intensity_target_bullseye_left, intensity_target_bullseye_right, intensity_strip, intensity_target, checkerboard_target, checkerboard_narrow_target = check_target_side(
        target_side)


    checkerboard_narrow = stimupy.stimuli.checkerboards.checkerboard(
        ppd=resolution["ppd"],
        visual_size=(6, resolution["visual_size"][1]),
        check_visual_size=target_size,
        target_indices=checkerboard_narrow_target,
    )

    checkerboard_narrow = stimupy.utils.pad_dict_to_visual_size(
        dct=checkerboard_narrow, **resolution, pad_value=intensity_background
    )

    return checkerboard_narrow

def checkerboard_separate(target_side):
    left_target, right_target, intensities, intensity_target_bullseye_left, intensity_target_bullseye_right, intensity_strip, intensity_target, checkerboard_target, checkerboard_narrow_target = check_target_side(
        target_side)

    bullseye_hfe = bullseye_high_freq(target_side)

    # Mask separation frame
    separate_mask = np.where(bullseye_hfe["grating_mask"] == 2, 1, 0)
    separate_mask = np.where(bullseye_hfe["grating_mask"] == 3, 1, separate_mask)
    separate_mask = np.where(bullseye_hfe["grating_mask"] == 7, 1, separate_mask)
    separate_mask = np.where(bullseye_hfe["grating_mask"] == 8, 1, separate_mask)
    separate_mask = np.where(bullseye_hfe["target_mask"], 1, separate_mask)

    checkerboard_separate = deepcopy(checkerboard_narrow(target_side))
    checkerboard_separate["img"] = np.where(
        separate_mask, checkerboard_separate["img"], intensity_background
    )

    return checkerboard_separate


def stims(stim, target_side):
    if stim == "sbc":
        return sbc(target_side)
    if stim == "bullseye_low_freq":
        return bullseye_low_freq(target_side)
    if stim == "bullseye_high_freq":
        return bullseye_high_freq(target_side)
    if stim == "sbc_separate":
        return sbc_separate(target_side)
    if stim == "sbc_separate_small":
        return sbc_separate_small(target_side)
    if stim == "bullseye_low_separate":
        return bullseye_low_separate(target_side)
    if stim == "whites":
        return whites(target_side)
    if stim == "whites_high_freq":
        return whites_high_freq(target_side)
    if stim == "whites_high_freq_equal_aspect":
        return whites_high_freq_equal_aspect(target_side)
    if stim == "whites_narrow":
        return whites_narrow(target_side)
    if stim == "whites_separate":
        return whites_separate(target_side)
    if stim == "strip":
        return strip(target_side)
    if stim == "checkerboard":
        return checkerboard(target_side)
    if stim == "checkerboard_narrow":
        return checkerboard_narrow(target_side)
    if stim == "checkerboard_separate":
        return checkerboard_separate(target_side)
    if stim == "catch_trial_1":
        return catch_trial_1()
    if stim == "catch_trial_2":
        return catch_trial_2()
    if stim == "catch_trial_3":
        return catch_trial_3()
    if stim == "catch_trial_4":
        return catch_trial_4()
    if stim == "catch_trial_5":
        return catch_trial_5()


def stimss(stim, target_side):
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

    elif target_side == "Both":
        left_target, right_target = 1, 1
        intensities = intensity_target, intensity_target
        intensity_target_bullseye_left = intensity_target
        intensity_target_bullseye_right = intensity_target
        intensity_strip = intensity_target, intensity_target
        checkerboard_target = ((2.0, 2.0), (2.0, 7.0))
        checkerboard_narrow_target = ((1.0, 2.0), (1.0, 7.0))


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


    if stim == "catch_trial_1": return catch_trial_1()
    if stim == "catch_trial_2": return catch_trial_2()
    if stim == "catch_trial_3": return catch_trial_3()
    if stim == "catch_trial_4": return catch_trial_4()
    if stim == "catch_trial_5": return catch_trial_5()


    return stims[stim]

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


def catch_trial_1():
    catch_trial_intensity_target_left = 0.3
    catch_trial_intensity_target_right = 0.1
    left = stimupy.stimuli.rings.rectangular_generalized(
        ppd=resolution["ppd"],
        visual_size=(resolution["visual_size"][0], resolution["visual_size"][1] / 2),
        radii=radii,
        target_indices=1,
        intensity_frames=(0, 0),
        intensity_target=catch_trial_intensity_target_left
    )
    right = stimupy.stimuli.rings.rectangular_generalized(
        ppd=resolution["ppd"],
        visual_size=(resolution["visual_size"][0], resolution["visual_size"][1] / 2),
        radii=radii,
        target_indices=1,
        intensity_frames=(0, 0),
        intensity_target=catch_trial_intensity_target_right
    )
    return stimupy.utils.stack_dicts(left, right, direction="horizontal")

def catch_trial_2():
    catch_trial_intensity_target_left = 0.5
    catch_trial_intensity_target_right = 0.4
    left = stimupy.stimuli.rings.rectangular_generalized(
        ppd=resolution["ppd"],
        visual_size=(resolution["visual_size"][0], resolution["visual_size"][1] / 2),
        radii=radii,
        target_indices=1,
        intensity_frames=(0, 0),
        intensity_target=catch_trial_intensity_target_left
    )
    right = stimupy.stimuli.rings.rectangular_generalized(
        ppd=resolution["ppd"],
        visual_size=(resolution["visual_size"][0], resolution["visual_size"][1] / 2),
        radii=radii,
        target_indices=1,
        intensity_frames=(0, 0),
        intensity_target=catch_trial_intensity_target_right
    )
    return stimupy.utils.stack_dicts(left, right, direction="horizontal")

def catch_trial_3():
    catch_trial_intensity_target_left = 0.2
    catch_trial_intensity_target_right = 0.2
    left = stimupy.stimuli.rings.rectangular_generalized(
        ppd=resolution["ppd"],
        visual_size=(resolution["visual_size"][0], resolution["visual_size"][1] / 2),
        radii=radii,
        target_indices=1,
        intensity_frames=(0, 0),
        intensity_target=catch_trial_intensity_target_left
    )
    right = stimupy.stimuli.rings.rectangular_generalized(
        ppd=resolution["ppd"],
        visual_size=(resolution["visual_size"][0], resolution["visual_size"][1] / 2),
        radii=radii,
        target_indices=1,
        intensity_frames=(0, 0),
        intensity_target=catch_trial_intensity_target_right
    )
    return stimupy.utils.stack_dicts(left, right, direction="horizontal")

def catch_trial_4():
    catch_trial_intensity_target_left = 0.4
    catch_trial_intensity_target_right = 0.5
    left = stimupy.stimuli.rings.rectangular_generalized(
        ppd=resolution["ppd"],
        visual_size=(resolution["visual_size"][0], resolution["visual_size"][1] / 2),
        radii=radii,
        target_indices=1,
        intensity_frames=(0, 0),
        intensity_target=catch_trial_intensity_target_left
    )
    right = stimupy.stimuli.rings.rectangular_generalized(
        ppd=resolution["ppd"],
        visual_size=(resolution["visual_size"][0], resolution["visual_size"][1] / 2),
        radii=radii,
        target_indices=1,
        intensity_frames=(0, 0),
        intensity_target=catch_trial_intensity_target_right
    )
    return stimupy.utils.stack_dicts(left, right, direction="horizontal")

def catch_trial_5():
    catch_trial_intensity_target_left = 0.1
    catch_trial_intensity_target_right = 0.3
    left = stimupy.stimuli.rings.rectangular_generalized(
        ppd=resolution["ppd"],
        visual_size=(resolution["visual_size"][0], resolution["visual_size"][1] / 2),
        radii=radii,
        target_indices=1,
        intensity_frames=(0, 0),
        intensity_target=catch_trial_intensity_target_left
    )
    right = stimupy.stimuli.rings.rectangular_generalized(
        ppd=resolution["ppd"],
        visual_size=(resolution["visual_size"][0], resolution["visual_size"][1] / 2),
        radii=radii,
        target_indices=1,
        intensity_frames=(0, 0),
        intensity_target=catch_trial_intensity_target_right
    )
    return stimupy.utils.stack_dicts(left, right, direction="horizontal")

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


def sbc(target_side):
    if target_side == "Left":
        left_target, right_target = 1, 0
    elif target_side == "Right":
        left_target, right_target = 0, 1
    elif target_side == "Both":
        left_target, right_target = 1, 1

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
    if target_side == "Left":
        left_target, right_target = 1, 0
    elif target_side == "Right":
        left_target, right_target = 0, 1
    elif target_side == "Both":
        left_target, right_target = 1, 1

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
    if target_side == "Left":
        intensity_target_bullseye_left = intensity_target
        intensity_target_bullseye_right = 0
    elif target_side == "Right":
        intensity_target_bullseye_left = 1
        intensity_target_bullseye_right = intensity_target
    elif target_side == "Both":
        intensity_target_bullseye_left = intensity_target
        intensity_target_bullseye_right = intensity_target

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
    if target_side == "Left":
        intensities = intensity_target, 1.0
    elif target_side == "Right":
        intensities = 0.0, intensity_target
    elif target_side == "Both":
        intensities = intensity_target, intensity_target

    return stimupy.stimuli.whites.white(
        **resolution,
        bar_width=target_size,
        target_indices=(2, -3),
        target_heights=target_size,
        intensity_bars=(0, 1),
        intensity_target=intensities
    )


def whites_high_freq(target_side):
    if target_side == "Left":
        intensities = intensity_target, 1.0
    elif target_side == "Right":
        intensities = 0.0, intensity_target
    elif target_side == "Both":
        intensities = intensity_target, intensity_target

    return stimupy.stimuli.whites.white(
        **resolution,
        bar_width=target_size / 2,
        target_indices=(4, -5),
        target_heights=target_size,
        intensity_bars=(0, 1),
        intensity_target=intensities
    )


def whites_high_freq_equal_aspect(target_side):
    if target_side == "Left":
        intensities = intensity_target, 1.0
    elif target_side == "Right":
        intensities = 0.0, intensity_target
    elif target_side == "Both":
        intensities = intensity_target, intensity_target

    return stimupy.stimuli.whites.white(
        **resolution,
        bar_width=target_size / 2,
        target_indices=(4, -5),
        target_heights=target_size,
        intensity_bars=(0, 1),
        intensity_target=intensities
    )


def whites_narrow(target_side):
    if target_side == "Left":
        intensities = intensity_target, 1.0
    elif target_side == "Right":
        intensities = 0.0, intensity_target
    elif target_side == "Both":
        intensities = intensity_target, intensity_target

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

    whites_s = deepcopy(whites(target_side))
    whites_s["img"] = np.where(separate_mask, whites_s["img"], intensity_background)

    return whites_s


def strip(target_side):
    if target_side == "Left":
        intensity_strip = intensity_target, 0.0
    elif target_side == "Right":
        intensity_strip = 1.0, intensity_target
    elif target_side == "Both":
        intensity_strip = intensity_target, intensity_target

    strip_stim = stimupy.stimuli.whites.white(
        ppd=resolution["ppd"],
        visual_size=(target_size, resolution["visual_size"][1]),
        bar_width=target_size,
        target_indices=(2, 7),
        target_heights=2,
        intensity_bars=(1, 0),
        intensity_target=intensity_strip,
    )
    strip_stim = stimupy.utils.pad_dict_to_visual_size(
        dct=strip_stim, **resolution, pad_value=intensity_background
    )

    return strip_stim


def checkerboard(target_side):
    if target_side == "Left":
        checkerboard_target = ((2.0, 2.0), (2.0, 2.0))
    elif target_side == "Right":
        checkerboard_target = ((2.0, 7.0), (2.0, 7.0))
    elif target_side == "Both":
        checkerboard_target = ((2.0, 2.0), (2.0, 7.0))

    return stimupy.stimuli.checkerboards.checkerboard(
        **resolution,
        check_visual_size=target_size,
        target_indices=checkerboard_target,
        intensity_checks=(1, 0),
    )


def checkerboard_narrow(target_side):
    if target_side == "Left":
        checkerboard_narrow_target = ((1.0, 2.0), (1.0, 2.0))
    elif target_side == "Right":
        checkerboard_narrow_target = ((1.0, 7.0), (1.0, 7.0))
    elif target_side == "Both":
        checkerboard_narrow_target = ((1.0, 2.0), (1.0, 7.0))

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
        return catch_trial(1)
    if stim == "catch_trial_2":
        return catch_trial(2)
    if stim == "catch_trial_3":
        return catch_trial(3)
    if stim == "catch_trial_4":
        return catch_trial(4)
    if stim == "catch_trial_5":
        return catch_trial(5)


def catch_trial(version):
    if version == 1:
        catch_trial_intensity_target_left = 0.3
        catch_trial_intensity_target_right = 0.1
    if version == 2:
        catch_trial_intensity_target_left = 0.5
        catch_trial_intensity_target_right = 0.4
    if version == 3:
        catch_trial_intensity_target_left = 0.2
        catch_trial_intensity_target_right = 0.2
    if version == 4:
        catch_trial_intensity_target_left = 0.4
        catch_trial_intensity_target_right = 0.5
    if version == 5:
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

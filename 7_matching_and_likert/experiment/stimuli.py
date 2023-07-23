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


def sbc(target_side, presented_intensity):
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
        intensity_target=presented_intensity
    )
    right = stimupy.stimuli.rings.rectangular_generalized(
        ppd=resolution["ppd"],
        visual_size=(resolution["visual_size"][0], resolution["visual_size"][1] / 2),
        radii=radii,
        target_indices=right_target,
        intensity_frames=(0, 0),
        intensity_target=presented_intensity
    )
    return stimupy.utils.stack_dicts(left, right, direction="horizontal")


def bullseye_low_freq(target_side, presented_intensity):
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
        intensity_target=presented_intensity
    )
    right = stimupy.stimuli.rings.rectangular_generalized(
        ppd=resolution["ppd"],
        visual_size=(resolution["visual_size"][0], resolution["visual_size"][1] / 2),
        radii=radii,
        target_indices=right_target,
        intensity_frames=(0, 1),
        intensity_target=presented_intensity
    )
    return stimupy.utils.stack_dicts(left, right, direction="horizontal")


def bullseye_high_freq(target_side, presented_intensity):
    if target_side == "Left":
        intensity_target_bullseye_left = presented_intensity
        intensity_target_bullseye_right = 0
    elif target_side == "Right":
        intensity_target_bullseye_left = 1
        intensity_target_bullseye_right = presented_intensity
    elif target_side == "Both":
        intensity_target_bullseye_left = presented_intensity
        intensity_target_bullseye_right = presented_intensity

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


def sbc_separate(target_side, presented_intensity):
    bullseye_hfe = bullseye_high_freq(target_side, presented_intensity)

    # Mask separation frame
    separate_mask = np.where(bullseye_hfe["grating_mask"] == 2, 1, 0)
    separate_mask = np.where(bullseye_hfe["grating_mask"] == 3, 1, separate_mask)
    separate_mask = np.where(bullseye_hfe["grating_mask"] == 7, 1, separate_mask)
    separate_mask = np.where(bullseye_hfe["grating_mask"] == 8, 1, separate_mask)
    separate_mask = np.where(bullseye_hfe["target_mask"], 1, separate_mask)

    # sbc_separated
    bullseye_lfe = bullseye_low_freq(target_side, presented_intensity)
    sbc_separate = deepcopy(bullseye_lfe)
    sbc_separate["img"] = np.where(separate_mask, sbc_separate["img"], intensity_background)
    return sbc_separate


def sbc_separate_small(target_side, presented_intensity):
    bullseye_hfe = bullseye_high_freq(target_side, presented_intensity)

    # Mask inner frame
    frame_mask = np.where(bullseye_hfe["grating_mask"] == 2, 1, 0)
    frame_mask = np.where(bullseye_hfe["grating_mask"] == 7, 1, frame_mask)
    frame_mask = np.where(bullseye_hfe["target_mask"], 1, frame_mask)

    sbc_smallest = deepcopy(bullseye_hfe)
    sbc_smallest["img"] = np.where(frame_mask, sbc_smallest["img"], intensity_background)
    return sbc_smallest


def bullseye_low_separate(target_side, presented_intensity):
    bullseye_hfe = bullseye_high_freq(target_side, presented_intensity)

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


def whites(target_side, presented_intensity):
    if target_side == "Left":
        intensities = presented_intensity, 1.0
    elif target_side == "Right":
        intensities = 0.0, presented_intensity
    elif target_side == "Both":
        intensities = presented_intensity, presented_intensity

    return stimupy.stimuli.whites.white(
        **resolution,
        bar_width=target_size,
        target_indices=(2, -3),
        target_heights=target_size,
        intensity_bars=(0, 1),
        intensity_target=intensities
    )


def whites_high_freq(target_side, presented_intensity):
    if target_side == "Left":
        intensities = presented_intensity, 1.0
    elif target_side == "Right":
        intensities = 0.0, presented_intensity
    elif target_side == "Both":
        intensities = presented_intensity, presented_intensity

    return stimupy.stimuli.whites.white(
        **resolution,
        bar_width=target_size / 2,
        target_indices=(4, -5),
        target_heights=target_size,
        intensity_bars=(0, 1),
        intensity_target=intensities
    )


def whites_high_freq_equal_aspect(target_side, presented_intensity):
    if target_side == "Left":
        intensities = presented_intensity, 1.0
    elif target_side == "Right":
        intensities = 0.0, presented_intensity
    elif target_side == "Both":
        intensities = presented_intensity, presented_intensity

    return stimupy.stimuli.whites.white(
        **resolution,
        bar_width=target_size / 2,
        target_indices=(4, -5),
        target_heights=target_size,
        intensity_bars=(0, 1),
        intensity_target=intensities
    )


def whites_narrow(target_side, presented_intensity):
    if target_side == "Left":
        intensities = presented_intensity, 1.0
    elif target_side == "Right":
        intensities = 0.0, presented_intensity
    elif target_side == "Both":
        intensities = presented_intensity, presented_intensity

    return stimupy.stimuli.whites.white(
        ppd=resolution["ppd"],
        visual_size=(6, resolution["visual_size"][1]),
        bar_width=target_size,
        target_indices=(2, -3),
        target_heights=target_size,
        intensity_bars=(0, 1),
        intensity_target=intensities
    )


def whites_separate(target_side, presented_intensity):
    bullseye_hfe = bullseye_high_freq(target_side, presented_intensity)

    # Mask separation frame
    separate_mask = np.where(bullseye_hfe["grating_mask"] == 2, 1, 0)
    separate_mask = np.where(bullseye_hfe["grating_mask"] == 3, 1, separate_mask)
    separate_mask = np.where(bullseye_hfe["grating_mask"] == 7, 1, separate_mask)
    separate_mask = np.where(bullseye_hfe["grating_mask"] == 8, 1, separate_mask)
    separate_mask = np.where(bullseye_hfe["target_mask"], 1, separate_mask)

    whites_s = deepcopy(whites(target_side, presented_intensity))
    whites_s["img"] = np.where(separate_mask, whites_s["img"], intensity_background)

    return whites_s


def strip(target_side, presented_intensity):
    if target_side == "Left":
        intensity_strip = presented_intensity, 0.0
    elif target_side == "Right":
        intensity_strip = 1.0, presented_intensity
    elif target_side == "Both":
        intensity_strip = presented_intensity, presented_intensity

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


def checkerboard(target_side, presented_intensity):
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


def checkerboard_narrow(target_side, presented_intensity):
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


def checkerboard_separate(target_side, presented_intensity):
    bullseye_hfe = bullseye_high_freq(target_side, presented_intensity)

    # Mask separation frame
    separate_mask = np.where(bullseye_hfe["grating_mask"] == 2, 1, 0)
    separate_mask = np.where(bullseye_hfe["grating_mask"] == 3, 1, separate_mask)
    separate_mask = np.where(bullseye_hfe["grating_mask"] == 7, 1, separate_mask)
    separate_mask = np.where(bullseye_hfe["grating_mask"] == 8, 1, separate_mask)
    separate_mask = np.where(bullseye_hfe["target_mask"], 1, separate_mask)

    checkerboard_separate = deepcopy(checkerboard_narrow(target_side, presented_intensity))
    checkerboard_separate["img"] = np.where(
        separate_mask, checkerboard_separate["img"], intensity_background
    )

    return checkerboard_separate


def stims(stim, target_side, flipped, presented_intensity):

    if stim == "sbc":
        stimulus = sbc(target_side, presented_intensity)
    elif stim == "bullseye_low_freq":
        stimulus = bullseye_low_freq(target_side, presented_intensity)
    elif stim == "bullseye_high_freq":
        stimulus = bullseye_high_freq(target_side, presented_intensity)
    elif stim == "sbc_separate":
        stimulus = sbc_separate(target_side, presented_intensity)
    elif stim == "sbc_separate_small":
        stimulus = sbc_separate_small(target_side, presented_intensity)
    elif stim == "bullseye_low_separate":
        stimulus = bullseye_low_separate(target_side, presented_intensity)
    elif stim == "whites":
        stimulus = whites(target_side, presented_intensity)
    elif stim == "whites_high_freq":
        stimulus = whites_high_freq(target_side, presented_intensity)
    elif stim == "whites_high_freq_equal_aspect":
        stimulus = whites_high_freq_equal_aspect(target_side, presented_intensity)
    elif stim == "whites_narrow":
        stimulus = whites_narrow(target_side, presented_intensity)
    elif stim == "whites_separate":
        stimulus = whites_separate(target_side, presented_intensity)
    elif stim == "strip":
        stimulus = strip(target_side, presented_intensity)
    elif stim == "checkerboard":
        stimulus = checkerboard(target_side, presented_intensity)
    elif stim == "checkerboard_narrow":
        stimulus = checkerboard_narrow(target_side, presented_intensity)
    elif stim == "checkerboard_separate":
        stimulus = checkerboard_separate(target_side, presented_intensity)
    elif stim == "catch_trial_1":
        stimulus = catch_trial(1)
    elif stim == "catch_trial_2":
        stimulus = catch_trial(2)
    elif stim == "catch_trial_3":
        stimulus = catch_trial(3)
    elif stim == "catch_trial_4":
        stimulus = catch_trial(4)
    elif stim == "catch_trial_5":
        stimulus = catch_trial(5)
    else:
        raise Exception("stim not found")

    if flipped:
        # TODO: unflip the responses of participants while evaluating
        return stimupy.utils.flip_dict(stimulus)
    else:
        return stimulus


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

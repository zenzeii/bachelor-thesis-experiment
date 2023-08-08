from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import stimupy

resolution = {
    "visual_size": (10, 20),
    "ppd": 32,
}
target_size = resolution["visual_size"][1] / 10
radii = np.array([0.5, 1.5, 2.5]) * target_size

INTENSITIES = {
    "black": 0.0,
    "white": 1.0,
}

__all__ = [
    "sbc",
    #"bullseye_low_freq",
    "bullseye_high_freq",
    "sbc_separate",
    #"sbc_separate_small",
    "bullseye_low_separate",
    "whites",
    #"whites_high_freq",
    #"whites_high_freq_equal_aspect",
    #"whites_narrow",
    "whites_separate",
    "strip",
    "checkerboard",
    #"checkerboard_narrow",
    "checkerboard_separate",
]

def sbc(target_side, intensity_target):
    if target_side == "Left":
        target_indices = ((1,), ())
    elif target_side == "Right":
        target_indices = ((), (1,))
    elif target_side == "Both":
        target_indices = ((1,), (1,))

    left = stimupy.stimuli.rings.rectangular_generalized(
        ppd=resolution["ppd"],
        visual_size=(resolution["visual_size"][0], resolution["visual_size"][1] / 2),
        radii=radii,
        target_indices=target_indices[0],
        intensity_frames=(1, 1),
        intensity_target=intensity_target
    )
    right = stimupy.stimuli.rings.rectangular_generalized(
        ppd=resolution["ppd"],
        visual_size=(resolution["visual_size"][0], resolution["visual_size"][1] / 2),
        radii=radii,
        target_indices=target_indices[1],
        intensity_frames=(0, 0),
        intensity_target=intensity_target
    )
    return stimupy.utils.stack_dicts(left, right, direction="horizontal")


def bullseye_low_freq(target_side, intensity_target):
    target_indices = (target_side in ["Left", "Both"], target_side in ["Right", "Both"])

    left = stimupy.stimuli.rings.rectangular_generalized(
        ppd=resolution["ppd"],
        visual_size=(resolution["visual_size"][0], resolution["visual_size"][1] / 2),
        radii=radii,
        target_indices=target_indices[0],
        intensity_frames=(1, 0),
        intensity_target=intensity_target
    )
    right = stimupy.stimuli.rings.rectangular_generalized(
        ppd=resolution["ppd"],
        visual_size=(resolution["visual_size"][0], resolution["visual_size"][1] / 2),
        radii=radii,
        target_indices=target_indices[1],
        intensity_frames=(0, 1),
        intensity_target=intensity_target
    )
    return stimupy.utils.stack_dicts(left, right, direction="horizontal")


def bullseye_high_freq(target_side, intensity_target):
    if target_side == "Left":
        intensity_targets = (intensity_target, 0.0)
    elif target_side == "Right":
        intensity_targets = (1.0, intensity_target)
    elif target_side == "Both":
        intensity_targets = (intensity_target, intensity_target)

    left = stimupy.stimuli.rings.rectangular(
        ppd=resolution["ppd"],
        visual_size=(resolution["visual_size"][0], resolution["visual_size"][1] / 2),
        target_indices=0,
        intensity_frames=(1.0, 0.0),
        n_frames=5,
        intensity_target=intensity_targets[0]
    )
    right = stimupy.stimuli.rings.rectangular(
        ppd=resolution["ppd"],
        visual_size=(resolution["visual_size"][0], resolution["visual_size"][1] / 2),
        target_indices=0,
        intensity_frames=(0.0, 1.0),
        n_frames=5,
        intensity_target=intensity_targets[1]
    )
    return stimupy.utils.stack_dicts(left, right, direction="horizontal")


def sbc_separate(target_side, intensity_target, intensity_background):
    bullseye_hfe = bullseye_high_freq(target_side, intensity_target)

    # Mask separation frame
    separate_mask = np.where(bullseye_hfe["grating_mask"] == 2, 1, 0)
    separate_mask = np.where(bullseye_hfe["grating_mask"] == 3, 1, separate_mask)
    separate_mask = np.where(bullseye_hfe["grating_mask"] == 7, 1, separate_mask)
    separate_mask = np.where(bullseye_hfe["grating_mask"] == 8, 1, separate_mask)
    separate_mask = np.where(bullseye_hfe["target_mask"], 1, separate_mask)

    # sbc_separated
    bullseye_lfe = bullseye_low_freq(target_side, intensity_target)
    sbc_separate = deepcopy(bullseye_lfe)
    sbc_separate["img"] = np.where(separate_mask, sbc_separate["img"], intensity_background)
    return sbc_separate


def sbc_separate_small(target_side, intensity_target, intensity_background):
    bullseye_hfe = bullseye_high_freq(target_side, intensity_target)

    # Mask inner frame
    frame_mask = np.where(bullseye_hfe["grating_mask"] == 2, 1, 0)
    frame_mask = np.where(bullseye_hfe["grating_mask"] == 7, 1, frame_mask)
    frame_mask = np.where(bullseye_hfe["target_mask"], 1, frame_mask)

    sbc_smallest = deepcopy(bullseye_hfe)
    sbc_smallest["img"] = np.where(frame_mask, sbc_smallest["img"], intensity_background)
    return sbc_smallest


def bullseye_low_separate(target_side, intensity_target, intensity_background):
    bullseye_hfe = bullseye_high_freq(target_side, intensity_target)

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


def whites(target_side, intensity_target):
    TARGET_INDICES = (2, -3)
    if target_side == "Left":
        target_indices = TARGET_INDICES[0]
    elif target_side == "Right":
        target_indices = TARGET_INDICES[1]
    elif target_side == "Both":
        target_indices = TARGET_INDICES

    return stimupy.stimuli.whites.white(
        **resolution,
        bar_width=target_size,
        target_indices=target_indices,
        target_heights=target_size,
        intensity_bars=(0, 1),
        intensity_target=intensity_target
    )


def whites_high_freq(target_side, intensity_target):
    TARGET_INDICES = (4, -5)
    if target_side == "Left":
        target_indices = TARGET_INDICES[0]
    elif target_side == "Right":
        target_indices = TARGET_INDICES[1]
    elif target_side == "Both":
        target_indices = TARGET_INDICES

    return stimupy.stimuli.whites.white(
        **resolution,
        bar_width=target_size / 2,
        target_indices=target_indices,
        target_heights=target_size,
        intensity_bars=(0, 1),
        intensity_target=intensity_target
    )


def whites_high_freq_equal_aspect(target_side, intensity_target):
    TARGET_INDICES = (4, -5)
    if target_side == "Left":
        target_indices = TARGET_INDICES[0]
    elif target_side == "Right":
        target_indices = TARGET_INDICES[1]
    elif target_side == "Both":
        target_indices = TARGET_INDICES

    return stimupy.stimuli.whites.white(
        **resolution,
        bar_width=target_size / 2,
        target_indices=target_indices,
        target_heights=target_size / 2,
        intensity_bars=(0, 1),
        intensity_target=intensity_target
    )


def whites_narrow(target_side, intensity_target, intensity_background):
    TARGET_INDICES = (2, -3)
    if target_side == "Left":
        target_indices = TARGET_INDICES[0]
    elif target_side == "Right":
        target_indices = TARGET_INDICES[1]
    elif target_side == "Both":
        target_indices = TARGET_INDICES

    whites_narrow = stimupy.stimuli.whites.white(
        ppd=resolution["ppd"],
        visual_size=(6, resolution["visual_size"][1]),
        bar_width=target_size,
        target_indices=target_indices,
        target_heights=target_size,
        intensity_bars=(0, 1),
        intensity_target=intensity_target
    )

    whites_narrow = stimupy.utils.pad_dict_to_visual_size(
        dct=whites_narrow, **resolution, pad_value=intensity_background
    )

    return whites_narrow

def whites_separate(target_side, intensity_target, intensity_background):
    bullseye_hfe = bullseye_high_freq(target_side, intensity_target)

    # Mask separation frame
    separate_mask = np.where(bullseye_hfe["grating_mask"] == 2, 1, 0)
    separate_mask = np.where(bullseye_hfe["grating_mask"] == 3, 1, separate_mask)
    separate_mask = np.where(bullseye_hfe["grating_mask"] == 7, 1, separate_mask)
    separate_mask = np.where(bullseye_hfe["grating_mask"] == 8, 1, separate_mask)
    separate_mask = np.where(bullseye_hfe["target_mask"], 1, separate_mask)

    whites_s = deepcopy(whites(target_side, intensity_target))
    whites_s["img"] = np.where(separate_mask, whites_s["img"], intensity_background)

    return whites_s


def strip(target_side, intensity_target, intensity_background):
    TARGET_INDICES = (2, -3)
    if target_side == "Left":
        target_indices = TARGET_INDICES[0]
    elif target_side == "Right":
        target_indices = TARGET_INDICES[1]
    elif target_side == "Both":
        target_indices = TARGET_INDICES

    strip_stim = stimupy.stimuli.whites.white(
        ppd=resolution["ppd"],
        visual_size=(target_size, resolution["visual_size"][1]),
        bar_width=target_size,
        target_indices=target_indices,
        target_heights=target_size,
        intensity_bars=(1, 0),
        intensity_target=intensity_target,
    )
    strip_stim = stimupy.utils.pad_dict_to_visual_size(
        dct=strip_stim, **resolution, pad_value=intensity_background
    )

    return strip_stim


def checkerboard(target_side, intensity_target):
    TARGET_INDICES = ((2, 2), (2, 7))
    if target_side == "Left":
        target_indices = (TARGET_INDICES[0],)
    elif target_side == "Right":
        target_indices = (TARGET_INDICES[1],)
    elif target_side == "Both":
        target_indices = TARGET_INDICES

    return stimupy.stimuli.checkerboards.checkerboard(
        **resolution,
        check_visual_size=target_size,
        target_indices=target_indices,
        intensity_checks=(1, 0),
        intensity_target=intensity_target,
    )


def checkerboard_narrow(target_side, intensity_target, intensity_background):
    TARGET_INDICES = ((1, 2), (1, 7))
    if target_side == "Left":
        target_indices = (TARGET_INDICES[0],)
    elif target_side == "Right":
        target_indices = (TARGET_INDICES[1],)
    elif target_side == "Both":
        target_indices = TARGET_INDICES

    checkerboard_narrow = stimupy.stimuli.checkerboards.checkerboard(
        ppd=resolution["ppd"],
        visual_size=(6, resolution["visual_size"][1]),
        check_visual_size=target_size,
        target_indices=target_indices,
        intensity_target=intensity_target
    )

    checkerboard_narrow = stimupy.utils.pad_dict_to_visual_size(
        dct=checkerboard_narrow, **resolution, pad_value=intensity_background
    )

    return checkerboard_narrow


def checkerboard_separate(target_side, intensity_target, intensity_background):
    bullseye_hfe = bullseye_high_freq(target_side, intensity_target)

    # Mask separation frame
    separate_mask = np.where(bullseye_hfe["grating_mask"] == 2, 1, 0)
    separate_mask = np.where(bullseye_hfe["grating_mask"] == 3, 1, separate_mask)
    separate_mask = np.where(bullseye_hfe["grating_mask"] == 7, 1, separate_mask)
    separate_mask = np.where(bullseye_hfe["grating_mask"] == 8, 1, separate_mask)
    separate_mask = np.where(bullseye_hfe["target_mask"], 1, separate_mask)

    checkerboard_separate = deepcopy(checkerboard_narrow(target_side, intensity_target, intensity_background))
    checkerboard_separate["img"] = np.where(
        separate_mask, checkerboard_separate["img"], intensity_background
    )

    return checkerboard_separate


def uniform(target_side, intensity_targets, intensity_background):
    left = stimupy.stimuli.rings.rectangular_generalized(
        ppd=resolution["ppd"],
        visual_size=(resolution["visual_size"][0], resolution["visual_size"][1] / 2),
        radii=radii,
        target_indices=target_side in ["Left", "Both"],
        intensity_frames=intensity_background,
        intensity_target=intensity_targets[0]
    )
    right = stimupy.stimuli.rings.rectangular_generalized(
        ppd=resolution["ppd"],
        visual_size=(resolution["visual_size"][0], resolution["visual_size"][1] / 2),
        radii=radii,
        target_indices=target_side in ["Right", "Both"],
        intensity_frames=intensity_background,
        intensity_target=intensity_targets[1]
    )

    return stimupy.utils.stack_dicts(left, right, direction="horizontal")


def stims(stim, target_side, flipped, intensity_target, intensity_background):

    if stim == "sbc":
        stimulus = sbc(target_side, intensity_target)
    elif stim == "bullseye_low_freq":
        stimulus = bullseye_low_freq(target_side, intensity_target)
    elif stim == "bullseye_high_freq":
        stimulus = bullseye_high_freq(target_side, intensity_target)
    elif stim == "sbc_separate":
        stimulus = sbc_separate(target_side, intensity_target, intensity_background)
    elif stim == "sbc_separate_small":
        stimulus = sbc_separate_small(target_side, intensity_target, intensity_background)
    elif stim == "bullseye_low_separate":
        stimulus = bullseye_low_separate(target_side, intensity_target, intensity_background)
    elif stim == "whites":
        stimulus = whites(target_side, intensity_target)
    elif stim == "whites_high_freq":
        stimulus = whites_high_freq(target_side, intensity_target)
    elif stim == "whites_high_freq_equal_aspect":
        stimulus = whites_high_freq_equal_aspect(target_side, intensity_target)
    elif stim == "whites_narrow":
        stimulus = whites_narrow(target_side, intensity_target, intensity_background)
    elif stim == "whites_separate":
        stimulus = whites_separate(target_side, intensity_target, intensity_background)
    elif stim == "strip":
        stimulus = strip(target_side, intensity_target, intensity_background)
    elif stim == "checkerboard":
        stimulus = checkerboard(target_side, intensity_target)
    elif stim == "checkerboard_narrow":
        stimulus = checkerboard_narrow(target_side, intensity_target, intensity_background)
    elif stim == "checkerboard_separate":
        stimulus = checkerboard_separate(target_side, intensity_target, intensity_background)
    elif "catch_trial" in stim:
        version = stim.split("_")[3]
        bg_color = stim.split("_")[2]
        stimulus = catch_trial(version, bg_color, target_side)
    else:
        raise Exception(f"stimulus {stim} not found")

    if flipped:
        # TODO: unflip the responses of participants while evaluating
        return stimupy.utils.flip_dict(stimulus)
    else:
        return stimulus


def catch_trial(version, background, side):
    if version == "1":
        intensity_targets = (.60, .40)
    elif version == "2":
        intensity_targets = (.55, .45)
    elif version == "3":
        intensity_targets = (.50, .50)
    elif version == "4":
        intensity_targets = (.45, .55)
    elif version == "5":
        intensity_targets = (.40, .60)
    else:
        raise Exception("version not found " + str(version))

    return uniform(side, intensity_targets=intensity_targets, intensity_background=INTENSITIES[background])



if __name__ == "__main__":
    target_side = "Both"
    intensity_target = 0.5

    num_cols = 3
    num_rows = 3

    stimulus_aspect_ratio = resolution["visual_size"][0] / (resolution["visual_size"][1] / 2)
    fig_width = 9
    fig_height = 7

    plt.figure(figsize=(fig_width, fig_height))

    for i, stim_name in enumerate(__all__):
        plt.subplot(num_rows, num_cols, i + 1)
        stimulus = stims(stim_name, target_side, False, intensity_target, intensity_background=0.27)

        plt.imshow(stimulus["img"], cmap="gray",)
        plt.axis("off")
        plt.title("(" + "ABCDEFGHIJKLMNOPQRSTUVW"[i] + ") " + stim_name)

    plt.tight_layout()
    plt.show()

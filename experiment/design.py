import sys
import numpy as np
import pandas as pd
import random
import data_management
import stimuli
from adjustment import adjust
from asymmetric_matching import matching_field, perturb_array
from text_displays import text_to_arr

VARIEGATED_ARRAY = np.loadtxt("matchsurround.txt")

SIDES = ("Left", "Right")
FLIPPED = ("False", "True")
rng = np.random.default_rng()


def display_stim_likert(ihrl, stim, likert_flipped):
    stimulus = stimuli.stims(stim, target_side="Both", flipped=likert_flipped, presented_intensity=0.5, intensity_background=ihrl.background)
    stim_texture = ihrl.graphics.newTexture(stimulus["img"])
    return stim_texture


def display_stim_matching(ihrl, stim, target_side, presented_intensity, matching_flipped):
    stimulus = stimuli.stims(stim, target_side=target_side, flipped=matching_flipped, presented_intensity=presented_intensity, intensity_background=ihrl.background)
    stim_texture = ihrl.graphics.newTexture(stimulus["img"])
    return stim_texture


def draw_match(ihrl, intensity_match, variegated_array):
    stim = matching_field(
        variegated_array=variegated_array,
        ppd=stimuli.resolution["ppd"],
        field_size=(stimuli.target_size, stimuli.target_size),
        field_intensity=intensity_match,
        check_visual_size=(stimuli.target_size / 2, stimuli.target_size / 2),
    )
    stim_texture = ihrl.graphics.newTexture(stim["img"])
    window_shape = (ihrl.height, ihrl.width)
    window_center = (window_shape[0] // 2, window_shape[1] // 2)  # Center of the drawing window
    pos = (window_center[1] - (stim_texture.wdth // 2), 0.6 * (stim_texture.hght))
    stim_texture.draw(pos=pos, sz=(stim_texture.wdth, stim_texture.hght))


def draw_options(ihrl, position):
    txt_ints = [0.0] * 5
    txt_ints[position - 1] = 1.0
    t1 = ihrl.graphics.newTexture(
        text_to_arr(
            "Left target is definitely brighter",
            intensity_background=ihrl.background,
            intensity_text=txt_ints[0],
            fontsize=25,
        ),
        "square",
    )
    t2 = ihrl.graphics.newTexture(
        text_to_arr(
            "Left target is maybe brighter",
            intensity_background=ihrl.background,
            intensity_text=txt_ints[1],
            fontsize=25,
        ),
        "square",
    )
    t3 = ihrl.graphics.newTexture(
        text_to_arr(
            "Targets are equally bright",
            intensity_background=ihrl.background,
            intensity_text=txt_ints[2],
            fontsize=25,
        ),
        "square",
    )
    t4 = ihrl.graphics.newTexture(
        text_to_arr(
            "Right target is maybe brighter",
            intensity_background=ihrl.background,
            intensity_text=txt_ints[3],
            fontsize=25,
        ),
        "square",
    )
    t5 = ihrl.graphics.newTexture(
        text_to_arr(
            "Right target is definitely brighter",
            intensity_background=ihrl.background,
            intensity_text=txt_ints[4],
            fontsize=25,
        ),
        "square",
    )

    t1.draw((102, 1010), (t1.wdth, t1.hght))
    t2.draw((102 + t1.wdth + 20, 1010), (t2.wdth, t2.hght))
    t3.draw((102 + t2.wdth + t1.wdth + 40, 1010), (t3.wdth, t3.hght))
    t4.draw((102 + t1.wdth + t2.wdth + t3.wdth + 60, 1010), (t4.wdth, t4.hght))
    t5.draw((102 + t1.wdth + t2.wdth + t3.wdth + t4.wdth + 80, 1010), (t5.wdth, t5.hght))


def select(ihrl, value, range):
    try:
        len(range)
    except:
        range = (0, range)

    accept = False
    press, _ = ihrl.inputs.readButton(btns=("Left", "Right", "Escape", "Space"))
    if press == "Escape":
        sys.exit("Participant terminated experiment.")
    elif press == "Left":
        value -= 1
        value = max(value, range[0])
    elif press == "Right":
        value += 1
        value = min(value, range[1])
    elif press == "Space":
        accept = True

    return value, accept


def run_trial(ihrl, stim, **kwargs):
    ihrl.graphics.flip(clr=True)
    if "target_side" in kwargs:
        target_side = kwargs.pop("target_side", None)
        presented_intensity = kwargs.pop("presented_intensity", None)
        matching_flipped = kwargs.pop("matching_flipped", None)
        return run_trial_matching(ihrl, stim, target_side, presented_intensity, matching_flipped, **kwargs)
    elif "likert_flipped" in kwargs:
        likert_flipped = kwargs.pop("likert_flipped", None)
        return run_trial_likert(ihrl, stim, likert_flipped, **kwargs)
    else:
        raise Exception("run trial went wrong")


def run_trial_likert(ihrl, stim, likert_flipped, **kwargs):
    response_position = 3
    accept = False
    stim_texture = display_stim_likert(
        ihrl,
        stim,
        likert_flipped,
    )

    window_shape = (ihrl.height, ihrl.width)
    window_center = (window_shape[0] // 2, window_shape[1] // 2)  # Center of the drawing window
    stimulus_position = (window_center[1] - (stim_texture.wdth // 2), window_center[0] - (stim_texture.hght // 2))
    while not accept:
        stim_texture.draw(pos=stimulus_position, sz=(stim_texture.wdth, stim_texture.hght))
        draw_options(ihrl, response_position)
        ihrl.graphics.flip(clr=True)
        response_position, accept = select(ihrl, value=response_position, range=(1, 5))

    return {"response": response_position}


def run_trial_matching(ihrl, stim, target_side, presented_intensity, matching_flipped, **kwargs):
    intensity_match = rng.random()
    accept = False
    stim_texture = display_stim_matching(
        ihrl,
        stim,
        target_side,
        presented_intensity,
        matching_flipped,
    )

    # create matching field (variegated checkerboard)
    variegated_array = perturb_array(VARIEGATED_ARRAY)

    window_shape = (ihrl.height, ihrl.width)
    window_center = (window_shape[0] // 2, window_shape[1] // 2)  # Center of the drawing window
    stimulus_position = (window_center[1] - (stim_texture.wdth // 2), window_center[0] - (stim_texture.hght // 2))
    while not accept:
        stim_texture.draw(pos=stimulus_position, sz=(stim_texture.wdth, stim_texture.hght))
        draw_match(ihrl, intensity_match=intensity_match, variegated_array=variegated_array)
        ihrl.graphics.flip(clr=True)
        intensity_match, accept = adjust(ihrl, value=intensity_match)

    return {"intensity_match": intensity_match}


def generate_session():
    order = ['a', 'b']
    random.shuffle(order)
    generate_session_likert(order[0])
    generate_session_matching(order[1])


def generate_session_likert(order, ):
    intensity_variation = [0.49, 0.5, 0.51]
    for i in range(len(intensity_variation)):
        block = generate_block_likert(intensity_variation, i)
        block_id = f"{order}-direction-{i}"
        filepath = data_management.design_filepath(block_id)
        block.to_csv(filepath)


def generate_block_likert(intensity_variation, stat_index):
    trials = []
    i = 0
    for likert_flipped in FLIPPED:
        for stim_name in stimuli.__all__:
            intensity = intensity_variation[(stat_index + i) % len(intensity_variation)]
            trials.append((stim_name, likert_flipped, intensity))
            i += 1

    random.shuffle(trials)

    catch_trials = [(("catch_trial_" + str(version)), "False") for version in range(1, 6)]
    random.shuffle(catch_trials)

    catch_trial_index = len(trials) // len(catch_trials)
    next_step = catch_trial_index
    for catch_trial in catch_trials:
        trials.insert(catch_trial_index - next_step, catch_trial)
        catch_trial_index += next_step + 1

    block = pd.DataFrame(
        trials,
        columns=["stim", "likert_flipped", "presented_intensity"],
    )

    block.index.name = "trial"
    return block


def generate_session_matching(order):
    intensity_variation = [0.49, 0.5, 0.51]
    for i in range(len(intensity_variation)):
        block = generate_block_matching(intensity_variation, i)
        block_id = f"{order}-matching-{i}"
        filepath = data_management.design_filepath(block_id)
        block.to_csv(filepath)


def generate_block_matching(intensity_variation, stat_index):

    trials = []
    i = 0
    for side in SIDES:
        for stim_name in stimuli.__all__:
            intensity = intensity_variation[(stat_index + i) % len(intensity_variation)]
            flip = bool(random.getrandbits(1))
            trials.append((stim_name, side, intensity, flip))
            i += 1

    random.shuffle(trials)

    block = pd.DataFrame(
        trials,
        columns=["stim", "target_side", "presented_intensity", "matching_flipped"],
    )

    block.index.name = "trial"
    return block

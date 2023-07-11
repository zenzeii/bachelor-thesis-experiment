import sys

import numpy as np
import pandas as pd

import data_management
import stimuli_likert
import stimuli_matching
from adjustment import adjust
from text_displays import text_to_arr

intensity_background = 0.3
LUMINANCES = (0.5,)
SIDES = ("Left", "Right")
stim_names_likert = stimuli_likert.stims.keys()
stim_names_matching = stimuli_matching.stimuli_names
rng = np.random.default_rng()
SHAPE = (768, 1024)  # Desired shape of the drawing window
CENTER = (SHAPE[0] // 2, SHAPE[1] // 2)  # Center of the drawing window


def display_stim_likert(ihrl, stim, response_position):
    stimulus = stimuli_likert.stims[stim]
    stim_texture = ihrl.graphics.newTexture(stimulus["img"])
    pos = (CENTER[1] - (stim_texture.wdth // 2), CENTER[0] - (stim_texture.hght // 2))
    stim_texture.draw(pos=pos, sz=(stim_texture.wdth, stim_texture.hght))
    draw_options(ihrl, response_position)
    ihrl.graphics.flip(clr=True)
    return


def display_stim_matching(ihrl, stim, intensity_target, target_side):
    stimulus = stimuli_matching.stims(
        stim, intensity_target=intensity_target, target_side=target_side
    )
    stim_texture = ihrl.graphics.newTexture(stimulus["img"])
    return stim_texture


def draw_match(ihrl, intensity_match):
    stim = stimuli_matching.matching_field(intensity_match=intensity_match)
    stim_texture = ihrl.graphics.newTexture(stim["img"])
    pos = (CENTER[1] - (stim_texture.wdth // 2), 0.6 * (stim_texture.hght))
    stim_texture.draw(pos=pos, sz=(stim_texture.wdth, stim_texture.hght))


def draw_options(ihrl, position):
    txt_ints = [0.0] * 5
    txt_ints[position - 1] = 1.0
    t1 = ihrl.graphics.newTexture(
        text_to_arr(
            "Left target is definitely brighter",
            intensity_background=intensity_background,
            intensity_text=txt_ints[0],
            fontsize=25,
        ),
        "square",
    )
    t2 = ihrl.graphics.newTexture(
        text_to_arr(
            "Left target is maybe brighter",
            intensity_background=intensity_background,
            intensity_text=txt_ints[1],
            fontsize=25,
        ),
        "square",
    )
    t3 = ihrl.graphics.newTexture(
        text_to_arr(
            "Targets are equally bright",
            intensity_background=intensity_background,
            intensity_text=txt_ints[2],
            fontsize=25,
        ),
        "square",
    )
    t4 = ihrl.graphics.newTexture(
        text_to_arr(
            "Right target is maybe brighter",
            intensity_background=intensity_background,
            intensity_text=txt_ints[3],
            fontsize=25,
        ),
        "square",
    )
    t5 = ihrl.graphics.newTexture(
        text_to_arr(
            "Right target is definitely brighter",
            intensity_background=intensity_background,
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
    if len(kwargs) < 3:
        return run_trial_likert(ihrl, stim, **kwargs)
    else:
        intensity_target = kwargs.pop("intensity_target", None)
        target_side = kwargs.pop("target_side", None)
        return run_trial_matching(ihrl, stim, intensity_target, target_side, **kwargs)


def run_trial_likert(ihrl, stim, **kwargs):
    response_position = 3
    accept = False
    while not accept:
        display_stim_likert(
            ihrl,
            stim,
            response_position=response_position,
        )
        response_position, accept = select(ihrl, value=response_position, range=(1, 5))
    return {"response": response_position}


def run_trial_matching(ihrl, stim, intensity_target, target_side, **kwargs):
    intensity_match = rng.random()
    accept = False
    stim_texture = display_stim_matching(
        ihrl,
        stim,
        intensity_target=intensity_target,
        target_side=target_side,
    )

    while not accept:
        pos = (CENTER[1] - (stim_texture.wdth // 2), CENTER[0] - (stim_texture.hght // 2))
        stim_texture.draw(pos=pos, sz=(stim_texture.wdth, stim_texture.hght))
        draw_match(ihrl, intensity_match=intensity_match)
        ihrl.graphics.flip(clr=True)
        intensity_match, accept = adjust(ihrl, value=intensity_match)

    return {"intensity_match": intensity_match}


def generate_session():
    generate_session_likert()
    generate_session_matching()


def generate_session_likert(Nrepeats=2):
    for i in range(Nrepeats):
        block = generate_block_likert()
        block_id = f"direction-{i}"
        filepath = data_management.design_filepath(block_id)
        block.to_csv(filepath)


def generate_block_likert():
    trials = [(name) for name in stim_names_likert]
    block = pd.DataFrame(
        trials,
        columns=["stim"],
    )
    block = block.reindex(np.random.permutation(block.index))
    block.reset_index(drop=True, inplace=True)
    block.index.name = "trial"
    return block


def generate_session_matching(Nrepeats=2):
    for i in range(Nrepeats):
        block = generate_block_matching()
        block_id = f"matching-{i}"
        filepath = data_management.design_filepath(block_id)
        block.to_csv(filepath)


def generate_block_matching():
    trials = [
        (stim_name, int_target, side)
        for stim_name in stim_names_matching
        for int_target in LUMINANCES
        for side in SIDES
    ]
    block = pd.DataFrame(
        trials,
        columns=["stim", "intensity_target", "target_side"],
    )
    block = block.reindex(np.random.permutation(block.index))
    block.reset_index(drop=True, inplace=True)
    block.index.name = "trial"
    return block

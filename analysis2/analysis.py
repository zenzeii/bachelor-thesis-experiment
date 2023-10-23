import os
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def combine_multiple_csv(skip_participants, invert_likert_response_participants):
    base_dir = '../data/results'
    dfs = []

    # Using os.walk to navigate through directories and subdirectories
    for dirpath, dirnames, filenames in os.walk(base_dir):

        for participant in skip_participants:
            # Remove the directory you want to skip
            if participant in dirnames:
                dirnames.remove(participant)

        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            df = pd.read_csv(file_path)

            # Extract the last part of the dirpath as the directory name
            dirname = os.path.basename(dirpath)

            # Insert a new column at the beginning with the value as the directory name
            df.insert(0, 'participant', dirname)

            # Modify the 'trial' column
            df['trial'] = df['trial'].apply(lambda x: f"{filename}-{x}")

            dfs.append(df)

    # Concatenate all DataFrames
    if dfs:  # Check if dfs is not empty

        # concat all
        final_df = pd.concat(dfs, axis=0, ignore_index=True, sort=False)

        # set scale from [1, 2 ,3, 4, 5] to [-2, -1, 0, 1, 2]
        final_df = final_df.assign(response=lambda x: x.response - 3)

        # scale the luminance to [0-250 cd/m²]
        final_df['presented_intensity'] *= 250
        final_df['intensity_match'] *= 250

        # correct naming for catch trials (during the experiment there is a mistake in the naming of the catching trials)
        final_df['stim'] = final_df['stim'].apply(transform_catch_value)

        # for flipped stim in likert -> flip responses too
        final_df.loc[final_df['likert_flipped'] == True, 'response'] = final_df.loc[final_df['likert_flipped'] == True, 'response'] * -1

        # fix sbc stim (in experiment the canonical version was flipped by mistake)
        final_df.loc[final_df['stim'] == 'sbc', 'response'] = final_df.loc[final_df['stim'] == 'sbc', 'response'] * -1
        mask = final_df['stim'] == 'sbc'
        final_df.loc[mask, 'target_side'] = np.where(final_df.loc[mask, 'target_side'] == 'Left', 'Right', 'Left')

        # invert likert responses for the set participant in invert_likert_response_participants
        for participant in invert_likert_response_participants:
            final_df.loc[final_df['participant'] == participant, 'response'] = final_df.loc[final_df['participant'] == participant, 'response'] * -1

        # add expected and tolerated catch trial columns
        final_df[['expected_catch_trial_response', 'tolerated_catch_trial_response']] = final_df.apply(compute_catch_trial_responses, axis=1)

        #final_df.to_csv('combined_data.csv', index=False)
        return final_df

    else:
        print("No CSV files found to concatenate.")

def transform_catch_value(value):
    parts = value.split("_")

    if len(parts) == 5 and 'catch' in parts:
        return "_".join([parts[0], parts[1], parts[3], parts[2]])
    else:
        return value

def compute_catch_trial_responses(row):
    if 'catch' in row['stim']:
        last_word = row['stim'].split('_')[-1]

        try:
            int_last_word = int(last_word) - 3
            int_response = int(row['response'])

            expected = 1 if int_last_word == int_response else 0
            tolerated = 1 if int_last_word in [int_response, int_response + 1, int_response - 1] else 0

            return pd.Series([expected, tolerated])

        except ValueError:
            # If the last word is not a number, return NaN for both columns
            return pd.Series([np.nan, np.nan])
    else:
        return pd.Series([np.nan, np.nan])


def extract_numeric(participant_str):
    return int(participant_str[1:].split("-")[0])



def matching_scatterplot():
    return

def matching_scatterplot_2():
    return

def matching_mean_adjustments():
    return

def matching_absolute_differences():
    return

def liker_heatmap(df, cmap, target):
    """
    Generate a heatmap illustrating the average response from each participant for each stimulus
    and presented intensity combined.
    """

    # Compute the average response for each participant
    df_filtered = df.copy()
    df_copy = df.copy()

    avg_responses = df_filtered.groupby('participant')['response'].median().sort_values()

    unique_participants = avg_responses.index.tolist()
    participant_mapping = {participant: f"s{i}-{participant}" for i, participant in enumerate(unique_participants)}
    df_copy['participant_num'] = df_copy['participant'].map(participant_mapping)

    # Combine intensities for the pivot
    concatenated = []
    for intensity_set in sorted(df_filtered['presented_intensity'].unique()):
        df_filtered = df_copy[df_copy['presented_intensity'] == intensity_set].copy()
        df_filtered['combined_stim'] = df_filtered['stim'] + ' (' + df_filtered['presented_intensity'].astype(str) + ')'
        pivot_data = df_filtered.pivot_table(index='combined_stim', columns='participant_num', values='response', aggfunc='median')
        concatenated.append(pivot_data)

    combined_data = pd.concat(concatenated)

    # Map old labels to correct sequence
    combined_data = combined_data[sorted(participant_mapping.values(), key=lambda x: int(x.split("-")[0][1:]))]

    # Filter out 'catch' and sequence labels
    df_no_catch = df_copy[~df_copy['stim'].str.contains('catch')]
    order = df_no_catch.groupby("stim")["response"].median().sort_values(ascending=True).index.tolist()
    combined_order = [f"{stim} ({intensity})" for stim in order for intensity in sorted(df_no_catch['presented_intensity'].unique())]
    combined_data = combined_data.reindex(combined_order)

    # Processing catch trial data
    filtered_catch_data = df[(df['stim'].str.contains('catch')) & (df['trial'].str.contains('direction'))].copy()
    expected_scores = filtered_catch_data.groupby('participant').apply(
        lambda x: x['expected_catch_trial_response'].sum() / len(x)).to_dict()
    tolerated_scores = filtered_catch_data.groupby('participant').apply(
        lambda x: x['tolerated_catch_trial_response'].sum() / len(x)).to_dict()
    filtered_catch_data['expected rate'] = filtered_catch_data['participant'].map(expected_scores)
    filtered_catch_data['tolerated rate'] = filtered_catch_data['participant'].map(tolerated_scores)
    filtered_catch_data['participant_num'] = filtered_catch_data['participant'].map(participant_mapping)
    filtered_catch_data = filtered_catch_data.iloc[filtered_catch_data['participant_num'].apply(extract_numeric).argsort()]

    # Create the heatmap
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [1, 1, 27]}, sharex=True)

    # custom color map for catch trial 'correctness'
    custom_cmap = sns.light_palette("green", as_cmap=True)

    # Use the custom colormap for the "Expected catch trial response" heatmap
    sns.heatmap(filtered_catch_data[['participant_num', 'expected rate']].drop_duplicates().set_index('participant_num').T,
                ax=ax1, cmap=custom_cmap, center=0, annot=True, fmt=".2f", linewidths=0.5, vmin=0, vmax=1, cbar=False)
    ax1.set_yticklabels([''])
    ax1.set_title("Expected catch trial response rate")
    ax1.set_xlabel("")
    ax1.set_ylabel("")
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)

    # Use the custom colormap for the "Tolerated catch trial response" heatmap
    sns.heatmap(filtered_catch_data[['participant_num', 'tolerated rate']].drop_duplicates().set_index('participant_num').T,
                ax=ax2, cmap=custom_cmap, center=0, annot=True, fmt=".2f", linewidths=0.5, vmin=0, vmax=1, cbar=False)
    ax2.set_yticklabels([''])
    ax2.set_title("Tolerated catch trial response rate")
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)

    ax3 = sns.heatmap(combined_data, cmap=cmap, center=0, annot=True, fmt=".2f", linewidths=0.5, vmin=-2, vmax=2,
                cbar=False)

    # Draw horizontal lines to separate stimuli
    for i in range(3, combined_data.shape[0]*3, 3):
        ax3.hlines(i, *ax3.get_xlim(), colors='black', linewidth=1)

    # Adjust the original y-axis labels
    y_labels = combined_data.index.tolist()
    y_positions = range(len(y_labels)+1)
    ax3.set_xlabel("Participant")
    ax3.set_ylabel("Stimulus")

    # Create a second y-axis for stimuli images
    ax4 = ax3.twinx()
    ax4.set_yticks(y_positions)
    ax4.set_yticklabels(["                                " for _ in y_positions])

    # Adding stimuli images next to y-labels on ax2
    for index, stimulus in enumerate(y_labels[::-1]):
        if (index+1) % 3 == 2:
            stimulus_name = stimulus.split(" ")[0]  # Assuming the format is "stim (intensity)"
            image = Image.open(f"../experiment/stim/{stimulus_name}.png")
            imagebox = OffsetImage(image, zoom=0.18)
            ab = AnnotationBbox(imagebox, (combined_data.shape[1], index), frameon=False,
                                boxcoords="data", box_alignment=(-0.05, 0.15), pad=0)
            ax4.add_artist(ab)

    plt.tight_layout()
    plt.savefig(f'{target}likert_heatmap_combined.svg')
    plt.close()

    return

def liker_median_responses():
    return

def liker_distribution():
    return







if __name__ == "__main__":

    # discard participants from analysis
    skip_participants = ["AA"]

    # invert likert responses for these participants
    invert_likert_response_participants = ["SP", "KP"]

    # Gather all data for the analysis
    df = combine_multiple_csv(skip_participants, invert_likert_response_participants)

    # Create common colormap
    cmap = sns.diverging_palette(250, 10, as_cmap=True)

    # directory to save all plots
    target = "../plots/"

    """
    START PLOTTING HERE
    """

    liker_heatmap(df, cmap, target)



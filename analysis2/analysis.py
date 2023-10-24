import math
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

def get_stim_order_for_matching(df, absolute=False):
    """
    Returns stim order from lowest to highest difference between value 'intensity_match' 'target_side=Left' and
    the value intensity_match' 'target_side=Right'

    Parameters:
    - df: DataFrame containing the data
    """

    # Calculate means
    df_filtered = df.copy()
    df_filtered = df_filtered[~df_filtered['stim'].str.contains('catch')]
    means = df_filtered.groupby(['stim', 'target_side'])['intensity_match'].mean().reset_index()

    # Calculate the difference between 'Right' and 'Left' for each stim
    diffs = means.pivot(index='stim', columns='target_side', values='intensity_match')
    if absolute:
        diffs['difference'] = (diffs['Right'] - diffs['Left']).abs()
    else:
        diffs['difference'] = (diffs['Right'] - diffs['Left'])

    # Order stimuli by the difference
    ordered_stimuli = diffs.sort_values(by='difference').index.tolist()

    return ordered_stimuli

def matching_scatterplot(df):
    """
    Generate a boxplot illustrating the results for each participant and stimulus.
    """

    # Define y-axis limits
    ymin = df['intensity_match'].min()
    ymax = df['intensity_match'].max()

    # get order
    order = get_stim_order_for_matching(df)

    # Filter rows based on given intensities
    df_filtered = df.copy()
    df_filtered = df_filtered[~df_filtered['stim'].str.contains('catch')]

    # Combine 'stim' and 'presented_intensity' for the x-axis
    df_filtered['stim_with_intensity'] = df_filtered['stim'] + '_' + df_filtered['presented_intensity'].astype(str).str.split(".").str[0]

    # Update the order list to reflect the new combined labels
    order_updated = [o + '_' + str(intensity).split(".")[0] for o in order for intensity in sorted(df_filtered['presented_intensity'].unique())]

    # Mapping target_side to colors
    palette_dict_dots = {'Right': cmap(0.99), 'Left': cmap(0.01)}
    palette_dict_box = {'Right': cmap(0.79), 'Left': cmap(0.21)}

    # Create a plot
    plt.figure(figsize=(15, 12))  # Adjust the figure size as needed
    sns.boxplot(x='stim_with_intensity', y='intensity_match', hue='target_side', data=df_filtered,
                orient='v', palette=palette_dict_box, hue_order=['Left', 'Right'], order=order_updated)
    sns.despine(left=True)

    # Add individual data points
    ax = sns.stripplot(x='stim_with_intensity', y='intensity_match', hue='target_side', data=df_filtered,
                       jitter=True, dodge=True, size=3.5, orient='v', palette=palette_dict_dots,
                       hue_order=['Left', 'Right'], order=order_updated)

    # Add stimuli images at the bottom
    ax2 = ax.twinx()
    ax2.tick_params(top=False, labeltop=False, left=False, labelleft=False, right=False, labelright=False,
                    bottom=False, labelbottom=False)

    for index, stim in enumerate(order):
        image = Image.open(f"../experiment/stim/{stim}.png")
        imagebox = OffsetImage(image, zoom=0.12)  # Adjust the zoom factor as needed
        ab = AnnotationBbox(imagebox, (((index*3)+1), 0), box_alignment=(0.5, 1.6), frameon=False)
        ax2.add_artist(ab)

    # Add horizontal lines for each intensity value
    #for intensity in intensities:
    #    ax.axhline(y=intensity, color='grey', linestyle='--', alpha=0.6, lw=1.5)

    # Create a proxy artist for the dashed line
    dash_line = plt.Line2D([0], [0], color='grey', linestyle='--', alpha=0.6, lw=1.5, label='Presented Intensity')
    # Add the proxy artist for the dashed line to the handles and its label to labels
    #handles.append(dash_line)
    #labels.append('Presented intensity')

    # Set legend
    handles, labels = ax.get_legend_handles_labels()
    labels = [label.replace('Left', 'Left target').replace('Right', 'Right target') for label in labels]
    ax2.legend(handles=handles, labels=labels, loc='upper left', ncol=5, bbox_to_anchor=(0, 1.1))

    xlables = sorted(df_filtered['presented_intensity'].unique()) *9

    ax.set_ylim(ymin, ymax)  # Set y-axis limits
    ax.set_ylabel('Adjusted luminance by participants in cd/m²')
    ax.set_xlabel('Stimulus')
    ax.get_legend().remove()
    ax.set_xlim(-0.5, len(order_updated) - 0.5)   # Set the x-axis limits to remove unused space
    ax.set_xticks(ticks=range(len(xlables)), labels=xlables, rotation=0)
    ax.set_yticks(ticks=range(0, 251, 10), labels=range(0, 251, 10), rotation=0)
    plt.tight_layout()

    x = range(54)
    osc = [0, 0, 0, 1, 1, 1] * 4 + [0, 0, 0]
    for x0, x1, os in zip(x[:-1], x[1:], osc):
        if os:
            plt.axvspan(x0-0.5, x1-0.5, color='gray', alpha=0.2, lw=0)

    plt.savefig(f'{target}matching_scatterplot_boxplot.png')
    plt.close()

    return

def matching_connected_scatterplot(df):
    df_copy = df.copy()
    filtered_df = df_copy[~df_copy['stim'].str.contains('catch')]
    filtered_df = filtered_df[filtered_df['trial'].str.contains('matching')]
    #filtered_df = filtered_df[filtered_df['stim'].str.contains('bullseye_high_freq')]

    ymin = filtered_df['intensity_match'].min()
    ymax = filtered_df['intensity_match'].max()

    # Create a combined column for x-axis
    filtered_df['combined'] = filtered_df['stim'] + "-" + filtered_df['presented_intensity'].astype(str) + "-" + filtered_df['target_side']

    # Sort dataframe by the combined column
    filtered_df = filtered_df.sort_values(by='combined')

    # Plot dots
    plt.figure(figsize=(24, 12))
    sns.stripplot(x='combined', y='intensity_match', hue='participant', data=filtered_df, jitter=False, dodge=False, marker='o', alpha=1, zorder=1)

    # Draw lines connecting dots
    for _, group in filtered_df.groupby(['participant', 'stim', 'presented_intensity']):
        if group.shape[0] == 2:  # only draw lines if there are two dots to connect
            plt.plot(group['combined'], group['intensity_match'], color='gray', linestyle=':', zorder=0)

    plt.ylabel('Intensity Match')
    plt.xlabel('Stim-Presented Intensity-Target Side')
    plt.xticks(rotation=45)
    plt.yticks(range(math.floor(int(ymin) / 10) * 10, int(ymax)+10, 10))
    plt.tight_layout()
    plt.savefig(f'{target}matching_scatterplot_raw.svg')


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

    avg_responses = df_filtered.groupby('participant')['response'].mean().sort_values()
    unique_participants = avg_responses.index.tolist()
    participant_mapping = {}
    participant_mapping_label = {}
    s = 0
    inverted = 0
    for i, participant in enumerate(unique_participants):
        participant_mapping[participant] = f"p{i+1}-{participant}"
        if i==4 and participant == "SP":
            participant_mapping_label[participant] = f"p12"
            inverted = inverted+1
        elif i+1-s-inverted == 12:
            s=s-1
            participant_mapping_label[participant] = f"p{i+1-s-inverted}"

        elif participant in ["AA", "KP"]:
            participant_mapping_label[participant] = f"s{s+1}"
            s=s+1
        else:
            participant_mapping_label[participant] = f"p{i+1-s-inverted}"

    df_copy['participant_num'] = df_copy['participant'].map(participant_mapping)
    df_copy['participant_label'] = df_copy['participant'].map(participant_mapping_label)

    print(participant_mapping_label)

    # Combine intensities for the pivot
    concatenated = []
    for intensity_set in sorted(df_filtered['presented_intensity'].unique()):
        df_filtered = df_copy[df_copy['presented_intensity'] == intensity_set].copy()
        df_filtered['combined_stim'] = df_filtered['stim'] + ' (' + df_filtered['presented_intensity'].astype(str) + ')'
        pivot_data = df_filtered.pivot_table(index='combined_stim', columns='participant_label', values='response', aggfunc='median')
        concatenated.append(pivot_data)

    combined_data = pd.concat(concatenated)

    # Map old labels to correct sequence
    ordered_labels = [participant_mapping_label[participant] for participant in unique_participants]
    combined_data = combined_data[ordered_labels]

    # Filter out 'catch' and sequence labels
    df_no_catch = df_copy[~df_copy['stim'].str.contains('catch')]
    df_no_catch = df_no_catch[~df_no_catch['trial'].str.contains('matching')]
    order = df_no_catch.groupby("stim")["response"].mean().sort_values(ascending=True).index.tolist()
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
    filtered_catch_data['participant_label'] = filtered_catch_data['participant'].map(participant_mapping)
    filtered_catch_data = filtered_catch_data.iloc[filtered_catch_data['participant_label'].apply(extract_numeric).argsort()]

    # Create the heatmap
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [1, 1, 27]}, sharex=True)

    # custom color map for catch trial 'correctness'
    custom_cmap = sns.light_palette("green", as_cmap=True)

    # Use the custom colormap for the "Expected catch trial response" heatmap
    sns.heatmap(filtered_catch_data[['participant_label', 'expected rate']].drop_duplicates().set_index('participant_label').T,
                ax=ax1, cmap=custom_cmap, center=0, annot=True, fmt=".2f", linewidths=0.5, vmin=0, vmax=1, cbar=False)
    ax1.set_yticklabels([''])
    ax1.set_title("Expected catch trial response rate")
    ax1.set_xlabel("")
    ax1.set_ylabel("")
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)

    # Use the custom colormap for the "Tolerated catch trial response" heatmap
    sns.heatmap(filtered_catch_data[['participant_label', 'tolerated rate']].drop_duplicates().set_index('participant_label').T,
                ax=ax2, cmap=custom_cmap, center=0, annot=True, fmt=".2f", linewidths=0.5, vmin=0, vmax=1, cbar=False)
    ax2.set_yticklabels([''])
    ax2.set_title("Tolerated catch trial response rate")
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)

    ax3 = sns.heatmap(combined_data, cmap=cmap, center=0, annot=True, fmt=".2f", linewidths=0.5, vmin=-2, vmax=2,
                cbar=False)

    # Get all the x-tick labels
    labels = ax3.get_xticklabels()

    # Mark certein labels with another color
    reversed_mapping = {v: k for k, v in participant_mapping_label.items()}
    for label in labels:
        if 's' in label.get_text():
            label.set_color('red')
        if 'SP' in reversed_mapping[label.get_text()]:
            label.set_color('blue')

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
    plt.savefig(f'{target}likert_heatmap_raw.svg')
    plt.close()

    return

def liker_median_responses():
    return

def liker_distribution():
    return







if __name__ == "__main__":

    # discard participants from analysis
    skip_participants = []#["AA"]

    # invert likert responses for these participants
    invert_likert_response_participants = []#["SP"]

    # Gather all data for the analysis
    df = combine_multiple_csv(skip_participants, invert_likert_response_participants)

    # Create common colormap
    cmap = sns.diverging_palette(250, 10, as_cmap=True)

    # directory to save all plots
    target = "../plots/"

    """
    START PLOTTING HERE
    """

    matching_connected_scatterplot(df)
    #matching_scatterplot(df)

    #matching_mean_adjustments()
    #matching_absolute_differences()

    #liker_heatmap(df, cmap, target)
    #liker_median_responses()
    #liker_distribution()


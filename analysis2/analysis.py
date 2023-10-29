import math
import os
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D

def combine_multiple_csv(skip_participants, invert_likert_response_participants):
    base_dir = '../data/results'
    dfs = []
    participant_mapping = {
        'MM': 'P1',
        'JV': 'P2',
        'HA': 'P3',
        'AK': 'P4',
        'AP': 'P5',
        'AR': 'P6',
        'DC': 'P7',
        'EW': 'P8',
        'GE': 'P9',
        'JL': 'P10',
        'JP': 'P11',
        'KF': 'P12',
        'KN': 'P13',
        'MA': 'P14',
        'MO': 'P15',
        'TT': 'P16',
        'YY': 'P17',
        'SP': 'P18',
        'AH': 'P19',
        'KP': 'P20',
        'AA': 'P21',
    }

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

        # Sort the unique participants and then create a mapping to "P1", "P2", etc.
        #sorted_unique_participants = sorted(final_df['participant'].unique())
        #participant_mapping = {participant: f'P{i + 1}' for i, participant in enumerate(sorted_unique_participants)}

        # Replace the 'participant' column values based on the custom mapping
        final_df['participant'] = final_df['participant'].map(participant_mapping)

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

def matching_scatterplot_old(df):
    """
    Generate a boxplot illustrating the results for each participant and stimulus.
    """

    # get order
    order = get_stim_order_for_matching(df)

    # Filter rows based on given intensities
    df_filtered = df.copy()
    df_filtered = df_filtered[~df_filtered['stim'].str.contains('catch')]

    # Define y-axis limits
    ymin = df_filtered['intensity_match'].min()
    ymax = df_filtered['intensity_match'].max()

    # Combine 'stim' and 'presented_intensity' for the x-axis
    df_filtered['stim_with_intensity'] = df_filtered['stim'] + '_' + df_filtered['presented_intensity'].astype(str).str.split(".").str[0]

    # Update the order list to reflect the new combined labels
    order_updated = [o + '_' + str(intensity).split(".")[0] for o in order for intensity in sorted(df_filtered['presented_intensity'].unique())]

    # Mapping target_side to colors
    palette_dict_dots = {'Right': cmap(0.99), 'Left': cmap(0.01)}
    palette_dict_box = {'Right': cmap(0.79), 'Left': cmap(0.21)}

    # Create a plot
    plt.figure(figsize=(12.5, 6))
    sns.boxplot(x='stim_with_intensity', y='intensity_match', hue='target_side', data=df_filtered,
                orient='v', palette=[(0, 0, 0, 0)], hue_order=['Left', 'Right'], order=order_updated)
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
    ax.set_ylabel('Adjustments by participants in cd/m²')
    ax.set_xlabel('Stimulus')
    ax.get_legend().remove()
    ax.set_xlim(-0.5, len(order_updated) - 0.5)   # Set the x-axis limits to remove unused space
    ax.set_xticks(ticks=range(len(xlables)), labels=xlables, rotation=0)
    ax.set_yticks(ticks=range(math.floor(int(ymin) / 10) * 10, int(ymax) + 10, 10), labels=range(math.floor(int(ymin) / 10) * 10, int(ymax) + 10, 10), rotation=0)
    plt.tight_layout()

    # alternating gray white background
    x = range(54)
    osc = [0, 0, 0, 1, 1, 1] * 4 + [0, 0, 0]
    for x0, x1, os in zip(x[:-1], x[1:], osc):
        if os:
            plt.axvspan(x0-0.5, x1-0.5, color='gray', alpha=0.2, lw=0)

    plt.savefig(f'{target}matching_scatterplot_boxplot_old.svg')
    plt.close()

    return

def matching_scatterplot(df):
    """
    Generate a boxplot illustrating the results for each participant and stimulus.
    """

    # get order
    order = get_stim_order_for_matching(df)

    # Filter rows based on given intensities
    df_filtered = df.copy()
    df_filtered = df_filtered[~df_filtered['stim'].str.contains('catch')]

    # Define x-axis limits
    xmin = df_filtered['intensity_match'].min()
    xmax = df_filtered['intensity_match'].max()

    # Combine 'stim' and 'presented_intensity' for the y-axis
    df_filtered['stim_with_intensity'] = df_filtered['stim'] + '_' + df_filtered['presented_intensity'].astype(str).str.split(".").str[0]

    # Update the order list to reflect the new combined labels
    order_updated = [o + '_' + str(intensity).split(".")[0] for o in order for intensity in sorted(df_filtered['presented_intensity'].unique())]

    # Mapping target_side to colors
    palette_dict_dots = {'Right': cmap(0.99), 'Left': cmap(0.01)}
    palette_dict_box = {'Right': cmap(0.79), 'Left': cmap(0.21)}

    # Create a plot
    plt.figure(figsize=(8, 11))
    sns.boxplot(y='stim_with_intensity', x='intensity_match', hue='target_side', data=df_filtered,
                orient='h', palette=palette_dict_box, hue_order=['Right', 'Left'], order=order_updated, showfliers = False)
    sns.despine(left=True)

    # Add individual data points
    ax = sns.stripplot(y='stim_with_intensity', x='intensity_match', hue='target_side', data=df_filtered,
                       jitter=True, dodge=True, size=3.5, orient='h', palette=palette_dict_dots,
                       hue_order=['Right', 'Left'], order=order_updated)

    # Add stimuli images on the right side
    ax2 = ax.twiny()
    ax2.tick_params(top=False, labeltop=False, left=False, labelleft=False, right=False, labelright=False,
                    bottom=False, labelbottom=False)

    for index, stim in enumerate(order):
        image = Image.open(f"../experiment/stim/{stim}.png")
        imagebox = OffsetImage(image, zoom=0.12)  # Adjust the zoom factor as needed
        ab = AnnotationBbox(imagebox, (1, ((index*3)+1)), box_alignment=(-0.1, 0.5), frameon=False)
        ax2.add_artist(ab)

    # Set legend
    handles, labels = ax.get_legend_handles_labels()
    labels = [label.replace('Left', 'Left target').replace('Right', 'Right target') for label in labels]
    labels = [labels[2],labels[3]]
    handles = [handles[2], handles[3]]
    ax.legend(handles=handles, labels=labels)

    ylables = sorted(df_filtered['presented_intensity'].unique()) * 9

    ax.set_xlim(xmin, xmax)  # Set x-axis limits
    ax.set_xlabel('Adjustments by participants in cd/m²')
    ax.set_ylabel('Stimulus')
    ax.set_ylim(-0.5, len(order_updated) - 0.5)   # Set the y-axis limits to remove unused space
    ax.set_yticks(ticks=range(len(ylables)))
    ax.set_yticklabels(ylables, rotation=0)
    ax.set_xticks(ticks=range(math.floor(int(xmin) / 10) * 10, int(xmax) + 10, 20))
    ax.set_xticklabels(range(math.floor(int(xmin) / 10) * 10, int(xmax) + 10, 20), rotation=0)
    plt.tight_layout()

    # alternating gray white background
    y = range(54)
    osc = [0, 0, 0, 1, 1, 1] * 4 + [0, 0, 0]
    for y0, y1, os in zip(y[:-1], y[1:], osc):
        if os:
            plt.axhspan(y0-0.5, y1-0.5, color='gray', alpha=0.2, lw=0)

    plt.savefig(f'{target}matching_scatterplot_boxplot.svg')
    plt.close()

def matching_scatterplot_combined(df):
    """
    Generate a boxplot illustrating the results for each participant and stimulus.
    """

    # get order
    order = get_stim_order_for_matching(df)

    # Filter rows based on given intensities
    df_filtered = df.copy()
    df_filtered = df_filtered[~df_filtered['stim'].str.contains('catch')]

    # Define x-axis limits
    xmin = df_filtered['intensity_match'].min()
    xmax = df_filtered['intensity_match'].max()

    # Combine 'stim' and 'presented_intensity' for the y-axis
    df_filtered['stim_with_intensity'] = df_filtered['stim'] + '_' + df_filtered['presented_intensity'].astype(str).str.split(".").str[0]

    # Update the order list to reflect the new combined labels
    order_updated = [o + '_' + str(intensity).split(".")[0] for o in order for intensity in sorted(df_filtered['presented_intensity'].unique())]

    # Mapping target_side to colors
    palette_dict_dots = {'Right': cmap(0.99), 'Left': cmap(0.01)}
    palette_dict_box = {'Right': cmap(0.79), 'Left': cmap(0.21)}

    # Create a plot
    plt.figure(figsize=(8, 11))
    sns.boxplot(y='stim', x='intensity_match', hue='target_side', data=df_filtered,
                orient='h', palette=palette_dict_box, hue_order=['Left', 'Right'], order=order, showfliers = False)
    sns.despine(left=True)

    # Add individual data points
    ax = sns.stripplot(y='stim', x='intensity_match', hue='target_side', data=df_filtered,
                       jitter=True, dodge=True, size=3.5, orient='h', palette=palette_dict_dots,
                       hue_order=['Left', 'Right'], order=order)

    # Add stimuli images on the right side
    ax2 = ax.twiny()
    ax2.tick_params(top=False, labeltop=False, left=False, labelleft=False, right=False, labelright=False,
                    bottom=False, labelbottom=False)

    for index, stim in enumerate(order):
        image = Image.open(f"../experiment/stim/{stim}.png")
        imagebox = OffsetImage(image, zoom=0.12)  # Adjust the zoom factor as needed
        ab = AnnotationBbox(imagebox, (1, ((index))), box_alignment=(-0.1, 0.5), frameon=False)
        ax2.add_artist(ab)

    # Set legend
    handles, labels = ax.get_legend_handles_labels()
    labels = [label.replace('Left', 'Left target').replace('Right', 'Right target') for label in labels]
    labels = [labels[2],labels[3]]
    handles = [handles[2], handles[3]]
    ax.legend(handles=handles, labels=labels)

    ylables = sorted(df_filtered['stim'].unique())

    ax.set_xlim(xmin, xmax)  # Set x-axis limits
    ax.set_xlabel('Adjustments by participants in cd/m²')
    ax.set_ylabel('Stimulus')
    ax.set_ylim(-0.5, len(order) - 0.5)   # Set the y-axis limits to remove unused space
    ax.set_yticks(ticks=range(len(ylables)))
    ax.set_yticklabels(ylables, rotation=0)
    ax.set_xticks(ticks=range(math.floor(int(xmin) / 10) * 10, int(xmax) + 10, 10))
    ax.set_xticklabels(range(math.floor(int(xmin) / 10) * 10, int(xmax) + 10, 10), rotation=0)
    plt.tight_layout()

    # alternating gray white background
    y = range(54)
    osc = [0, 1] * 4 + [0]
    for y0, y1, os in zip(y[:-1], y[1:], osc):
        if os:
            plt.axhspan(y0-0.5, y1-0.5, color='gray', alpha=0.2, lw=0)

    plt.savefig(f'{target}matching_scatterplot_boxplot_combined.svg')
    plt.close()



def matching_connected_scatterplot_old(df):
    df_copy = df.copy()
    filtered_df = df_copy[~df_copy['stim'].str.contains('catch')]
    filtered_df = filtered_df[filtered_df['trial'].str.contains('matching')]
    # filtered_df = filtered_df[filtered_df['stim'].str.contains('bullseye_high_freq')]

    ymin = filtered_df['intensity_match'].min()
    ymax = filtered_df['intensity_match'].max()

    # Create a combined column for x-axis
    filtered_df['combined'] = filtered_df['stim'] + "-" + filtered_df['presented_intensity'].astype(str) + "-" + \
                              filtered_df['target_side']
    # filtered_df['combined'] = filtered_df['stim'] + "-" + filtered_df['target_side']

    # Get the ordered stimuli
    ordered_stimuli = get_stim_order_for_matching(df_copy)

    # Use the pd.Categorical type to order the 'combined' column based on ordered_stimuli
    filtered_df['combined'] = pd.Categorical(filtered_df['combined'], categories=[stim + "-" + str(intensity) + "-" + side for stim in ordered_stimuli for intensity in filtered_df['presented_intensity'].unique() for side in filtered_df['target_side'].unique()], ordered=True)
    filtered_df = filtered_df.sort_values('combined')

    # Plot dots
    plt.figure(figsize=(23, 10))
    ax = sns.stripplot(x='combined', y='intensity_match', hue='participant', data=filtered_df, jitter=False,
                       dodge=False, marker='o', alpha=1, zorder=1)

    # Draw lines connecting dots
    for _, group in filtered_df.groupby(['participant', 'stim', 'presented_intensity']):
        if group.shape[0] == 2:  # only draw lines if there are two dots to connect
            ax.plot(group['combined'], group['intensity_match'], color='gray', linestyle=':', zorder=0)

    # alternating gray white background
    x = range(108)
    osc = ([0] * 6 + [1] * 6) * 4 + [0] * 6
    for x0, x1, os in zip(x[:-1], x[1:], osc):
        if os:
            plt.axvspan(x0 - 0.5, x1 - 0.5, color='gray', alpha=0.2, lw=0)

    ax.set_ylabel('Adjustments by participants in cd/m²')
    ax.set_xlabel('Stimulus')
    plt.xticks(rotation=45)
    plt.yticks(range(math.floor(int(ymin) / 10) * 10, int(ymax) + 10, 10))
    plt.tight_layout()
    plt.savefig(f'{target}matching_connected_scatterplot_old.svg')
    return

def matching_connected_scatterplot_colored_lines(df):
    df_copy = df.copy()
    filtered_df = df_copy[~df_copy['stim'].str.contains('catch')]
    filtered_df = filtered_df[filtered_df['trial'].str.contains('matching')]

    xmin = filtered_df['intensity_match'].min()
    xmax = filtered_df['intensity_match'].max()

    filtered_df['combined'] = filtered_df['stim'] + "-" + filtered_df['presented_intensity'].astype(str) + "-" + \
                              filtered_df['target_side']
    ordered_stimuli = get_stim_order_for_matching(df_copy)[::-1]
    ordered_categories = [stim + "-" + str(intensity) + "-" + side for stim in ordered_stimuli for intensity in
                          filtered_df['presented_intensity'].unique() for side in filtered_df['target_side'].unique()]
    filtered_df['combined'] = pd.Categorical(filtered_df['combined'], categories=ordered_categories, ordered=True)
    filtered_df = filtered_df.sort_values('combined')

    plt.figure(figsize=(10, 15))
    ax = sns.stripplot(y='combined', x='intensity_match', hue='participant', data=filtered_df, jitter=False,
                       dodge=False, marker='o', alpha=1, zorder=1)

    for _, group in filtered_df.groupby(['participant', 'stim', 'presented_intensity']):
        if group.shape[0] == 2:
            left_val = group[group['target_side'] == 'Left']['intensity_match'].values[0]
            right_val = group[group['target_side'] == 'Right']['intensity_match'].values[0]
            diff = right_val - left_val

            color = 'gray'
            if diff > 0:
                color = 'red'
            elif diff < 0:
                color = 'blue'

            ax.plot(group['intensity_match'], group['combined'], color=color, linestyle=':', zorder=0)

    y = range(108)
    osc = ([0] * 6 + [1] * 6) * 4 + [0] * 6
    for y0, y1, os in zip(y[:-1], y[1:], osc):
        if os:
            plt.axhspan(y0 - 0.5, y1 - 0.5, color='gray', alpha=0.2, lw=0)

    ax.set_xlabel('Adjustments by participants in cd/m²')
    ax.set_ylabel('Stimulus')
    ax.set_ylim(-0.5, len(ordered_categories) - 0.5)
    plt.yticks(rotation=45)
    plt.xticks(range(math.floor(int(xmin) / 10) * 10, int(xmax) + 10, 20))

    # Add stimuli images on the right side
    ax2 = ax.twiny()
    ax2.tick_params(top=False, labeltop=False, left=False, labelleft=False, right=False, labelright=False,
                    bottom=False, labelbottom=False)
    for index, stim in enumerate(ordered_stimuli):
        image = Image.open(f"../experiment/stim/{stim}.png")
        imagebox = OffsetImage(image, zoom=0.12)  # Adjust the zoom factor as needed
        ab = AnnotationBbox(imagebox, (1, (index * 6 + 3)), box_alignment=(-0.1, 0.5), frameon=False)
        ax2.add_artist(ab)

    plt.tight_layout()
    plt.savefig(f'{target}matching_connected_scatterplot_colored_lines.svg')


def matching_connected_scatterplot_colored_lines_2(df):
    df_copy = df.copy()
    filtered_df = df_copy[~df_copy['stim'].str.contains('catch')]
    filtered_df = filtered_df[filtered_df['trial'].str.contains('matching')]

    xmin = filtered_df['intensity_match'].min()
    xmax = filtered_df['intensity_match'].max()

    # Removing presented intensity from the combined column
    filtered_df['combined'] = filtered_df['stim'] + "-" + filtered_df['target_side']

    ordered_stimuli = get_stim_order_for_matching(df_copy)

    # Removing presented intensity from the ordered_categories
    ordered_categories = [stim + "-" + side for stim in ordered_stimuli for side in filtered_df['target_side'].unique()]

    filtered_df['combined'] = pd.Categorical(filtered_df['combined'], categories=ordered_categories, ordered=True)
    filtered_df = filtered_df.sort_values('combined')

    plt.figure(figsize=(10, 13))
    ax = sns.stripplot(y='combined', x='intensity_match', hue='participant', data=filtered_df, jitter=False,
                       dodge=False, marker='o', alpha=1, zorder=1)

    # After plotting the scatterplot, retrieve the existing legend handles and labels
    handles, labels = ax.get_legend_handles_labels()

    # Create two dummy Line2D objects for blue and red dashed lines.
    direction_elements = [
        Line2D([0], [0], linestyle='none', markersize=0, color='none'),
        Line2D([0], [0], linestyle='none', markersize=0, color='none'),
        Line2D([0], [0], color='blue', linestyle=':', label='Left'),
        Line2D([0], [0], color='red', linestyle=':', label='Right')
    ]

    # Append these new legend elements to the existing ones

    handles = [Line2D([0], [0], linestyle='none', markersize=0, color='none')] + handles
    labels = ['Participant'] + labels

    handles += direction_elements
    labels += ['', 'Direction', 'Left', 'Right']

    # Create a single legend with the combined handles and labels
    legend = ax.legend(handles=handles, labels=labels, loc='upper right')

    # Removing presented intensity from the grouping
    for _, group in filtered_df.groupby(['participant', 'stim', 'presented_intensity']):
        if group.shape[0] == 2:
            left_val = group[group['target_side'] == 'Left']['intensity_match'].values[0]
            right_val = group[group['target_side'] == 'Right']['intensity_match'].values[0]
            diff = right_val - left_val

            color = 'gray'
            if diff > 0:
                color = 'red'
            elif diff < 0:
                color = 'blue'

            ax.plot(group['intensity_match'], group['combined'], color=color, linestyle=':', zorder=0)

    y = range(18)  # 9 unique stim and 2 unique target_side
    osc = ([0] * 2 + [1] * 2) * 4 + [0] * 2

    for y0, y1, os in zip(y[:-1], y[1:], osc):
        if os:
            plt.axhspan(y0 - 0.5, y1 - 0.5, color='gray', alpha=0.2, lw=0)

    ax.set_xlabel('Adjustments by participants in cd/m²')
    ax.set_ylabel('Stimulus')
    ax.set_ylim(-0.5, len(ordered_categories) - 0.5)
    plt.yticks(rotation=45)
    plt.xticks(range(math.floor(int(xmin) / 10) * 10, int(xmax) + 10, 20))

    # Add stimuli images on the right side
    ax2 = ax.twiny()
    ax2.tick_params(top=False, labeltop=False, left=False, labelleft=False, right=False, labelright=False,
                    bottom=False, labelbottom=False)
    for index, stim in enumerate(ordered_stimuli):
        image = Image.open(f"../experiment/stim/{stim}.png")
        imagebox = OffsetImage(image, zoom=0.12)  # Adjust the zoom factor as needed
        ab = AnnotationBbox(imagebox, (1, (index * 2 + 1)), box_alignment=(-0.1, 0.5), frameon=False)
        ax2.add_artist(ab)

    plt.tight_layout()
    plt.savefig(f'{target}matching_connected_scatterplot_colored_lines.svg')


def matching_connected_scatterplot_participant(df):
    df_copy = df.copy()
    filtered_df = df_copy[~df_copy['stim'].str.contains('catch')]
    filtered_df = filtered_df[filtered_df['trial'].str.contains('matching')]

    xmin = filtered_df['intensity_match'].min()
    xmax = filtered_df['intensity_match'].max()

    # Create a combined column for y-axis
    filtered_df['combined'] = filtered_df['stim'] + "-" + filtered_df['target_side']

    # Get the ordered stimuli
    ordered_stimuli = get_stim_order_for_matching(df_copy)

    # Use the pd.Categorical type to order the 'combined' column based on ordered_stimuli
    ordered_categories = [stim + "-" + side for stim in ordered_stimuli for side in filtered_df['target_side'].unique()]
    filtered_df['combined'] = pd.Categorical(filtered_df['combined'], categories=ordered_categories, ordered=True)
    filtered_df = filtered_df.sort_values('combined')

    # Plot dots
    plt.figure(figsize=(10, 46))
    sns.set_palette("husl", 21)  # 'husl' is a palette with distinct colors
    ax = sns.stripplot(y='combined', x='intensity_match', hue='participant', data=filtered_df, jitter=False,
                       dodge=False, marker='o', alpha=1, zorder=1)

    # Creating the FacetGrid
    g = sns.FacetGrid(filtered_df, col="participant", col_wrap=4, height=6)
    g.map(sns.stripplot, 'intensity_match', 'combined', jitter=False, dodge=False, marker='o', alpha=1, zorder=1,
          color='blue')

    # Define a dictionary for markers based on unique values of 'presented_intensity'
    unique_intensities = filtered_df['presented_intensity'].unique()
    markers = ['o', 's', 'v', '^', '<', '>', 'p', '*', '+', 'x', 'D']  # add more if needed
    intensity_marker_dict = {intensity: marker for intensity, marker in zip(unique_intensities, markers)}

    # Plot using different markers based on 'presented_intensity'
    for intensity, marker in intensity_marker_dict.items():
        subset_data = filtered_df[filtered_df['presented_intensity'] == intensity]
        g.map(sns.stripplot, 'intensity_match', 'combined', data=subset_data,
              jitter=False, dodge=False, alpha=1, zorder=1, color='blue', marker=marker)

    # Extract the participant names from the FacetGrid in order
    facet_participants = g.col_names

    # Loop through each participant name and its corresponding axis
    for participant_name, ax in zip(facet_participants, g.axes.flat):
        participant_data = filtered_df[filtered_df['participant'] == participant_name]

        # Draw lines connecting dots for each participant
        for _, group in participant_data.groupby(['stim', 'presented_intensity']):
            if group.shape[0] == 2:  # only draw lines if there are two dots to connect
                left_val = group[group['target_side'] == 'Left']['intensity_match'].values[0]
                right_val = group[group['target_side'] == 'Right']['intensity_match'].values[0]
                diff = right_val - left_val

                color = 'gray'
                if diff > 0:
                    color = 'red'
                elif diff < 0:
                    color = 'blue'

                ax.plot(group['intensity_match'], group['combined'].cat.codes, color=color, linestyle=':', zorder=0)

    # alternating gray white background
    y = range(108)
    osc = ([0] * 2 + [1] * 2) * 4 + [0] * 2
    for y0, y1, os in zip(y[:-1], y[1:], osc):
        if os:
            plt.axhspan(y0 - 0.5, y1 - 0.5, color='gray', alpha=0.2, lw=0)

    ax.set_xlabel('Adjustments by participants in cd/m²')
    ax.set_ylabel('Stimulus')
    plt.yticks(rotation=45)
    plt.xticks(range(math.floor(int(xmin) / 10) * 10, int(xmax) + 10, 10))
    plt.tight_layout()
    plt.savefig(f'{target}matching_connected_scatterplot_participant.svg')

def matching_mean_adjustments():
    """
        Generate a scatter plot showing the average adjustment from participants for each stimulus.
    """

    # Filter rows based on given intensities
    df_filtered = df.copy()
    df_filtered = df_filtered[~df_filtered['stim'].str.contains('catch')]
    order = get_stim_order_for_matching(df)

    # Calculate means
    means = df_filtered.groupby(['stim', 'target_side'])['intensity_match'].mean().reset_index()

    # Group by 'presented_intensity' and compute the mean for 'intensity_match'
    means_all = df_filtered.groupby(['target_side', 'stim'])['intensity_match'].mean()

    # Extract y_min and y_max
    y_min = means_all.min()
    y_max = means_all.max()+8

    # Map each unique 'stim' to an index
    stim_to_index = {stim: index for index, stim in enumerate(order)}

    # Adjust x-values for 'Right' and 'Left' to avoid overlap
    means['x_adjust'] = means.apply(lambda row: stim_to_index[row['stim']] + 0.1 if row['target_side'] == 'Right' else stim_to_index[row['stim']] - 0.1, axis=1)

    # Create a new column with modified target side labels
    means['label_target_side'] = means['target_side'].replace({'Right': 'Right target', 'Left': 'Left target'})

    # Create a plot
    plt.figure(figsize=(10, 6))
    ax = sns.scatterplot(x='x_adjust', y='intensity_match', hue='label_target_side', data=means, palette={'Right target': cmap(0.99), 'Left target': cmap(0.01)}, s=200, zorder=1)

    sns.despine(left=True)
    ax.set_ylim(y_min-8, y_max)


    # Add stimuli images at the bottom
    ax2 = ax.twinx()
    ax2.tick_params(top=False, labeltop=False, left=False, labelleft=False, right=False, labelright=False, bottom=False, labelbottom=False)


    # Add horizontal lines for each intensity value
    for intensity in intensities:
        ax.axhline(y=intensity, color='grey', linestyle='--', alpha=0.6, lw=1.5)

    # Draw vertical bars for each 'stim'
    for stim, x_val in stim_to_index.items():
        # Existing vertical bar code
        ax.axvline(x=x_val - 0.1, color='#f0f0f0', ymin=0, ymax=1, lw=10, zorder=0)
        ax.axvline(x=x_val + 0.1, color='#f0f0f0', ymin=0, ymax=1, lw=10, zorder=0)

        # Extract intensity_match values for the current stim and each target side
        left_intensity = means[(means['stim'] == stim) & (means['target_side'] == 'Left')]['intensity_match'].values[0]
        right_intensity = means[(means['stim'] == stim) & (means['target_side'] == 'Right')]['intensity_match'].values[0]

        # Add the intensity_match values on the vertical bars
        ax.text(x_val - 0.1, y_max-1, f"{left_intensity:.2f}", ha="center", va="center", rotation=90, color="black", fontsize=8)
        ax.text(x_val + 0.1, y_max-1, f"{right_intensity:.2f}", ha="center", va="center", rotation=90, color="black", fontsize=8)

        # Add stimuli fig
        image = Image.open(f"../experiment/stim/{stim}.png")
        imagebox = OffsetImage(image, zoom=0.1)  # Adjust the zoom factor as needed
        ab = AnnotationBbox(imagebox, (x_val, 0), box_alignment=(0.5, -0.1), frameon=False)
        ax2.add_artist(ab)

    # Customize the plot appearance
    ax.set_ylabel('Average adjustments by participants in cd/m²')
    ax.set_xlabel('Stimulus')
    ax.set_xticks(range(len(stim_to_index)))
    ax.set_xticklabels(stim_to_index.keys(), rotation=45, ha='right')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)
    y_ticks = np.arange(y_min - (y_min % 5), y_max + 5,5)
    ax.set_yticks(y_ticks)

    # Customize the plot appearance
    plt.tight_layout()
    plt.savefig(f'{target}matching_mean_adjustments.svg')
    plt.close()
    return

def matching_absolute_differences():
    """
    Generate a scatter plot showing the average adjust_diff for each stimulus.
    """

    df_filtered = df.copy()
    df_filtered = df_filtered[~df_filtered['stim'].str.contains('catch')]
    df_filtered = df_filtered[~df_filtered['trial'].str.contains('direction')]

    # Group by 'participant', 'stim', 'presented_intensity' and calculate difference in 'intensity_match'
    means = df_filtered.groupby(['participant', 'stim', 'presented_intensity'])['intensity_match'].diff().dropna()
    print(means)

    # Group by 'stim' and calculate the mean difference
    #avg_adjust_diff = means.groupby(level='stim').mean()
    avg_adjust_diff = means.groupby(level=1).mean()

    sorted_stim = avg_adjust_diff.sort_values(ascending=False).index.tolist()

    # Plotting
    fig, ax = plt.subplots(figsize=(9, 6))

    for i, stim in enumerate(sorted_stim, start=1):
        x = avg_adjust_diff[stim]
        y = i
        norm_value = (x - (0)) / (10 - (-10))  # Normalizing between -0.1 to 0.1
        color = LinearSegmentedColormap.from_list("luminance", ["white", "black"])(norm_value)

        ax.scatter(x, y, s=1500, c=[color], alpha=0.5, label=stim)
        ax.text(x-1, y, round(x, 2), va='center')

    ax.set_ylabel("Stimulus")
    ax.set_xlabel("Average difference of adjustments in cd/m²")
    ax.set_yticks(range(1, len(sorted_stim) + 1))
    ax.set_yticklabels(sorted_stim)
    ticks = np.arange(0, 40 + 1, 5)
    ax.set_xticks(ticks)

    # Create a second y-axis for stimuli images
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(ax.get_yticks())
    ax2.set_yticklabels(["" for _ in sorted_stim])

    # Adding stimuli images next to y-labels on ax2
    for index, stimulus in enumerate(sorted_stim):
        if 'catch' in stimulus:
            continue
        image = Image.open(f"../experiment/stim/{stimulus}.png")
        imagebox = OffsetImage(image, zoom=0.13)
        ab = AnnotationBbox(imagebox, (40, ax2.get_yticks()[index]), frameon=False, boxcoords="data",
                            pad=0, box_alignment=(-0.05, 0.5))
        ax2.add_artist(ab)

    ax.set_zorder(ax2.get_zorder() + 1)
    plt.tight_layout()
    ax.grid(True, axis='both')
    plt.subplots_adjust(left=0.25)
    plt.savefig(f'{target}matching_absolute_differences.svg')
    plt.close()

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

    for i, participant in enumerate(unique_participants):
        participant_mapping[participant] = f"{participant}"

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
    ordered_labels = [participant_mapping[participant] for participant in unique_participants]
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
    filtered_catch_data['participant_num'] = filtered_catch_data['participant'].map(participant_mapping)
    filtered_catch_data = filtered_catch_data.set_index('participant_num').loc[ordered_labels].reset_index()

    # Create the heatmap
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [1, 1, 27]}, sharex=True)

    # custom color map for catch trial 'correctness'
    custom_cmap = sns.light_palette("green", as_cmap=True)

    # Use the custom colormap for the "Expected catch trial response" heatmap
    sns.heatmap(filtered_catch_data[['participant_num', 'expected rate']].drop_duplicates().set_index('participant_num').T,
                ax=ax1, cmap=custom_cmap, center=0, annot=True, fmt=".2f", linewidths=0.5, vmin=0, vmax=1, cbar=False)
    ax1.set_yticklabels([''])
    ax1.set_title("Exactly correct catch trial response rate")
    ax1.set_xlabel("")
    ax1.set_ylabel("")
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)

    # Use the custom colormap for the "Tolerated catch trial response" heatmap
    sns.heatmap(filtered_catch_data[['participant_num', 'tolerated rate']].drop_duplicates().set_index('participant_num').T,
                ax=ax2, cmap=custom_cmap, center=0, annot=True, fmt=".2f", linewidths=0.5, vmin=0, vmax=1, cbar=False)
    ax2.set_yticklabels([''])
    ax2.set_title("Exactly or one-off correct catch trial response rate")
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)

    ax3 = sns.heatmap(combined_data, cmap=cmap, center=0, annot=True, fmt=".2f", linewidths=0.5, vmin=-2, vmax=2,
                cbar=False)

    # Get all the x-tick labels
    labels = ax3.get_xticklabels()

    # Mark certain labels with another color
    """
    for label in labels:
        if 'P18' in label.get_text():
            label.set_color('blue')
        if 'P19' in label.get_text():
            label.set_color('red')
        if 'P20' in label.get_text():
            label.set_color('red')
        if 'P21' in label.get_text():
            label.set_color('red')
    """

    # Draw horizontal lines to separate stimuli
    for i in range(3, combined_data.shape[0]*3, 3):
        ax3.hlines(i, *ax3.get_xlim(), colors='black', linewidth=1)

    # Adjust the original y-axis labels
    y_labels = combined_data.index.tolist()
    y_positions = range(len(y_labels)+1)
    ax3.set_xlabel("Participant")
    ax3.set_ylabel("Stimulus (target's intensity in cd/m²)")

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
    plt.savefig(f'{target}likert_heatmap.svg')
    plt.close()

    return

def liker_median_responses(df):
    """
    Generate a scatter plot showing the median response for each stimulus.
    """

    # Filter rows based on given intensities
    df_filtered = df.copy()
    df_filtered = df_filtered[~df_filtered['stim'].str.contains('catch')]

    # Group by 'stim' and compute average response
    avg_response = df_filtered.groupby("stim")["response"].median()

    # Sort stimuli by the computed average response
    sorted_stim = avg_response.sort_values(ascending=False).index.tolist()

    # Plotting
    fig, ax = plt.subplots(figsize=(9, 6))

    for i, stim in enumerate(sorted_stim, start=1):
        x = avg_response[stim]
        y = i
        norm_value = (x - (-2)) / (2 - (-2))  # Normalizing between -2 to 2
        color = cmap(norm_value)

        ax.scatter(x, y, s=1500, c=[color], alpha=0.5, label=stim)
        ax.text(x-0.1, y, round(x, 2), va='center')

    ax.set_ylabel("Stimulus")
    ax.set_xlabel("Median response")
    ax.set_yticks(range(1, len(sorted_stim) + 1))
    ax.set_yticklabels(sorted_stim)
    ax.set_xticks(range(-2, 3))
    ax.axvline(x=0, color="black", linestyle="--")

    # Create a second y-axis for stimuli images
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(ax.get_yticks())
    ax2.set_yticklabels(["" for _ in sorted_stim])

    # Adding stimuli images next to y-labels on ax2
    for index, stimulus in enumerate(sorted_stim):
        image = Image.open(f"../experiment/stim/{stimulus}.png")
        imagebox = OffsetImage(image, zoom=0.13)
        ab = AnnotationBbox(imagebox, (2, ax2.get_yticks()[index]), frameon=False, boxcoords="data",
                            pad=0, box_alignment=(-0.05, 0.5))
        ax2.add_artist(ab)

    ax.set_zorder(ax2.get_zorder() + 1)
    plt.tight_layout()
    ax.grid(True, axis='both')
    plt.savefig(f'{target}likert_median_response.svg')
    plt.close()

    return sorted_stim

    return

def liker_distribution(df):
    """
    Display the distribution of responses for each stimulus.
    """

    # Filter data based on intensities
    df_filtered = df.copy()
    df_filtered = df_filtered[~df_filtered['stim'].str.contains('catch')]

    # Colors for the likert scale responses
    color_map = {
        -2: cmap(0.1),
        -1: cmap(0.3),
        0: cmap(0.5),
        1: cmap(0.7),
        2: cmap(0.9)
    }

    # Legend for the responses
    legend = ['Left target is definitely brighter', 'Left target is maybe brighter', 'Targets are equally bright',
              'Right target is maybe brighter', 'Right target is definitely brighter']

    # Aggregate data
    grouped_data = df_filtered.groupby(['stim', 'response']).size().unstack(fill_value=0)


    sorted_data = grouped_data.sort_values(by=list(color_map.keys()), ascending=True)

    # Bar chart for visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    y_labels = sorted_data.index.tolist()
    bar_width = 0.6
    bottoms = np.zeros(len(y_labels))

    for i, likert_value in enumerate(sorted_data.columns):
        color = color_map.get(likert_value, 'white')
        counts = sorted_data[likert_value]
        bars = ax.barh(np.arange(len(y_labels)), counts, color=color, label=legend[int(likert_value + 2)],
                       height=bar_width, left=bottoms)

        # Labeling bars
        for bar, value in zip(bars, counts):
            if value != 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, str(value), ha='center',
                        va='center', color='black', fontsize=10)

        bottoms += counts  # Update for stacked bar's position

    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)

    # Now set the y-ticks and labels
    ax2 = ax.twinx()
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    y_position_stim = range(0, (len(y_labels)+1))
    ax2.set_yticks(y_position_stim)
    ax2.set_yticklabels(["                       " for _ in y_position_stim])

    # Calculate the maximum x-value from the data
    max_x_value = sorted_data.sum(axis=1).max()

    # Adding stimuli images next to y-labels
    for index, stimulus in enumerate(y_labels):
        image = Image.open(f"../experiment/stim/{stimulus}.png")
        imagebox = OffsetImage(image, zoom=0.12)
        ab = AnnotationBbox(imagebox, (max_x_value, y_position_stim[index]), frameon=False, boxcoords="data",
                            box_alignment=(-0.05, -0.3), pad=0)
        ax2.add_artist(ab)

    # Set the x-axis limits to end at
    ax.set_xlim(right=max_x_value)
    ax.set_xlabel("Number of responses from participants")
    ax.get_xaxis().set_visible(False)

    # Set y-axis limits
    ax.set_ylim(-0.3, len(y_labels) - 0.7)

    plt.ylabel("Stimulus")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout()
    plt.savefig(f'{target}likert_distribution.svg')
    plt.close()
    return

def iqr2(df):
    # Function to compute IQR
    def compute_iqr(data, column_name):
        return data.groupby('stim')[column_name].agg(lambda x: round((x.quantile(0.75) - x.quantile(0.25)), 2))

    # Filter out rows that contain 'catch' in 'stim'
    df_copy = df.copy()
    df_copy = df_copy[~df_copy['stim'].str.contains('catch')]

    # Compute IQR for 'matching'
    matching_df = df_copy[df_copy['trial'].str.contains('matching')]
    matching_iqr = compute_iqr_for_matching(matching_df).rename('iqr_matching')

    # Compute IQR for 'direction'
    direction_df = df_copy[df_copy['trial'].str.contains('direction')]
    direction_iqr = compute_iqr(direction_df, 'response')

    # Merge the two IQR values on 'stim'
    result = pd.DataFrame(matching_iqr).rename(columns={'intensity_match': 'iqr_matching'})
    result['iqr_direction'] = direction_iqr
    result = result.reset_index()
    result = result.sort_values(by='iqr_matching')

    print(result)


def compute_iqr_for_matching(data):
    # Store differences for each combination of stim and presented_intensity
    differences = []

    for _, group in data.groupby(['participant', 'stim', 'presented_intensity']):
        if group.shape[0] == 2:  # only consider if there are two dots to connect
            left_val = group[group['target_side'] == 'Left']['intensity_match'].values[0]
            right_val = group[group['target_side'] == 'Right']['intensity_match'].values[0]
            diff = right_val - left_val
            differences.append({'stim': group['stim'].values[0],
                                'presented_intensity': group['presented_intensity'].values[0],
                                'difference': diff})

    # Create DataFrame from differences
    diff_df = pd.DataFrame(differences)


    # Calculate IQR for each combination of stim and presented_intensity
    iqr_df = diff_df.groupby(['stim'])['difference'].agg(lambda x: round((x.quantile(0.75) - x.quantile(0.25)), 2))
    return iqr_df


def participant_oriented(df):
    # Separate out "direction" and "matching" trials
    df_copy = df[~df['stim'].str.contains('catch')]
    direction_df = df_copy[df_copy['trial'].str.contains('direction')]
    matching_df = df_copy[df_copy['trial'].str.contains('matching')]

    # For direction trials, group by the necessary columns and compute the median
    direction_agg = (
        direction_df.groupby(['participant', 'stim', 'presented_intensity'])['response']
        .median()
        .reset_index()
        .rename(columns={'response': 'likert_median'})
    )

    # For matching trials, process the data to get 'match_left' and 'match_right' columns
    matching_df['match_left'] = matching_df.apply(
        lambda x: x['intensity_match'] if x['target_side'] == 'Left' else None, axis=1)
    matching_df['match_right'] = matching_df.apply(
        lambda x: x['intensity_match'] if x['target_side'] == 'Right' else None, axis=1)

    # Aggregate the matching dataframe
    matching_agg = (
        matching_df.groupby(['participant', 'stim', 'presented_intensity'])
        .agg(match_left=('match_left', 'first'), match_right=('match_right', 'first'))
        .reset_index()
    )

    # Merge direction_agg and matching_agg
    final_df = pd.merge(direction_agg, matching_agg, on=['participant', 'stim', 'presented_intensity'], how='outer')

    # Calculate 'match_diff' column
    final_df['match_diff'] = final_df['match_right'] - final_df['match_left']


    final_df.to_csv('all_participants.results.csv', index=False)
    return final_df

def participant_oriented_combined(df):
    # Separate out "direction" and "matching" trials
    df_copy = df[~df['stim'].str.contains('catch')]
    direction_df = df_copy[df_copy['trial'].str.contains('direction')]
    matching_df = df_copy[df_copy['trial'].str.contains('matching')]

    # For direction trials, group by the necessary columns and compute the median
    direction_agg = (
        direction_df.groupby(['participant', 'stim'])['response']
        .median()
        .reset_index()
        .rename(columns={'response': 'likert_median'})
    )

    # For matching trials, process the data to get 'match_left' and 'match_right' columns
    matching_df.loc[matching_df['target_side'] == 'Left', 'match_left'] = matching_df['intensity_match']
    matching_df.loc[matching_df['target_side'] == 'Right', 'match_right'] = matching_df['intensity_match']

    # Aggregate the matching dataframe to compute the average
    matching_agg = (
        matching_df.groupby(['participant', 'stim'])
        .agg(match_left=('match_left', 'mean'), match_right=('match_right', 'mean'))
        .reset_index()
    )

    # Merge direction_agg and matching_agg
    final_df = pd.merge(direction_agg, matching_agg, on=['participant', 'stim'], how='outer')

    # Calculate 'match_diff' column
    final_df['match_diff'] = final_df['match_right'] - final_df['match_left']

    final_df.to_csv('all_participants.results.csv', index=False)
    return final_df

def average_direction_both_methods(df_participants, cmap):
    # Ensure no NaN values are in the dataframe
    df_participants = df_participants.dropna()

    # Reshape dataframe to make it suitable for the heatmap
    df_melted_likert = df_participants.melt(id_vars=['participant', 'stim'], value_vars=['likert_median'],
                               var_name='method', value_name='value')
    df_melted_likert['method'] = 'brightness_ratings'

    df_melted_match = df_participants.melt(id_vars=['participant', 'stim'], value_vars=['match_diff'],
                              var_name='method', value_name='value')
    df_melted_match['method'] = 'method_of_adjustment'

    df_melted = pd.concat([df_melted_likert, df_melted_match])

    # Convert values to categorical representation
    df_melted['value_cat'] = df_melted['value'].apply(lambda x: -1 if x < 0 else (1 if x > 0 else 0))

    # Pivot the data to get a matrix format suitable for heatmap
    df_pivot = df_melted.pivot_table(index=['participant'], columns=['stim', 'method'], values='value_cat')

    # Sorting Rows based on the sum of their values
    df_pivot = df_pivot.reindex(df_pivot.sum(axis=1).sort_values().index)

    # Calculate average values for each 'stim' across both methods
    avg_values = df_pivot.mean(axis=0).unstack().mean(axis=1)

    # Sort 'stim' based on the average values
    sorted_stims = avg_values.sort_values().index.tolist()

    # Order columns based on sorted 'stim' values and methods
    ordered_cols = [(stim, method) for stim in sorted_stims for method in
                    ['brightness_ratings', 'method_of_adjustment']]
    df_pivot = df_pivot[ordered_cols]

    # Extract colors from the diverging_palette
    original_cmap = sns.diverging_palette(250, 10, as_cmap=True)
    blue = original_cmap(0)
    red = original_cmap(255)

    # Custom colormap
    colors = [blue, 'grey', red]
    cmap = ListedColormap(colors)

    # Plot the heatmap
    plt.figure(figsize=(10, 10))
    ax = sns.heatmap(df_pivot, annot=False, cmap=cmap, center=0, cbar=False)

    # Add stimuli images at the bottom
    ax2 = ax.twinx()
    ax2.tick_params(top=False, labeltop=False, left=False, labelleft=False, right=False, labelright=False, bottom=False, labelbottom=False)

    # Add vertical lines every two columns
    for i in range(0, df_pivot.shape[1], 2):
        ax.axvline(i + 2, color='black', lw=1)

        # Add stimuli fig
        image = Image.open(f"../experiment/stim/{sorted_stims[i//2]}.png")
        imagebox = OffsetImage(image, zoom=0.1)  # Adjust the zoom factor as needed
        ab = AnnotationBbox(imagebox, (i+1,1), box_alignment=(0.5, -0.1), frameon=False)
        ax2.add_artist(ab)

    legend_patches = [
        Patch(color=blue, label="Left average direction of effect"),
        Patch(color='grey', label="No average direction of effect"),
        Patch(color=red, label="Right average direction of effect")
    ]

    # Add the legend
    #ax.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)
    ax.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, 1.07), ncol=3)

    ax.set_ylabel("Participant")
    ax.set_xlabel("Stimulus - Method")
    plt.tight_layout()
    plt.savefig(f'{target}average_direction_both_methods.svg')
    plt.show()

if __name__ == "__main__":

    # discard participants from analysis
    #skip_participants = []
    skip_participants = ["AA", "KP", "AH"]

    # invert likert responses for these participants
    #invert_likert_response_participants = []
    invert_likert_response_participants = ["SP"]

    # Gather all data for the analysis
    df = combine_multiple_csv(skip_participants, invert_likert_response_participants)
    intensities = sorted(df['presented_intensity'].unique())

    # Create common colormap
    cmap = sns.diverging_palette(250, 10, as_cmap=True)

    # directory to save all plots
    target = "../plots/"

    """
    START PLOTTING HERE
    Remove '#' when needed
    """

    """
    #matching_mean_adjustments() # Done
    #matching_absolute_differences() # Done
    #matching_scatterplot(df) # Done
    #matching_connected_scatterplot_colored_lines_2(df)
    """

    """
    liker_median_responses(df) # Done
    liker_distribution(df) # Done
    liker_heatmap(df, cmap, target) # Done
    """


    #iqr2(df)

    #convert to participant oriented
    #participants_df = participant_oriented(df)
    #participants_df = participant_oriented_combined(df)
    #average_direction_both_methods(participants_df, cmap)
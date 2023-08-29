import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image, ImageOps


def avg_response_per_stimulus(df, intensities, cmap, target, order=None):
    """
    Generate a scatter plot showing the average response for each stimulus.

    Parameters:
    - df: DataFrame containing the data
    - intensities: List of intensities to filter by
    - cmap : common colormap
    - target : target path
    - order : order of stimuli
    """

    # Filter rows based on given intensities
    df_filtered = df[df['presented_intensity'].isin(intensities)].copy()

    # Group by 'stim' and compute average response
    avg_response = df_filtered.groupby("stim")["response"].mean()

    if order:
        # Use existing order
        sorted_stim = order
    else:
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
    ax.set_xlabel("Average Response")
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
        image = Image.open(f"../../experiment/stim/{stimulus}.png")
        if stimulus == "sbc":
            image = ImageOps.mirror(image)
        imagebox = OffsetImage(image, zoom=0.13)
        ab = AnnotationBbox(imagebox, (2, ax2.get_yticks()[index]), frameon=False, boxcoords="data",
                            pad=0, box_alignment=(-0.05, 0.5))
        ax2.add_artist(ab)

    ax.set_zorder(ax2.get_zorder() + 1)
    plt.tight_layout()
    ax.grid(True, axis='both')
    plt.savefig(f'{target}likert_avg_response_per_stimulus_{intensities}.png')
    plt.close()

    return sorted_stim


def avg_response_per_stimulus_combined(df, multi_intensities, cmap, target, order=None):
    """
    Generate a scatter plot showing the average response for each stimulus for combined intensities.

    Parameters:
    - df: DataFrame containing the data
    - multi_intensities: List of lists of intensities to filter by
    - cmap : common colormap
    - target : target path
    - order : order of stimuli
    """

    # Plotting
    fig, ax = plt.subplots(figsize=(9, 12))

    # Determine color normalization bounds
    vmin = -2
    vmax = 2

    # Placeholder to keep track of y position
    y_pos = 0

    for intensity_set in multi_intensities:
        # Filter rows based on given intensities
        df_filtered = df[df['presented_intensity'].isin(intensity_set)].copy()

        # Group by 'stim' and compute average response
        avg_response = df_filtered.groupby("stim")["response"].mean()

        if order:
            # Use existing order
            sorted_stim = order
        else:
            # Sort stimuli by the computed average response
            sorted_stim = avg_response.sort_values(ascending=False).index.tolist()

        for stim in sorted_stim:
            x = avg_response[stim]
            y = y_pos
            norm_value = (x - vmin) / (vmax - vmin)
            color = cmap(norm_value)

            ax.scatter(x, y, s=1500, c=[color], alpha=1, label=f"{stim} ({intensity_set[0]})")
            ax.text(x-0.1, y-0.1, round(x, 2))

            y_pos += 1

    ax.set_ylabel("Stimulus")
    ax.set_xlabel("Average Response")
    ax.set_xticks(range(-2, 3))
    ax.axvline(x=0, color="black", linestyle="--")
    ax.set_yticks(range(len(sorted_stim) * len(multi_intensities)),
              [f"{stim} ({intensity_set[0]})" for intensity_set in multi_intensities for stim in sorted_stim])
    ax.set_ylim(-0.5, len(sorted_stim) * len(multi_intensities) - 0.5)

    # Create a second y-axis for stimuli images
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(range(y_pos))
    ax2.set_yticklabels(["" for _ in order * len(multi_intensities)])

    # Adding stimuli images next to y-labels on ax2
    for index, stimulus in enumerate(order * len(multi_intensities)):
        image = Image.open(f"../../experiment/stim/{stimulus}.png")
        if stimulus == "sbc":
            image = ImageOps.mirror(image)
        imagebox = OffsetImage(image, zoom=0.1)
        ab = AnnotationBbox(imagebox, (2, ax2.get_yticks()[index]), frameon=False, boxcoords="data",
                            pad=0, box_alignment=(-0.05, 0.5))
        ax2.add_artist(ab)

    plt.tight_layout()
    ax.grid(True)
    plt.savefig(f'{target}likert_avg_response_per_stimulus_combined_ver2_{multi_intensities}.png')
    plt.close()


def responses_on_heatmap(df, intensities, cmap, target, order=None):
    """
    Generate a heatmap illustrating the average response from each participant for each stimulus.

    Parameters:
    - df: DataFrame containing the data
    - intensities: List of intensities to filter by
    - cmap : common colormap
    - target : target path
    - order : order of stimuli
    """

    # Filter and preprocess the data
    df_filtered = df[df['presented_intensity'].isin(intensities)].copy()
    df_filtered['participant'] = df_filtered['trial'].str[:2]

    # Compute the average response for each participant
    avg_responses = df_filtered.groupby('participant')['response'].mean().sort_values()

    # Generate the new participant labels based on the sorted order
    participant_mapping = {participant: f"s{i}-{participant}" for i, participant in enumerate(avg_responses.index)}

    df_filtered['participant_num'] = df_filtered['participant'].map(participant_mapping)
    pivot_data = df_filtered.pivot_table(index='stim', columns='participant_num', values='response', aggfunc='mean')

    # Sort columns of pivot_data for the heatmap x-axis
    pivot_data = pivot_data[sorted(pivot_data.columns, key=lambda x: int(x.split("-")[0][1:]))]

    if order:
        pivot_data = pivot_data.reindex(order).iloc[::-1]

    # Create the heatmap
    plt.figure(figsize=(12, 6))
    ax = sns.heatmap(pivot_data, cmap=cmap, center=0, annot=True, fmt=".2f", linewidths=0.5, vmin=-2, vmax=2, cbar=False)

    # Adjust the color bar ticks and labels
    """
    color_bar = ax.collections[0].colorbar
    color_bar.set_ticks([-2, -1, 0, 1, 2])
    color_bar.set_ticklabels(['-2: Left target is definitely brighter', '-1: Left target is maybe brighter',
                              '0: Targets are equally bright', '1: Right target is maybe brighter',
                              '2: Right target is definitely brighter'])
    """

    # Adjust the original y-axis labels
    y_labels = pivot_data.index.tolist()
    y_positions = range(len(y_labels)+1)
    ax.set_xlabel("Subject")
    ax.set_ylabel("Stimulus")

    # Create a second y-axis for stimuli images
    ax2 = ax.twinx()
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels(["                    " for _ in y_positions])

    # Adding stimuli images next to y-labels on ax2
    for index, stimulus in enumerate(y_labels[::-1]):
        stimulus_name = stimulus.split(" ")[0]  # Assuming the format is "stim (intensity)"
        image = Image.open(f"../../experiment/stim/{stimulus_name}.png")
        if stimulus_name == "sbc":
            image = ImageOps.mirror(image)
        imagebox = OffsetImage(image, zoom=0.13)
        ab = AnnotationBbox(imagebox, (pivot_data.shape[1], y_positions[index]), frameon=False,
                            boxcoords="data", box_alignment=(-0.05, -0.05), pad=0)
        ax2.add_artist(ab)

    plt.tight_layout()
    plt.savefig(f'{target}likert_heatmap_{intensities}.png')
    plt.close()


def responses_on_heatmap_combined(df, multi_intensities, cmap, target, order=None):
    """
    Generate a heatmap illustrating the average response from each participant for each stimulus
    and presented intensity combined.

    Parameters:
    - df: DataFrame containing the data
    - multi_intensities: List of List of intensities to filter by
    - cmap : common colormap
    - target : target path
    - order (optional): order of stimuli
    """

    # Filter and preprocess the data
    df['participant'] = df['trial'].str[:2]
    unique_participants = df['participant'].unique()
    participant_mapping = {participant: f"s{i}-{participant}" for i, participant in enumerate(unique_participants)}
    df['participant_num'] = df['participant'].map(participant_mapping)

    # Combine intensities for the pivot
    concatenated = []
    for intensity_set in multi_intensities:
        df_filtered = df[df['presented_intensity'].isin(intensity_set)].copy()
        df_filtered['combined_stim'] = df_filtered['stim'] + ' (' + df_filtered['presented_intensity'].astype(str) + ')'
        pivot_data = df_filtered.pivot_table(index='combined_stim', columns='participant_num', values='response', aggfunc='mean')
        concatenated.append(pivot_data)

    combined_data = pd.concat(concatenated)

    # Compute the average response for each participant
    avg_responses = combined_data.mean().sort_values()

    # Reorder the columns (participants) of the `combined_data` based on the computed average responses
    combined_data = combined_data[avg_responses.index]

    # Map old labels to correct sequence
    label_order = sorted(participant_mapping.values(), key=lambda x: int(x.split("-")[0][1:]))
    rename_mapping = {old: new for old, new in zip(avg_responses.index, label_order)}
    combined_data = combined_data.rename(columns=rename_mapping)

    if order:
        order = order[::-1]
        combined_order = [f"{stim} ({intensity})" for stim in order for intensity in np.concatenate(multi_intensities)]
        combined_data = combined_data.reindex(combined_order)

    # Create the heatmap
    plt.figure(figsize=(12, 12))
    ax = sns.heatmap(combined_data, cmap=cmap, center=0, annot=True, fmt=".2f", linewidths=0.5, vmin=-2, vmax=2,
                cbar=False)

    # Draw horizontal lines to separate stimuli
    for i in range(3, combined_data.shape[0]*3, 3):
        ax.hlines(i, *ax.get_xlim(), colors='black', linewidth=1)

    # Adjust the original y-axis labels
    y_labels = combined_data.index.tolist()
    y_positions = range(len(y_labels)+1)
    ax.set_xlabel("Subject")
    ax.set_ylabel("Stimulus")

    # Create a second y-axis for stimuli images
    ax2 = ax.twinx()
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels(["                                " for _ in y_positions])

    # Adding stimuli images next to y-labels on ax2
    for index, stimulus in enumerate(y_labels[::-1]):
        if (index+1) % 3 == 2:
            stimulus_name = stimulus.split(" ")[0]  # Assuming the format is "stim (intensity)"
            image = Image.open(f"../../experiment/stim/{stimulus_name}.png")
            if stimulus_name == "sbc":
                image = ImageOps.mirror(image)
            imagebox = OffsetImage(image, zoom=0.18)
            ab = AnnotationBbox(imagebox, (combined_data.shape[1], index), frameon=False,
                                boxcoords="data", box_alignment=(-0.05, 0.15), pad=0)
            ax2.add_artist(ab)

    plt.tight_layout()
    plt.savefig(f'{target}likert_heatmap_combined_{multi_intensities}.png')
    plt.close()


def response_distribution(df, intensities, cmap, target, order=None):
    """
    Display the distribution of responses for each stimulus.

    Parameters:
    - df: DataFrame containing the data
    - intensities: List of intensities to filter by
    - cmap : common colormap
    - target : target path
    - order : order of stimuli
    """

    # Filter data based on intensities
    df_filtered = df[df['presented_intensity'].isin(intensities)].copy()

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

    if order:
        sorted_data = grouped_data.reindex(order)
    else:
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
    ax.get_xaxis().set_visible(False)
    plt.ylabel("Stimulus")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0))
    plt.tight_layout()
    plt.savefig(f'{target}likert_response_distribution_{intensities}.png')
    plt.close()


def response_distribution_combined(df, multi_intensities, cmap, target):
    """
    Display the distribution of responses for each stimulus and intensities combined

    Parameters:
    - df: DataFrame containing the data
    - multi_intensities: List of intensities to filter by
    - cmap : common colormap
    - target : target path
    """

    color_map = {
        -2: cmap(0.1),
        -1: cmap(0.3),
        0: cmap(0.5),
        1: cmap(0.7),
        2: cmap(0.9)
    }

    legend = ['Left target is definitely brighter', 'Left target is maybe brighter', 'Targets are equally bright',
              'Right target is maybe brighter', 'Right target is definitely brighter']

    fig, ax = plt.subplots(figsize=(12, 15))
    ax2 = ax.twinx()

    y_labels = None
    y_positions = []
    y_tick_labels = []
    intensity_number = 0

    group = df.groupby(['stim', 'response']).size().unstack(fill_value=0)
    order = group.sort_values(by=list(color_map.keys()), ascending=True).index.tolist()

    # Iterate over each intensity set and plot
    for intensity_set in multi_intensities:
        df_filtered = df[df['presented_intensity'].isin(intensity_set)].copy()
        grouped_data = df_filtered.groupby(['stim', 'response']).size().unstack(fill_value=0)
        sorted_data = grouped_data.reindex(order)

        if y_labels is None:
            y_labels = sorted_data.index.tolist()

        # Append y-tick positions and labels for each intensity
        for i, label in enumerate(y_labels):
            y_positions.append(i + intensity_number)
            y_tick_labels.append(f"{label} ({intensity_set[0]*100})")

        bottoms = np.zeros(len(y_labels))

        for i, likert_value in enumerate(sorted_data.columns):
            color = color_map.get(likert_value, 'white')
            counts = sorted_data[likert_value]
            if intensity_number == 0:
                bars = ax.barh(np.arange(len(y_labels)) + intensity_number, counts, color=color,
                               label=legend[int(likert_value + 2)], height=0.3, left=bottoms)
            else:
                bars = ax.barh(np.arange(len(y_labels)) + intensity_number, counts, color=color,
                               height=0.3, left=bottoms)

            for bar, value in zip(bars, counts):
                if value != 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, str(value),
                            ha='center',
                            va='center', color='black', fontsize=10)

            bottoms += counts
        intensity_number = intensity_number + 0.31

    # Now set the y-ticks and labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_tick_labels)
    y_position_stim = range(0, (len(y_tick_labels)+1), 3)
    ax2.set_yticks(y_position_stim)
    ax2.set_yticklabels(["                                         " for _ in y_position_stim])

    # Adding stimuli images next to y-labels
    for index, stimulus in enumerate(y_labels):
        image = Image.open(f"../../experiment/stim/{stimulus}.png")
        if stimulus == "sbc":
            image = ImageOps.mirror(image)
        imagebox = OffsetImage(image, zoom=0.25)
        ab = AnnotationBbox(imagebox, (26, y_position_stim[index]), frameon=False, boxcoords="data",
                            box_alignment=(-0.05, -0.3), pad=0)
        ax2.add_artist(ab)

    # Set y-axis limits
    lower_limit = 0 - 0.5 * 0.3  # Half bar height below the first bar
    upper_limit = len(y_labels) - 1 + 0.5 * 0.3 + (len(multi_intensities) - 1) * 0.31  # Top of the last group of bars
    ax.set_ylim(lower_limit, upper_limit)

    ax.get_xaxis().set_visible(False)
    plt.ylabel("Stimulus")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0))
    plt.tight_layout()
    plt.savefig(f'{target}likert_response_distribution_combined{multi_intensities}.png')
    plt.close()


def main(source="../format_correction/merge/likert_merged.csv", target=""):

    # Load data
    df = pd.read_csv(source)

    # Create common colormap
    cmap = sns.diverging_palette(250, 10, as_cmap=True)

    # Specified variations of intensities
    intensities_variation = [[0.49, 0.5, 0.51], [0.49], [0.5], [0.51]]

    # Process each variation

    for intensities in intensities_variation:
        # Scatterplot; Average response per stimulus
        if intensities == [0.49, 0.5, 0.51]:
            # Determine order that is going to be used for all other plots
            order = avg_response_per_stimulus(df, intensities, cmap, target)
        else:
            avg_response_per_stimulus(df, intensities, cmap, target, order)

    # Scatterplot; for separate intensities combined in one chart
    avg_response_per_stimulus_combined(df, [[0.49], [0.5], [0.51]], cmap, target, order)

    for intensities in intensities_variation:
        # Heatmap; average response per participant per stimulus
        responses_on_heatmap(df, intensities, cmap, target, order)

    responses_on_heatmap_combined(df, [[0.49], [0.5], [0.51]], cmap, target, order)

    for intensities in intensities_variation:
        # Discrete distribution as horizontal bar chart
        if intensities == [0.49, 0.5, 0.51]:
            # Determine order that is going to be used for all other plots
            order = response_distribution(df, intensities, cmap, target)
        else:
            response_distribution(df, intensities, cmap, target, order)

    # Discrete distribution as horizontal bar chart for separate intensities combined in one chart
    response_distribution_combined(df, [[0.49], [0.5], [0.51]], cmap, target)


if __name__ == "__main__":
    main()

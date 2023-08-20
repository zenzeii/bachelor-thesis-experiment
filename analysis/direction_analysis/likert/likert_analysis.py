import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def avg_response_per_stimulus(df, intensities):
    """
    Generate a scatter plot showing the average response for each stimulus.

    Parameters:
    - df: DataFrame containing the data
    - intensities: List of intensities to filter by
    """

    # Filter rows based on given intensities
    df_filtered = df[df['presented_intensity'].isin(intensities)]

    # Group by 'stim' and compute average response
    avg_response = df_filtered.groupby("stim")["response"].mean()

    # Sort stimuli by the computed average response
    sorted_stim = avg_response.sort_values(ascending=False).index.tolist()

    # Define a red-blue gradient colormap
    cmap = sns.diverging_palette(250, 10, as_cmap=True)

    # Determine color normalization bounds
    vmin = avg_response.min()
    vmax = avg_response.max()

    # Plotting
    plt.figure(figsize=(10, 6))
    for i, stim in enumerate(sorted_stim, start=1):
        x = avg_response[stim]
        y = i
        norm_value = (x - vmin) / (vmax - vmin)
        color = cmap(norm_value)

        plt.scatter(x, y, s=2000, c=[color], alpha=0.5, label=stim)
        plt.text(x - (0.04), y - (0.06), round(x, 2))

    plt.ylabel("Stimulus")
    plt.xlabel("Average Response")
    plt.title(f"Average Response per Stimulus for intensities: {intensities}")
    plt.yticks(range(1, len(sorted_stim) + 1), sorted_stim)
    plt.xticks(range(1, 6), range(1, 6))
    plt.axvline(x=3, color="black", linestyle="--", label="Threshold")
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(f'avg_response_per_stimulus_{intensities}.png')


def avg_response_per_participant(df, intensities):
    """
    Generate a heatmap illustrating the average response for each participant and stimulus.

    Parameters:
    - df: DataFrame containing the data
    - intensities: List of intensities to filter by
    """

    # Filter and preprocess the data
    df_filtered = df[df['presented_intensity'].isin(intensities)].copy()
    df_filtered['participant'] = df_filtered['trial'].str[:2]
    pivot_data = df_filtered.pivot_table(index='stim', columns='participant', values='response', aggfunc='mean')

    # Define colormap
    cmap = sns.diverging_palette(250, 10, as_cmap=True)

    # Create the heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot_data, cmap=cmap, center=3, annot=True, fmt=".2f", linewidths=0.5)
    plt.title(f"Average Response Heatmap for intensities: {intensities}")
    plt.xlabel("Participant")
    plt.ylabel("Stimulus")
    plt.tight_layout()
    plt.savefig(f'avg_response_per_participant_{intensities}.png')


def response_distribution(df, intensities):
    """
    Display the distribution of responses for each stimulus.

    Parameters:
    - df: DataFrame containing the data
    - intensities: List of intensities to filter by
    """

    # Filter data based on intensities
    df_filtered = df[df['presented_intensity'].isin(intensities)].copy()

    # Colors for the likert scale responses
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    color_map = {
        1: cmap(0.1),
        2: cmap(0.4),
        3: cmap(0.5),
        4: cmap(0.6),
        5: cmap(0.9)
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
        bars = ax.barh(np.arange(len(y_labels)), counts, color=color, label=legend[int(likert_value) - 1],
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
    ax.set_title(f'Distribution of Responses for Each Stimulus with Intensities: {intensities}')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0))
    plt.tight_layout()
    plt.savefig(f'response_distribution_{intensities}.png')


if __name__ == "__main__":
    # Load data
    df = pd.read_csv("../../merge/likert_merged.csv")

    # Specified variations of intensities
    intensities_variation = [[0.49, 0.5, 0.51], [0.49], [0.5], [0.51]]

    # Process each variation
    for intensities in intensities_variation:
        avg_response_per_stimulus(df, intensities)
        avg_response_per_participant(df, intensities)
        response_distribution(df, intensities)

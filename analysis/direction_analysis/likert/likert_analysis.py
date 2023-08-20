import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def avg_response(df, intensities):
    # Filter rows with intensities
    df = df[(df['presented_intensity']).isin(intensities)]

    # Group the data by 'stim' and calculate the average response
    avg_response = df.groupby("stim")["response"].mean()

    # Sort stimuli by average response
    sorted_stim = avg_response.sort_values().index.tolist()

    # Create a scatter plot with circles
    plt.figure(figsize=(10, 6))
    for i, stim in enumerate(sorted_stim, start=1):
        x = avg_response[stim]
        y = i
        color = "red" if x > 3 else "blue"
        plt.scatter(x, y, s=2000, c=color, alpha=0.5, label=stim)
        plt.text(x-(0.04), y-(0.06), round(x, 2))

    plt.ylabel("Stimulus")
    plt.xlabel("Average Response")
    plt.title("Average Response for intensities: " + str(intensities))
    plt.yticks(range(1, len(sorted_stim) + 1), sorted_stim)
    plt.xticks(range(1, 6), range(1, 6))
    plt.axvline(x=3, color="black", linestyle="--", label="Threshold")
    plt.tight_layout()
    plt.grid(True)

    # Show the plot
    plt.savefig(f'avg_response_overall_{intensities}.png')

def avg_response_per_participant(df, intensities):
    # Filter rows with intensities
    df = df[df['presented_intensity'].isin(intensities)].copy()

    # Extract participant ID from trial
    df['participant'] = df['trial'].str[:2]

    # Pivot the data to get average response values for each participant and stimulus
    pivot_data = df.pivot_table(index='stim', columns='participant', values='response', aggfunc='mean')

    # Define a colormap for red-blue gradient
    cmap = sns.diverging_palette(250, 10, as_cmap=True)

    # Create the heatmap
    plt.figure(figsize=(10, 6))
    heatmap = sns.heatmap(pivot_data, cmap=cmap, center=3, annot=True, fmt=".2f", linewidths=0.5)

    # Set heatmap title and labels
    plt.title("Average Response Heatmap for intensities: " + str(intensities))
    plt.xlabel("Participant")
    plt.ylabel("Stimulus")

    # Show the plot
    plt.savefig(f'avg_response_per_participant_{intensities}.png')


def avg_response_distribution(df, intensities):
    # Filter rows with intensities
    df = df[df['presented_intensity'].isin(intensities)].copy()

    # Define a colormap for red-blue gradient
    cmap = sns.diverging_palette(250, 10, as_cmap=True)

    # Define the color map based on likert_flipped values
    color_map = {
        1: cmap(0.1),  # Far start of the gradient for extreme values
        2: cmap(0.4),
        3: cmap(0.5),  # Middle of the gradient
        4: cmap(0.6),
        5: cmap(0.9)   # Pick colors from the far end of the gradient for extreme values
    }

    legend = ['Left target is definitely brighter', 'Left target is maybe brighter', 'Targets are equally bright', 'Right target is maybe brighter', 'Right target is definitely brighter']

    # Process the data to group and aggregate by stim and response
    grouped_data = df.groupby(['stim', 'response']).size().unstack(fill_value=0)

    # Sort the data by response values
    sorted_data = grouped_data.sort_values(by=list(color_map.keys()), ascending=True)

    # Create a horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 8))

    y_labels = sorted_data.index.tolist()

    bar_width = 0.6
    bottoms = np.zeros(len(y_labels))  # To keep track of the bottom positions for stacking

    for i, likert_value in enumerate(sorted_data.columns):
        color = color_map.get(likert_value, 'white')
        counts = sorted_data[likert_value]
        bars = ax.barh(np.arange(len(y_labels)), counts, color=color, label=legend[int(likert_value)-1], height=bar_width,
                       left=bottoms)

        for bar, value in zip(bars, counts):
            if value != 0:  # Only label bars that have a count
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, str(value),
                        ha='center', va='center', color='black', fontsize=10)

        bottoms += counts  # Update the bottoms for the next iteration

    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.get_xaxis().set_visible(False)
    ax.set_title('Distribution of Responses for Each Stimulus')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0))

    plt.tight_layout()

    # Show the plot
    plt.savefig(f'avg_response_distribution_{intensities}.png')


if __name__ == "__main__":

    # Read the CSV file
    df = pd.read_csv("../../merge/likert_merged.csv")

    intensities_variation = [[0.49, 0.5, 0.51], [0.49], [0.5], [0.51]]
    for intensities in intensities_variation:
        avg_response(df, intensities)
        avg_response_per_participant(df, intensities)
        avg_response_distribution(df, intensities)


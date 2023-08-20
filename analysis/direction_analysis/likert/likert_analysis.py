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

    # Define color mapping based on response values
    color_mapping = {
        5: 'red',
        4: 'lightcoral',
        3: 'lightgrey',
        2: 'lightblue',
        1: 'blue'
    }

    # Process the data to group and aggregate by stim and response
    grouped_data = df.groupby(['stim', 'response']).size().unstack(fill_value=0)

    # Sort the data by response values
    sorted_data = grouped_data.sort_values(by=list(color_mapping.keys()), ascending=True)

    # Plot the horizontal bar graph
    plt.figure(figsize=(10, 8))
    sns.set()

    for idx, (stim, row) in enumerate(sorted_data.iterrows()):
        color = [color_mapping[resp] for resp in row.index]
        plt.barh(idx, row, color=color, edgecolor='black')

    plt.yticks(range(len(sorted_data)), sorted_data.index)
    plt.ylabel("Stimulus")
    plt.xlabel("Number of Trials")
    plt.title("Response Distribution by Stimulus")

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


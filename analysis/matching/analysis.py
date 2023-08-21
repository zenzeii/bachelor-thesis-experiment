import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


def plot_matching_res_to_boxplot(df, intensities, cmap, target):
    """
    Generate a boxplot illustrating the results for each participant and stimulus.

    Parameters:
    - df: DataFrame containing the data
    - intensities: List of intensities to filter by
    - cmap : common colormap
    - target : target path
    """

    # Define y-axis limits
    ymin = df['intensity_match'].min()
    ymax = df['intensity_match'].max()

    # Filter rows based on given intensities
    df_filtered = df[df['presented_intensity'].isin(intensities)].copy()

    # Mapping target_side to colors
    palette_dict_dots = {'Right': cmap(0.99), 'Left': cmap(0.01)}
    palette_dict_box = {'Right': cmap(0.79), 'Left': cmap(0.21)}

    # Group data by 'stim' and 'target_side', then create box plots
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed

    sns.boxplot(x='stim', y='intensity_match', hue='target_side', data=df_filtered, orient='v', palette=palette_dict_box)
    sns.despine(left=True)
    # Add individual data points
    sns.stripplot(x='stim', y='intensity_match', hue='target_side', data=df_filtered, jitter=False,
                  dodge=True, size=3.5, orient='v', palette=palette_dict_dots)

    plt.ylim(ymin, ymax)  # Set y-axis limits
    plt.title(f'Results as Box Plots for Each Stimulus and Target Side With Presented Intensities: {intensities}')
    plt.ylabel('Adjusted luminance by subjects in cd/m²')
    plt.xlabel('Stimulus')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
    plt.tight_layout()
    plt.savefig(f'{target}matching_box_plots_{intensities}.png')
    plt.close()


def avg_adjusted_luminance(df, intensities, cmap, cmap_luminance, target):
    """
    Generate a scatter plot showing the average adjustment from subjects for each stimulus.

    Parameters:
    - df: DataFrame containing the data
    - intensities: List of intensities to filter by
    - cmap : common colormap
    - target : target path
    """
    # Group by 'presented_intensity' and compute the mean for 'intensity_match'
    means_all = df.groupby(['presented_intensity', 'target_side', 'stim'])['intensity_match'].mean()

    # Extract y_min and y_max
    y_min = means_all.min()
    y_max = means_all.max() + 3

    # Filter rows based on given intensities
    df_filtered = df[df['presented_intensity'].isin(intensities)].copy()

    # Calculate means
    means = df_filtered.groupby(['stim', 'target_side'])['intensity_match'].mean().reset_index()

    # Map each unique 'stim' to an index
    stim_to_index = {stim: index for index, stim in enumerate(means['stim'].unique())}

    # Adjust x-values for 'Right' and 'Left' to avoid overlap
    means['x_adjust'] = means.apply(
        lambda row: stim_to_index[row['stim']] + 0.1 if row['target_side'] == 'Right' else stim_to_index[row['stim']] - 0.1,
        axis=1)

    # Mapping target_side to colors
    palette_dict_avg = {'Right': cmap(0.99), 'Left': cmap(0.01)}

    # Create a plot
    plt.figure(figsize=(10, 6))
    ax = sns.scatterplot(x='x_adjust', y='intensity_match', hue='target_side', data=means, palette=palette_dict_avg, s=200, zorder=1)
    sns.despine(left=True)
    ax.set_ylim(y_min, y_max)

    # Add horizontal lines for each intensity value
    for intensity in intensities:
        ax.axhline(y=intensity, color='grey', linestyle='--', alpha=0.6, lw=1.5)

    # Normalize intensity_match values to [0, 1]
    means['normalized_intensity'] = means['intensity_match'] / 100.0

    # Extract colors for each 'stim' based on normalized 'intensity_match' and 'target_side'
    left_colors = means[means['target_side'] == 'Left'].set_index('stim')['normalized_intensity'].map(
        cmap_luminance).to_dict()
    right_colors = means[means['target_side'] == 'Right'].set_index('stim')['normalized_intensity'].map(
        cmap_luminance).to_dict()

    # Draw vertical bars for each 'stim'
    for stim, x_val in stim_to_index.items():
        # Existing vertical bar code
        ax.axvline(x=x_val - 0.1, color=left_colors[stim], ymin=0, ymax=1, lw=10, zorder=0)
        ax.axvline(x=x_val + 0.1, color=right_colors[stim], ymin=0, ymax=1, lw=10, zorder=0)

        # Extract intensity_match values for the current stim and each target side
        left_intensity = means[(means['stim'] == stim) & (means['target_side'] == 'Left')]['intensity_match'].values[0]
        right_intensity = means[(means['stim'] == stim) & (means['target_side'] == 'Right')]['intensity_match'].values[
            0]

        # Add the intensity_match values on the vertical bars
        ax.text(x_val - 0.1, left_intensity + 1.75, f"{left_intensity:.2f}", ha="center", va="center", rotation=90,
                color="white", fontsize=8)
        ax.text(x_val + 0.1, right_intensity + 1.75, f"{right_intensity:.2f}", ha="center", va="center", rotation=90,
                color="white", fontsize=8)

    # Customize the plot appearance
    plt.title(f'Average Adjusted Luminance by Subjects for Each Stimulus and Target Side With Presented Intensities:{intensities}')
    plt.ylabel('Average adjusted luminance by subjects in cd/m²')
    plt.xlabel('Stimulus')
    plt.xticks(rotation=45, ha='right')
    plt.xticks(ticks=range(len(stim_to_index)), labels=stim_to_index.keys())  # Set x-tick labels to 'stim' values
    plt.tight_layout()
    plt.savefig(f'{target}matching_avg_adjusted_luminance_{intensities}.png')
    plt.close()


def adjustments_on_heatmap(df, intensities, cmap, target):
    """
    Generate a heatmap illustrating the (average) adjusted value from each participant and stimulus.

    Parameters:
    - df: DataFrame containing the data
    - intensities: List of intensities to filter by
    - cmap : common colormap
    - target : target path
    """

    # Filter and preprocess the data
    df_filtered = df[df['presented_intensity'].isin(intensities)].copy()
    df_filtered['participant'] = df_filtered['trial'].str[:2]

    # Create a combined column for stim and target_side
    df_filtered['stim_target'] = df_filtered['stim'] + " (" + df_filtered['target_side'] + ")"

    # Pivot data using the combined column
    pivot_data = df_filtered.pivot_table(index='participant', columns='stim_target', values='intensity_match',
                                         aggfunc='mean')

    # Create the heatmap
    plt.figure(figsize=(15, 6))
    ax = sns.heatmap(pivot_data, cmap=cmap, center=50, annot=True, fmt=".2f", linewidths=0.5, vmin=0, vmax=100)

    # Adjust the color bar ticks
    color_bar = ax.collections[0].colorbar
    color_bar.set_ticks([0, 20, 40, 60, 80, 100])

    # Adjust x-tick labels and positions
    unique_stims = df['stim'].unique()
    ax.set_xticks(
        [i for i in range(1, 2 * len(unique_stims), 2)])  # Position ticks between the two squares for each stim
    ax.set_xticklabels(unique_stims, rotation=45, ha='right')  # rotation pivot at the end

    # Add target_side labels on top
    target_sides = df['target_side'].unique()  # Assuming there are two unique sides
    for i in range(0, pivot_data.shape[1], 2):
        ax.text(i+0.5, -0.4, target_sides[1], ha='center', va='center', fontsize=10)
        ax.text(i+1.5, -0.4, target_sides[0], ha='center', va='center', fontsize=10)

    # Add horizontal lines every 2 column
    for i in range(2, pivot_data.shape[1], 2):
        ax.vlines(i, *ax.get_xlim(), colors='white', linewidth=5)  # Draw horizontal lines with specified linewidth and color

    plt.title(f"Average Adjusted Luminance Heatmap (in cd/m²) With Presented Intensities:{intensities}", y=1.08)
    plt.ylabel("Participant")
    plt.xlabel("Stimulus")
    plt.tight_layout()
    plt.savefig(f'{target}matching_heatmap_{intensities}.png')
    plt.close()


def main(source="../format_correction/merge/matching_merged.csv", target=""):

    # Load data
    df = pd.read_csv(source)

    # Filter out rows with 'catch_trial' in 'stim' column
    df = df[~df['stim'].str.contains('catch_trial')]

    # Convert the 'intensity_match' column to the required format
    df['intensity_match'] = df['intensity_match'] * 100
    df['presented_intensity'] = df['presented_intensity'] * 100

    # Create common colormap
    cmap = sns.diverging_palette(250, 10, as_cmap=True)

    # Create common colormap for luminance
    cmap_luminance = LinearSegmentedColormap.from_list("luminance", ["black", "white"])

    # Specified variations of intensities
    intensities_variation = [[49, 50, 51], [49], [50], [51]]

    # Process each variation
    for intensities in intensities_variation:
        plot_matching_res_to_boxplot(df, intensities, cmap, target)    # Boxplot; showing (avg) adjustment value from subjects
        avg_adjusted_luminance(df, intensities, cmap, cmap_luminance, target)   # Scatterplot; Average adjustment per stimulus
        adjustments_on_heatmap(df, intensities, cmap_luminance, target)    # Heatmap; (avg) adjustment per subject per stimulus


if __name__ == "__main__":
    main()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


def plot_matching_res_to_boxplot(df, intensities, cmap, target, order):
    """
    Generate a boxplot illustrating the results for each participant and stimulus.

    Parameters:
    - df: DataFrame containing the data
    - intensities: List of intensities to filter by
    - cmap : common colormap
    - target : target path
    - order : order of stimuli
    """

    # Define y-axis limits
    ymin = df['intensity_match'].min()
    ymax = df['intensity_match'].max()

    # Filter rows based on given intensities
    df_filtered = df[df['presented_intensity'].isin(intensities)].copy()

    # Mapping target_side to colors
    palette_dict_dots = {'Right': cmap(0.99), 'Left': cmap(0.01)}
    palette_dict_box = {'Right': cmap(0.79), 'Left': cmap(0.21)}

    # Create a plot
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    sns.boxplot(x='stim', y='intensity_match', hue='target_side', data=df_filtered,
                orient='v', palette=palette_dict_box, hue_order=['Left', 'Right'], order=order)
    sns.despine(left=True)

    # Add individual data points
    ax = sns.stripplot(x='stim', y='intensity_match', hue='target_side', data=df_filtered,
                       jitter=True, dodge=True, size=3.5, orient='v', palette=palette_dict_dots,
                       hue_order=['Left', 'Right'], order=order)

    # Add horizontal lines for each intensity value
    for intensity in intensities:
        ax.axhline(y=intensity, color='grey', linestyle='--', alpha=0.6, lw=1.5)

    # Create a proxy artist for the dashed line
    dash_line = plt.Line2D([0], [0], color='grey', linestyle='--', alpha=0.6, lw=1.5, label='Presented Intensity')

    # Set legend
    handles, labels = ax.get_legend_handles_labels()

    # Add the proxy artist for the dashed line to the handles and its label to labels
    handles.append(dash_line)
    labels.append('Presented intensity')

    # Set legend
    labels = [label.replace('Left', 'Left target').replace('Right', 'Right target') for label in labels]
    plt.legend(handles=handles, labels=labels, loc='upper left', ncol=5, bbox_to_anchor=(0, 1.15))

    plt.ylim(ymin, ymax)  # Set y-axis limits
    title_text = f'Results as Box Plots for Each Stimulus and Target Side With Presented Intensities: {intensities}'
    title = ax.set_title(title_text, y=1.2)
    title.set_position((0.44, 1.2))
    plt.ylabel('Adjusted luminance by subjects in cd/m²')
    plt.xlabel('Stimulus')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
    plt.tight_layout()
    plt.savefig(f'{target}matching_box_plots_{intensities}.png')
    plt.close()


def avg_adjusted_luminance(df, intensities, cmap, cmap_luminance, target, order):
    """
    Generate a scatter plot showing the average adjustment from subjects for each stimulus.

    Parameters:
    - df: DataFrame containing the data
    - intensities: List of intensities to filter by
    - cmap : common colormap
    - target : target path
    - order : order of stimuli
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
    means = means[means['stim'].isin(order)]

    # Map each unique 'stim' to an index
    stim_to_index = {stim: index for index, stim in enumerate(order)}

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


def adjustments_on_heatmap(df, intensities, cmap, target, order):
    """
    Generate a heatmap illustrating the (average) adjusted value from each participant and stimulus.

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

    # Create a combined column for stim and target_side
    df_filtered['stim_target'] = df_filtered['stim'] + " (" + df_filtered['target_side'] + ")"

    # Pivot data using the combined column
    pivot_data = df_filtered.pivot_table(index='participant', columns='stim_target', values='intensity_match',
                                         aggfunc='mean')
    column_order = [f"{stim} (Left)" for stim in order] + [f"{stim} (Right)" for stim in order]
    pivot_data = pivot_data.reindex(column_order, axis=1)

    # Create the heatmap
    plt.figure(figsize=(15, 6))
    ax = sns.heatmap(pivot_data, cmap=cmap, center=50, annot=True, fmt=".2f", linewidths=0.5, vmin=0, vmax=100)

    # Adjust the color bar ticks
    color_bar = ax.collections[0].colorbar
    color_bar.set_ticks([0, 20, 40, 60, 80, 100])
    color_bar.remove()

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


def get_stim_order(df):
    """
    Returns stim order from lowest to highest difference between value 'intensity_match' 'target_side=Left' and
    the value intensity_match' 'target_side=Right'

    Parameters:
    - df: DataFrame containing the data
    """

    # Calculate means
    means = df.groupby(['stim', 'target_side'])['intensity_match'].mean().reset_index()

    # Calculate the absolute difference between 'Right' and 'Left' for each stim
    diffs = means.pivot(index='stim', columns='target_side', values='intensity_match')
    diffs['abs_difference'] = (diffs['Right'] - diffs['Left']).abs()

    # Order stims by the absolute difference
    ordered_stims = diffs.sort_values(by='abs_difference').index.tolist()

    return ordered_stims


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

    # Determine stim order
    order = get_stim_order(df)

    # Specified variations of intensities
    intensities_variation = [[49, 50, 51], [49], [50], [51]]

    # Process each variation
    for intensities in intensities_variation:

        # Boxplot; showing (avg) adjustment value from subjects
        plot_matching_res_to_boxplot(df, intensities, cmap, target, order)

        # Scatterplot; Average adjustment per stimulus
        avg_adjusted_luminance(df, intensities, cmap, cmap_luminance, target, order)

        # Heatmap; (avg) adjustment per subject per stimulus
        adjustments_on_heatmap(df, intensities, cmap_luminance, target, order)


if __name__ == "__main__":
    main()

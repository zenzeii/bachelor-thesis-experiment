import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image, ImageOps


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
    #title_text = f'Results as Box Plots for Each Stimulus and Target Side With Presented Intensities: {intensities}'
    #title = ax.set_title(title_text, y=1.2)
    #title.set_position((0.44, 1.2))
    plt.ylabel('Adjusted luminance by participants in cd/m²')
    plt.xlabel('Stimulus')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
    plt.tight_layout()
    plt.savefig(f'{target}matching_box_plots_{intensities}.png')
    plt.close()


def plot_matching_res_to_boxplot_combined(df, intensities, cmap, target, order):
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

    # Combine 'stim' and 'presented_intensity' for the x-axis
    df_filtered['stim_with_intensity'] = df_filtered['stim'] + '_' + df_filtered['presented_intensity'].astype(str).str.split(".").str[0]

    # Update the order list to reflect the new combined labels
    order_updated = [o + '_' + str(intensity).split(".")[0] for o in order for intensity in intensities]
    # Mapping target_side to colors
    palette_dict_dots = {'Right': cmap(0.99), 'Left': cmap(0.01)}
    palette_dict_box = {'Right': cmap(0.79), 'Left': cmap(0.21)}

    # Create a plot
    plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
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
        image = Image.open(f"../../experiment/stim/{stim}.png")
        if stim == "sbc":
            image = ImageOps.mirror(image)
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

    xlables = intensities*9

    ax.set_ylim(ymin, ymax)  # Set y-axis limits
    ax.set_ylabel('Adjusted luminance by participants in cd/m²')
    ax.set_xlabel('Stimulus')
    ax.get_legend().remove()
    ax.set_xticks(ticks=range(len(xlables)), labels=xlables, rotation=0)
    ax.set_yticks(ticks=range(0, 101, 5), labels=range(0, 101, 5), rotation=0)
    plt.tight_layout()

    x = range(54)
    osc = [0, 0, 0, 1, 1, 1] * 4 + [0, 0, 0]
    for x0, x1, os in zip(x[:-1], x[1:], osc):
        if os:
            plt.axvspan(x0-0.5, x1-0.5, color='gray', alpha=0.2, lw=0)

    plt.savefig(f'{target}matching_box_plots_combined.png')
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
    #plt.title(f'Average Adjusted Luminance by Subjects for Each Stimulus and Target Side With Presented Intensities:{intensities}')
    plt.ylabel('Average adjusted luminance by subjects in cd/m²')
    plt.xlabel('Stimulus')
    plt.xticks(rotation=45, ha='right')
    plt.xticks(ticks=range(len(stim_to_index)), labels=stim_to_index.keys())  # Set x-tick labels to 'stim' values
    plt.tight_layout()
    plt.savefig(f'{target}matching_avg_adjusted_luminance_{intensities}.png')
    plt.close()


def avg_adjusted_luminance_combined(df, intensities, cmap, cmap_luminance, target, order):
    """
    Generate a scatter plot showing the average adjustment from subjects for each stimulus for individual intensities.

    Parameters:
    - df: DataFrame containing the data
    - intensities: List of intensities to filter by (should contain multiple intensities for a combined plot)
    - cmap : common colormap
    - target : target path
    - order : order of stimuli
    """

    # Filter rows based on given intensities
    df_filtered = df[df['presented_intensity'].isin(intensities)].copy()

    # Calculate means
    means = df_filtered.groupby(['stim', 'target_side', 'presented_intensity'])['intensity_match'].mean().reset_index()
    means = means[means['stim'].isin(order)]

    # Define a dictionary to map intensity_index to buffer values
    buffer_dict = {0: 0, 1: -0.4, 2: -0.8}

    # Calculate x_adjust values using a list comprehension
    means['x_adjust'] = [
        (order.index(row['stim']) * len(intensities) * 2) +
        intensities.index(row['presented_intensity']) * 2 +
        (0 if row['target_side'] == 'Left' else 0.65) +
        buffer_dict.get(intensities.index(row['presented_intensity']), 0)
        for _, row in means.iterrows()
    ]

    # Mapping target_side to colors
    palette_dict_avg = {'Right': cmap(0.99), 'Left': cmap(0.01)}

    # Create a plot
    plt.figure(figsize=(14, 7))
    ax = sns.scatterplot(x='x_adjust', y='intensity_match', hue='target_side', data=means, palette=palette_dict_avg, s=200, zorder=1)
    sns.despine(left=True)
    ax.set_ylim(means['intensity_match'].min() - 0.5, means['intensity_match'].max() + 2)
    ax.set_xlim(-0.5, 52.5)

    # Normalize intensity_match values to [0, 1]
    means['normalized_intensity'] = means['intensity_match'] / 100.0

    # Add stimuli images at the bottom
    ax2 = ax.twinx()
    ax2.tick_params(top=False, labeltop=False, left=False, labelleft=False, right=False, labelright=False,
                    bottom=False, labelbottom=False)

    # Create vertical bars for each combination of stim and intensity
    for index, stim in enumerate(order):
        for intensity_index, intensity in enumerate(intensities):

            buffer = buffer_dict.get(intensity_index, 0)

            # Left bar
            left_bar_x = (index * len(intensities) * 2) + intensity_index * 2 + buffer
            ax.axvline(x=left_bar_x, color='#f0f0f0',
                       ymin=0, ymax=1, lw=10, zorder=0)

            # Right bar
            right_bar_x = (index * len(intensities) * 2) + intensity_index * 2 + 0.65 + buffer
            ax.axvline(x=right_bar_x, color='#f0f0f0',
                       ymin=0, ymax=1, lw=10, zorder=0)

            # Extract intensity_match values for the current stim and each target side
            left_intensity = \
            means[(means['stim'] == stim) & (means['target_side'] == 'Left') & (means['presented_intensity'] == intensity)]['intensity_match'].values[0]
            right_intensity = \
            means[(means['stim'] == stim) & (means['target_side'] == 'Right') & (means['presented_intensity'] == intensity)]['intensity_match'].values[
                0]

            # Add the intensity_match values on the vertical bars
            ax.text(left_bar_x, 55.5, f"{left_intensity:.2f}", ha="center", va="center", rotation=90,
                    color="black", fontsize=10)
            ax.text(right_bar_x, 55.5, f"{right_intensity:.2f}", ha="center", va="center",
                    rotation=90,
                    color="black", fontsize=10)

            if intensity_index == 0:
                image = Image.open(f"../../experiment/stim/{stim}.png")
                if stim == "sbc":
                    image = ImageOps.mirror(image)
                imagebox = OffsetImage(image, zoom=0.15)  # Adjust the zoom factor as needed
                ab = AnnotationBbox(imagebox, (index * len(intensities) * 2 + 1.9, 0), box_alignment=(0.5, 1.5), frameon=False)
                ax2.add_artist(ab)

    # Add horizontal lines for each intensity value
    for intensity in intensities:
        ax.axhline(y=intensity, color='grey', linestyle='--', alpha=0.6, lw=1.5)

    # Create a proxy artist for the dashed line
    dash_line = plt.Line2D([0], [0], color='grey', linestyle='--', alpha=0.6, lw=1.5, label='Presented Intensity')

    # Add the proxy artist for the dashed line to the handles and its label to labels
    handles, labels = ax.get_legend_handles_labels()
    handles.append(dash_line)
    labels.append('Presented intensity')

    # Set legend
    labels = [label.replace('Left', 'Left target').replace('Right', 'Right target') for label in labels]
    ax2.legend(handles=handles, labels=labels, loc='upper left', ncol=5, bbox_to_anchor=(0, 1.1))

    # Calculate the x-tick positions based on means['x_adjust']
    xtick_positions = [row['x_adjust'] for _, row in means.iterrows()]
    xtick_positions = sorted(xtick_positions)

    # Generate x-tick labels that represent each combination of stim and intensity for the left side
    xtick_labels_left = [f"{intensity}" for stim in order for intensity in intensities]

    # Create an empty list for the right side x-tick labels
    xtick_labels_right = [""] * len(xtick_labels_left)

    # Combine left and right x-tick labels
    xtick_labels_combined = [label for pair in zip(xtick_labels_left, xtick_labels_right) for label in pair]

    # Customize the plot appearance
    ax.set_ylabel('Average adjusted luminance by participants in cd/m²')
    ax.set_xlabel('Stimulus')
    ax.get_legend().remove()
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels_combined, ha='left')

    plt.tight_layout()
    plt.savefig(f'{target}matching_avg_adjusted_luminance_combined.png')
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

    #plt.title(f"Average Adjusted Luminance Heatmap (in cd/m²) With Presented Intensities:{intensities}", y=1.08)
    plt.ylabel("Participant")
    plt.xlabel("Stimulus")
    plt.tight_layout()
    plt.savefig(f'{target}matching_heatmap_{intensities}.png')
    plt.close()


def get_stim_order(df, absolute=False):
    """
    Returns stim order from lowest to highest difference between value 'intensity_match' 'target_side=Left' and
    the value intensity_match' 'target_side=Right'

    Parameters:
    - df: DataFrame containing the data
    """

    # Calculate means
    means = df.groupby(['stim', 'target_side'])['intensity_match'].mean().reset_index()

    # Calculate the difference between 'Right' and 'Left' for each stim
    diffs = means.pivot(index='stim', columns='target_side', values='intensity_match')
    if absolute:
        diffs['difference'] = (diffs['Right'] - diffs['Left']).abs()
    else:
        diffs['difference'] = (diffs['Right'] - diffs['Left'])

    # Order stimuli by the difference
    ordered_stimuli = diffs.sort_values(by='difference').index.tolist()

    return ordered_stimuli


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
    order_absolute = get_stim_order(df, absolute=True)
    order = get_stim_order(df, absolute=False)

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

    avg_adjusted_luminance_combined(df, [49, 50, 51], cmap, cmap_luminance, target, order)

    plot_matching_res_to_boxplot_combined(df, [49, 50, 51], cmap, target, order)


if __name__ == "__main__":
    main()

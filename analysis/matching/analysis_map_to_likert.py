import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image, ImageOps


def avg_adjust_diff_per_stimulus(df, intensities, cmap, target, order=None):
    """
    Generate a scatter plot showing the average adjust_diff for each stimulus.

    Parameters:
    - df: DataFrame containing the data
    - intensities: List of intensities to filter by
    - cmap : common colormap
    - target : target path
    - order : order of stimuli
    """

    # Filter rows based on given intensities
    #df_filtered = df[df['presented_intensity'].isin(intensities)].copy()
    df_filtered = df.copy()

    # Group by 'stim' and compute average adjust_diff
    avg_adjust_diff = df_filtered.groupby("stim")["adjust_diff"].mean()

    if order:
        # Use existing order
        sorted_stim = order
    else:
        # Sort stimuli by the computed average adjust_diff
        sorted_stim = avg_adjust_diff.sort_values(ascending=False).index.tolist()

    # Plotting
    fig, ax = plt.subplots(figsize=(9, 6))

    for i, stim in enumerate(sorted_stim, start=1):
        x = avg_adjust_diff[stim]
        y = i
        norm_value = (x - (-10)) / (10 - (-10))  # Normalizing between -0.1 to 0.1
        color = cmap(norm_value)

        ax.scatter(x, y, s=1500, c=[color], alpha=0.5, label=stim)
        ax.text(x-0.5, y, round(x, 2), va='center')

    ax.set_ylabel("Stimulus")
    ax.set_xlabel("Average adjust_diff")
    ax.set_yticks(range(1, len(sorted_stim) + 1))
    ax.set_yticklabels(sorted_stim)
    ticks = np.arange(-10, 10 + 1, 5)
    ax.set_xticks(ticks)
    ax.axvline(x=0, color="black", linestyle="--")

    # Create a second y-axis for stimuli images
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(ax.get_yticks())
    ax2.set_yticklabels(["" for _ in sorted_stim])

    # Adding stimuli images next to y-labels on ax2
    for index, stimulus in enumerate(sorted_stim):
        if 'catch' in stimulus:
            continue
        image = Image.open(f"../../experiment/stim/{stimulus}.png")
        if stimulus == "sbc":
            image = ImageOps.mirror(image)
        imagebox = OffsetImage(image, zoom=0.13)
        ab = AnnotationBbox(imagebox, (10, ax2.get_yticks()[index]), frameon=False, boxcoords="data",
                            pad=0, box_alignment=(-0.05, 0.5))
        ax2.add_artist(ab)

    ax.set_zorder(ax2.get_zorder() + 1)
    plt.tight_layout()
    ax.grid(True, axis='both')
    plt.subplots_adjust(left=0.25)
    plt.savefig(f'{target}matching_avg_diff_adj_per_stimulus_{intensities}.png')
    plt.close()

    return sorted_stim


def avg_adjust_diff_per_stimulus_combined(df, multi_intensities, cmap, target, order=None):
    """
    Generate a scatter plot showing the average adjust_diff for each stimulus for combined intensities.

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
    vmin = -0.1
    vmax = 0.1

    # Placeholder to keep track of y position
    y_pos = 0

    for intensity_set in multi_intensities:
        # Filter rows based on given intensities
        df_filtered = df[df['presented_intensity'].isin(intensity_set)].copy()

        # Group by 'stim' and compute average adjust_diff
        avg_adjust_diff = df_filtered.groupby("stim")["adjust_diff"].mean()

        if order:
            # Use existing order
            sorted_stim = order
        else:
            # Sort stimuli by the computed average adjust_diff
            sorted_stim = avg_adjust_diff.sort_values(ascending=False).index.tolist()

        for stim in sorted_stim:
            x = avg_adjust_diff[stim]
            y = y_pos
            norm_value = (x - vmin) / (vmax - vmin)
            color = cmap(norm_value)

            ax.scatter(x, y, s=1500, c=[color], alpha=1, label=f"{stim} ({intensity_set[0]})")
            ax.text(x-0.01, y-0.1, round(x, 3))

            y_pos += 1

    ax.set_ylabel("Stimulus")
    ax.set_xlabel("Average adjust_diff")
    ticks = np.arange(-0.1, 0.1 + 0.01, 0.05)
    ax.set_xticks(ticks)
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
        ab = AnnotationBbox(imagebox, (0.1, ax2.get_yticks()[index]), frameon=False, boxcoords="data",
                            pad=0, box_alignment=(-0.05, 0.5))
        ax2.add_artist(ab)

    plt.tight_layout()
    ax.grid(True)
    plt.subplots_adjust(left=0.25)
    plt.savefig(f'{target}matching_avg_diff_adj_per_stimulus_combined_{multi_intensities}.png')
    plt.close()


def main(source="../format_correction/merge/matching_merged_direction_0.csv", target=""):

    # Load data
    df = pd.read_csv(source)

    # Create common colormap
    cmap = sns.diverging_palette(250, 10, as_cmap=True)

    # Specified variations of intensities
    intensities_variation = [[0.49, 0.5, 0.51], [0.49], [0.5], [0.51]]

    # order like in likert
    order = ['checkerboard', 'strip', 'checkerboard_separate', 'bullseye_high_freq', 'bullseye_low_separate', 'sbc_separate', 'whites', 'sbc', 'whites_separate']
    #order = avg_adjust_diff_per_stimulus(df, [0.49, 0.5, 0.51], cmap, target)

    # Process each variation
    for intensities in intensities_variation:
        # Scatterplot; Average adjust_diff per stimulus
        avg_adjust_diff_per_stimulus(df, intensities, cmap, target, order)

    # Scatterplot; for separate intensities combined in one chart
    #avg_adjust_diff_per_stimulus_combined(df, [[0.49], [0.5], [0.51]], cmap, target, order)


if __name__ == "__main__":
    main()

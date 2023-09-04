import os
import pandas as pd
import shutil


def convert_matching_results(source, target):
    # Read the old csv
    df = pd.read_csv(source)

    # New colum to group participant
    df['participant'] = df['trial'].str[:2]

    # Compute avg_right and avg_left adjusted luminance for each stim and trial
    avg_right = df[df['target_side'] == 'Right'].groupby(['participant', 'stim'])['intensity_match'].mean()
    avg_left = df[df['target_side'] == 'Left'].groupby(['participant', 'stim'])['intensity_match'].mean()

    # Prepare a list to store new data
    data = []

    # Iterate over unique trials and stimuli to calculate adjust_diff
    for (trial, stim), _ in df.groupby(['participant', 'stim']):
        print(trial, stim)
        adjust_diff = avg_right.get((trial, stim), 0) - avg_left.get((trial, stim), 0)
        data.append({
            'trial': trial,
            'stim': stim,
            'adjust_diff': adjust_diff
        })

    # Convert list to DataFrame
    new_df = pd.DataFrame(data)

    # Save to the target path
    new_df.to_csv(target, index=False)


def main(source="../format_correction/merge/matching_merged.csv",
         target="../format_correction/merge/matching_merged_direction.csv"):
    convert_matching_results(source, target)
    print("Results converted successfully: format_correction/merge/matching_merged_direction.csv")


if __name__ == "__main__":
    main()

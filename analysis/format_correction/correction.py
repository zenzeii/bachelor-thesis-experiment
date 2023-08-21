import os
import pandas as pd
import shutil

def replace_nan_in_matching_csv(file_path):
    df = pd.read_csv(file_path)
    if 'matching' in file_path:
        df['matching_flipped'] = df['matching_flipped'].fillna(False)
        df['presented_intensity'] = df['presented_intensity'].fillna(0.5)
    df.to_csv(file_path, index=False)

def replace_target_side_in_matching_csv(file_path):
    df = pd.read_csv(file_path)
    if 'matching' in file_path:
        for index, row in df.iterrows():
            if row['target_side'] == 'False':
                df.at[index, 'target_side'] = row['stim'].split('_')[-1].capitalize()
                df.at[index, 'stim'] = row['stim'].split('_')[0] + '_' + row['stim'].split('_')[1] + '_' + row['stim'].split('_')[3] + '_' + row['stim'].split('_')[2]
        df.to_csv(file_path, index=False)

def unflip_response_values_in_likert_csv(file_path):
    df = pd.read_csv(file_path)
    if 'direction' in file_path:
        for index, row in df.iterrows():
            if row['likert_flipped'] == True:
                df.at[index, 'response'] = flip_values(row['response'])
        df.to_csv(file_path, index=False)

def flip_values(value):
    if value == '1':
        return '5'
    elif value == '2':
        return '4'
    elif value == '3':
        return '3'
    elif value == '4':
        return '2'
    elif value == '5':
        return '1'


def process_folders(source_folder, target_folder):
    for root, dirs, fireplace_nan_in_matching_csvles in os.walk(source_folder):
        for dir_name in dirs:
            source_dir = os.path.join(root, dir_name)
            target_dir = os.path.join(target_folder, source_dir[len(source_folder) + 1:])
            os.makedirs(target_dir, exist_ok=True)
            for root2, dirs2, files2 in os.walk(source_dir):
                for file_name in files2:
                    if file_name.endswith('.csv'):
                        source_file = os.path.join(root2, file_name)
                        target_file = os.path.join(target_dir, file_name)
                        shutil.copy(source_file, target_file)
                        replace_nan_in_matching_csv(target_file)
                        replace_target_side_in_matching_csv(target_file)
                        unflip_response_values_in_likert_csv(target_file)


def main(source_folder="../../data/results", target_folder="results_corrected_format"):
    process_folders(source_folder, target_folder)
    print("Results formatted successfully: format_correction/results_corrected_format/")


if __name__ == "__main__":
    main()

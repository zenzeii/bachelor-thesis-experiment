import os
import pandas as pd
import shutil

def replace_nan_in_csv(file_path):
    df = pd.read_csv(file_path)
    if 'matching' in file_path:
        df['matching_flipped'] = df['matching_flipped'].fillna(False)
        df['presented_intensity'] = df['presented_intensity'].fillna(0.5)
    df.to_csv(file_path, index=False)

def replace_target_side_in_csv(file_path):
    df = pd.read_csv(file_path)
    if 'matching' in file_path:
        for index, row in df.iterrows():
            if row['target_side'] == 'False':
                df.at[index, 'target_side'] = row['stim'].split('_')[-1].capitalize()
                df.at[index, 'stim'] = row['stim'].split('_')[0] + '_' + row['stim'].split('_')[1] + '_' + row['stim'].split('_')[3] + '_' + row['stim'].split('_')[2]
        df.to_csv(file_path, index=False)

def process_folders(source_folder, target_folder):
    for root, dirs, files in os.walk(source_folder):
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
                        replace_nan_in_csv(target_file)
                        replace_target_side_in_csv(target_file)


if __name__ == "__main__":
    source_folder = "../data/results"
    target_folder = "../data/results_corrected_format_3"
    process_folders(source_folder, target_folder)

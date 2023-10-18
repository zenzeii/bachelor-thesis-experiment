import os
import pandas as pd
import shutil


def replace_nan_in_matching_csv(file_path):
    df = pd.read_csv(file_path)
    if 'matching' in file_path:
        df['matching_flipped'] = df['matching_flipped'].fillna(False)
        df['presented_intensity'] = df['presented_intensity'].fillna(0.5)
    df.to_csv(file_path, index=False)


def replace_empty_fields_in_likert_csv(file_path):
    df = pd.read_csv(file_path)
    if 'direction' in file_path:
        # For rows where 'stim' contains 'catch', set 'presented_intensity' to '0.0'
        mask = df['stim'].str.contains('catch', na=False)
        df.loc[mask, 'presented_intensity'] = '0.0'
    df.to_csv(file_path, index=False)


def replace_target_side_in_matching_csv(file_path):
    df = pd.read_csv(file_path)
    if 'matching' in file_path:
        for index, row in df.iterrows():
            if row['target_side'] == 'False':
                df.at[index, 'target_side'] = row['stim'].split('_')[-1].capitalize()
                df.at[index, 'stim'] = row['stim'].split('_')[0] + '_' + row['stim'].split('_')[1] + '_' + row['stim'].split('_')[3] + '_' + row['stim'].split('_')[2]
        df.to_csv(file_path, index=False)


def rename_stim_in_likert_csv(file_path):
    df = pd.read_csv(file_path)
    if 'direction' in file_path:
        #mask = df['stim'].str.contains('both', na=False)
        #df.loc[mask, 'stim'] = df.loc[mask, 'stim'].split('_')[0] + '_' + df.loc[mask, 'stim'].split('_')[1] + '_' + df.loc[mask, 'stim'].split('_')[3] + '_' + df.loc[mask, 'stim'].split('_')[2]
        for index, row in df.iterrows():
            if 'both' in row['stim']:
                df.at[index, 'stim'] = row['stim'].split('_')[0] + '_' + row['stim'].split('_')[1] + '_' + row['stim'].split('_')[3] + '_' + row['stim'].split('_')[2]
        df.to_csv(file_path, index=False)


def unflip_response_values_in_likert_csv(file_path):
    df = pd.read_csv(file_path)
    if 'direction' in file_path:
        for index, row in df.iterrows():
            if row['likert_flipped'] == True:
                df.at[index, 'response'] = flip_value(str(row['response']))
        df.to_csv(file_path, index=False)


def flip_value(value):
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
    elif value == '1.0':
        return '5.0'
    elif value == '2.0':
        return '4.0'
    elif value == '3.0':
        return '3.0'
    elif value == '4.0':
        return '2.0'
    elif value == '5.0':
        return '1.0'
    else:
        return "error flip value " + str(value)


def fix_sbc_in_matching(file_path):
    df = pd.read_csv(file_path)
    if 'matching' in file_path:
        for index, row in df.iterrows():
            if row['stim'] == 'sbc':
                if row['target_side'] == 'Left':
                    df.at[index, 'target_side'] = 'Right'
                if row['target_side'] == 'Right':
                    df.at[index, 'target_side'] = 'Left'
        df.to_csv(file_path, index=False)


def fix_sbc_in_likert(file_path):
    df = pd.read_csv(file_path)
    if 'direction' in file_path:
        for index, row in df.iterrows():
            if row['stim'] == 'sbc':
                df.at[index, 'response'] = flip_value(str(row['response']))
        df.to_csv(file_path, index=False)


def convert_responses_in_likert_csv(file_path):
    df = pd.read_csv(file_path)
    if 'direction' in file_path:
        for index, row in df.iterrows():
            df.at[index, 'response'] = convert_response(str(row['response']))
        df.to_csv(file_path, index=False)


def convert_response(value):
    if value == '1':
        return '-2'
    elif value == '2':
        return '-1'
    elif value == '3':
        return '0'
    elif value == '4':
        return '1'
    elif value == '5':
        return '2'
    elif value == '1.0':
        return '-2.0'
    elif value == '2.0':
        return '-1.0'
    elif value == '3.0':
        return '0.0'
    elif value == '4.0':
        return '1.0'
    elif value == '5.0':
        return '2.0'
    else:
        return "error convert_response " + str(value)


def flip_response_of_participant_in_likert(file_path, participant):
    df = pd.read_csv(file_path)
    if 'direction' in file_path and participant in file_path:
        for index, row in df.iterrows():
            df.at[index, 'response'] = flip_converted_response_in_likert(str(row['response']))
        df.to_csv(file_path, index=False)


def flip_converted_response_in_likert(value):
    if value == '-2':
        return '2'
    elif value == '-1':
        return '1'
    elif value == '0':
        return '0'
    elif value == '1':
        return '-1'
    elif value == '2':
        return '-2'
    else:
        return "error convert_response " + str(value)


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

                        # Fixing rows with catch trial and NaN values in matching
                        replace_nan_in_matching_csv(target_file)

                        # Fixing rows with catch trial and NaN values in likert
                        replace_empty_fields_in_likert_csv(target_file)

                        # Fixing rows with catch trial and False as target_side
                        replace_target_side_in_matching_csv(target_file)

                        # unify stim name in likert
                        rename_stim_in_likert_csv(target_file)

                        # Flip the responses where stim has been flipped
                        unflip_response_values_in_likert_csv(target_file)

                        # Flip sbc target_side
                        fix_sbc_in_matching(target_file)

                        # Flip sbc target_side
                        fix_sbc_in_likert(target_file)

                        # Convert from [1, 2 ,3, 4, 5] to [-2, -1, 0, 1, 2]
                        convert_responses_in_likert_csv(target_file)

                        # Flip response from certain participant in likert
                        flip_response_of_participant_in_likert(target_file, 'SP')
                        flip_response_of_participant_in_likert(target_file, 'KP')


def main(source_folder="../../data/results", target_folder="results_corrected_format"):
    process_folders(source_folder, target_folder)
    print("Results formatted successfully: format_correction/results_corrected_format/")


if __name__ == "__main__":
    main()

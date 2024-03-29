import os
import csv
import math


def calculate_duration(start_time, stop_time):
    # Assuming start_time and stop_time are in the format: YYYYMMDD:HHMMSS.MMMMMM
    start_seconds = int(start_time[9:11]) * 60 * 60 + int(start_time[11:13]) * 60 + int(start_time[13:15])
    stop_seconds = int(stop_time[9:11]) * 60 * 60 + int(stop_time[11:13]) * 60 + int(stop_time[13:15])
    return stop_seconds - start_seconds


def get_time_of_day(start_time):
    # Assuming start_time is in the format: YYYYMMDD:HHMMSS.MMMMMM
    hour = int(start_time[9:11])

    if 6 <= hour < 10:
        return "Morning"
    elif 10 <= hour < 12:
        return "Late Morning"
    elif 12 <= hour < 15:
        return "Noon"
    elif 15 <= hour < 18:
        return "Afternoon"
    elif 18 <= hour < 22:
        return "Evening"
    else:
        return "Night"


def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier


def is_row_complete(row):
    for value in row.values():
        if value is None or value == '':
            return False
    if len(row) == 7:
        return True
    else:
        return False


def check_catch_trial_response(stim, response):
    if int(stim.split('_')[-1]) == int(response)+3:
        return 1
    else:
        return 0


def check_catch_trial_response_tolerated(stim, response):
    if int(stim.split('_')[-1]) == int(response)+2 or int(stim.split('_')[-1]) == int(response)+3 or int(stim.split('_')[-1]) == int(response)+4:
        return 1
    else:
        return 0


def merge_csv_files(directory, target_path):
    merged_data = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as csv_file:
                    reader = csv.DictReader(csv_file)
                    for row in reader:
                        # Check if the 'stim' column has the value "catch"
                        if 'catch' in row.get('stim', '').lower():
                            continue  # skip this row

                        if is_row_complete(row):
                            row['trial'] = f"{os.path.splitext(file)[0]}_{row['trial']}"

                            row['duration'] = calculate_duration(row['start_time'], row['stop_time'])

                            row['time_of_day'] = get_time_of_day(row['start_time'])

                            hour = int(row['start_time'][9:11])
                            minute = int(row['start_time'][11:13])
                            row['start_time'] =f"{hour}:{minute:02d}"

                            hour = int(row['stop_time'][9:11])
                            minute = int(row['stop_time'][11:13])
                            row['stop_time'] =f"{hour}:{minute:02d}"

                            merged_data.append(row)

    merged_file_path = target_path + 'likert_merged.csv'
    with open(merged_file_path, 'w', newline='') as merged_file:
        fieldnames = ['trial', 'stim', 'likert_flipped', 'presented_intensity', 'response', 'start_time', 'stop_time', 'duration', 'time_of_day']
        writer = csv.DictWriter(merged_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_data)

    print(f"Results merged successfully: {merged_file_path}")


def merge_csv_files_catching(directory, target_path):
    merged_data = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as csv_file:
                    reader = csv.DictReader(csv_file)
                    for row in reader:
                        # Check if the 'stim' column has the value "catch"
                        if 'catch' not in row.get('stim', '').lower():
                            continue  # skip this row

                        if is_row_complete(row):
                            row['trial'] = f"{os.path.splitext(file)[0]}_{row['trial']}"

                            row['duration'] = calculate_duration(row['start_time'], row['stop_time'])

                            row['expected_catch_trial_response'] = check_catch_trial_response(row['stim'], row['response'])
                            row['tolerated_expected_catch_trial_response'] = check_catch_trial_response_tolerated(row['stim'], row['response'])

                            hour = int(row['start_time'][9:11])
                            minute = int(row['start_time'][11:13])
                            row['start_time'] =f"{hour}:{minute:02d}"

                            hour = int(row['stop_time'][9:11])
                            minute = int(row['stop_time'][11:13])
                            row['stop_time'] =f"{hour}:{minute:02d}"

                            merged_data.append(row)

    merged_file_path = target_path + 'likert_merged_catching.csv'
    with open(merged_file_path, 'w', newline='') as merged_file:
        fieldnames = ['trial', 'stim', 'likert_flipped', 'presented_intensity', 'response', 'start_time', 'stop_time', 'duration', 'expected_catch_trial_response', 'tolerated_expected_catch_trial_response']
        writer = csv.DictWriter(merged_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_data)

    print(f"Results merged successfully: {merged_file_path}")


def main(directory_path='../results_corrected_format', target_path=''):
    merge_csv_files(directory_path, target_path)
    merge_csv_files_catching(directory_path,target_path)


if __name__ == "__main__":
    main()

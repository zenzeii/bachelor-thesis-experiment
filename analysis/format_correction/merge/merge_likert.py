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


def merge_csv_files(directory, target_path):
    merged_data = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as csv_file:
                    reader = csv.DictReader(csv_file)
                    for row in reader:
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


def main(directory_path='../results_corrected_format', target_path=''):
    merge_csv_files(directory_path, target_path)


if __name__ == "__main__":
    main()

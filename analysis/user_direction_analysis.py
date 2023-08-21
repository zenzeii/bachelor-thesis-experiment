import csv
import math


def analyse_matching(intensities):
    # Read matching_merges.csv and store intensity_match data
    intensity_match_data = {}
    with open("format_correction/merge/matching_merged.csv", "r") as matching_file:
        matching_reader = csv.DictReader(matching_file)
        for row in matching_reader:
            if row["presented_intensity"] in intensities:
                stim = row["stim"]
                target_side = row["target_side"]
                intensity_match = float(row["intensity_match"])

                if stim not in intensity_match_data:
                    intensity_match_data[stim] = {"Left": [], "Right": []}

                intensity_match_data[stim][target_side].append(intensity_match)

    # Calculate average intensity_match values and determine method of adjustment directions
    method_of_adjustment_avg_intensity_left = {}
    method_of_adjustment_avg_intensity_right = {}
    method_of_adjustment_directions = {}
    for stim, values in intensity_match_data.items():
        method_of_adjustment_avg_intensity_left[stim] = sum(values["Left"]) / len(values["Left"])
        method_of_adjustment_avg_intensity_right[stim] = sum(values["Right"]) / len(values["Right"])

        if abs(method_of_adjustment_avg_intensity_left[stim]) > abs(method_of_adjustment_avg_intensity_right[stim]):
            method_of_adjustment_directions[stim] = "Left"
        else:
            method_of_adjustment_directions[stim] = "Right"

    return method_of_adjustment_directions, method_of_adjustment_avg_intensity_left, method_of_adjustment_avg_intensity_right


def analyse_likert(intensities):
    # Read likert_merged.csv and store response data
    response_data = {}
    with open("format_correction/merge/likert_merged.csv", "r") as likert_file:
        likert_reader = csv.DictReader(likert_file)
        for row in likert_reader:
            if row["presented_intensity"] in intensities:
                stim = row["stim"]
                response = float(row["response"])

                if stim not in response_data:
                    response_data[stim] = []

                response_data[stim].append(response)

    # Calculate average response values and determine brightness ratings directions
    brightness_ratings_avg_response = {}
    brightness_ratings_directions = {}
    for stim, responses in response_data.items():
        brightness_ratings_avg_response[stim] = sum(responses) / len(responses)

        if brightness_ratings_avg_response[stim] > 3:
            brightness_ratings_directions[stim] = "Right"
        else:
            brightness_ratings_directions[stim] = "Left"

    return brightness_ratings_directions, brightness_ratings_avg_response


def check_both_direction(stimulus_names, method_of_adjustment_directions, brightness_ratings_directions):
    # Check if direction for a stimulus is the same using both methods
    identical_directions = {}
    for stim_name in stimulus_names:
        method_of_adjustment_dir = method_of_adjustment_directions.get(stim_name, "")
        brightness_ratings_dir = brightness_ratings_directions.get(stim_name, "")
        identical_directions[stim_name] = method_of_adjustment_dir == brightness_ratings_dir

    return identical_directions


def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier

def analyse_direction(intensities):

    # Create the table
    table = [["Stimulus name", "Method of adjustment", "", "", "Brightness ratings", "", "Identical direction?"], ["", "Avg luminance left", "Avg luminance right", "Avg direction", "Avg response", "Avg direction", ""]]
    stimulus_names = [
        "sbc", "bullseye_high_freq", "sbc_separate", "bullseye_low_separate",
        "whites", "whites_separate", "strip", "checkerboard", "checkerboard_separate"
    ]

    method_of_adjustment_directions, method_of_adjustment_avg_intensity_left, method_of_adjustment_avg_intensity_right = analyse_matching(intensities)
    brightness_ratings_directions, brightness_ratings_avg_response = analyse_likert(intensities)
    identical_directions = check_both_direction(stimulus_names, method_of_adjustment_directions, brightness_ratings_directions)

    for stim_name in stimulus_names:
        table.append([
            stim_name,
            round_up(float(method_of_adjustment_avg_intensity_left.get(stim_name, "")*100), 2),
            round_up(float(method_of_adjustment_avg_intensity_right.get(stim_name, "")*100), 2),
            method_of_adjustment_directions.get(stim_name, ""),
            round_up(float(brightness_ratings_avg_response.get(stim_name, "")), 2),
            brightness_ratings_directions.get(stim_name, ""),
            identical_directions.get(stim_name, "")
        ])

    # Print the table
    for row in table:
        print("{:<25} {:<20} {:<20} {:<15} {:<40} {:<40} {:<25} ".format(*row))
    print()
    print()
    print()
    print()

def main():

    print()
    print("Tables showing the average direction for each stimulus using each method. Each Table shows data with different presented intensities to subjects.")
    print()

    intensities_variation = [["0.49", "0.5", "0.51"], ["0.49"], ["0.5"], ["0.51"]]
    for intensities in intensities_variation:
        if len(intensities) > 1:
            print("No Filter/All intensities: " + str(intensities))
        else:
            print("Filtered with intensity:" + str(intensities))
        analyse_direction(intensities)


if __name__ == "__main__":
    main()

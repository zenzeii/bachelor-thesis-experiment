import format_correction.correction
import format_correction.merge.merge_likert
import format_correction.merge.merge_matching
import likert.analysis
import matching.analysis
import user_direction_analysis

if __name__ == "__main__":
    print("\n\n")

    source_folder = "../data/results"
    target_folder = "format_correction/results_corrected_format"
    format_correction.correction.main(source_folder, target_folder)

    source_folder = "format_correction/results_corrected_format/"
    target_folder = "format_correction/merge/"
    format_correction.merge.merge_likert.main(source_folder, target_folder)
    format_correction.merge.merge_matching.main(source_folder, target_folder)

    source_folder = "format_correction/merge/likert_merged.csv"
    target_folder = "likert/"
    likert.analysis.main(source_folder, target_folder)
    print("Results plotted successfully: likert/")

    source_folder = "format_correction/merge/matching_merged.csv"
    target_folder = "matching/"
    matching.analysis.main(source_folder, target_folder)
    print("Results plotted successfully: matching/")

    print("\n\n")
    user_direction_analysis.main()
    print("\nHave a look at these folders: \n- 'analysis/likert/' \n- 'analysis/matching'\n\n")


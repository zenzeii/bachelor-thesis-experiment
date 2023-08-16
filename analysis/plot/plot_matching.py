import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



if __name__ == "__main__":
    # Read the CSV file into a pandas DataFrame
    csv_file = '../matching_merged.csv'
    data = pd.read_csv(csv_file)

    # Set up Seaborn for better aesthetics
    sns.set(style="whitegrid")

    # Group data by 'stim' and 'target_side', then create box plots
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    sns.boxplot(x='intensity_match', y='stim', hue='target_side', data=data, orient='h')
    sns.despine(left=True)
    # Add individual data points
    sns.stripplot(x='intensity_match', y='stim', hue='target_side', data=data, jitter=False, dodge=True, size=3.5, orient='h')

    plt.xlabel('Adjusted brightness by subjects')
    plt.ylabel('Stimulus')
    plt.title('Results as box plots for each stimulus and target side')

    # Adjust layout for better y-axis label visibility
    plt.subplots_adjust(left=0.35)

    # Save the plot to a file
    plt.savefig('box_plots_matching.png')

    # Show the plot
    plt.close()

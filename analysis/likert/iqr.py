import pandas as pd
from io import StringIO



if __name__ == "__main__":
    # Given CSV data
    csv_data = "../format_correction/merge/likert_merged.csv"

    # Read CSV data into a pandas DataFrame
    df = pd.read_csv(csv_data)
    print(df)

    # Calculate IQR for each 'stim'
    iqr_values = df.groupby('stim')['response'].agg(lambda x: x.quantile(0.75) - x.quantile(0.25))

    # Display the results
    print(iqr_values)



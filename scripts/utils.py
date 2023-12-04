# Function to create sequences
import numpy as np
import pandas as pd
from typing import List
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler

def create_sequences(df_day, n_steps):
    X, y = [], []
    data_array = df_day.values.astype(float)  # Convert the entire data array to float
    for i in range(len(data_array) - n_steps):
        X.append(data_array[i:i + n_steps, :-1])
        y.append(data_array[i + n_steps, -1])
    return np.array(X), np.array(y)

def preprocess(df: pd.DataFrame, columns_to_keep: List[str] = [], IP: str = '', days: List[str] = [], sample_size: int = -1) -> pd.DataFrame:
    # Removed Unnamed Columns
    df.drop(columns=[col for col in df.columns if 'Unnamed' in col], inplace=True)

    # Convert times to datetime
    df['TIME_FIRST'] = pd.to_datetime(df['TIME_FIRST'], unit='s')
    df['TIME_LAST'] = pd.to_datetime(df['TIME_LAST'], unit='s')

    # Filter by IP if specified
    if IP:
        df = df[df['SRC_IP'] == IP].copy()

    # Convert to Binary Classification for Generalizability
    df['LABEL'] = df['LABEL'].apply(lambda x: 0 if x == 'clear' else 1)

    # Filter by specified days if given
    if days:
        days_datetime = pd.to_datetime(days)
        df = df[df['TIME_FIRST'].dt.date.isin([d.date() for d in days_datetime])]

    # Sample if a specific sample size is given
    if sample_size != -1 and sample_size < df.shape[0]:
        df = df.sample(n=sample_size, random_state=42)

    # Select specified columns if given
    if columns_to_keep:
        df = df[columns_to_keep]

    # Create a copy of the DataFrame to avoid SettingWithCopyWarning
    df = df.copy()

    # Fill missing values with column means
    column_means = df.mean()
    df.fillna(column_means, inplace=True)

    # Normalize or standardize the dataframe excluding 'LABEL' column
    scaler = StandardScaler()
    feature_columns = [col for col in df.columns if col != 'LABEL']

    # Use .loc to avoid SettingWithCopyWarning
    df.loc[:, feature_columns] = scaler.fit_transform(df[feature_columns])

    return df.reset_index(drop=True)

def perform_anova(df, target_column):
    """
    Perform ANOVA on all numeric columns of a DataFrame against a target column.

    Parameters:
    df (DataFrame): The DataFrame containing the data.
    target_column (str): The name of the column in df to use as the target.

    Returns:
    DataFrame: A DataFrame with the ANOVA F-Value and P-Value for each column.
    """
    anova_results = {}

    # Iterate over all columns in the dataframe
    for col in df.columns:
        # We only want to perform ANOVA on numeric columns, excluding the target column
        if df[col].dtype in ['int64', 'float64'] and col != target_column:
            # Perform ANOVA using the f_oneway function from SciPy
            # This requires splitting the data into groups based on the unique values in the target column
            groups = [group[col].dropna() for name, group in df.groupby(target_column)]
            fvalue, pvalue = stats.f_oneway(*groups)
            anova_results[col] = {'F-Value': fvalue, 'P-Value': pvalue}

    # Convert the ANOVA results into a pandas DataFrame for better visualization
    anova_results_df = pd.DataFrame(anova_results).T

    # Return the results, sorted by the P-Value
    return anova_results_df.sort_values(by='P-Value')


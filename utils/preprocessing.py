import pandas as pd
import numpy as np


df_raw = pd.read_csv("data/creditcard.csv")




def check_nulls(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Check for null values in the DataFrame and return a DataFrame with the count of nulls per column.
    Args:
        df (pd.DataFrame): The input DataFrame to check for null values.
    Returns:
        pd.DataFrame: A DataFrame with the count of null values per column.
    '''
    null_counts = df.isna().sum()
    return pd.DataFrame(null_counts[null_counts > 0], columns=['Null Count'])

def replace_nulls(df_raw: pd.DataFrame) -> pd.DataFrame:
    '''
    Replace null values in the DataFrame with the mean of each column.
    Args:
        df_raw (pd.DataFrame): The input DataFrame with potential null values.
    '''
    df = df_raw.copy()
    for col in df.columns:
        if df[col].isnull().any():
            mean_value = df[col].mean()
            df[col].fillna(mean_value, inplace=True)
    return df

def check_outliers(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Check for outliers in the DataFrame using the IQR method and return a DataFrame with the count of outliers per column.
    Args:
        df (pd.DataFrame): The input DataFrame to check for outliers.
    Returns:
        pd.DataFrame: A DataFrame with the count of outliers per column.
    '''
    outlier_counts = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_counts[col] = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
    
    return pd.DataFrame.from_dict(outlier_counts, orient='index', columns=['Outlier Count'])

def check_outliers_five_sigma(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Check for outliers in the DataFrame using the five-sigma rule and return a DataFrame with the count of outliers per column.
    Args:
        df (pd.DataFrame): The input DataFrame to check for outliers.
    Returns:
        pd.DataFrame: A DataFrame with the count of outliers per column.
    '''
    outlier_counts = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        mean_value = df[col].mean()
        std_value = df[col].std()
        lower_bound = mean_value - 5 * std_value
        upper_bound = mean_value + 5 * std_value
        outlier_counts[col] = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
    
    return pd.DataFrame.from_dict(outlier_counts, orient='index', columns=['Outlier Count'])

def replace_outliers(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Replace outliers in the DataFrame using the IQR method.
    Args:
        df (pd.DataFrame): The input DataFrame with potential outliers.
    Returns:
        pd.DataFrame: A DataFrame with outliers replaced by the median of each column.
    '''
    df_cleaned = df.copy()
    for col in df_cleaned.select_dtypes(include=[np.number]).columns:
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        median_value = df_cleaned[col].median()
        df_cleaned[col] = np.where((df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound), median_value, df_cleaned[col])
    
    return df_cleaned

def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Normalize the numerical columns in the DataFrame using Min-Max scaling.
    Args:
        df (pd.DataFrame): The input DataFrame to normalize.
    Returns:
        pd.DataFrame: A DataFrame with normalized numerical columns.
    '''
    df_normalized = df.copy()
    for col in df_normalized.select_dtypes(include=[np.number]).columns:
        min_value = df_normalized[col].min()
        max_value = df_normalized[col].max()
        df_normalized[col] = (df_normalized[col] - min_value) / (max_value - min_value)
    
    return df_normalized

def standardize_data(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Standardize the numerical columns in the DataFrame using Z-score normalization.
    Args:
        df (pd.DataFrame): The input DataFrame to standardize.
    Returns:
        pd.DataFrame: A DataFrame with standardized numerical columns.
    '''
    df_standardized = df.copy()
    for col in df_standardized.select_dtypes(include=[np.number]).columns:
        mean_value = df_standardized[col].mean()
        std_value = df_standardized[col].std()
        df_standardized[col] = (df_standardized[col] - mean_value) / std_value
    
    return df_standardized

# check for nulls and outliers in df
nulls_df = check_nulls(df_raw)
outliers_df = check_outliers(df_raw)
outliers_five_sigma_df = check_outliers_five_sigma(df_raw)

print("Null Values in DataFrame:")
print(nulls_df)
print("\nOutliers in DataFrame:")   
print(outliers_five_sigma_df)

# outliers exist in the data but in this case is probably expected due to the nature of the data
# no nulls exist in the data
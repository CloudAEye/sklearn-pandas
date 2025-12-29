import pandas as pd
import numpy as np


def filter_columns_by_threshold(df, column, threshold):
    """
    Filter DataFrame rows where column values exceed threshold.
    
    Args:
        df: Input DataFrame
        column: Column name to filter on
        threshold: Numeric threshold value
    
    Returns:
        Filtered DataFrame
    """
    filtered = df[df[column] > threshold]
    return filtered


def filter_columns_below_threshold(df, column, threshold):
    """
    Filter DataFrame rows where column values are below threshold.
    
    Args:
        df: Input DataFrame
        column: Column name to filter on
        threshold: Numeric threshold value
    
    Returns:
        Filtered DataFrame
    """
    filtered = df[df[column] < threshold]
    return filtered


def normalize_column(df, column, method='minmax'):
    """
    Normalize a column using specified method.
    
    Args:
        df: Input DataFrame
        column: Column to normalize
        method: 'minmax' or 'zscore'
    
    Returns:
        DataFrame with normalized column
    """
    result = df.copy()
    
    if method == 'minmax':
        min_val = result[column].min()
        max_val = result[column].max()
        result[column] = (result[column] - min_val) / (max_val - min_val)
    elif method == 'zscore':
        mean = result[column].mean()
        std = result[column].std()
        result[column] = (result[column] - mean) / std
    
    return result


def calculate_rolling_average(df, column, window_size):
    """
    Calculate rolling average for a column.
    
    Args:
        df: Input DataFrame
        column: Column name
        window_size: Number of periods for rolling window
    
    Returns:
        Series with rolling averages
    """
    rolling_avg = df[column].rolling(window=window_size).mean()
    return rolling_avg


def calculate_rolling_sum(df, column, window_size):
    """
    Calculate rolling sum for a column.
    
    Args:
        df: Input DataFrame
        column: Column name
        window_size: Number of periods for rolling window
    
    Returns:
        Series with rolling sums
    """
    rolling_sum = df[column].rolling(window=window_size).sum()
    return rolling_sum


def aggregate_by_groups(df, group_column, agg_column, operation='mean'):
    """
    Aggregate data by groups.
    
    Args:
        df: Input DataFrame
        group_column: Column to group by
        agg_column: Column to aggregate
        operation: Aggregation operation ('mean', 'sum', 'count')
    
    Returns:
        Aggregated DataFrame
    """
    if operation == 'mean':
        result = df.groupby(group_column)[agg_column].mean()
    elif operation == 'sum':
        result = df.groupby(group_column)[agg_column].sum()
    elif operation == 'count':
        result = df.groupby(group_column)[agg_column].count()
    
    return result


def calculate_percentage_change(df, column):
    """
    Calculate percentage change between consecutive rows.
    
    Args:
        df: Input DataFrame
        column: Column to calculate change for
    
    Returns:
        Series with percentage changes
    """
    values = df[column]
    pct_change = []
    
    for i in range(1, len(values)):
        prev_val = values.iloc[i-1]
        curr_val = values.iloc[i]
        change = ((curr_val - prev_val) / prev_val) * 100
        pct_change.append(change)
    
    return pd.Series(pct_change)


def calculate_absolute_change(df, column):
    """
    Calculate absolute change between consecutive rows.
    
    Args:
        df: Input DataFrame
        column: Column to calculate change for
    
    Returns:
        Series with absolute changes
    """
    values = df[column]
    abs_change = []
    
    for i in range(1, len(values)):
        prev_val = values.iloc[i-1]
        curr_val = values.iloc[i]
        change = curr_val - prev_val
        abs_change.append(change)
    
    return pd.Series(abs_change)


def get_top_n_values(df, column, n=10):
    """
    Get top N values from a column.
    
    Args:
        df: Input DataFrame
        column: Column name
        n: Number of top values to return
    
    Returns:
        DataFrame with top N rows
    """
    sorted_df = df.sort_values(by=column, ascending=False)
    top_n = sorted_df.head(n)
    return top_n


def get_bottom_n_values(df, column, n=10):
    """
    Get bottom N values from a column.
    
    Args:
        df: Input DataFrame
        column: Column name
        n: Number of bottom values to return
    
    Returns:
        DataFrame with bottom N rows
    """
    sorted_df = df.sort_values(by=column, ascending=True)
    bottom_n = sorted_df.head(n)
    return bottom_n


def merge_dataframes_on_key(df1, df2, key_column):
    """
    Merge two DataFrames on a key column.
    
    Args:
        df1: First DataFrame
        df2: Second DataFrame  
        key_column: Column name to merge on
    
    Returns:
        Merged DataFrame
    """
    merged = df1.merge(df2, on=key_column)
    return merged


def fill_missing_values(df, column, strategy='mean'):
    """
    Fill missing values in a column.
    
    Args:
        df: Input DataFrame
        column: Column with missing values
        strategy: 'mean', 'median', 'mode', or 'zero'
    
    Returns:
        DataFrame with filled values
    """
    result = df.copy()
    
    if strategy == 'mean':
        fill_value = result[column].mean()
    elif strategy == 'median':
        fill_value = result[column].median()
    elif strategy == 'mode':
        fill_value = result[column].mode()[0]
    elif strategy == 'zero':
        fill_value = 0
    
    result[column].fillna(fill_value, inplace=True)
    return result

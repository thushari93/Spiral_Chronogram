import numpy as np
import pandas as pd
import os
import re
import math

def create_lag_matrix(distance_matrix, dates_df, date_type):
    """
    Create a lag matrix from the distance matrix and dates.

    Parameters:
    distance_matrix (np.ndarray): Distance matrix.
    dates_df (pd.DataFrame): DataFrame containing the dates.

    Returns:
    np.ndarray: Lag matrix.
    """

    # Initialize the lag matrix with zeros, having the same shape as the distance matrix
    lag = np.zeros(distance_matrix.shape)

    # Iterate over the dates in the Dates DataFrame
    dates = dates_df['Date'].to_list()

    if date_type == 'D':
        for i, date1 in enumerate(dates):
            for j, date2 in enumerate(dates[:i]):
                # Compute the lag (difference in days) between the dates
                lag[i, j] = lag[j, i] = (date1 - date2).days
    if date_type == 'M':
        for i, date1 in enumerate(dates):
            for j, date2 in enumerate(dates[:i]):
                # Compute the lag (difference in months) between the dates
                lag_days = (date1 - date2).days
                lag_years = float(lag_days/365)
                lag[i, j] = lag[j, i] = int((date1 - date2).days/365)*12 +int((lag_years - int(lag_years))*365/30)

    if date_type == 'Y':
        for i, date1 in enumerate(dates):
            for j, date2 in enumerate(dates[:i]):
                # Compute the lag (difference in years) between the dates
                lag_days = (date1 - date2).days
                lag[i, j] = lag[j, i] = int(lag_days/365)

    # Convert the lag matrix to integers and return it
    return lag.astype(int)

def create_dates_matrix(dates_df, date_type):
    """
    Create a dates matrix by tiling the dates.

    Parameters:
    dates_df (pd.DataFrame): DataFrame containing the dates.

    Returns:
    np.ndarray: Dates matrix.
    """
    return np.tile(dates_df['Date'], (len(dates_df), 1))


def extract_upper_triangle(matrix):
    """
    Extract the upper triangle of a matrix including the diagonal.

    Parameters:
    matrix (np.ndarray): Input matrix.

    Returns:
    np.ndarray: Upper triangle of the matrix.
    """

    # Extract the upper triangle of the matrix (including the main diagonal) and return it as a numpy array
    return matrix[np.triu_indices(matrix.shape[0])]


def check_date(file_name, expected_extensions=['.csv', '.txt', '.npy']):
    """
    Check if the file has one of the expected formats (extensions) and contains dates in YYYY-MM-DD or YYYY-MM format
    in the first column.

    Parameters:
    file_name (str): The name or path of the file.
    expected_extensions (list of str): The list of expected file extensions (e.g., ['.csv', '.txt', '.npy']).

    Returns:
    DataFrame: Returns a dataframe with dates in YYYY-MM-DD format
    """
    # Check file extension
    _, file_extension = os.path.splitext(file_name)
    if file_extension.lower() not in [ext.lower() for ext in expected_extensions]:
        return False

    # Define the regex patterns for YYYY-MM-DD and YYYY-MM
    date_pattern_ymd = re.compile(r'\b\d{4}-\d{2}-\d{2}\b')
    date_pattern_ym = re.compile(r'\b\d{4}-\d{2}\b')
    date_pattern_y = re.compile(r'\b\d{4}')

    def validate_and_convert_dates(series):
        try:
            series_str = series.astype(str)
            # Check if the series matches the YYYY-MM-DD pattern
            if series_str.astype(str).str.match(date_pattern_ymd).all():
                return pd.to_datetime(series_str).dt.date

            # Check if the series matches the YYYY-MM pattern
            if series_str.astype(str).str.match(date_pattern_ym).all():
                # Convert to datetime and fill day with the first day of the month
                return pd.to_datetime(series_str + '-01').dt.date

            # Check if the series matches the YYYY pattern
            if series_str.astype(str).str.match(date_pattern_y).all():
                # Convert to datetime and fill day with the first day of the month
                return pd.to_datetime(series_str + '-01' + '-01').dt.date

            return None
        except Exception as e:
            print(f"An error occurred during date conversion: {e}")
            return None

    try:
        if file_extension.lower() in ['.csv', '.txt']:
            # Read the file into a DataFrame
            df = pd.read_csv(file_name)

            # Check if the first column exists
            if df.shape[1] == 0:
                print("No columns found in the file.")
                return None

            # Get the first column
            first_column = df.iloc[:, 0]

            # Validate and convert dates
            converted_dates = validate_and_convert_dates(first_column)
            if converted_dates is None:
                print("The first column does not contain dates in a recognized format.")
                return None

            df['Date'] = converted_dates

            # Check if dates are in ascending order
            if not df['Date'].is_monotonic_increasing:
                print("Dates are not in ascending order.")
                return None

        elif file_extension.lower() == '.npy':
            # Load the NumPy array from the file
            array = np.load(file_name, allow_pickle=True)

            # If the array is not already a DataFrame, convert it
            if not isinstance(array, pd.DataFrame):
                array = pd.DataFrame(array)

            # Check if the first column exists
            if array.shape[1] == 0:
                print("No columns found in the file.")
                return None

            # Get the first column
            first_column = array.iloc[:, 0]

            # Validate and convert dates
            converted_dates = validate_and_convert_dates(first_column)
            if converted_dates is None:
                print("The first column does not contain dates in a recognized format.")
                return None

            array['Date'] = converted_dates

            # Check if dates are in ascending order
            if not array['Date'].is_monotonic_increasing:
                print("Dates are not in ascending order.")
                return None
            df=array
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    return df



def check_distance(file_name):
    """
    Check if a NumPy array file is an n by n array with real non-negative numbers.

    Parameters:
    file_name (str): The path to the NumPy array file.

    Returns:
    bool: True if the array is n by n with real non-negative numbers, False otherwise.
    """
    try:
        # Load the NumPy array from the file
        array = np.load(file_name, allow_pickle=True)

        # Check if the array is 2-dimensional
        if len(array.shape) != 2:
            return False

        # Check if the array is n by n
        if array.shape[0] != array.shape[1]:
            return False

        # Check if all elements are real and non-negative
        if not np.all(np.isreal(array)) or not np.all(array >= 0):
            return False

        return array
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def check_and_load_files(distance_file_name, date_file_name):
    """
    Check if one file contains dates in YYYY-MM-DD format and another file contains a 2D array with real non-negative numbers.

    Parameters:
    date_file_name (str): The path to the date file.
    distance_file_name (str): The path to the distance file.

    Returns:
    dataframes: Dates and Distance Dataframes.
    """
    _, date_file_extension = os.path.splitext(date_file_name)
    _, distance_file_extension = os.path.splitext(distance_file_name)

    if date_file_extension.lower() in ['.csv', '.txt'] and distance_file_extension.lower() in ['.npy', '.csv', '.txt'] and check_date(date_file_name) is not None and check_distance(distance_file_name) is not None:

        dates_df =check_date(date_file_name)
        dates_df['Date'] = pd.to_datetime(dates_df['Date']).dt.date

        distance_matrix=check_distance(distance_file_name)
        return dates_df, distance_matrix
    else:
        print("Unsupported file format.")
        return None



def create_final_df(distance_matrix_path, dates_file_path, date_type):
    """
    Create the final DataFrame from the distance matrix and dates file.

    Parameters:
    distance_matrix_path (str): Path to the distance matrix file.
    dates_file_path (str): Path to the dates file.

    Returns:
    pd.DataFrame: Final DataFrame containing Date, Lag, and Dist columns.
    """
    dates_df, distance_matrix = check_and_load_files(distance_matrix_path, dates_file_path)
    lag_matrix = create_lag_matrix(distance_matrix, dates_df, date_type)
    dates_matrix = create_dates_matrix(dates_df,date_type)
    upper_dates = extract_upper_triangle(dates_matrix)
    upper_lag = extract_upper_triangle(lag_matrix)
    upper_dist = extract_upper_triangle(distance_matrix)
    return pd.DataFrame({'Date': upper_dates, 'Lag': upper_lag, 'Dist': upper_dist})

def aggregate(final_df, date="Date", lag="Lag", dist="Dist"):
    """
    Aggregate the final DataFrame by Date and Lag, computing the mean of Dist.

    Parameters:
    final_df (pd.DataFrame): Final DataFrame containing Date, Lag, and Dist columns.
    date (str): Name of the date column.
    lag (str): Name of the lag column.
    dist (str): Name of the dist column.

    Returns:
    pd.DataFrame: Aggregated DataFrame.
    """
    if type(date) == str and type(lag) == str and type(dist) == str:
        final_df[date] = final_df[date].astype(str)
        final_df[lag] = final_df[lag].astype(float)
        final_df[dist] = final_df[dist].astype(float)
    else:
        raise TypeError("date, lag, dist should be string column name")
    df = final_df.groupby([date, lag]).agg({dist: "mean"}).reset_index(drop=False)
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    return df

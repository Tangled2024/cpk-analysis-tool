"""
Data utilities module for CPK Analysis Tool
Handles CSV file processing, data cleaning, and validation
"""

import os
import csv
import logging
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from config import config

def csv_to_list(filename: str) -> List[List[str]]:
    """
    Read CSV file to list with robust encoding handling
    
    Parameters:
    -----------
    filename : str
        Path to the CSV file
    
    Returns:
    --------
    List[List[str]]
        List of rows containing list of column values
    """
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    
    # Try different encodings
    for encoding in encodings:
        try:
            with open(filename, newline="", encoding=encoding) as csvfile:
                reader = csv.reader(csvfile, delimiter=",", quotechar='"')
                return list(reader)
        except UnicodeDecodeError:
            continue
    
    # Fallback with error handling
    try:
        with open(filename, newline="", encoding='utf-8', errors='replace') as csvfile:
            reader = csv.reader(csvfile, delimiter=",", quotechar='"')
            return list(reader)
    except Exception as e:
        logging.error(f"Failed to read file {filename} with error: {e}")
        return []

def csv_to_df(filename: str) -> pd.DataFrame:
    """
    Read CSV file to DataFrame with robust parsing
    
    Parameters:
    -----------
    filename : str
        Path to the CSV file
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the CSV data
    """
    try:
        return pd.read_csv(filename, encoding='utf-8', engine='python', on_bad_lines='warn')
    except Exception as e:
        logging.error(f"Error reading {filename}: {e}")
        return pd.DataFrame()

def find_known_words(search_words: List[List[str]], known_words: List[str]) -> Tuple[Optional[str], Optional[List[int]]]:
    """
    Find known words in a 2D list (like CSV data)
    
    Parameters:
    -----------
    search_words : List[List[str]]
        2D list of strings to search in
    known_words : List[str]
        List of strings to search for
    
    Returns:
    --------
    Tuple[Optional[str], Optional[List[int]]]
        The found word and its position [row, col], or (None, None) if not found
    """
    for know_word in known_words:
        for row in range(len(search_words)):
            for col in range(len(search_words[row])):
                if search_words[row][col] == know_word:
                    return know_word, [row, col]
    
    logging.error("Could not find any of these words in the CSV:")
    for know_word in known_words:
        logging.error(f'"{know_word}"')
    return None, None

def pd_str_to_num_with_nan(data_df: pd.DataFrame, start_col: Optional[int] = None) -> pd.DataFrame:
    """
    Replace all non-numeric values with NaN
    
    Parameters:
    -----------
    data_df : pd.DataFrame
        DataFrame to process
    start_col : Optional[int]
        Index of the first column to convert
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with numeric values
    """
    if start_col is None:
        start_col = 0
    
    cols = slice(start_col, data_df.shape[1])
    data_df.iloc[:, cols] = data_df.iloc[:, cols].apply(pd.to_numeric, errors="coerce", axis=1)
    return data_df.infer_objects()

def is_number(s: Any) -> bool:
    """
    Check if value is a number
    
    Parameters:
    -----------
    s : Any
        Value to check
    
    Returns:
    --------
    bool
        True if value is a number, False otherwise
    """
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False

def get_limits(data_head: List[List[str]], limits_row: int, index_col: int, 
               usl_string: str, lsl_string: str) -> pd.DataFrame:
    """
    Extract upper and lower limits from data headers
    
    Parameters:
    -----------
    data_head : List[List[str]]
        2D list containing the data headers
    limits_row : int
        Row index of the limits
    index_col : int
        Column index to use as index
    usl_string : str
        String identifying upper limit row
    lsl_string : str
        String identifying lower limit row
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing upper and lower limits
    """
    try:
        data_head_df = pd.DataFrame(data_head)
        data_head_df.columns = data_head_df.iloc[limits_row].tolist()
        data_head_df.set_index(data_head_df.iloc[limits_row, index_col], drop=False, inplace=True)
        return data_head_df.loc[[lsl_string, usl_string]]
    except Exception as e:
        logging.error(f"Error getting limits: {e}")
        return pd.DataFrame()

def get_valid_data(log_data: List[List[str]], valid_col1: int, valid_col2: int, 
                  min_starting_row: int, column_name_row: int) -> Optional[pd.DataFrame]:
    """
    Extract and validate data from raw CSV content
    
    Parameters:
    -----------
    log_data : List[List[str]]
        2D list containing the raw CSV data
    valid_col1 : int
        Index of the first validation column (string)
    valid_col2 : int
        Index of the second validation column (numeric)
    min_starting_row : int
        Minimum row index to start searching from
    column_name_row : int
        Row index containing column names
    
    Returns:
    --------
    Optional[pd.DataFrame]
        DataFrame containing valid data or None if invalid
    """
    # Find first valid data row
    start_row = None
    for row in range(min_starting_row, len(log_data)):
        if (len(log_data[row][valid_col1].strip()) != 0 and
                len(log_data[row][valid_col2]) != 0 and
                is_number(log_data[row][valid_col2])):
            start_row = row
            break
    
    if start_row is None:
        logging.error("Could not find start row with valid data")
        return None
    
    try:
        unit_data_df = pd.DataFrame(log_data[start_row:])
        unit_data_df.columns = log_data[column_name_row]
        unit_data_df.set_index(log_data[column_name_row][valid_col1], drop=False, inplace=True)
        return pd_str_to_num_with_nan(unit_data_df, valid_col2)
    except Exception as e:
        logging.error(f"Error creating valid data DataFrame: {e}")
        return None

def find_strings_in_list(data_list: List[str], searching_strs: List[str]) -> Tuple[List[str], List[str]]:
    """
    Find specific strings in list
    
    Parameters:
    -----------
    data_list : List[str]
        List of strings to search in
    searching_strs : List[str]
        List of strings to search for
    
    Returns:
    --------
    Tuple[List[str], List[str]]
        Lists of matching and non-matching strings
    """
    metrics_with_strs = []
    metrics_without_strs = []
    
    for elem in data_list:
        if any(string in elem for string in searching_strs):
            metrics_with_strs.append(elem)
        else:
            metrics_without_strs.append(elem)
            
    return metrics_with_strs, metrics_without_strs

def remove_thousand_sep(data_list: List[List[str]], chars: Optional[List[str]] = None) -> List[List[str]]:
    """
    Remove thousand separators from data
    
    Parameters:
    -----------
    data_list : List[List[str]]
        2D list of data
    chars : Optional[List[str]]
        List of characters to remove
    
    Returns:
    --------
    List[List[str]]
        Cleaned 2D list
    """
    if chars is None:
        chars = [","]  # Default to comma if no chars provided
    
    for row in range(len(data_list)):
        for col in range(len(data_list[row])):
            if isinstance(data_list[row][col], str):
                for char in chars:
                    data_list[row][col] = data_list[row][col].replace(char, "")
                    
    return data_list

def filter_metrics_with_limits(limits_df: pd.DataFrame) -> List[str]:
    """
    Filter out metrics with limits
    
    Parameters:
    -----------
    limits_df : pd.DataFrame
        DataFrame containing limits
    
    Returns:
    --------
    List[str]
        List of metrics with valid limits
    """
    metrics_with_limits = []
    
    for column in limits_df.columns:
        if any(is_number(limits_df[column].loc[index]) for index in limits_df.index):
            metrics_with_limits.append(column)
            
    return metrics_with_limits

def get_smt_data_from_csv(csv_name: str, label: str = 'log') -> Tuple:
    """
    Extract and process data from CSV file
    
    Parameters:
    -----------
    csv_name : str
        Path to the CSV file
    label : str
        Label for the data source
    
    Returns:
    --------
    Tuple
        (data_df, limits_df, metrics_with_limits, sn_string, usl_string, lsl_string)
    """
    try:
        # Use configuration from config object
        sn_strings = config.SN_STRINGS
        usl_strings = config.USL_STRINGS
        lsl_strings = config.LSL_STRINGS
        drop_dup = config.DROP_DUP
        pass_only = config.PASS_ONLY
        
        # Read and clean CSV data
        csv_raw_data = csv_to_list(csv_name)
        if not csv_raw_data:
            raise ValueError("Empty or invalid CSV file")
        
        log_raw_data = remove_thousand_sep(csv_raw_data, [","])
        
        # Find required headers
        sn_string, index_sn = find_known_words(log_raw_data, sn_strings)
        usl_string, index_usl = find_known_words(log_raw_data, usl_strings)
        lsl_string, index_lsl = find_known_words(log_raw_data, lsl_strings)
        
        if None in [index_sn, index_usl, index_lsl]:
            raise ValueError("Could not find required headers in CSV")
        
        limits_df = get_limits(log_raw_data, index_sn[0], index_usl[1], usl_string, lsl_string)
        if limits_df.empty:
            raise ValueError("Could not extract limits from data")
        
        # Handle duplicated columns
        if limits_df.columns.duplicated().any():
            logging.warning("Found duplicated columns in data")
            limits_df = limits_df.loc[:, ~limits_df.columns.duplicated()].copy()
        
        metrics_with_limits = filter_metrics_with_limits(limits_df)
        if not metrics_with_limits:
            raise ValueError("No metrics with valid limits found")
        
        first_metric = metrics_with_limits[0]
        _, index_first_metric = find_known_words(log_raw_data, [first_metric])
        if index_first_metric is None:
            raise ValueError(f"Could not locate column for first metric: {first_metric}")
        
        # Convert to numeric
        limits_nan_df = pd_str_to_num_with_nan(limits_df, index_first_metric[1])
        
        # Get valid data
        log_data_df = get_valid_data(
            log_raw_data,
            index_sn[1],
            index_first_metric[1],
            max(index_usl[0], index_lsl[0]) + 1,
            index_sn[0]
        )
        
        if log_data_df is None:
            raise ValueError("No valid data rows found")
        
        # Handle duplicates
        if log_data_df.columns.duplicated().any():
            logging.warning("Found duplicated columns in log data")
            log_data_df = log_data_df.loc[:, ~log_data_df.columns.duplicated()].copy()
        
        # Handle row duplicates
        if drop_dup in ["first", "last"]:
            log_data_df_dropdup = log_data_df.drop_duplicates(subset=sn_string, keep=drop_dup)
        else:
            log_data_df_dropdup = log_data_df.copy(deep=True)
        
        # Filter pass-only data if requested
        if pass_only:
            log_data_passonly_df = log_data_df_dropdup.copy(deep=True)
            numeric_metrics = metrics_with_limits
            
            # Convert to float for comparison
            data_values = log_data_df_dropdup[numeric_metrics].astype(float)
            usl_values = limits_df.loc[usl_string, numeric_metrics].astype(float)
            lsl_values = limits_df.loc[lsl_string, numeric_metrics].astype(float)
            
            # Apply pass/fail filters
            usl_fail = data_values > usl_values
            lsl_fail = data_values < lsl_values
            log_data_passonly_df[numeric_metrics] = data_values.mask(usl_fail | lsl_fail)
        else:
            log_data_passonly_df = log_data_df_dropdup
        
        # Add label columns
        limits_nan_df = pd.concat([pd.Series([label] * len(limits_nan_df), name="Label"), limits_nan_df], axis=1)
        log_data_passonly_df = pd.concat(
            [pd.Series([label] * len(log_data_passonly_df), name="Label"), log_data_passonly_df], axis=1)
        
        return log_data_passonly_df, limits_nan_df, metrics_with_limits, sn_string, usl_string, lsl_string
    
    except Exception as e:
        logging.error(f"Error processing {csv_name}: {str(e)}")
        raise

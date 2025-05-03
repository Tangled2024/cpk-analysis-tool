"""
Statistics module for CPK Analysis Tool
Handles CPK calculations and statistical analysis
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Dict, Any, Optional

from config import config
from data_utils import find_strings_in_list

def calc_cpk(data_df: pd.DataFrame,
             limits_df: pd.DataFrame,
             metrics_with_limits: List[str],
             usl_string: str,
             lsl_string: str,
             if_sort: bool = True,
             sort_column: str = 'cpk',
             sort_axis: int = 0,
             label: str = 'log') -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """
    Calculate CPK statistics with proper error handling
    
    Parameters:
    -----------
    data_df : pd.DataFrame
        DataFrame containing measurement data
    limits_df : pd.DataFrame
        DataFrame containing USL and LSL limits
    metrics_with_limits : List[str]
        List of metrics that have defined limits
    usl_string : str
        String identifying the USL row
    lsl_string : str
        String identifying the LSL row
    if_sort : bool
        Whether to sort the results
    sort_column : str
        Column to sort by
    sort_axis : int
        Axis to sort on (0 for index, 1 for columns)
    label : str
        Label for the data source
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]
        (metrics_df, waived_df, metrics_with_cpk, metrics_without_cpk)
    """
    try:
        # Get configuration values
        usl_only_strings = config.USL_ONLY_STRINGS
        lsl_only_strings = config.LSL_ONLY_STRINGS
        waived_strings = config.CPK_WAIVED_STRINGS
        
        # Initialize DataFrame with proper typing
        stat_metrics_df = pd.DataFrame(index=metrics_with_limits)
        stat_metrics_df["Label"] = label
        
        # Safely get limits with explicit type conversion
        usl_values = limits_df.loc[usl_string, metrics_with_limits]
        lsl_values = limits_df.loc[lsl_string, metrics_with_limits]
        stat_metrics_df["usl"] = pd.to_numeric(usl_values, errors='coerce')
        stat_metrics_df["lsl"] = pd.to_numeric(lsl_values, errors='coerce')
        
        # Skip parameters where USL == LSL with proper type handling
        comparison_result = stat_metrics_df["usl"] == stat_metrics_df["lsl"]
        if isinstance(comparison_result, pd.Series):
            same_limits_mask = comparison_result
        else:
            # Handle case where comparison returns a scalar
            same_limits_mask = pd.Series([bool(comparison_result)] * len(metrics_with_limits),
                                        index=metrics_with_limits)
        
        metrics_with_limits = [m for m in metrics_with_limits
                              if not same_limits_mask.get(m, False)]
        
        if not metrics_with_limits:
            logging.warning("No parameters with different USL/LSL values found")
            return pd.DataFrame(), pd.DataFrame(), [], []
        
        # Reinitialize DataFrame with filtered metrics
        stat_metrics_df = pd.DataFrame(index=metrics_with_limits)
        stat_metrics_df["Label"] = label
        stat_metrics_df["usl"] = limits_df.loc[usl_string, metrics_with_limits].astype(float)
        stat_metrics_df["lsl"] = limits_df.loc[lsl_string, metrics_with_limits].astype(float)
        
        # Calculate basic statistics
        stat_metrics_df["mean"] = data_df[metrics_with_limits].mean(axis=0)
        stat_metrics_df["min"] = data_df[metrics_with_limits].min(axis=0)
        stat_metrics_df["max"] = data_df[metrics_with_limits].max(axis=0)
        stat_metrics_df["std"] = data_df[metrics_with_limits].std(axis=0)
        
        # Skip parameters where max == min (no variation) with proper type handling
        max_min_comparison = stat_metrics_df["max"] == stat_metrics_df["min"]
        if isinstance(max_min_comparison, pd.Series):
            no_variation_mask = max_min_comparison
        else:
            no_variation_mask = pd.Series([bool(max_min_comparison)] * len(metrics_with_limits),
                                         index=metrics_with_limits)
        
        metrics_with_variation = [m for m in metrics_with_limits
                                 if not no_variation_mask.get(m, False)]
        metrics_with_limits = metrics_with_variation
        
        if not metrics_with_limits:
            logging.warning("No parameters with variation found")
            return pd.DataFrame(), pd.DataFrame(), [], []
        
        # Reinitialize DataFrame again with metrics that have variation
        stat_metrics_df = stat_metrics_df.loc[metrics_with_limits]
        
        # Calculate CPU and CPL with division by zero protection
        with np.errstate(divide='ignore', invalid='ignore'):
            # Vectorized calculation for efficiency
            stat_metrics_df["cpu"] = ((stat_metrics_df["usl"] - stat_metrics_df["mean"]) / 
                                     (3.0 * stat_metrics_df["std"])).astype(float)
            
            stat_metrics_df["cpl"] = ((stat_metrics_df["mean"] - stat_metrics_df["lsl"]) / 
                                     (3.0 * stat_metrics_df["std"])).astype(float)
        
        # Handle special cases with explicit type checking
        if usl_only_strings and len(usl_only_strings[0].strip()) > 1:
            metrics_usl_only, _ = find_strings_in_list(metrics_with_limits, usl_only_strings)
            stat_metrics_df.loc[metrics_usl_only, "cpl"] = np.nan
        
        if lsl_only_strings and len(lsl_only_strings[0].strip()) > 1:
            metrics_lsl_only, _ = find_strings_in_list(metrics_with_limits, lsl_only_strings)
            stat_metrics_df.loc[metrics_lsl_only, "cpu"] = np.nan
        
        # Calculate CPK with explicit typing
        cpk_values = stat_metrics_df[["cpu", "cpl"]].min(axis=1)
        stat_metrics_df["cpk"] = cpk_values.astype(float)
        stat_metrics_df["N"] = data_df[metrics_with_limits].count(axis=0).astype(int)
        
        # Handle waived metrics with explicit type checking
        metrics_without_cpk, metrics_with_cpk = find_strings_in_list(metrics_with_limits, waived_strings)
        
        # Sort results if requested
        if if_sort:
            stat_metrics_bycpk_df = stat_metrics_df.loc[metrics_with_cpk].sort_values(
                by=sort_column, axis=sort_axis)
            stat_waived_bycpk_df = stat_metrics_df.loc[metrics_without_cpk].sort_values(
                by=sort_column, axis=sort_axis)
        else:
            stat_metrics_bycpk_df = stat_metrics_df.loc[metrics_with_cpk]
            stat_waived_bycpk_df = stat_metrics_df.loc[metrics_without_cpk]
        
        return (stat_metrics_bycpk_df, stat_waived_bycpk_df,
                metrics_with_cpk, metrics_without_cpk)
    
    except Exception as e:
        logging.error(f"Error calculating CPK: {e}")
        raise

def analyze_low_variance_metrics(all_log_data: Dict[str, pd.DataFrame], 
                                all_stat_data: Dict[str, pd.DataFrame],
                                labels: List[str],
                                variance_threshold: float = 1e-8) -> List[str]:
    """
    Identify metrics with low variance that might cause issues with KDE
    
    Parameters:
    -----------
    all_log_data : Dict[str, pd.DataFrame]
        Dictionary of DataFrames containing measurement data
    all_stat_data : Dict[str, pd.DataFrame]
        Dictionary of DataFrames containing statistic data
    labels : List[str]
        List of data source labels
    variance_threshold : float
        Threshold below which variance is considered too low
        
    Returns:
    --------
    List[str]
        List of metrics with low variance
    """
    problematic_metrics = []
    
    # Get all unique metrics from all stat data
    all_metrics = set().union(*[set(df.index) for df in all_stat_data.values()])
    
    for metric in all_metrics:
        for label in labels:
            if metric in all_log_data[label].columns:
                data = all_log_data[label][metric].dropna()
                if len(data) > 1 and np.ptp(data) <= variance_threshold:
                    problematic_metrics.append(metric)
                    break
    
    return problematic_metrics

def sort_metrics_by_cpk(all_stat_data: Dict[str, pd.DataFrame], 
                       labels: List[str], 
                       cpk_threshold: float = 1.67) -> Tuple[List[str], List[str]]:
    """
    Sort metrics by CPK value from the last file
    
    Parameters:
    -----------
    all_stat_data : Dict[str, pd.DataFrame]
        Dictionary of DataFrames containing statistic data
    labels : List[str]
        List of data source labels
    cpk_threshold : float
        Threshold for filtering metrics if not plotting all
        
    Returns:
    --------
    Tuple[List[str], List[str]]
        (sorted_metrics, filtered_metrics)
    """
    # Get all metrics that appear in any file
    all_metrics = set().union(*[set(df.index) for df in all_stat_data.values()])
    
    # Create list of (metric, last_file_cpk) tuples for sorting
    metric_cpk_pairs = []
    for metric in all_metrics:
        # Try to get CPK from last file first
        if metric in all_stat_data[labels[-1]].index:
            cpk = all_stat_data[labels[-1]].loc[metric, 'cpk']
        else:
            # Fall back to first available CPK if not in last file
            cpk = None
            for label in labels:
                if metric in all_stat_data[label].index:
                    cpk = all_stat_data[label].loc[metric, 'cpk']
                    break
        
        # Handle NaN values by putting them last
        sort_value = cpk if not pd.isna(cpk) else float('inf')
        metric_cpk_pairs.append((metric, sort_value))
    
    # Sort by CPK (ascending) - worst CPK first, NaN values last
    sorted_metrics = [m[0] for m in sorted(metric_cpk_pairs, key=lambda x: x[1])]
    
    # Create filtered version for low CPK values
    filtered_metrics = [m for m in sorted_metrics
                      if any(all_stat_data[label].loc[m, 'cpk'] <= cpk_threshold
                            for label in labels if m in all_stat_data[label].index)]
    
    return sorted_metrics, filtered_metrics

"""
Visualization module for CPK Analysis Tool
Handles plotting and PDF report generation
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import gaussian_kde
from typing import Dict, List, Tuple, Optional, Any, Set

from config import config
from stats import analyze_low_variance_metrics

# Set global font settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 8,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'figure.titlesize': 12
})

def format_value(value):
    """
    Format numerical values with appropriate decimal points or scientific notation
    
    Parameters:
    -----------
    value : float or int
        Value to format
        
    Returns:
    --------
    str
        Formatted value string:
        - Very large numbers (≥1E9) shown in scientific notation (e.g. 2E+09)
        - Medium large numbers (≥1E7) shown in shorter form (e.g. 24.0M)
        - Integers or values with zeros after decimal shown without decimal points
        - Normal decimal values shown with 2 decimal places
    """
    if pd.isna(value):
        return "N/A"
    
    # For integers or values with zeros after decimal, don't show decimal points
    if isinstance(value, (int, float)):
        # Use scientific notation for very large numbers
        abs_value = abs(value)
        if abs_value >= 1e9:  # For values ≥ 1 billion
            # Format with E notation and no decimal places
            return f"{value:.0e}".replace("e+0", "E+").replace("e+", "E+").replace("e-0", "E-").replace("e-", "E-")
        # Use millions format for medium large numbers
        elif abs_value >= 1e7:  # For values ≥ 10 million
            value_in_millions = value / 1e6
            if value_in_millions == int(value_in_millions):
                return f"{int(value_in_millions)}M"
            else:
                return f"{value_in_millions:.1f}M"
        # For integers or values that are effectively integers
        elif value == int(value):
            return f"{int(value)}"
        # For normal decimal values
        else:
            return f"{value:.2f}"
    else:
        return str(value)

def plot_to_pdf(output_path: str, 
                all_log_data: Dict[str, pd.DataFrame], 
                all_limits_data: Dict[str, pd.DataFrame], 
                all_stat_data: Dict[str, pd.DataFrame], 
                labels: List[str], 
                sorted_metrics: List[str]) -> Tuple[bool, str]:
    """
    Generate comprehensive PDF report with dynamic-width summary tables
    
    Parameters:
    -----------
    output_path : str
        Directory path for output files
    all_log_data : Dict[str, pd.DataFrame]
        Dictionary of DataFrames containing measurement data
    all_limits_data : Dict[str, pd.DataFrame]
        Dictionary of DataFrames containing limit data
    all_stat_data : Dict[str, pd.DataFrame]
        Dictionary of DataFrames containing statistic data
    labels : List[str]
        List of data source labels
    sorted_metrics : List[str]
        List of metrics sorted by CPK
        
    Returns:
    --------
    Tuple[bool, str]
        (success, pdf_path)
    """
    pdf_path = ""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Update filename to reflect filter status
        filter_status = f"_Pass{int(config.PASS_ONLY)}_Last{int(config.DROP_DUP == 'last')}"
        pdf_path = os.path.join(output_path, config.PDF_NAME.replace('.pdf', f'{filter_status}.pdf'))
        
        # Check for metrics with low variance (skip KDE for these)
        logging.info("Checking for metrics with low variance...")
        problematic_metrics = analyze_low_variance_metrics(all_log_data, all_stat_data, labels)
        
        if problematic_metrics:
            logging.info("These metrics have low variance (KDE will be skipped):")
            for metric in problematic_metrics:
                logging.info(f"- {metric}")
        else:
            logging.info("No low-variance metrics detected")
        
        with PdfPages(pdf_path) as pdf:
            logging.info(f"Generating PDF report at: {pdf_path} [Pass Only: {config.PASS_ONLY}, Last Only: {config.DROP_DUP}]")
            
            # Filter metrics if not in plot-all mode
            plotting_metrics = sorted_metrics
            if not config.IF_PLOT_ALL:
                plotting_metrics = [m for m in sorted_metrics 
                                    if any(all_stat_data[label].loc[m, 'cpk'] <= 1.67
                                          for label in labels if m in all_stat_data[label].index)]
            
            # Filter metrics that have no limits unless Plot All is selected
            if not config.IF_PLOT_ALL:
                filtered_metrics = []
                for metric in plotting_metrics:
                    # Check if at least one file has limits for this metric
                    has_limits = False
                    for label in labels:
                        if metric in all_stat_data[label].index:
                            usl = all_stat_data[label].loc[metric, 'usl']
                            lsl = all_stat_data[label].loc[metric, 'lsl']
                            if not (pd.isna(usl) and pd.isna(lsl)):
                                has_limits = True
                                break
                    
                    # Only include metrics that have limits or if Plot All is enabled
                    if has_limits:
                        filtered_metrics.append(metric)
                    else:
                        logging.info(f"Skipping metric without limits: {metric} (use Plot All to include)")
                
                metrics_to_plot = filtered_metrics
            else:
                metrics_to_plot = sorted_metrics
            
            # ==================================================================
            # 1. MULTI-FILE COMPARISON TABLE (FOR 2+ FILES)
            # ==================================================================
            if len(labels) >= 2:
                generate_comparison_tables(pdf, all_stat_data, labels, metrics_to_plot)
            
            # ==================================================================
            # 2. SUMMARY TABLE (ALL CASES)
            # ==================================================================
            generate_summary_tables(pdf, all_stat_data, labels, metrics_to_plot)
            
            # ==================================================================
            # 3. INDIVIDUAL METRIC PLOTS
            # ==================================================================
            for metric in metrics_to_plot:
                try:
                    # Skip if we shouldn't plot all and metric is passing in all files
                    if not config.IF_PLOT_ALL:
                        all_passing = True
                        for label in labels:
                            if metric in all_stat_data[label].index:
                                if all_stat_data[label].loc[metric, 'cpk'] <= 1.67:
                                    all_passing = False
                                    break
                        if all_passing:
                            continue
                    
                    generate_metric_plot(pdf, all_log_data, all_stat_data, labels, metric, problematic_metrics)
                    
                except Exception as e:
                    logging.error(f"Skipping {metric}: {str(e)}")
                    plt.close('all')  # Ensure all figures are closed
                    continue
        
        logging.info(f"PDF created at: {pdf_path}")
        return True, pdf_path
    
    except Exception as e:
        logging.error(f"PDF generation failed: {str(e)}")
        if 'pdf_path' in locals() and os.path.exists(pdf_path):
            os.remove(pdf_path)
        return False, ""

def generate_comparison_tables(pdf: PdfPages, 
                              all_stat_data: Dict[str, pd.DataFrame], 
                              labels: List[str], 
                              sorted_metrics: List[str]) -> None:
    """
    Generate comparison tables for multiple files
    
    Parameters:
    -----------
    pdf : PdfPages
        PDF pages object to save to
    all_stat_data : Dict[str, pd.DataFrame]
        Dictionary of DataFrames containing statistic data
    labels : List[str]
        List of data source labels
    sorted_metrics : List[str]
        List of metrics sorted by CPK
    """
    # Find parameters common to ALL files with matching USL/LSL
    common_params = set(all_stat_data[labels[0]].index)
    for label in labels[1:]:
        common_params.intersection_update(all_stat_data[label].index)
    
    # Filter to only include sorted metrics
    common_sorted_metrics = [m for m in sorted_metrics if m in common_params]
    
    # Generate comparison data
    comparison_data = []
    for metric in common_sorted_metrics:
        # Verify USL/LSL match across all files
        base_usl = all_stat_data[labels[0]].loc[metric, 'usl']
        base_lsl = all_stat_data[labels[0]].loc[metric, 'lsl']
        
        usl_match = all(all_stat_data[label].loc[metric, 'usl'] == base_usl for label in labels)
        lsl_match = all(all_stat_data[label].loc[metric, 'lsl'] == base_lsl for label in labels)
        
        if usl_match and lsl_match:
            # Get N from last file
            n = all_stat_data[labels[-1]].loc[metric, 'N']
            
            # Build row with numbered columns
            row = [metric, format_value(base_usl), format_value(base_lsl)]
            
            # Add all means (mean, mean1, mean2...)
            for label in labels:
                row.append(f"{all_stat_data[label].loc[metric, 'mean']:.2f}")
            
            # Add all mins (min, min1, min2...)
            for label in labels:
                row.append(f"{all_stat_data[label].loc[metric, 'min']:.2f}")
            
            # Add all maxes (max, max1, max2...)
            for label in labels:
                row.append(f"{all_stat_data[label].loc[metric, 'max']:.2f}")
            
            # Add all stds (std, std1, std2...)
            for label in labels:
                row.append(f"{all_stat_data[label].loc[metric, 'std']:.2f}")
            
            # Add all CPUs (cpu, cpu1, cpu2...)
            for label in labels:
                row.append(f"{all_stat_data[label].loc[metric, 'cpu']:.2f}")
            
            # Add all CPLs (cpl, cpl1, cpl2...)
            for label in labels:
                row.append(f"{all_stat_data[label].loc[metric, 'cpl']:.2f}")
            
            # Add all CPKs (cpk, cpk1, cpk2...)
            for label in labels:
                row.append(f"{all_stat_data[label].loc[metric, 'cpk']:.2f}")
            
            row.append(str(n))
            comparison_data.append(row)
    
    if comparison_data:
        # Generate column headers with numbered format
        col_labels = (
            ["Parameter", "USL", "LSL"] +
            ["Mean"] + [f"Mean{i}" for i in range(1, len(labels))] +
            ["Min"] + [f"Min{i}" for i in range(1, len(labels))] +
            ["Max"] + [f"Max{i}" for i in range(1, len(labels))] +
            ["Std"] + [f"Std{i}" for i in range(1, len(labels))] +
            ["CPU"] + [f"CPU{i}" for i in range(1, len(labels))] +
            ["CPL"] + [f"CPL{i}" for i in range(1, len(labels))] +
            ["CPK"] + [f"CPK{i}" for i in range(1, len(labels))] +
            ["N"]
        )
        
        # Split into chunks for pagination
        chunks = [comparison_data[i:i + 20] for i in range(0, len(comparison_data), 20)]
        
        for page_num, chunk in enumerate(chunks, 1):
            fig = plt.figure(figsize=(16, 9), dpi=100)
            ax = fig.add_subplot(111)
            ax.axis('off')
            
            # Create table with numbered columns
            table = ax.table(
                cellText=chunk,
                colLabels=col_labels,
                cellLoc='center',
                loc='center',
                colWidths=[0.45] + [0.04] * 2 + [0.05] * (len(col_labels) - 3)
            )
            
            # Style table with CPK highlighting
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            for (i, j), cell in table.get_celld().items():
                cell.set_edgecolor('black')
                cell.set_linewidth(0.5)
                if i == 0:  # Header row
                    cell.set_facecolor('#f0f0f0')
                    cell.set_text_props(weight='bold', color='black')
                
                # Highlight low CPK values in red (CPK < 1.50)
                if j >= len(col_labels) - len(labels) - 1 and j < len(col_labels) - 1:  # CPK columns
                    try:
                        cpk_value = float(cell.get_text().get_text())
                        if cpk_value < 1.50:
                            cell.set_facecolor('#ffcccc')  # Light red
                            cell.set_text_props(weight='bold', color='red')
                    except (ValueError, AttributeError):
                        pass
                
                cell.set_height(0.05)
            
            plt.title(
                f"Parameter Comparison: {' vs '.join(labels)}\n(Sorted by {labels[-1]} CPK)\nPage {page_num} of {len(chunks)}",
                fontsize=12, pad=20)
            plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

def generate_summary_tables(pdf: PdfPages, 
                           all_stat_data: Dict[str, pd.DataFrame], 
                           labels: List[str], 
                           sorted_metrics: List[str]) -> None:
    """
    Generate summary tables for all metrics
    
    Parameters:
    -----------
    pdf : PdfPages
        PDF pages object to save to
    all_stat_data : Dict[str, pd.DataFrame]
        Dictionary of DataFrames containing statistic data
    labels : List[str]
        List of data source labels
    sorted_metrics : List[str]
        List of metrics sorted by CPK
    """
    # Prepare summary data - group by metric (not by file)
    summary_data = []
    for metric in sorted_metrics:
        for label in labels:
            if metric in all_stat_data[label].index:
                stat_df = all_stat_data[label]
                row = [label, metric] + [
                    format_value(stat_df.loc[metric, col])
                    if col in ['usl', 'lsl'] else
                    f"{stat_df.loc[metric, col]:.2f}"
                    if isinstance(stat_df.loc[metric, col], (int, float)) and not pd.isna(stat_df.loc[metric, col])
                    else str(stat_df.loc[metric, col])
                    for col in ['usl', 'lsl', 'mean', 'min', 'max', 'std', 'cpu', 'cpl', 'cpk', 'N']
                ]
                summary_data.append(row)
            else:
                row = [label, metric] + ["N/A"] * 10
                summary_data.append(row)
    
    # Split into chunks of max 20 metrics per page
    chunks = [summary_data[i:i + 20] for i in range(0, len(summary_data), 20)]
    
    for page_num, data_chunk in enumerate(chunks, 1):
        fig = plt.figure(figsize=(16, 9), dpi=100)
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        # Calculate dynamic width for Parameter column
        page_params = [row[1] for row in data_chunk]
        max_param_len = max(len(str(param)) for param in page_params) if page_params else 20
        param_width = min(0.1 + max_param_len * 0.01, 0.35)
        
        # Create table with adjusted column widths
        table = ax.table(
            cellText=data_chunk,
            colLabels=["Label", "Parameter", "USL", "LSL", "Mean", "Min", "Max",
                      "Std", "CPU", "CPL", "CPK", "N"],
            cellLoc='center',
            loc='center',
            colWidths=[
                0.08,  # Label
                param_width,  # Parameter (dynamic)
                0.07, 0.07,  # USL/LSL
                0.08,  # Mean
                0.07, 0.07,  # Min/Max
                0.07,  # Std
                0.07, 0.07,  # CPU/CPL
                0.07,  # CPK
                0.05  # N
            ]
        )
        
        # Style table with CPK highlighting
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        for (i, j), cell in table.get_celld().items():
            cell.set_edgecolor('black')
            cell.set_linewidth(0.5)
            if i == 0:  # Header row
                cell.set_facecolor('#dddddd')
                cell.set_text_props(weight='bold')
            
            # Highlight low CPK values in red (CPK < 1.50)
            if j == 10 and i > 0:  # CPK column (0-based index 10), skip header row
                try:
                    cpk_value = float(cell.get_text().get_text())
                    if cpk_value < 1.50:
                        cell.set_facecolor('#ffcccc')  # Light red
                        cell.set_text_props(weight='bold', color='red')
                except (ValueError, AttributeError):
                    pass
            
            cell.set_height(0.05)
        
        # Dynamic title
        title_parts = ["Test Parameters Summary (Sorted by Last File CPK)"]
        if config.IF_PLOT_ALL:
            title_parts.append("All Parameters")
        else:
            title_parts.append("CPK ≤ 1.67")
        title_parts.append(f"Page {page_num} of {len(chunks)}")
        
        plt.title("\n".join(title_parts), y=1.02, fontsize=12)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

def generate_metric_plot(pdf: PdfPages, 
                        all_log_data: Dict[str, pd.DataFrame], 
                        all_stat_data: Dict[str, pd.DataFrame], 
                        labels: List[str], 
                        metric: str, 
                        problematic_metrics: List[str]) -> None:
    """
    Generate detailed plot for a single metric
    
    Parameters:
    -----------
    pdf : PdfPages
        PDF pages object to save to
    all_log_data : Dict[str, pd.DataFrame]
        Dictionary of DataFrames containing measurement data
    all_stat_data : Dict[str, pd.DataFrame]
        Dictionary of DataFrames containing statistic data
    labels : List[str]
        List of data source labels
    metric : str
        Metric to plot
    problematic_metrics : List[str]
        List of metrics with low variance to skip KDE
    """
    # Create figure with grid layout
    fig = plt.figure(figsize=(16, 9), dpi=100)
    
    # Define a different grid layout that places tables under each plot
    gs = fig.add_gridspec(2, 2, 
                         height_ratios=[2, 1],  # Adjusted ratio for more space for tables
                         hspace=0.4, wspace=0.3,
                         left=0.08, right=0.95,
                         top=0.92, bottom=0.05)  # Extend bottom to make room
    
    # Create subplots - boxplot and table on left, histogram and table on right
    ax_box = fig.add_subplot(gs[0, 0])      # Boxplot (top left)
    ax_box_table = fig.add_subplot(gs[1, 0])  # Table for boxplot (bottom left)
    ax_box_table.axis('off')
    
    ax_hist = fig.add_subplot(gs[0, 1])     # Histogram with KDE (top right)
    ax_hist_table = fig.add_subplot(gs[1, 1]) # Table for histogram (bottom right)
    ax_hist_table.axis('off')
    
    # Title with CPK from last file
    if metric in all_stat_data[labels[-1]].index:
        last_cpk = all_stat_data[labels[-1]].loc[metric, 'cpk']
        cpk_str = f"CPK: {last_cpk:.2f}" if not pd.isna(last_cpk) else "CPK: N/A"
        
        # Check if limits exist
        usl = all_stat_data[labels[-1]].loc[metric, 'usl']
        lsl = all_stat_data[labels[-1]].loc[metric, 'lsl']
        has_limits = not (pd.isna(usl) and pd.isna(lsl))
        limits_str = "" if has_limits else " (No Limits)"
        
        fig.suptitle(
            f"{metric}\n(Sorted by {labels[-1]} {cpk_str}{limits_str})",
            fontsize=14, y=0.98)
    else:
        fig.suptitle(f"{metric}", fontsize=14, y=0.98)
    
    # --- BOXPLOT ---
    box_data = []
    box_labels = []
    file_usl_lsl = {}  # Store USL/LSL per file
    
    for i, label in enumerate(labels):
        if metric in all_log_data[label].columns:
            metric_data = all_log_data[label][metric].dropna()
            if len(metric_data) > 0:
                box_data.append(metric_data)
                box_labels.append(label)
                
                # Store USL/LSL for this file
                if metric in all_stat_data[label].index:
                    file_usl_lsl[label] = {
                        'usl': all_stat_data[label].loc[metric, 'usl'],
                        'lsl': all_stat_data[label].loc[metric, 'lsl']
                    }
    
    if box_data:
        boxplot = ax_box.boxplot(box_data, patch_artist=True)
        ax_box.set_xticklabels(box_labels)
        
        # Set colors for boxes
        for i, box in enumerate(boxplot['boxes']):
            box.set_facecolor(config.COLOR_PALETTE[i % len(config.COLOR_PALETTE)])
        
        # Add USL/LSL lines for each file
        for i, label in enumerate(file_usl_lsl.keys(), 1):
            color = config.COLOR_PALETTE[(i - 1) % len(config.COLOR_PALETTE)]
            usl = file_usl_lsl[label]['usl']
            lsl = file_usl_lsl[label]['lsl']
            
            ax_box.axhline(usl, color=color, linestyle='--', linewidth=1,
                          alpha=0.7, label=f'{label} USL')
            ax_box.axhline(lsl, color=color, linestyle=':', linewidth=1,
                          alpha=0.7, label=f'{label} LSL')
        
        ax_box.set_ylabel("Value")
        ax_box.set_title("Boxplot Comparison", pad=10)
        ax_box.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_box.grid(True, linestyle='--', alpha=0.3)
    
    # --- HISTOGRAM with KDE ---
    hist_data = {}  # Store histogram data for table
    
    for i, label in enumerate(labels):
        if metric in all_log_data[label].columns:
            metric_data = all_log_data[label][metric].dropna()
            hist_data[label] = metric_data  # Store for table
            
            if len(metric_data) > 1:
                # Plot histogram
                n_bins = min(30, len(metric_data) // 5 + 1)  # Ensure at least 1 bin
                hist_color = config.COLOR_PALETTE[i % len(config.COLOR_PALETTE)]
                ax_hist.hist(metric_data, bins=n_bins, density=True,
                            alpha=0.5, color=hist_color,
                            label=f"{label} Histogram")
                
                # Add KDE if sufficient variance and not problematic
                if metric not in problematic_metrics and np.ptp(metric_data) > 1e-8:
                    try:
                        kde = gaussian_kde(metric_data)
                        xmin, xmax = ax_hist.get_xlim()
                        x = np.linspace(xmin, xmax, 300)
                        ax_hist.plot(x, kde(x),
                                    color=hist_color,
                                    linestyle='-',
                                    linewidth=1.5,
                                    label=f"{label} KDE")
                    except Exception as e:
                        logging.warning(f"Could not generate KDE for {metric}: {e}")
                
                # Add USL/LSL lines
                if metric in all_stat_data[label].index:
                    usl = all_stat_data[label].loc[metric, 'usl']
                    lsl = all_stat_data[label].loc[metric, 'lsl']
                    
                    has_limits = not (pd.isna(usl) and pd.isna(lsl))
                    
                    if has_limits:
                        if not pd.isna(usl):
                            ax_hist.axvline(usl, color=hist_color, linestyle='--',
                                        linewidth=1, alpha=0.7, label=f'{label} USL')
                        if not pd.isna(lsl):
                            ax_hist.axvline(lsl, color=hist_color, linestyle=':',
                                        linewidth=1, alpha=0.7, label=f'{label} LSL')
                    else:
                        # For distributions without limits, automatically adjust the x-axis
                        # to focus on the data (removing extreme outliers)
                        if len(metric_data) > 10:  # Only if we have enough data points
                            # Get percentiles to exclude extreme outliers
                            lower = np.percentile(metric_data, 1)  # 1st percentile
                            upper = np.percentile(metric_data, 99)  # 99th percentile
                            
                            # Add some margin
                            margin = (upper - lower) * 0.1
                            ax_hist.set_xlim(lower - margin, upper + margin)
                            
                            # Add note about auto-scaling
                            if i == 0:  # Only add once
                                ax_hist.text(0.98, 0.02, "Auto-scaled (no limits)", 
                                            transform=ax_hist.transAxes,
                                            ha='right', va='bottom', 
                                            fontsize=7, style='italic',
                                            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
    
    ax_hist.set_xlabel("Value")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("Distribution", pad=10)
    ax_hist.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_hist.grid(True, linestyle='--', alpha=0.3)
    
    # --- Create Summary Statistics Tables ---
    try:
        # Base table data with headers - same for both tables
        table_headers = ["Label", "USL", "LSL", "Mean", "Min", "Max", "Std", "CPU", "CPL", "CPK", "N"]
        box_table_data = [table_headers.copy()]
        hist_table_data = [table_headers.copy()]
        
        # Add data for each label
        for label in labels:
            if metric in all_stat_data[label].index:
                stats_row = all_stat_data[label].loc[metric]
                
                # Format stats data
                row_data = [
                    label,
                    format_value(stats_row['usl']),
                    format_value(stats_row['lsl']), 
                    f"{stats_row['mean']:.2f}" if not pd.isna(stats_row['mean']) else "N/A",
                    f"{stats_row['min']:.2f}" if not pd.isna(stats_row['min']) else "N/A",
                    f"{stats_row['max']:.2f}" if not pd.isna(stats_row['max']) else "N/A",
                    f"{stats_row['std']:.2f}" if not pd.isna(stats_row['std']) else "N/A",
                    f"{stats_row['cpu']:.2f}" if not pd.isna(stats_row['cpu']) else "N/A",
                    f"{stats_row['cpl']:.2f}" if not pd.isna(stats_row['cpl']) else "N/A",
                    f"{stats_row['cpk']:.2f}" if not pd.isna(stats_row['cpk']) else "N/A",
                    f"{stats_row['N']}" if not pd.isna(stats_row['N']) else "N/A"
                ]
                
                # Add same data to both tables
                box_table_data.append(row_data)
                hist_table_data.append(row_data)
        
        # Calculate column widths (same for both tables)
        col_widths = [0.1] * len(table_headers)  # Equal width for each column
        
        # Create the boxplot statistics table
        if len(box_table_data) > 1:
            box_table = ax_box_table.table(
                cellText=box_table_data,
                colWidths=col_widths,
                cellLoc='center',
                loc='center'
            )
            
            # Style the table
            for (i, j), cell in box_table.get_celld().items():
                if i == 0:  # Header row
                    cell.set_facecolor('#dddddd')
                    cell.set_text_props(weight='bold')
                
                # Highlight low CPK values
                if j == 9 and i > 0:  # CPK column (index 9)
                    try:
                        cpk_value = float(cell.get_text().get_text())
                        if cpk_value < 1.50:
                            cell.set_facecolor('#ffcccc')  # Light red
                            cell.set_text_props(weight='bold', color='red')
                    except (ValueError, AttributeError):
                        pass
                        
                cell.set_edgecolor('black')
                cell.set_linewidth(0.5)
                
            box_table.scale(1, 1.5)
            ax_box_table.set_title("Boxplot Statistics", pad=10)
        
        # Create the histogram statistics table
        if len(hist_table_data) > 1:
            hist_table = ax_hist_table.table(
                cellText=hist_table_data,
                colWidths=col_widths,
                cellLoc='center',
                loc='center'
            )
            
            # Style the table
            for (i, j), cell in hist_table.get_celld().items():
                if i == 0:  # Header row
                    cell.set_facecolor('#dddddd')
                    cell.set_text_props(weight='bold')
                
                # Highlight low CPK values
                if j == 9 and i > 0:  # CPK column (index 9)
                    try:
                        cpk_value = float(cell.get_text().get_text())
                        if cpk_value < 1.50:
                            cell.set_facecolor('#ffcccc')  # Light red
                            cell.set_text_props(weight='bold', color='red')
                    except (ValueError, AttributeError):
                        pass
                        
                cell.set_edgecolor('black')
                cell.set_linewidth(0.5)
                
            hist_table.scale(1, 1.5)
            ax_hist_table.set_title("Distribution Statistics", pad=10)
            
    except Exception as e:
        logging.error(f"Error creating tables for {metric}: {str(e)}")
    
    # Save the figure
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

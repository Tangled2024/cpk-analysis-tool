"""
Main entry point for the CPK Analysis Tool
Coordinates the user interface, data processing, and reporting
"""

import os
import sys
import time
import logging
import traceback
from typing import List, Dict, Tuple, Optional

import pandas as pd
from PyQt5.QtWidgets import QApplication

from config import config
from data_utils import get_smt_data_from_csv
from stats import calc_cpk, sort_metrics_by_cpk
from visualization import plot_to_pdf
from gui import MainWindow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    filename='cpk_analysis.log',
    filemode='w'
)
logger = logging.getLogger(__name__)

# Add error code handler
class ErrorCodeFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, 'error_code'):
            record.error_code = '00000'
        return True

logger.addFilter(ErrorCodeFilter())

def cpk_analysis_main(file_paths: List[str], main_window: Optional[MainWindow] = None) -> bool:
    """
    Main CPK analysis function
    
    Parameters:
    -----------
    file_paths : List[str]
        List of CSV file paths to analyze
    main_window : Optional[MainWindow]
        Main window for updating UI status
        
    Returns:
    --------
    bool
        True if analysis completed successfully, False otherwise
    """
    # Initialize timing variables
    time_start = time_end = None
    
    try:
        if not file_paths:
            if main_window:
                main_window.show_error("Input Error", "No log files specified")
            logger.error("No log files specified", extra={'error_code': 'E001'})
            return False
        
        # Update UI status if available
        if main_window:
            main_window.set_status("Analyzing files...")
        
        # Generate sequential labels (log, log1, log2, etc.)
        labels = []
        for i in range(len(file_paths)):
            if i == 0:
                labels.append("log")  # First file is just "log"
            else:
                labels.append(f"log{i}")  # Subsequent files are log1, log2, etc.
        
        # Set up output directory
        last_file_name = file_paths[-1]
        log_file_path = os.path.dirname(last_file_name)
        output_folder = os.path.join(log_file_path, config.OUTPUT_FOLDER_NAME)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Verify output directory is writable
        if not os.access(output_folder, os.W_OK):
            error_msg = f"Cannot write to output directory {output_folder}"
            if main_window:
                main_window.show_error("Permission Error", error_msg)
            logger.error(error_msg, extra={'error_code': 'E002'})
            return False
        
        # Data collection and processing
        all_log_data = {}
        all_limits_data = {}
        all_stat_data = {}
        
        logger.info("Getting data from log files...")
        if main_window:
            main_window.set_status("Loading and processing data...")
        
        time_start = time.time()
        
        # Process each file
        for i, (log_file, label) in enumerate(zip(file_paths, labels)):
            logger.info(f"Processing {label}: {log_file}")
            if main_window:
                main_window.set_status(f"Processing {label}: {os.path.basename(log_file)}")
            
            try:
                # Get data from CSV using the generated label
                (data_comp_temp_df, limits_df_comp_temp, metrics_with_limits_comp_temp,
                 sn_string_comp_temp, usl_string_comp_temp, lsl_string_comp_temp) = get_smt_data_from_csv(
                    log_file, label)
                
                # Calculate CPK stats
                (stat_comp_temp_df, stat_comp_wavied_temp_df, metrics_with_cpk_comp_temp,
                 metrics_without_cpk_comp_temp) = calc_cpk(
                    data_comp_temp_df, limits_df_comp_temp, metrics_with_limits_comp_temp,
                    usl_string_comp_temp, lsl_string_comp_temp, True, "cpk", 0, label)
                
                # Store results
                all_log_data[label] = data_comp_temp_df
                all_limits_data[label] = limits_df_comp_temp
                all_stat_data[label] = stat_comp_temp_df
                
                # Save individual CSV files
                csv_path = os.path.join(output_folder, f"cpk_{label}.csv")
                stat_comp_temp_df.to_csv(csv_path, na_rep="")
                logger.info(f"Saved CSV: {csv_path}")
                
                waived_csv_path = os.path.join(output_folder, f"cpk_waived_{label}.csv")
                stat_comp_wavied_temp_df.to_csv(waived_csv_path, na_rep="")
                logger.info(f"Saved waived CSV: {waived_csv_path}")
                
            except Exception as e:
                error_msg = f"Error processing file {log_file}: {str(e)}"
                logger.error(error_msg, extra={'error_code': 'E003'})
                if main_window:
                    main_window.set_status(f"Error processing {os.path.basename(log_file)}")
                continue
        
        time_end = time.time()
        logger.info(f"Data processing completed in {time_end - time_start:.2f} seconds")
        
        if not all_log_data:
            error_msg = "No data could be extracted from any of the input files"
            if main_window:
                main_window.show_error("Data Error", error_msg)
            logger.error(error_msg, extra={'error_code': 'E004'})
            return False
        
        # Sort metrics by CPK
        sorted_metrics, _ = sort_metrics_by_cpk(all_stat_data, labels)
        
        # Generate PDF report
        logger.info("Generating PDF report...")
        if main_window:
            main_window.set_status("Generating PDF report...")
        
        pdf_time_start = time.time()
        
        # Generate the combined PDF report
        success, pdf_path = plot_to_pdf(
            output_folder, all_log_data, all_limits_data, all_stat_data, labels, sorted_metrics)
        
        pdf_time_end = time.time()
        if success:
            logger.info(f"PDF generation completed in {pdf_time_end - pdf_time_start:.2f} seconds")
            logger.info("Analysis completed successfully")
            
            success_msg = f"Analysis completed successfully!\nResults saved to:\n{output_folder}"
            if main_window:
                main_window.set_status("Analysis completed successfully!")
                main_window.show_info("Analysis Complete", success_msg)
            return True
        else:
            error_msg = "PDF generation failed"
            if main_window:
                main_window.show_error("PDF Error", error_msg)
            logger.error(error_msg, extra={'error_code': 'E005'})
            return False
    
    except Exception as e:
        error_msg = f"Critical error in analysis: {str(e)}"
        logger.error(error_msg, extra={'error_code': 'E006'})
        if time_start and time_end:
            logger.info(f"Processing time: {time_end - time_start:.2f} seconds")
        
        # Log full traceback
        logger.error(traceback.format_exc())
        
        if main_window:
            main_window.set_status("Analysis failed!")
            main_window.show_error("Analysis Error", f"Error: {str(e)}")
        
        return False

def start_gui():
    """Launch the graphical user interface"""
    app = QApplication(sys.argv)
    
    # Load config from file if exists
    config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.conf")
    config.load_from_file(config_file)
    
    # Create main window
    main_window = MainWindow()
    
    # Connect analysis function
    def start_analysis():
        """Callback for the start button"""
        file_paths = main_window.get_file_paths()
        if not file_paths:
            main_window.show_error("Input Error", "No valid files found")
            return
        
        # Start analysis
        cpk_analysis_main(file_paths, main_window)
    
    # Connect start button to analysis function
    main_window.set_start_button_handler(start_analysis)
    
    # Show window
    main_window.show()
    
    # Start application event loop
    return app.exec_()

def start_cli(file_paths: List[str]):
    """
    Run analysis in command line mode
    
    Parameters:
    -----------
    file_paths : List[str]
        List of CSV file paths to analyze
    """
    # Load config from file if exists
    config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.conf")
    config.load_from_file(config_file)
    
    # Print configuration
    print("CPK Analysis Tool - Command Line Mode")
    print(f"Using configuration from: {config_file if os.path.exists(config_file) else 'defaults'}")
    print(f"Processing {len(file_paths)} files...")
    
    # Run analysis
    success = cpk_analysis_main(file_paths)
    
    # Print result
    if success:
        print("Analysis completed successfully!")
    else:
        print("Analysis failed. See log file for details.")
    
    return 0 if success else 1

if __name__ == "__main__":
    # Check if files are provided as arguments
    if len(sys.argv) > 1:
        # Command line mode
        file_paths = sys.argv[1:]
        sys.exit(start_cli(file_paths))
    else:
        # GUI mode
        sys.exit(start_gui())

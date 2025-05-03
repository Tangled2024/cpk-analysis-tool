"""
Configuration module for CPK Analysis Tool
Contains all constants and configuration parameters
"""

# Default configuration values
DEFAULT_CONFIG = {
    # String matches for identifying columns in CSV files
    "USL_STRINGS": [
        'Upper Limit ----->', 'Upper Limit----->', 'UpLimit ---->', 
        'Upper Limited----------->', 'UPPER LIMIT----->'
    ],
    "LSL_STRINGS": [
        'Lower Limit ----->', 'Lower Limit----->', 'LowLimit ---->', 
        'Lower Limited----------->', 'LOWER LIMIT----->'
    ],
    "SN_STRINGS": ["SerialNumber", "SERIALNUMBER"],
    "CP_STRINGS": ['Check Point', 'Check Points', 'check point', 'check points', 'cp', 'CP', 'Cp'],
    "T_ZERO_STRS": ["T0"],
    "CFG_STRINGS": "Config",
    "SN_PREFIX_STRS": "H0",
    
    # Analysis parameters
    "PLOT_T0": False,
    "PLOT_DRIFT": True,
    "DRIFT_SIGMA": 4,
    "IF_PLOT_BY_LOC": True,
    "IGNORE_SN": True,
    "DROP_DUP": "last",
    "PASS_ONLY": True,
    "CHECK_DRIFT": False,
    "IF_PLOT_ALL": False,
    "PEAK_VAR_USL": 15,
    
    # Special parameter lists
    "CPK_WAIVED_STRINGS": ["can list here", "can llist here"],
    "USL_ONLY_STRINGS": [""],
    "LSL_ONLY_STRINGS": [""],
    
    # Output settings
    "OUTPUT_FOLDER_NAME": "CPK_ana_outputs",
    "PDF_NAME": "CPK_Ana.pdf",
    "SAVE_FIGURES": 0,
    
    # Visualization settings
    "COLOR_PALETTE": [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
}

from typing import Dict, Any, List
from pydantic import BaseModel, validator
import os

class ConfigModel(BaseModel):
    """Main configuration model with validation"""
    OUTPUT_FOLDER_NAME: str = 'cpk_results'
    PDF_NAME: str = 'cpk_report.pdf'
    SN_STRINGS: List[str] = ['SN', 'S/N', 'SerialNumber']
    USL_STRINGS: List[str] = ['USL', 'UpperSpecLimit']
    LSL_STRINGS: List[str] = ['LSL', 'LowerSpecLimit']
    DROP_DUP: str = 'first'  # 'first', 'last', or None
    PASS_ONLY: bool = False
    CPK_WAIVED_STRINGS: List[str] = ['Waived', 'NotMeasured']

    @validator('DROP_DUP')
    def validate_drop_dup(cls, v):
        if v not in ['first', 'last', None]:
            raise ValueError('DROP_DUP must be "first", "last" or None')
        return v

class Config:
    """Configuration class for CPK Analysis Tool"""
    
    def __init__(self):
        """Initialize with default settings"""
        # Copy all default settings to instance variables
        for key, value in DEFAULT_CONFIG.items():
            setattr(self, key, value)
        
        # Initialize ConfigModel
        self.config_model = ConfigModel()
    
    def load_from_file(self, file_path):
        """
        Load configuration from file with fallback to defaults
        
        Parameters:
        -----------
        file_path : str
            Path to the configuration file
        
        Returns:
        --------
        bool
            True if file was loaded successfully, False otherwise
        """
        import os
        
        if not os.path.exists(file_path):
            print(f"Config file {file_path} not found. Using default settings.")
            return False
            
        try:
            with open(file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                        
                    parts = [p.strip() for p in line.split(",", 1)]
                    if len(parts) != 2:
                        continue
                        
                    name, value = parts
                    if name in DEFAULT_CONFIG:
                        # Convert string representation to proper type
                        if isinstance(DEFAULT_CONFIG[name], list):
                            # Handle lists - split by special delimiter
                            if value.startswith('[') and value.endswith(']'):
                                value = value[1:-1]  # Remove brackets
                            setattr(self, name, [v.strip() for v in value.split('|')])
                        elif isinstance(DEFAULT_CONFIG[name], bool):
                            # Handle booleans
                            setattr(self, name, value.lower() in ('true', 'yes', '1'))
                        elif isinstance(DEFAULT_CONFIG[name], int):
                            # Handle integers
                            try:
                                setattr(self, name, int(value))
                            except ValueError:
                                print(f"Warning: could not convert {value} to int for {name}")
                        else:
                            # Handle strings
                            setattr(self, name, value)
                            
                        print(f"Loaded {name} = {getattr(self, name)}")
                        
            return True
        except Exception as e:
            print(f"Error loading config file: {e}")
            return False
    
    def __str__(self):
        """String representation of configuration"""
        return "\n".join(f"{key}: {getattr(self, key)}" for key in DEFAULT_CONFIG.keys())

# Create global configuration instance
config = Config()

def load_from_file(file_path: str) -> None:
    """Load configuration from file"""
    if os.path.exists(file_path):
        # Implementation would parse file and update config
        pass

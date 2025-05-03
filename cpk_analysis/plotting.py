"""
Plotting module for CPK Analysis Tool
Handles individual plot generation
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List

from config import config

# Set global font settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 8,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'figure.titlesize': 12
})

def generate_boxplot(data: Dict[str, pd.DataFrame], 
                    metric: str,
                    ax: plt.Axes) -> None:
    """Generate boxplot for a single metric"""
    # Implementation would go here
    pass

def generate_histogram(data: Dict[str, pd.DataFrame],
                      metric: str,
                      ax: plt.Axes) -> None:
    """Generate histogram for a single metric"""
    # Implementation would go here
    pass

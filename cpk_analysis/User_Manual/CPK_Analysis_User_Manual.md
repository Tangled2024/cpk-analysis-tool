# CPK Analysis Tool - Complete User Manual

## Video Tutorials

### Installation Guide
[![Installation](https://img.youtube.com/vi/EXAMPLE1/0.jpg)](https://youtu.be/EXAMPLE1)

### Basic Usage
[![Basic Usage](https://img.youtube.com/vi/EXAMPLE2/0.jpg)](https://youtu.be/EXAMPLE2)

### Advanced Features
[![Advanced Features](https://img.youtube.com/vi/EXAMPLE3/0.jpg)](https://youtu.be/EXAMPLE3)

## Core Functions

### 1. Main Analysis (`cpk_analysis_main`)
```python
def cpk_analysis_main(file_paths: List[str], main_window=None) -> bool
```
**Purpose**: Coordinates the entire analysis workflow
**Parameters**:
- `file_paths`: List of CSV files to analyze
- `main_window`: Optional GUI window for progress updates
**Returns**: True if analysis succeeded

### 2. Data Processing (`data_utils.py`)
#### `get_smt_data_from_csv()`
```python
def get_smt_data_from_csv(csv_name: str, label: str = 'log')
```
**Purpose**: Extracts measurement data and limits from CSV
**Outputs**:
- Data DataFrame
- Limits DataFrame
- Valid metrics list

### 3. Statistical Calculations (`stats.py`)
#### `calc_cpk()`
```python
def calc_cpk(data_df, limits_df, metrics, usl_str, lsl_str)
```
**Purpose**: Calculates process capability indices
**Key Metrics**:
- CPK, CPU, CPL
- Mean, Std Dev, Min/Max

## Usage Guide

### GUI Mode
1. Launch: `python main.py`
2. Select files via file dialog
3. View real-time progress
4. Access PDF report in output folder

### CLI Mode
```bash
python main.py file1.csv file2.csv --output custom_report.pdf
```

## Quick Reference Cheat Sheet

### CLI Commands
```bash
# Basic analysis
python main.py data.csv

# Multiple files with custom output
python main.py file1.csv file2.csv -o custom_report.pdf

# Enable debug mode
python main.py --debug data.csv
```

### Common Config Settings
| Parameter | Example Value | Description |
|-----------|---------------|-------------|
| output_dir | ./reports | Output location |
| usl_string | USL,UpperLimit | USL identifiers |
| ml_enabled | true | Enable AI detection |

### Keyboard Shortcuts (GUI)
- Ctrl+O: Open files
- Ctrl+R: Run analysis
- F1: Help menu

## Input Requirements
CSV files must contain:
1. Header row with parameter names
2. USL/LSL rows
3. Measurement data rows

Example:
```
SN,Voltage,Current
USL,5.1,2.5
LSL,4.9,2.1
1001,5.05,2.3
```

## Configuration
Edit `config.conf` to customize:
- Output locations
- Column headers
- Analysis parameters

## Troubleshooting
Common issues:
- **Missing headers**: Update config with alternate names
- **Encoding errors**: Use UTF-8 formatted CSVs
- **Memory issues**: Enable chunked processing

## Advanced Features
- Anomaly detection (ML)
- Custom report templates
- Plugin system for extensions

## Plugin Developer API

### Base Plugin Class
```python
class AnalysisPlugin:
    @abstractmethod
    def process(self, data: Dict) -> Dict:
        """Process raw measurement data"""
        
    @abstractmethod    
    def visualize(self, results: Dict, report):
        """Add visualizations to report"""
```

### Example Plugin
```python
class SPCChartPlugin(AnalysisPlugin):
    def process(self, data):
        return calculate_spc_stats(data)
        
    def visualize(self, results, report):
        add_spc_charts(report, results)
```

### Registration
Add to `setup.cfg`:
```ini
[options.entry_points]
cpk.plugins =
    spc = my_plugin:SPCChartPlugin
```

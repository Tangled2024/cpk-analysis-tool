"""
GUI module for CPK Analysis Tool
Handles the graphical user interface components
"""

import os
import logging
from typing import Callable, List

from PyQt5.QtWidgets import (QApplication, QWidget, QCheckBox, QTextEdit, QPushButton,
                            QVBoxLayout, QHBoxLayout, QLabel, QFileDialog, QMessageBox)
from PyQt5.QtCore import pyqtSignal, Qt

from config import config

class TextEditLog(QTextEdit):
    """Text area for file input with drag and drop capability"""
    
    def __init__(self, parent):
        """Initialize the text edit widget"""
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.init_edit_log()
    
    def init_edit_log(self):
        """Configure the text edit appearance"""
        self.setPlaceholderText("Drag and drop CSV files here.\nPlot order follows file order.")
    
    def dropEvent(self, e):
        """Handle file drop events"""
        self.setText(self.toPlainText() + e.mimeData().text() + "\n")

class ConfigCheckBox(QCheckBox):
    """Base class for configuration checkboxes"""
    
    valueChanged = pyqtSignal(object)
    
    def __init__(self, parent, text: str, config_attr: str, default_value: bool = False, 
                 convert_to_bool: bool = False, special_value: str = None):
        """
        Initialize a configuration checkbox
        
        Parameters:
        -----------
        parent : QWidget
            Parent widget
        text : str
            Checkbox label text
        config_attr : str
            Name of the configuration attribute to modify
        default_value : bool
            Default value for the checkbox
        convert_to_bool : bool
            Whether to convert the config value to a boolean for display
        special_value : str
            Special string value to treat as True (for string configs)
        """
        super().__init__(parent)
        self.config_attr = config_attr
        self.convert_to_bool = convert_to_bool
        self.special_value = special_value
        
        self.setText(text)
        
        # Get current value from config
        config_value = getattr(config, config_attr, default_value)
        
        # Convert to boolean for checkbox if needed
        if self.convert_to_bool and isinstance(config_value, str):
            is_checked = config_value == self.special_value
        else:
            is_checked = bool(config_value)
            
        self.setChecked(is_checked)
        self.stateChanged.connect(self._on_state_changed)
    
    def _on_state_changed(self, state: int) -> None:
        """
        Handle state change events
        
        Parameters:
        -----------
        state : int
            New checkbox state (0 or 2)
        """
        is_checked = bool(state)
        
        # Convert boolean to appropriate config value type
        if self.convert_to_bool:
            value = self.special_value if is_checked else False
        else:
            value = is_checked
            
        setattr(config, self.config_attr, value)
        self.valueChanged.emit(value)

class MainWindow(QWidget):
    """Main application window for CPK Analysis Tool"""
    
    def __init__(self):
        """Initialize the main window"""
        super().__init__()
        self.setWindowTitle("CPK Analysis Tool")
        self.resize(600, 480)
        self.setup_ui()
    
    def setup_ui(self):
        """Configure the UI layout"""
        # Create main layout
        main_layout = QVBoxLayout()
        
        # Title
        title_label = QLabel("CPK Analysis Tool")
        title_label.setAlignment(Qt.AlignCenter)
        title_font = title_label.font()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        main_layout.addWidget(title_label)
        
        # Control checkboxes
        control_layout = QHBoxLayout()
        
        # Create checkboxes with config binding
        self.ckbox_last_only = ConfigCheckBox(
            self, "Last Only", "DROP_DUP", True, 
            convert_to_bool=True, special_value="last"
        )
        self.ckbox_pass_only = ConfigCheckBox(self, "Pass Only", "PASS_ONLY", True)
        self.check_drift_ckb = ConfigCheckBox(self, "Drift by SN", "CHECK_DRIFT", False)
        self.check_drift_ckb.setEnabled(False)  # Disabled feature
        self.plot_all = ConfigCheckBox(self, "Plot All", "IF_PLOT_ALL", False)
        
        # Add checkboxes to layout
        control_layout.addWidget(self.ckbox_last_only)
        control_layout.addWidget(self.ckbox_pass_only)
        control_layout.addWidget(self.check_drift_ckb)
        control_layout.addWidget(self.plot_all)
        control_layout.addStretch()
        
        # Add file selection button
        self.select_files_btn = QPushButton("Select CSV Files", self)
        self.select_files_btn.clicked.connect(self.select_files)
        control_layout.addWidget(self.select_files_btn)
        
        main_layout.addLayout(control_layout)
        
        # Text area for file paths
        self.text_log = TextEditLog(self)
        main_layout.addWidget(self.text_log)
        
        # Status info label
        self.status_label = QLabel("Ready. Add CSV files above, then click Start.")
        self.status_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.status_label)
        
        # Start button
        self.button_start = QPushButton("Start Analysis", self)
        self.button_start.setMinimumHeight(40)
        self.button_start.setEnabled(False)  # Disabled until files are added
        
        # Connect text change to update button state
        self.text_log.textChanged.connect(self.update_start_button)
        
        # Add to layout
        main_layout.addWidget(self.button_start)
        
        # Set the layout
        self.setLayout(main_layout)
    
    def select_files(self):
        """Open file dialog to select CSV files"""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select CSV Files",
            os.path.expanduser("~"),  # Start in home directory
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if files:
            # Add file paths to text area
            current_text = self.text_log.toPlainText().strip()
            new_files = "\n".join([f"file://{file}" for file in files])
            
            if current_text:
                self.text_log.setText(f"{current_text}\n{new_files}")
            else:
                self.text_log.setText(new_files)
    
    def update_start_button(self):
        """Enable/disable start button based on text content"""
        has_content = bool(self.text_log.toPlainText().strip())
        self.button_start.setEnabled(has_content)
    
    def set_start_button_handler(self, handler: Callable) -> None:
        """
        Set the callback function for the start button
        
        Parameters:
        -----------
        handler : Callable
            Function to call when the start button is clicked
        """
        self.button_start.clicked.connect(handler)
    
    def set_status(self, message: str) -> None:
        """
        Update the status label
        
        Parameters:
        -----------
        message : str
            Status message to display
        """
        self.status_label.setText(message)
    
    def get_file_paths(self) -> List[str]:
        """
        Get the list of file paths from the text area
        
        Returns:
        --------
        List[str]
            List of file paths
        """
        file_paths = []
        for line in self.text_log.toPlainText().strip().split("\n"):
            line = line.strip()
            if not line:
                continue
                
            file_pos = line.find("file://") + 7
            if file_pos != 6:  # find returns -1 if not found, +7 makes it 6
                clean_path = line[file_pos:].rstrip()
                if os.path.exists(clean_path):
                    file_paths.append(clean_path)
                else:
                    logging.warning(f"File not found: {clean_path}")
        
        return file_paths
    
    def show_error(self, title: str, message: str) -> None:
        """
        Show error message box
        
        Parameters:
        -----------
        title : str
            Error dialog title
        message : str
            Error message
        """
        QMessageBox.critical(self, title, message)
    
    def show_info(self, title: str, message: str) -> None:
        """
        Show information message box
        
        Parameters:
        -----------
        title : str
            Information dialog title
        message : str
            Information message
        """
        QMessageBox.information(self, title, message)

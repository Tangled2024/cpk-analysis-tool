import os
import pdfkit
from pathlib import Path

MANUAL_DIR = Path(__file__).parent

files_to_convert = [
    'CPK_Analysis_User_Manual.md',
    'config_reference.md'
]

for md_file in files_to_convert:
    input_path = MANUAL_DIR / md_file
    output_path = MANUAL_DIR / f'{input_path.stem}.pdf'
    
    # Convert with styling
    pdfkit.from_file(
        str(input_path), 
        str(output_path),
        options={
            'encoding': 'UTF-8',
            'quiet': '',
            'margin-top': '20mm',
            'margin-right': '20mm',
            'margin-bottom': '20mm',
            'margin-left': '20mm'
        },
        css='manual_style.css'
    )

print(f'Generated PDFs in {MANUAL_DIR}')

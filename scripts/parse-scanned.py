import os
import pandas as pd
import numpy as np
import re
import json
import sys
from pathlib import Path
from rapidfuzz import process, fuzz
from markdown_it import MarkdownIt
from bs4 import BeautifulSoup
from utils import calculate_MFR, calculate_HRR, calculate_HRR_O2_only, colorize
import shutil
from dateutil import parser
from datetime import datetime

# First argument is the input directory, second argument is the output directory
args = sys.argv[1:]
if len(args) > 2:
    print("""Too many arguments
          Usage: python parse-scanned.py <input_dir> <output_dir>
          Leave empty to use defaults.""")
    sys.exit()

# Relative to the path the script is being run from
# Assumes the script is being run from the root of the repo
## ex path matching INPUT_DIR: C:Users/user-id/path-to-repo-folder/cone-db/data/markdowns
INPUT_DIR = Path(r"./data/markdowns")
OUTPUT_DIR = Path(r"./data/auto-processed/archive")
BAD_SCANS_DIR = Path("./data/bad_scans")

# Ensure output directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BAD_SCANS_DIR.mkdir(parents=True, exist_ok=True)

if len(args) == 2:
    INPUT_DIR = Path(args[0])
    OUTPUT_DIR = Path(args[1])
#################################
# Global variables
test_data = {}
total_parsed = 0
total_bad_scans = 0
found_tests = []  # Add this line

# Define expected metadata fields and their variations
EXPECTED_METADATA_KEYS = {
    "initial_mass_g": ["Initial Mass", "INITIAL WEIGHT OF SAMPLE", "Initial Weight"],
    "calibration_factor": ["CALIBRATION FACTOR", "Calibration Factor", "Cal Factor"],
    "conversion_factor": ["CONVERSION FACTOR", "Conversion Factor", "E Factor"],
    "carbon_fraction": ["FRACTION OF CARBON IN FUEL", "Carbon Fraction", "Fraction of Carbon"],
    "operator": ["Operator", "Test Operator", "Tested By"],
    "laboratory": ["Laboratory", "Lab", "Testing Facility"],
    "date": ["Test Date", "Date", "Date of Test"]
}

# Define expected data fields and their variations
DATA_FIELDS = {
    'time_s': ['Time', 'Time s', 'Time (s)', 's'],
    'qdot_kw_m2': ['Q-Dot', 'Q-Dot kW/m2', 'Heat Flux', 'kW/m2'],
    'sumq_mj_m2': ['Sum Q', 'Sum Q MJ/m2', 'Total Heat', 'MJ/m2'],
    'mdot_g_sm2': ['M-Dot', 'M-Dot g/s-m2', 'Mass Loss Rate', 'g/s-m2'],
    'mass_loss_kg_m2': ['Mass Loss', 'Mass Loss kg/m2', 'Total Mass Loss', 'kg/m2'],
    'ht_comb_mj_kg': ['HtofCmb', 'HT Comb', 'HT Comb MJ/kg', 'Heat of Combustion', 'MJ/kg'],
    'ex_area_m2_kg': ['Sp Ext Area', 'Ex Area', 'Ex Area m2/kg', 'Extinction Area', 'm2/kg'],
    'co2_kg_kg': ['CO2', 'CO2 kg/kg', 'Carbon Dioxide', 'kg/kg'],
    'co_kg_kg': ['CO', 'CO kg/kg', 'Carbon Monoxide', 'kg/kg'],
    'h2o_kg_kg': ['H2O', 'H2O kg/kg', 'Water', 'kg/kg'],
    'hc_kg_kg': ["H'carbs", 'Hydrocarbons kg/kg', 'Hydrocarbons', 'kg/kg'],
    'hcl_kg_kg': ['HCL', 'HCl', 'HCl kg/kg', 'Hydrogen Chloride', 'kg/kg']
}

COLUMN_VARIATIONS = {
    'Q-Dot kW/m2': [('Q-DOT', 'KW/M2'), ('W-DOT', 'KW/M2'), ('Q DOT', 'KW/M2'), ('W DOT', 'KW/M2')],
    'Sum Q MJ/m2': [('SUM Q', 'MJ/M2'), ('TOTAL Q', 'MJ/M2')],
    'M-Dot g/s-m2': [('M-DOT', 'G/S-M2'), ('M-DOT', 'G/S'), ('M LOSS', 'G/S'), ('MASS LOSS RATE', 'G/S')],
    'Mass Loss kg/m2': [('MASS LOSS', 'KG/M2'), ('MASS', 'KG/M2')],
    'HT Comb MJ/kg': [('H COMB', 'MJ/KG'), ('HT COMB', 'MJ/KG'), ('HTOFCMB', 'MJ/KG')],
    'Ex Area m2/kg': [('EX AREA', 'M2/KG'), ('SP EXT AREA', 'M2/KG')],
    'CO2 kg/kg': [('CO2', 'KG/KG')],
    'CO kg/kg': [('CO', 'KG/KG')],
    'H2O kg/kg': [('H2O', 'KG/KG')],
    'HC kg/kg': [('HC', 'KG/KG'), ("H'CARBS", 'KG/KG')],
    'HCl kg/kg': [('HCL', 'KG/KG')]
}

def get_page_header_info(text):
    """
    Extract test identification information from a page section of a markdown file.
    
    This function searches the first 10 lines of text for test numbers and material names
    using three different formats:
    1. "Test ####" format (e.g., "Test 1181")
    2. "(####)" format (e.g., "Material Name (1170)")
    3. "T####" format (e.g., "T1524")
    
    Args:
        text (str): The text content of a page section
        
    Returns:
        dict: A dictionary containing:
            - test_number (str): The identified test number, or None if not found
            - material_name (str): The extracted material name, or None if not found
            - header_text (str): The full line containing the test information
    """
    header_info = {
        "test_number": None,
        "material_name": None,
        "header_text": None
    }
    
    # Examine first 10 lines for header information
    lines = text.strip().split('\n')
    for line in lines[:10]:
        # Skip empty lines or table rows (which start with |)
        if not line.strip() or line.strip().startswith('|'):
            continue
            
        # Check for "Test ####" format (common in 1181-style files)
        test_match = re.search(r'Test\s+(\d{4})', line, re.IGNORECASE)
        if test_match:
            test_num = test_match.group(1)
            header_info["test_number"] = test_num
            found_tests.append(test_num)
            header_info["header_text"] = line.strip()
            
            # Extract material name by taking everything before "KW/M2"
            parts = line.split()
            material_parts = []
            for part in parts:
                if "KW/M2" in part.upper():
                    break
                material_parts.append(part)
            if material_parts:
                header_info["material_name"] = ' '.join(material_parts).strip()
            break
        
        # Check for "(####)" format (common in 1170-style files)
        test_match = re.search(r'\((\d{4})\)', line)
        if test_match:
            test_num = test_match.group(1)
            header_info["test_number"] = test_num
            found_tests.append(test_num)
            header_info["header_text"] = line.strip()
            
            # Extract material name from everything before the heat flux
            heat_flux_idx = line.upper().find("KW/M2")
            if heat_flux_idx != -1:
                material_name = line[:heat_flux_idx].strip()
                # Remove the test number from material name
                material_name = re.sub(r'\(\d{4}\)', '', material_name).strip()
                header_info["material_name"] = material_name
            break
            
        # Check for "T####" format
        test_match = re.search(r'T(\d{4})', line)
        if test_match:
            test_num = test_match.group(1)
            header_info["test_number"] = test_num
            found_tests.append(test_num)
            header_info["header_text"] = line.strip()
            
            # Extract material name from text before heat flux
            parts = line.split()
            material_parts = []
            for part in parts:
                if "KW/M2" in part.upper():
                    break
                material_parts.append(part)
            if material_parts:
                header_info["material_name"] = ' '.join(material_parts).strip()
            break
    
    return header_info

def convert_md_to_html(md_content):
    """Convert markdown content to HTML."""
    md = MarkdownIt()
    return md.render(md_content)

def extract_metadata_from_html(html_content):
    """Extract metadata from HTML content using BeautifulSoup."""
    metadata = {
        "laboratory": "NBS",
        "test_number": None,
        "date": None,
        "heat_flux_kW/m2": None,
        "orientation": None,
        "sample_area_m2": None,
        "initial_mass_g": None,
        "calibration_factor": None,
        "conversion_factor": None,
        "carbon_fraction": None,
        "comments": [],
        "material_name": None,
        "operator": None
    }
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    text_content = ''
    for element in soup.find_all(string=True):
        if not element.parent.name == 'table':
            text_content += element.strip() + '\n'
    
    lines = text_content.split('\n')
    
    # First try to get test number from header
    for line in lines[:10]:  # Check first 10 lines
        test_matches = [
            re.search(r'Test\s+(\d+)', line, re.IGNORECASE),
            re.search(r'\((\d{4})\)', line),
            re.search(r'T(\d+)', line)
        ]
        for match in test_matches:
            if match:
                metadata["test_number"] = match.group(1)
                break
        if metadata["test_number"]:
            break
    
    # Process each line for metadata
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        
        # Extract test info from header line
        if "KW/M2" in line.upper():
            parts = line.strip().split()
            
            # Try to extract material name based on different formats
            if not metadata["material_name"]:
                # For 1170-style files
                heat_flux_idx = line.upper().find("KW/M2")
                if heat_flux_idx != -1:
                    material_name = line[:heat_flux_idx].strip()
                    # Remove any test numbers
                    material_name = re.sub(r'\(\d{4}\)', '', material_name).strip()
                    metadata["material_name"] = material_name
                
                # For 1181-style files
                material_match = re.search(r'(FAA\s*-\s*\d+F)', line)
                if material_match:
                    metadata["material_name"] = material_match.group(1)
            
            # Extract orientation
            if "VERT" in line.upper():
                metadata["orientation"] = "Vertical"
            elif "HOR" in line.upper():
                metadata["orientation"] = "Horizontal"
            
            # Extract heat flux
            flux_match = re.search(r'(\d+)\s*KW/M2', line.upper())
            if flux_match:
                metadata["heat_flux_kW/m2"] = float(flux_match.group(1))
            
            continue
        
        # Extract date
        date_match = re.search(r'(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})', line)
        if date_match and not metadata["date"]:
            try:
                date_str = f"{date_match.group(1)} {date_match.group(2)} {date_match.group(3)}"
                metadata["date"] = pd.to_datetime(date_str).strftime('%Y-%m-%d')
            except:
                pass
        
        # Extract pre-test comments
        if process.extractOne("PRE-TEST COMMENTS", [line.upper()], scorer=fuzz.partial_ratio)[1] > 85:
            next_lines = []
            j = i + 1
            while j < len(lines) and lines[j].strip() and not any(
                process.extractOne(keyword, [lines[j].upper()], scorer=fuzz.partial_ratio)[1] > 85 
                for keyword in ["Mass ratio", "Soot average", "Initial Mass"]
            ):
                next_lines.append(lines[j].strip())
                j += 1
            if next_lines:
                metadata["comments"].append(f"Pre-test: {' '.join(next_lines)}")
        
        # Process other metadata fields using fuzzy matching
        for key, variations in EXPECTED_METADATA_KEYS.items():
            best_match = process.extractOne(line, variations, scorer=fuzz.partial_ratio)
            if best_match and best_match[1] > 85:
                # Look for value after the key using regex
                match = re.search(rf"{best_match[0]}[:=\s]+(.+?)(?:\s*$|\s*[A-Z])", line, re.IGNORECASE)
                if match:
                    value = match.group(1).strip()
                    
                    # Convert numeric values
                    if key in ["heat_flux_kW/m2", "sample_area_m2", "initial_mass_g", 
                             "calibration_factor", "conversion_factor", "carbon_fraction"]:
                        try:
                            value = float(re.search(r'[\d.]+', value).group())
                        except:
                            continue
                    
                    metadata[key] = value
                    
                # Special handling for carbon fraction which might be on next line
                elif key == "carbon_fraction" and i + 1 < len(lines):
                    try:
                        next_line = lines[i + 1]
                        value = float(re.search(r'([\d.]+)', next_line).group(1))
                        metadata[key] = value
                    except:
                        continue
    
    # Clean up metadata
    metadata = {k: v for k, v in metadata.items() if v is not None}
    
    return metadata

def clean_cell_value(cell):
    """Clean and convert cell values, handling asterisks and missing data."""
    cell = cell.strip()
    if cell in ['*****', '****', '', '-']:
        return None
    
    # Remove any asterisks
    cell = cell.replace('*', '')
    
    try:
        # Try to convert to float, handling negative values
        if cell.startswith('-.'):  # Handle cases like "-.66"
            cell = '-0' + cell[1:]
        return float(cell)
    except ValueError:
        return cell

def should_skip_table(first_line):
    """Check if table should be skipped based on its content."""
    skip_keywords = [
        "TEST AVG",
        "YIELD",
        "AVERAGE",
        "MAXIMUM",
        "PEAK",
        "TOTAL",
        "Mass Weighted Average",
        "Mass ratio",
        "Soot average"
    ]
    return any(keyword.upper() in first_line.upper() for keyword in skip_keywords)

def extract_tables_from_html(html_content, test_number):
    """Extract tables from HTML content using BeautifulSoup."""
    global test_data
    
    soup = BeautifulSoup(html_content, 'html.parser')
    tables = soup.find_all('table')
    
    if not tables:
        text_content = soup.get_text()
        lines = text_content.split('\n')
        table_lines = []
        current_table = []
        in_table = False
        
        for line in lines:
            if '|' in line:
                if not in_table:
                    if should_skip_table(line):
                        continue
                    in_table = True
                current_table.append(line.strip())
            elif in_table and current_table:
                if len(current_table) > 2:  # Ensure we have at least header + separator
                    table_lines.append(current_table)
                current_table = []
                in_table = False
        
        if current_table and len(current_table) > 2:
            table_lines.append(current_table)
        
        # Dictionary to store data by time point and table source
        time_data_by_table = {}
        
        for table_idx, table in enumerate(table_lines):
            try:
                # Get headers from first row
                header_row = table[0]
                headers = [col.strip() for col in header_row.split('|')[1:-1]]
                
                # Skip tables without time column
                if not any('time' in h.lower() for h in headers):
                    continue
                
                # Store data for this table separately
                current_table_data = {}
                
                # Find the data rows (skip headers and separators)
                data_rows = []
                for row in table:
                    if ('---' in row or 
                        any(unit in row.lower() for unit in ['s', 'kw/m2', 'mj/m2', 'g/s-m2', 'kg/kg']) or 
                        row == header_row):
                        continue
                    data_rows.append(row)
                
                # Process each data row
                for row in data_rows:
                    cells = [clean_cell_value(cell) for cell in row.split('|')[1:-1]]
                    if len(cells) != len(headers):
                        continue
                        
                    # Find time value
                    time_idx = next(i for i, h in enumerate(headers) if 'time' in h.lower())
                    time_value = cells[time_idx]
                    
                    if time_value is not None:
                        try:
                            time_value = float(time_value)
                        except (ValueError, TypeError):
                            continue
                            
                        if time_value not in current_table_data:
                            current_table_data[time_value] = {}
                            
                        # Map data to standardized column names
                        for header, value in zip(headers, cells):
                            if value is None or value == '*****':  # Skip missing data
                                continue
                                
                            header_lower = header.lower()
                            if 'time' in header_lower:
                                continue
                            elif 'co2' in header_lower:
                                current_table_data[time_value]['CO2 kg/kg'] = value
                            elif header_lower == 'co':
                                current_table_data[time_value]['CO kg/kg'] = value
                            elif 'h2o' in header_lower:
                                current_table_data[time_value]['H2O kg/kg'] = value
                            elif "h'carbs" in header_lower or 'hc' in header_lower:
                                current_table_data[time_value]['HC kg/kg'] = value
                            elif 'hcl' in header_lower:
                                current_table_data[time_value]['HCl kg/kg'] = value
                            elif 'q-dot' in header_lower:
                                current_table_data[time_value]['Q-Dot kW/m2'] = value
                            elif 'sum q' in header_lower:
                                current_table_data[time_value]['Sum Q MJ/m2'] = value
                            elif 'm-dot' in header_lower:
                                current_table_data[time_value]['M-Dot g/s-m2'] = value
                            elif 'mass loss' in header_lower:
                                current_table_data[time_value]['Mass Loss kg/m2'] = value
                            elif 'ht comb' in header_lower or 'htofcmb' in header_lower:
                                current_table_data[time_value]['HT Comb MJ/kg'] = value
                            elif 'ex area' in header_lower or 'sp ext area' in header_lower:
                                current_table_data[time_value]['Ex Area m2/kg'] = value
                
                time_data_by_table[table_idx] = current_table_data
                
            except Exception as e:
                print(f"Error processing table {table_idx}: {str(e)}")
                continue
        
        if time_data_by_table:
            # Get all unique time points across all tables
            all_times = set()
            for table_data in time_data_by_table.values():
                all_times.update(table_data.keys())
            all_times = sorted(all_times)
            
            # Merge data from all tables
            merged_data = []
            for time_value in all_times:
                row = {'Time': time_value}
                for table_data in time_data_by_table.values():
                    if time_value in table_data:
                        row.update(table_data[time_value])
                merged_data.append(row)
            
            # Convert to DataFrame
            combined_df = pd.DataFrame(merged_data)
            
            # Standardize column names - find the time column
            time_column = None
            for col in combined_df.columns:
                if col.upper().replace(' ', '').replace('(S)', '').replace('(SEC)', '') == 'TIME':
                    time_column = col
                    break
            
            if time_column is None:
                print(colorize("No time column found in data", "red"))
                print(colorize(f"Available columns: {', '.join(combined_df.columns)}", "yellow"))
                return None
                
            # Rename the time column to standardized format
            combined_df = combined_df.rename(columns={time_column: 'Time'})
            
            # Convert numeric columns
            for col in combined_df.columns:
                if col != 'Time':
                    combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
            
            # Sort by time
            combined_df = combined_df.sort_values('Time')
            
            # Store in global test_data dictionary
            test_data[test_number] = combined_df
            
            # Save the data
            output_path = OUTPUT_DIR / f"{test_number}.csv"
            combined_df.to_csv(output_path, index=False)
            print(f"Successfully processed and saved data for test {test_number}")
            
            # Print time range info for validation
            print(f"Time range: {combined_df['Time'].min():.1f}s to {combined_df['Time'].max():.1f}s")
            print(f"Number of time points: {len(combined_df)}")
            
            return combined_df
    
    return None

def process_markdown_file(filepath):
    """
    Process a markdown file containing scanned test data.
    
    This function:
    1. Reads the markdown file
    2. Splits it into page sections
    3. Groups related pages into individual tests
    4. Extracts tables and metadata for each test
    5. Saves the processed data as CSV and JSON files
    
    Args:
        filepath (Path): Path to the markdown file to process
        
    Side effects:
        - Creates CSV files containing test data
        - Creates JSON files containing test metadata
        - Updates global counters for processed files and bad scans
        - Moves problematic files to bad scans directory
    """
    global total_parsed, total_bad_scans, found_tests
    
    print(colorize(f"\nProcessing file: {filepath}", "cyan"))
    
    # Read the markdown file
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Split content into individual pages
    page_sections = content.split('---')
    
    # Initialize structure to hold current test information
    current_test = {
        "content": [],
        "test_number": None,
        "material_name": None,
        "header_text": None
    }
    
    tests = []
    
    # Process each page section
    for section in page_sections:
        if not section.strip():
            continue
            
        # Get header information from the current section
        header_info = get_page_header_info(section)
        
        # Determine if this section starts a new test
        is_new_test = True
        if current_test["test_number"]:
            if (header_info["test_number"] == current_test["test_number"] or 
                header_info["material_name"] == current_test["material_name"]):
                is_new_test = False
            elif not header_info["header_text"]:
                is_new_test = False
        
        # Handle new test detection
        if is_new_test and current_test["content"]:
            if len(tests) >= 6:
                print(colorize(f"WARNING: Found more than 6 tests in {filepath}. This might indicate an issue with test separation.", "yellow"))
                break
            tests.append(current_test.copy())
            current_test = {
                "content": [section],
                "test_number": header_info["test_number"],
                "material_name": header_info["material_name"],
                "header_text": header_info["header_text"]
            }
        else:
            # Continue adding content to current test
            current_test["content"].append(section)
            if not current_test["test_number"] and header_info["test_number"]:
                current_test["test_number"] = header_info["test_number"]
                current_test["material_name"] = header_info["material_name"]
                current_test["header_text"] = header_info["header_text"]
    
    # Add the last test if it exists and we haven't exceeded the limit
    if current_test["content"] and len(tests) < 6:
        tests.append(current_test)
    
    print(colorize(f"Found {len(tests)} tests in file", "purple"))
    
    # Process each identified test
    for test in tests:
        if not test["test_number"]:
            print(colorize("No test number found, moving to bad scans", "red"))
            shutil.copy(filepath, BAD_SCANS_DIR / Path(filepath).name)
            total_bad_scans += 1
            continue
        
        print(colorize(f"\nProcessing test number: {test['test_number']}", "cyan"))
        
        # Combine all content for this test
        combined_content = '\n'.join(test["content"])
        html_content = convert_md_to_html(combined_content)
        
        # Extract metadata and tables
        metadata = extract_metadata_from_html(html_content)
        test_number = metadata.get("test_number", "unknown_test")
        
        # Add source filename to metadata for tracking
        metadata["source_file"] = str(Path(filepath).name)
        
        if test_number == "unknown_test":
            print(colorize("Unknown test number in metadata, moving to bad scans", "red"))
            shutil.copy(filepath, BAD_SCANS_DIR / Path(filepath).name)
            total_bad_scans += 1
            continue
        
        # Save metadata and extract tables
        df = extract_tables_from_html(html_content, test_number)
        
        # Save metadata to JSON file
        metadata_path = OUTPUT_DIR / f"{test_number}.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        print(colorize(f"Saved metadata to {metadata_path}", "green"))
        
        # Report success or failure of data extraction
        if df is not None and not df.empty:
            total_parsed += 1
            print(colorize(f"Successfully saved data for test {test_number}", "green"))
            print(colorize(f"Time range: {df['Time'].min():.1f}s to {df['Time'].max():.1f}s", "purple"))
            print(colorize(f"Number of time points: {len(df)}", "purple"))
        else:
            print(colorize(f"No valid data found for test {test_number}", "red"))

def calculate_and_append_values(csv_path):
    """
    Read a CSV file, calculate additional values using utils.py functions,
    and append the new columns to the CSV.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        calculations_added = False
        
        # Debug: Print original columns
        print(colorize(f"\nOriginal columns in {csv_path}:", "cyan"))
        print(colorize(", ".join(df.columns), "cyan"))
        
        # Rename columns to match expected format
        column_mapping = {
            'time_s': 'Time (s)',
            'mass_g': 'Mass (g)',
            'o2_%': 'O2 (Vol fr)',
            'co2_%': 'CO2 (Vol fr)',
            'co_%': 'CO (Vol fr)',
            'dpt_pa': 'DPT (Pa)',
            'stack_tc_k': 'Te (K)',
        }
        
        # Try both original and lowercase column names
        for orig_col in df.columns:
            lower_col = orig_col.lower()
            if lower_col in column_mapping:
                df = df.rename(columns={orig_col: column_mapping[lower_col]})
        
        # Convert percentages to volume fractions if needed
        for col in ['O2 (Vol fr)', 'CO2 (Vol fr)', 'CO (Vol fr)']:
            if col in df.columns and df[col].mean() > 1:  # If values appear to be percentages
                df[col] = df[col] / 100
        
        # Calculate initial values from baseline (first 30 seconds)
        baseline_end = 30
        if 'O2 (Vol fr)' in df.columns:
            df['X_O2_initial'] = df['O2 (Vol fr)'][:baseline_end].mean()
        if 'CO2 (Vol fr)' in df.columns:
            df['X_CO2_initial'] = df['CO2 (Vol fr)'][:baseline_end].mean()
        
        # Required columns for each calculation
        mfr_cols = ['c', 'DPT (Pa)', 'Te (K)']
        k_cols = ['I_0', 'I', 'L']
        hrr_cols = ['O2 (Vol fr)', 'CO2 (Vol fr)', 'CO (Vol fr)', 
                   'X_O2_initial', 'X_CO2_initial',
                   'DPT (Pa)', 'Te (K)', 'c', 'e', 'area']
        
        # Debug: Print missing columns for each calculation
        print("\nMissing columns:")
        print(f"MFR calculation: {[col for col in mfr_cols if col not in df.columns]}")
        print(f"k calculation: {[col for col in k_cols if col not in df.columns]}")
        print(f"HRR calculation: {[col for col in hrr_cols if col not in df.columns]}")
        
        # Initialize new columns with None
        df['MFR_calc'] = None
        df['k_calc'] = None
        df['HRR_calc'] = None
        
        # Calculate MFR (Mass Flow Rate)
        if all(col in df.columns for col in mfr_cols):
            df['MFR_calc'] = df.apply(
                lambda row: calculate_MFR(
                    row['c'],
                    row['DPT (Pa)'],
                    row['Te (K)']
                ) if pd.notnull(row['c']) and pd.notnull(row['DPT (Pa)']) and pd.notnull(row['Te (K)']) else None,
                axis=1
            )
            if df['MFR_calc'].notna().any():
                calculations_added = True

        # Calculate k (extinction coefficient)
        if all(col in df.columns for col in k_cols):
            df['k_calc'] = df.apply(
                lambda row: calculate_k(
                    row['I_0'],
                    row['I'],
                    row['L']
                ) if pd.notnull(row['I_0']) and pd.notnull(row['I']) and pd.notnull(row['L']) else None,
                axis=1
            )
            if df['k_calc'].notna().any():
                calculations_added = True

        # Calculate HRR (Heat Release Rate)
        if all(col in df.columns for col in hrr_cols):
            df['HRR_calc'] = df.apply(
                lambda row: calculate_HRR(
                    row['O2 (Vol fr)'],
                    row['CO2 (Vol fr)'],
                    row['CO (Vol fr)'],
                    row['X_O2_initial'],
                    row['X_CO2_initial'],
                    row['DPT (Pa)'],
                    row['Te (K)'],
                    row['c'],
                    row['e'],
                    row['area']
                ) if all(pd.notnull(row[col]) for col in hrr_cols) else None,
                axis=1
            )
            if df['HRR_calc'].notna().any():
                calculations_added = True

        # Save the updated DataFrame back to CSV
        df.to_csv(csv_path, index=False)
        if calculations_added:
            print(colorize(f"Added calculated values to {csv_path}", "green"))
        else:
            print(colorize(f"No calculations possible for {csv_path}", "yellow"))
        
        return calculations_added

    except Exception as e:
        print(colorize(f"Error processing {csv_path}: {str(e)}", "red"))
        return False

def process_all_csvs(directory):
    """
    Process all CSV files in a directory to add calculated values.
    
    Args:
        directory (str or Path): Directory containing CSV files to process
    """
    directory = Path(directory)
    csv_files = list(directory.glob("*.csv"))
    
    if not csv_files:
        print(colorize(f"No CSV files found in {directory}", "yellow"))
        return
    
    total_files = len(csv_files)
    successful_calculations = 0
    
    print(f"Found {total_files} CSV files to process")
    for csv_file in csv_files:
        if calculate_and_append_values(csv_file):
            successful_calculations += 1
    
    print(colorize(f"\nCalculations Summary:", "blue"))
    print(colorize(f"Total CSV files processed: {total_files}", "blue"))
    print(colorize(f"Files with successful calculations: {successful_calculations}", "blue"))
    print(colorize(f"Success rate: {(successful_calculations/total_files)*100:.1f}%", "blue"))

def sort_files_by_year():
    """
    Sort processed files into year folders based on the date in metadata.
    Should be called after all files have been processed.
    """
    print("\nSorting files by year...")
    
    # Get all JSON files in the output directory
    json_files = list(OUTPUT_DIR.glob("*.json"))
    
    # Skip materials_index.json
    json_files = [f for f in json_files if f.stem != "materials_index"]
    
    files_moved = 0
    for json_file in json_files:
        try:
            # Read the metadata
            with open(json_file, "r") as f:
                metadata = json.load(f)
            
            # Get the date and parse the year
            if "date" in metadata:
                # Try different date formats
                try:
                    # Try ISO format first
                    year = parser.parse(metadata["date"]).year
                except:
                    # If that fails, try other common formats
                    try:
                        year = datetime.strptime(metadata["date"], "%d/%m/%Y").year
                    except:
                        print(colorize(f"Could not parse date in {json_file.name}", "yellow"))
                        continue
                
                # Create year directory
                year_dir = OUTPUT_DIR / str(year)
                year_dir.mkdir(exist_ok=True)
                
                # Move the JSON file
                new_json_path = year_dir / json_file.name
                shutil.move(str(json_file), str(new_json_path))
                
                # Move the corresponding CSV file
                csv_file = json_file.with_suffix(".csv")
                if csv_file.exists():
                    new_csv_path = year_dir / csv_file.name
                    shutil.move(str(csv_file), str(new_csv_path))
                    files_moved += 2
                
        except Exception as e:
            print(colorize(f"Error sorting {json_file.name}: {e}", "red"))
            continue
    
    print(colorize(f"Moved {files_moved} files into year folders", "green"))

def create_material_index():
    """
    Create a comprehensive index of all materials and their associated files.
    Includes full file paths and organizes by material name.
    """
    print("\nCreating material index...")
    
    material_index = {}
    
    # Recursively find all JSON files in output directory (including year subdirectories)
    json_files = list(OUTPUT_DIR.rglob("*.json"))
    
    # Skip materials_index.json
    json_files = [f for f in json_files if f.stem != "materials_index"]
    
    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                metadata = json.load(f)
            
            material_name = metadata.get("material_name", "Unknown Material")
            
            # Create entry for this material if it doesn't exist
            if material_name not in material_index:
                material_index[material_name] = {
                    "test_count": 0,
                    "tests": []
                }
            
            # Add file information
            test_info = {
                "test_number": metadata.get("test_number", "unknown"),
                "date": metadata.get("date", "unknown"),
                "json_path": str(json_file.relative_to(OUTPUT_DIR)),
                "csv_path": str(json_file.with_suffix(".csv").relative_to(OUTPUT_DIR)),
                "original_file": metadata.get("source_file", "unknown")
            }
            
            material_index[material_name]["tests"].append(test_info)
            material_index[material_name]["test_count"] += 1
            
        except Exception as e:
            print(colorize(f"Error processing {json_file}: {e}", "red"))
            continue
    
    # Sort tests within each material by date
    for material in material_index.values():
        material["tests"].sort(key=lambda x: x["date"])
    
    # Save the comprehensive index
    index_path = OUTPUT_DIR / "material_index_detailed.json"
    with open(index_path, "w") as f:
        json.dump(material_index, f, indent=4)
    
    # Print summary
    print("\n=== Material Index Summary ===")
    print(f"Total unique materials: {len(material_index)}")
    print("\nMaterial breakdown:")
    for material, info in sorted(material_index.items()):
        print(f"- {material}: {info['test_count']} tests")
    print(f"\nDetailed index saved to: {index_path}")
    
    return material_index

def main():
    """Main execution function."""
    global materials_index, found_tests
    
    print("Starting processing of files...")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Bad scans directory: {BAD_SCANS_DIR}\n")
    
    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BAD_SCANS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize tracking variables
    materials_index = {}
    found_tests = []
    
    # Process all markdown files
    for filepath in INPUT_DIR.glob("*.md"):
        process_markdown_file(filepath)
    
    # Save materials index
    with open(OUTPUT_DIR / "materials_index.json", "w") as f:
        json.dump(materials_index, f, indent=4)
    
    # Sort files into year folders
    sort_files_by_year()
    
    # Create comprehensive material index
    detailed_index = create_material_index()
    
    print("\n=== Processing Summary ===")
    print(f"Total files processed: {total_parsed}")
    print(f"Total bad scans: {total_bad_scans}")
    print(f"Materials index saved to: {OUTPUT_DIR}/materials_index.json")
    print(f"Detailed material index saved to: {OUTPUT_DIR}/material_index_detailed.json")

if __name__ == "__main__":
    main()

import pandas as pd
import re
import os
import logging
import subprocess
import sys
from pathlib import Path
import warnings
from collections import defaultdict
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_java_installation():
    """Check if Java is installed and accessible"""
    try:
        result = subprocess.run(['java', '-version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            logger.info("✅ Java is available")
            return True
        else:
            logger.warning("⚠️ Java command failed")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        logger.warning("⚠️ Java not found in PATH")
        return False

def try_import_tabula():
    """Try to import tabula-py, return None if not available"""
    try:
        import tabula
        return tabula
    except ImportError:
        logger.warning("tabula-py not available. Install with: pip install tabula-py")
        return None

def try_import_camelot():
    """Try to import camelot-py, return None if not available"""
    try:
        import camelot
        return camelot
    except ImportError:
        logger.warning("camelot-py not available. Install with: pip install camelot-py[cv]")
        return None

def identify_technology_category(table_title):
    """
    Identify the technology category from a table title
    """
    if not table_title:
        return "Unknown"
    
    title_lower = table_title.lower()
    
    # Define technology keywords and their categories
    tech_keywords = {
        'solar': ['solar', 'photovoltaic', 'pv'],
        'wind': ['wind', 'onshore', 'offshore'],
        'hydro': ['hydro', 'hydropower', 'hydroelectric'],
        'thermal': ['thermal', 'coal', 'lng', 'gas-fired', 'ccgt'],
        'nuclear': ['nuclear'],
        'biomass': ['biomass', 'bio'],
        'waste': ['waste-to-energy', 'waste'],
        'transmission': ['transmission', 'power line', 'grid', 'substation'],
        'storage': ['storage', 'battery', 'pumped-storage'],
        'cogeneration': ['cogeneration', 'chp'],
        'flexible': ['flexible', 'peaking']
    }
    
    for category, keywords in tech_keywords.items():
        for keyword in keywords:
            if keyword in title_lower:
                return category
    
    return "Unknown"

def extract_table_name_from_text(text):
    """
    Extract table name from text that contains table headers like 'Table 1: List of LNG thermal power plants'
    """
    if not text:
        return "Unknown_Table"
    
    # Look for patterns like "Table X: Description" or "Table X - Description"
    table_patterns = [
        r'Table\s+(\d+):\s*(.+)',
        r'Table\s+(\d+)\s*-\s*(.+)',
        r'Table\s+(\d+)\s+(.+)',
    ]
    
    for pattern in table_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            table_num = match.group(1)
            table_desc = match.group(2).strip()
            return f"Table_{table_num}_{table_desc}"
    
    return "Unknown_Table"

def clean_table_data(df):
    """
    Clean and process the extracted table data
    """
    if df is None or df.empty:
        return df
    
    # Remove completely empty rows and columns
    df = df.dropna(how='all').dropna(axis=1, how='all')
    
    # Reset index
    df = df.reset_index(drop=True)
    
    # Clean column names
    if not df.columns.empty:
        df.columns = [str(col).strip() if pd.notna(col) else f'Column_{i}' 
                     for i, col in enumerate(df.columns)]
    
    # Clean cell values
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace(['nan', 'None', ''], pd.NA)
    
    return df

def extract_tables_with_tabula(pdf_path):
    """Extract tables using tabula-py with header-based grouping"""
    tabula = try_import_tabula()
    if tabula is None:
        return []
    
    # Check if Java is available
    if not check_java_installation():
        logger.warning("Java not available, skipping tabula-py")
        return []
    
    try:
        logger.info("Extracting tables with tabula-py (header-based grouping)...")
        
        # First, extract all content to find table headers
        all_tables = []
        
        # Method 1: Lattice tables
        try:
            lattice_tables = tabula.read_pdf(
                pdf_path,
                pages='all',
                multiple_tables=True,
                guess=False,
                lattice=True,
                stream=False,
                pandas_options={'header': None}
            )
            if lattice_tables:
                all_tables.extend(lattice_tables)
                logger.info(f"Found {len(lattice_tables)} lattice tables")
        except Exception as e:
            logger.warning(f"Lattice extraction failed: {e}")
        
        # Method 2: Stream tables
        try:
            stream_tables = tabula.read_pdf(
                pdf_path,
                pages='all',
                multiple_tables=True,
                guess=False,
                lattice=False,
                stream=True,
                pandas_options={'header': None}
            )
            if stream_tables:
                all_tables.extend(stream_tables)
                logger.info(f"Found {len(stream_tables)} stream tables")
        except Exception as e:
            logger.warning(f"Stream extraction failed: {e}")
        
        return all_tables
        
    except Exception as e:
        logger.error(f"Error with tabula-py: {e}")
        return []

def extract_tables_with_camelot(pdf_path):
    """Extract tables using camelot-py with header-based grouping"""
    camelot = try_import_camelot()
    if camelot is None:
        return []
    
    try:
        logger.info("Extracting tables with camelot-py (header-based grouping)...")
        
        tables = []
        
        # Method 1: Stream flavor
        try:
            stream_tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')
            if stream_tables:
                tables.extend([table.df for table in stream_tables])
                logger.info(f"Found {len(stream_tables)} stream tables with camelot")
        except Exception as e:
            logger.warning(f"Camelot stream extraction failed: {e}")
        
        # Method 2: Lattice flavor
        try:
            lattice_tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
            if lattice_tables:
                tables.extend([table.df for table in lattice_tables])
                logger.info(f"Found {len(lattice_tables)} lattice tables with camelot")
        except Exception as e:
            logger.warning(f"Camelot lattice extraction failed: {e}")
        
        return tables
        
    except Exception as e:
        logger.error(f"Error with camelot-py: {e}")
        return []

def find_table_headers(all_tables):
    """
    Scan through all tables to find table headers and their positions
    """
    logger.info("Scanning for table headers...")
    
    table_headers = []
    
    for i, table in enumerate(all_tables):
        if table is None or table.empty:
            continue
        
        # Clean the table
        cleaned_table = clean_table_data(table)
        if cleaned_table is None or cleaned_table.empty:
            continue
        
        # Look for table headers in the first few rows
        for row_idx in range(min(3, len(cleaned_table))):
            row_text = ' '.join(str(cell) for cell in cleaned_table.iloc[row_idx] if pd.notna(cell) and str(cell).strip())
            
            # Check for table header patterns
            table_match = re.search(r'Table\s+(\d+):\s*(.+)', row_text, re.IGNORECASE)
            if table_match:
                table_num = table_match.group(1)
                table_desc = table_match.group(2).strip()
                tech_category = identify_technology_category(table_desc)
                
                table_headers.append({
                    'index': i,
                    'table_num': table_num,
                    'description': table_desc,
                    'technology': tech_category,
                    'table': cleaned_table
                })
                
                logger.info(f"Found table header: Table {table_num} - {table_desc} -> {tech_category}")
                break
    
    logger.info(f"Found {len(table_headers)} table headers")
    return table_headers

def extract_integrated_tables(all_tables, table_headers):
    """
    Extract integrated tables based on header positions
    """
    logger.info("Extracting integrated tables based on headers...")
    
    integrated_tables = []
    
    for i, header in enumerate(table_headers):
        start_index = header['index']
        
        # Find the end index (next header or end of all tables)
        if i + 1 < len(table_headers):
            end_index = table_headers[i + 1]['index']
        else:
            end_index = len(all_tables)
        
        logger.info(f"Processing Table {header['table_num']} ({header['technology']}): tables {start_index} to {end_index-1}")
        
        # Collect all tables between this header and the next header
        tables_in_group = []
        for j in range(start_index, end_index):
            if j < len(all_tables) and all_tables[j] is not None and not all_tables[j].empty:
                cleaned_table = clean_table_data(all_tables[j])
                if cleaned_table is not None and not cleaned_table.empty:
                    tables_in_group.append(cleaned_table)
        
        if tables_in_group:
            # Merge all tables in this group
            if len(tables_in_group) == 1:
                integrated_table = tables_in_group[0]
            else:
                logger.info(f"Merging {len(tables_in_group)} tables for Table {header['table_num']}")
                integrated_table = pd.concat(tables_in_group, ignore_index=True)
                integrated_table = integrated_table.drop_duplicates()
            
            integrated_tables.append({
                'table_num': header['table_num'],
                'description': header['description'],
                'technology': header['technology'],
                'data': integrated_table
            })
            
            logger.info(f"Created integrated table: {header['table_num']} - {header['description']} ({len(integrated_table)} rows)")
    
    return integrated_tables

def analyze_table_content(df):
    """
    Analyze table content to extract table title and determine if it's a header table
    """
    if df is None or df.empty:
        return None, False
    
    # Look at the first few rows for table titles
    header_text = ""
    for i in range(min(3, len(df))):
        row_text = ' '.join(str(cell) for cell in df.iloc[i] if pd.notna(cell) and str(cell).strip())
        header_text += " " + row_text
    
    # Extract table title
    table_title = extract_table_name_from_text(header_text)
    
    # Check if this looks like a header table (contains "Table X:" pattern)
    is_header = bool(re.search(r'Table\s+\d+:', header_text, re.IGNORECASE))
    
    return table_title, is_header

def group_tables_by_technology(all_tables):
    """
    Group tables by technology category based on table titles
    """
    logger.info("Analyzing and grouping tables by technology category...")
    
    # Group tables by technology category
    tech_groups = defaultdict(list)
    header_tables = []
    unknown_tables = []
    
    for i, table in enumerate(all_tables):
        if table is None or table.empty:
            continue
        
        # Clean the table
        cleaned_table = clean_table_data(table)
        if cleaned_table is None or cleaned_table.empty:
            continue
        
        # Analyze table content
        table_title, is_header = analyze_table_content(cleaned_table)
        
        if is_header:
            # This is a header table - identify technology category
            tech_category = identify_technology_category(table_title)
            logger.info(f"Found header table: {table_title} -> Category: {tech_category}")
            header_tables.append((tech_category, table_title, cleaned_table))
        else:
            # This might be a continuation table - try to match with previous headers
            tech_category = identify_technology_category(table_title)
            if tech_category != "Unknown":
                logger.info(f"Found continuation table: {table_title} -> Category: {tech_category}")
                tech_groups[tech_category].append(cleaned_table)
            else:
                unknown_tables.append(cleaned_table)
    
    # Process header tables and group related tables
    for tech_category, table_title, header_table in header_tables:
        tech_groups[tech_category].append(header_table)
    
    # Add unknown tables to a separate group
    if unknown_tables:
        tech_groups["Unknown"].extend(unknown_tables)
    
    logger.info(f"Grouped tables into {len(tech_groups)} technology categories:")
    for category, tables in tech_groups.items():
        logger.info(f"  {category}: {len(tables)} tables")
    
    return tech_groups

def identify_table_continuation(table1, table2):
    """
    Check if table2 is a continuation of table1
    """
    if table1 is None or table2 is None or table1.empty or table2.empty:
        return False
    
    # Check if column structures are similar
    if len(table1.columns) != len(table2.columns):
        return False
    
    # Check if the first row of table2 looks like a continuation (no header-like content)
    first_row_text = ' '.join(str(cell) for cell in table2.iloc[0] if pd.notna(cell))
    
    # If first row contains table headers, it's not a continuation
    if re.search(r'Table\s+\d+:', first_row_text, re.IGNORECASE):
        return False
    
    # Check if the last row of table1 and first row of table2 have similar data types
    # (e.g., both contain numbers, both contain text, etc.)
    last_row_table1 = table1.iloc[-1]
    first_row_table2 = table2.iloc[0]
    
    # Simple heuristic: if both rows have similar non-empty cell patterns
    non_empty_table1 = sum(1 for cell in last_row_table1 if pd.notna(cell) and str(cell).strip())
    non_empty_table2 = sum(1 for cell in first_row_table2 if pd.notna(cell) and str(cell).strip())
    
    if abs(non_empty_table1 - non_empty_table2) <= 1:  # Allow for slight differences
        return True
    
    return False

def merge_related_tables(tables_in_category):
    """
    Merge related tables within a technology category with intelligent continuation detection
    """
    if not tables_in_category:
        return None
    
    if len(tables_in_category) == 1:
        return tables_in_category[0]
    
    logger.info(f"Merging {len(tables_in_category)} tables in category...")
    
    # Sort tables by their content to try to put related tables together
    sorted_tables = sorted(tables_in_category, key=lambda x: str(x.iloc[0, 0]) if not x.empty else "")
    
    merged_tables = []
    current_group = [sorted_tables[0]]
    
    for i in range(1, len(sorted_tables)):
        current_table = sorted_tables[i]
        last_table = current_group[-1]
        
        # Check if this table is a continuation of the previous one
        if identify_table_continuation(last_table, current_table):
            current_group.append(current_table)
            logger.info(f"Identified table continuation, group now has {len(current_group)} tables")
        else:
            # Merge the current group
            if len(current_group) > 1:
                group_merged = pd.concat(current_group, ignore_index=True)
                group_merged = group_merged.drop_duplicates()
                merged_tables.append(group_merged)
                logger.info(f"Merged group of {len(current_group)} tables")
            else:
                merged_tables.append(current_group[0])
            
            # Start new group
            current_group = [current_table]
    
    # Handle the last group
    if len(current_group) > 1:
        group_merged = pd.concat(current_group, ignore_index=True)
        group_merged = group_merged.drop_duplicates()
        merged_tables.append(group_merged)
        logger.info(f"Merged final group of {len(current_group)} tables")
    else:
        merged_tables.append(current_group[0])
    
    # If we have multiple merged tables, try to combine them
    if len(merged_tables) > 1:
        logger.info(f"Combining {len(merged_tables)} merged table groups")
        final_merged = pd.concat(merged_tables, ignore_index=True)
        final_merged = final_merged.drop_duplicates()
        return final_merged
    else:
        return merged_tables[0]

# Manual mapping of table names to page ranges (1-based, inclusive)
TABLE_PAGE_MAPPING = {
    'Table 7: List of large hydropower sources': '13-15',
    'Table 8: List of hydropower plants with capacity under 50 MW connected at voltage level 220 kV or higher': '16-17',
    'Table 9: List of pumped storage hydropower plants': '18-19',
    'Table 11: Projected portfolio of battery storage projects (MW)': '20',
    'Table 12: List of proposed onshore and nearshore wind power projects approved in Power Plan VIII, Power Plan VIII Implementation Plan': '21-33',
    'Table 13: List of onshore and nearshore wind power projects allocated': '34-45',
    'Table 14: List of concentrated solar power projects': '46-71',
    'Table 15: List of biomass power projects with capacity of 50 MW or more and projects with capacity less than 50 MW connected at voltage level of 220 kV or more': '72-73',
    'Table 16: List of waste-to-energy projects with capacity of 50 MW or more and projects with capacity of less than 50 MW connected at voltage level of 220 kV or more': '74',
    'Table 1: List of UHVDC projects in the period 2031-2035': '79',
    'Table 2: Orientation of UHVAC lines and transformer stations 765 ÷1000K period 2031-2035': '80',
    'Table 3: List of newly built and renovated 500 kV transformer stations in the Northern region': '81-83',
    'Table 4: List of newly built and renovated 500 kV lines in the Northern region': '84-88',
    'Table 5: List of newly built and renovated 220 kV transformer stations in the Northern region': '89-94',
    'Table 6: List of newly built and renovated 220 kV lines in the Northern region': '95-107',
    'Table 7: List of newly built and renovated 500 kV substations in the Central region': '108-109',
    'Table 8: List of newly built and renovated 500 kV lines in the Central region': '110-112',
    'Table 9: List of newly built and renovated 220 kV substations in the Central region': '112-115',
    'Table 10: List of newly built and renovated 220 kV lines in the Central region': '116-121',
    'Table 11: List of newly built and renovated 500 kV substations in the Southern region': '122-123',
    'Table 12: List of newly built and renovated 500 kV lines in the Southern region': '124-127',
    'Table 13: List of newly built and renovated 220 kV substations in the Southern region': '128-131',
    'Table 14: List of renovated and newly constructed 220 kV lines in the Southern region': '132-142',
}

def extract_tables_for_page_range(pdf_path, page_range):
    """
    Extract and aggregate all tables from a given page range using tabula-py.
    Returns a list of DataFrames.
    """
    tabula = try_import_tabula()
    if tabula is None:
        return []
    if not check_java_installation():
        logger.warning("Java not available, skipping tabula-py")
        return []
    try:
        logger.info(f"Extracting tables from pages {page_range}...")
        tables = tabula.read_pdf(
            pdf_path,
            pages=page_range,
            multiple_tables=True,
            guess=True,  # Use guess mode for better structure
            lattice=False,
            stream=True,
            pandas_options={'header': None}
        )
        return tables
    except Exception as e:
        logger.error(f"Tabula extraction failed for pages {page_range}: {e}")
        return []

def extract_tables_from_pdf(pdf_path, output_path):
    """
    Extract tables from PDF using manual page mapping for each table.
    Aggregate, clean, and export each table to a separate worksheet.
    """
    logger.info(f"Starting extraction from: {pdf_path}")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        successful_tables = 0
        for table_name, page_range in TABLE_PAGE_MAPPING.items():
            tables = extract_tables_for_page_range(pdf_path, page_range)
            if not tables:
                logger.warning(f"No tables found for {table_name} (pages {page_range})")
                continue
            # Clean and aggregate
            cleaned_tables = [clean_table_data(df) for df in tables if df is not None and not df.empty]
            if not cleaned_tables:
                logger.warning(f"No valid data for {table_name} (pages {page_range})")
                continue
            merged_table = pd.concat(cleaned_tables, ignore_index=True)
            merged_table = merged_table.drop_duplicates()
            if merged_table is None or merged_table.empty:
                logger.warning(f"Merged table is empty for {table_name} (pages {page_range})")
                continue
            # Worksheet name: use table_name, truncate to 31 chars, clean for Excel
            worksheet_name = table_name[:31]
            worksheet_name = re.sub(r'[\\/*?:\[\]]', '_', worksheet_name)
            # Handle duplicate worksheet names
            original_name = worksheet_name
            counter = 1
            while worksheet_name in [sheet.title for sheet in writer.book.worksheets]:
                worksheet_name = f"{original_name}_{counter}"[:31]
                counter += 1
            try:
                merged_table.to_excel(writer, sheet_name=worksheet_name, index=False)
                logger.info(f"Successfully wrote table: {worksheet_name}")
                successful_tables += 1
            except Exception as e:
                logger.error(f"Error writing table {worksheet_name} to Excel: {e}")
                simple_name = f"Table_{successful_tables + 1}"
                merged_table.to_excel(writer, sheet_name=simple_name, index=False)
                logger.info(f"Wrote table with simple name: {simple_name}")
                successful_tables += 1
    logger.info(f"Extraction complete. Successfully processed {successful_tables} tables.")
    logger.info(f"Output saved to: {output_path}")
    return output_path

def main():
    """
    Main function to run the PDF table extraction
    """
    # Define file paths
    pdf_path = r"C:\Users\SamClissold\MCCF_Git_Repo\MCCF\PDP8\data\PDP8 power projects (english translation).pdf"
    output_path = r"C:\Users\SamClissold\MCCF_Git_Repo\MCCF\PDP8\data\PDP8_power_projects_by_technology.xlsx"
    
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Check Java installation
        java_available = check_java_installation()
        if not java_available:
            print("⚠️  Java not found. tabula-py will not work.")
            print("   Download Java from: https://www.java.com/download/")
            print("   Or try the simple extraction script: extract_pdf_tables_simple.py")
        
        # Extract tables from PDF
        result_path = extract_tables_from_pdf(pdf_path, output_path)
        
        print(f"\n✅ Successfully extracted tables from PDF!")
        print(f"📁 Output file: {result_path}")
        print(f"📊 Tables grouped by technology category in separate worksheets.")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"❌ Error: {e}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 
import os
import pandas as pd
import win32com.client
import openpyxl
import subprocess
import gc


excel_path = r"C:\Users\SamClissold\OneDrive - Carbon Tracker Initiative\Desktop\MCCF\Models\Ninh Binh Gas Model_V8.xlsm"

def extract_all_vba(file_path):
    print("\n=== VBA CODE EXTRACTION ===")
    excel = win32com.client.Dispatch("Excel.Application")
    excel.Visible = False
    excel.DisplayAlerts = False

    wb = excel.Workbooks.Open(file_path)

    try:
        vba_output = []
        for vb_comp in wb.VBProject.VBComponents:
            name = vb_comp.Name
            line_count = vb_comp.CodeModule.CountOfLines
            if line_count > 0:
                code = vb_comp.CodeModule.Lines(1, line_count)
                vba_output.append(f"\n--- {name} ---\n{code}")
                print(f"\n--- {name} ---\n{code}")

        
        # Optionally save to file
        with open("extracted_vba.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(vba_output))

    finally:
        wb.Close(SaveChanges=False)
        excel.Quit()

def extract_all_data(file_path):
    print("\n=== WORKSHEET DATA EXTRACTION ===")
    wb = openpyxl.load_workbook(file_path, data_only=True)

    for sheet_name in wb.sheetnames:
        sheet = wb[sheet_name]
        print(f"\n--- Sheet: {sheet_name} ---")
        
        data = sheet.values
        df = pd.DataFrame(data)
        print(df.head(10))  # Print only top 10 rows per sheet

        # Save complete sheet to CSV (removed the [:30] slice)
        df.to_csv(f"{sheet_name}_complete.csv", index=False, header=False)

    print("\n✅ All sheets exported as complete CSV files.")



def extract_named_ranges(file_path):
    print("\n=== NAMED RANGES EXTRACTION ===")
    excel = win32com.client.Dispatch("Excel.Application")
    excel.Visible = False
    excel.DisplayAlerts = False

    wb = excel.Workbooks.Open(file_path)

    try:
        named_output = []
        for name in wb.Names:
            try:
                named_output.append(f"{name.Name} = {name.RefersTo}")
                print(f"{name.Name} = {name.RefersTo}")
            except Exception as e:
                named_output.append(f"{name.Name} = <Error: {e}>")

        # Optionally save to file
        with open("extracted_named_ranges.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(named_output))

    finally:
        wb.Close(SaveChanges=False)
        excel.Quit()


def extract_formulas_from_sheets(excel_path, sheet_names, output_path):
    excel = win32com.client.Dispatch("Excel.Application")
    excel.Visible = False
    wb = excel.Workbooks.Open(os.path.abspath(excel_path))

    with open(output_path, "w", encoding="utf-8") as f:
        for sheet_name in sheet_names:
            try:
                ws = wb.Sheets(sheet_name)
            except Exception as e:
                f.write(f"\nSheet '{sheet_name}' not found. Skipping.\n")
                continue

            f.write(f"\n--- Sheet: {sheet_name} ---\n")

            used_range = ws.UsedRange
            row_count = used_range.Rows.Count
            col_count = used_range.Columns.Count

            for row in range(1, row_count + 1):
                for col in range(1, col_count + 1):
                    cell = ws.Cells(row, col)
                    formula = cell.Formula
                    if isinstance(formula, str) and formula.startswith("="):
                        address = cell.Address
                        f.write(f"{address}: {formula}\n")
    
    wb.Close(False)
    excel.Quit()

def extract_all_formulas_from_workbook(excel_path, output_path):
    """Extract formulas from ALL sheets in the workbook"""
    print("\n=== COMPLETE FORMULA EXTRACTION ===")
    
    # Kill any existing Excel processes first
    try:
        subprocess.run(['taskkill', '/f', '/im', 'excel.exe'], capture_output=True)
        print("Killed existing Excel processes")
    except:
        pass
    
    excel = None
    wb = None
    
    try:
        excel = win32com.client.Dispatch("Excel.Application")
        excel.Visible = False
        excel.DisplayAlerts = False
        
        print(f"Opening workbook: {excel_path}")
        wb = excel.Workbooks.Open(os.path.abspath(excel_path))
        
        # Get sheet names first to avoid COM issues
        sheet_names = []
        for i in range(1, wb.Sheets.Count + 1):
            try:
                sheet_names.append(wb.Sheets(i).Name)
            except Exception as e:
                print(f"Error getting sheet {i}: {e}")
                continue
        
        print(f"Found {len(sheet_names)} sheets: {sheet_names}")
        
        with open(output_path, "w", encoding="utf-8") as f:
            for sheet_name in sheet_names:
                print(f"Processing sheet: {sheet_name}")
                f.write(f"\n--- Sheet: {sheet_name} ---\n")

                try:
                    sheet = wb.Sheets(sheet_name)
                    
                    # Get the used range
                    used_range = sheet.UsedRange
                    row_count = used_range.Rows.Count
                    col_count = used_range.Columns.Count
                    
                    print(f"Sheet {sheet_name}: {row_count} rows, {col_count} columns")

                    # Process in smaller chunks to avoid COM errors
                    chunk_size = 50  # Reduced chunk size
                    formula_count = 0
                    
                    for start_row in range(1, row_count + 1, chunk_size):
                        end_row = min(start_row + chunk_size - 1, row_count)
                        
                        for row in range(start_row, end_row + 1):
                            for col in range(1, col_count + 1):
                                try:
                                    cell = sheet.Cells(row, col)
                                    formula = cell.Formula
                                    
                                    # Check if it's a formula (starts with =)
                                    if isinstance(formula, str) and formula.startswith("="):
                                        address = cell.Address
                                        f.write(f"{address}: {formula}\n")
                                        formula_count += 1
                                        
                                except Exception as e:
                                    print(f"Error accessing cell {row},{col} in sheet {sheet_name}: {e}")
                                    continue
                    
                    print(f"Found {formula_count} formulas in sheet {sheet_name}")
                    
                except Exception as e:
                    print(f"Error processing sheet {sheet_name}: {e}")
                    f.write(f"Error processing sheet {sheet_name}: {e}\n")
                    continue
                
                # Force garbage collection after each sheet
                gc.collect()
        
        print(f"✅ All formulas extracted to {output_path}")
        
    except Exception as e:
        print(f"Error during extraction: {e}")
        raise
        
    finally:
        # Clean up
        if wb:
            try:
                wb.Close(SaveChanges=False)
            except:
                pass
        if excel:
            try:
                excel.Quit()
            except:
                pass
        
        # Kill Excel process again to ensure cleanup
        try:
            subprocess.run(['taskkill', '/f', '/im', 'excel.exe'], capture_output=True)
        except:
            pass




if __name__ == "__main__":
    extract_all_vba(excel_path)
    extract_all_data(excel_path)
    extract_named_ranges(excel_path)
    #Use the new function to extract ALL formulas from ALL sheets
    extract_all_formulas_from_workbook(excel_path, "extracted_formulas.txt")
    print("\n✅ All extractions complete.")



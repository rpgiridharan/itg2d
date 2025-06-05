import subprocess
import os
import numpy as np
import re
import sys

# --- Configuration ---
# Set to True to skip runs if the output file already exists.
# Set to False to always run, potentially overwriting existing files.
skip_existing_runs = True
# --- End Configuration ---

# Path to the original ITG2D script
original_script_path = "itg2d.py"
# Name for the temporary script that will be executed
temp_script_path = "itg2d_temp_run.py"

# Define the range of kapt values
kapt_values = np.arange(0.3, 1.5 + 0.01, 0.1) # Add epsilon to include 1.5

# Function to format numbers in scientific notation for filenames (must match itg2d.py)
def format_exp(d):
    dstr = f"{d:.1e}"
    base, exp_val = dstr.split("e")
    base = base.replace(".", "_")
    if "-" in exp_val:
        exp_val = exp_val.replace("-", "")
        prefix = "em"
    else:
        prefix = "e"
    exp_val = str(int(exp_val))
    return f"{base}_{prefix}{exp_val}"

# Read the original script content
try:
    with open(original_script_path, 'r') as f:
        original_content = f.read()
except FileNotFoundError:
    print(f"Error: Original script '{original_script_path}' not found.")
    print(f"Please ensure it is in the same directory as this script: {os.path.dirname(os.path.abspath(__file__))}")
    sys.exit(1)

# Helper function to extract the filename expression from the script content
def extract_filename_expression(content):
    # Look for the line that defines filename
    pattern = r"^filename\s*=\s*(.+)$"
    match = re.search(pattern, content, re.MULTILINE)
    if match:
        return match.group(1).strip()
    else:
        print(f"Warning: Could not extract 'filename' from '{original_script_path}'.")
        print(f"Ensure 'filename' is defined on its own line like 'filename = ...'.")
        return None

# Extract filename expression if skip_existing_runs is True
filename_expression = None
if skip_existing_runs:
    print(f"Extracting filename expression from '{original_script_path}' for file existence checking...")
    filename_expression = extract_filename_expression(original_content)
    
    if filename_expression is None:
        print(f"Error: Could not extract filename expression from '{original_script_path}'.")
        print(f"Cannot proceed with skip_existing_runs=True. Please check '{original_script_path}' or set skip_existing_runs=False.")
        sys.exit(1)
    
    print("Successfully extracted filename expression for checking.")

print(f"Starting simulations for kapt values: {[float(f'{k:.1f}') for k in kapt_values]}\n")

for kapt_val_raw in kapt_values:
    kapt_val = round(kapt_val_raw, 1) # Round to one decimal place

    current_kapt_str_for_script = f"{kapt_val:.1f}"  # e.g., "0.3", "1.0"
    current_kapt_str_for_filename = str(kapt_val).replace(".", "_") # e.g., "0_3", "1_0"

    if skip_existing_runs:
        # Create a modified version of the filename expression with the current kapt value
        # Replace str(kapt) in the filename expression with the current kapt value
        modified_filename_expr = filename_expression.replace("str(kapt)", f"str({kapt_val})")
        
        # We need to evaluate this expression to get the actual filename
        # First, we need to extract the required variables from the original script
        try:
            # Create a local namespace with the required variables
            local_vars = {}
            
            # Extract the variables needed for filename construction
            for var_name in ['output_dir', 'chi', 'DPhi', 'HPhi']:
                # Extract string variables (like output_dir)
                if var_name == 'output_dir':
                    pattern = rf"^{var_name}\s*=\s*\"([^\"]*)\""
                    match = re.search(pattern, original_content, re.MULTILINE)
                    if match:
                        local_vars[var_name] = match.group(1)
                # Extract numeric variables
                else:
                    pattern = rf"^{var_name}\s*=\s*([0-9.eE\-+]+)"
                    match = re.search(pattern, original_content, re.MULTILINE)
                    if match:
                        local_vars[var_name] = float(match.group(1))
                
                if var_name not in local_vars:
                    raise ValueError(f"Could not extract {var_name}")
            
            # Add the format_exp function to local namespace
            local_vars['format_exp'] = format_exp
            local_vars['str'] = str
            local_vars['kapt'] = kapt_val
            
            # Evaluate the filename expression
            expected_filename = eval(modified_filename_expr, {"__builtins__": {}}, local_vars)
            
        except Exception as e:
            print(f"Error evaluating filename expression for kapt = {current_kapt_str_for_script}: {e}")
            print("Proceeding with simulation...")
        else:
            if os.path.exists(expected_filename):
                print(f"-----------------------------------------------------")
                print(f"Skipping simulation for kapt = {current_kapt_str_for_script}: Output file exists.")
                print(f"  File: {expected_filename}")
                print(f"-----------------------------------------------------\n")
                continue

    print(f"-----------------------------------------------------")
    print(f"Preparing to run simulation for kapt = {current_kapt_str_for_script}")

    # Modify the original script content to set the current kapt_val
    # Pattern: kapt = VALUE # Optional comment
    kapt_line_pattern = re.compile(r"^(kapt\s*=\s*)([0-9\.]+)(.*)$", re.MULTILINE)
    replacement_str = fr"\g<1>{current_kapt_str_for_script}\g<3>"
    
    modified_content, num_replacements = kapt_line_pattern.subn(replacement_str, original_content, count=1)

    if num_replacements == 0:
        print(f"Warning: Could not find or modify the 'kapt' line in '{original_script_path}' (e.g., 'kapt = 1.0').")
        print(f"The script '{temp_script_path}' will run with the original kapt value from '{original_script_path}'.")
        # modified_content will be the same as original_content if no substitution occurred
    
    try:
        with open(temp_script_path, 'w') as f:
            f.write(modified_content)
    except IOError as e:
        print(f"Error writing temporary script '{temp_script_path}': {e}. Skipping this run.")
        continue

    print(f"Running '{temp_script_path}' with kapt = {current_kapt_str_for_script}...")
    
    try:
        process = subprocess.Popen(
            [sys.executable, temp_script_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)) # Run from script's directory
        )
        stdout, stderr = process.communicate()

        if process.returncode == 0:
            print(f"Successfully ran for kapt = {current_kapt_str_for_script}")
            if stdout.strip(): print(f"Output from script:\n{stdout}")
        else:
            print(f"Error running script for kapt = {current_kapt_str_for_script}. Return code: {process.returncode}")
            if stdout.strip(): print(f"Standard Output:\n{stdout}")
            if stderr.strip(): print(f"Standard Error:\n{stderr}")
            # print("Stopping further simulations due to error.") # Optional: uncomment to stop on error
            # break
    except FileNotFoundError:
        print(f"Critical Error: Python interpreter '{sys.executable}' or script '{temp_script_path}' not found.")
        print("Stopping all simulations.")
        break
    except Exception as e:
        print(f"An unexpected error occurred while running script for kapt = {current_kapt_str_for_script}: {e}")
    finally:
        if os.path.exists(temp_script_path):
            try:
                os.remove(temp_script_path)
            except OSError as e:
                print(f"Warning: Could not remove temporary file '{temp_script_path}': {e}")
    print(f"-----------------------------------------------------\n")

print("All simulations finished.")

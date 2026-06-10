import json
import sys
import os
from pathlib import Path

def run_notebook(nb_path):
    print(f"\n==========================================")
    print(f"RUNNING NOTEBOOK: {nb_path}")
    print(f"==========================================")
    
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    # Extract code cells and combine them
    code_blocks = []
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = cell['source']
            if isinstance(source, list):
                code = "".join(source)
            else:
                code = source
            code_blocks.append(code)
            
    # Execute each block in the global context
    global_env = {
        '__name__': '__main__',
        '__file__': str(nb_path)
    }
    
    for i, code in enumerate(code_blocks):
        if not code.strip():
            continue
        try:
            exec(code, global_env)
        except Exception as e:
            print(f"ERROR: Exception in cell {i+1} of {nb_path}:", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            raise e
            
    print(f"✓ Notebook {nb_path} completed successfully.")

def main():
    # Set CWD to project directory
    project_dir = Path(__file__).resolve().parent
    os.chdir(project_dir)
    sys.path.append(str(project_dir))
    
    notebooks = [
        '01_notebook_eda_preprocessing.ipynb',
        '02_notebook_direct_classification.ipynb',
        '03_notebook_cascade.ipynb',
        '04_notebook_multitask.ipynb'
    ]
    
    for nb in notebooks:
        run_notebook(nb)
        
    print("\n✓ ALL PIPELINE NOTEBOOKS EXECUTED SUCCESSFULLY.")

if __name__ == '__main__':
    main()

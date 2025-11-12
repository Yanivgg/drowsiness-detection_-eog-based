"""
Prepare files for upload to Google Colab
Creates a zip file with all necessary code and data
"""

import os
import zipfile
from pathlib import Path

def create_colab_package():
    """
    Create a zip file with all necessary files for Colab training
    """
    
    # Define what to include (updated for Phase1_CNN structure)
    data_files = Path('../data/processed/files')
    code_dir = Path('../src/models')
    file_sets = Path('../data/file_sets.mat')
    
    # Output zip file
    zip_path = 'microsleep_colab_package.zip'
    
    # Verify required files exist
    print("Checking required files...")
    
    if not data_files.exists():
        print("ERROR: data/processed/files/ directory not found!")
        return False
    
    mat_files = list(data_files.glob('*.mat'))
    if len(mat_files) != 20:
        print(f"WARNING: Expected 20 .mat files, found {len(mat_files)}")
    else:
        print(f"Found {len(mat_files)} .mat files")
    
    if not file_sets.exists():
        print("ERROR: data/file_sets.mat not found!")
        return False
    else:
        print("Found file_sets.mat")
    
    if not code_dir.exists():
        print("ERROR: src/models/ directory not found!")
        return False
    else:
        print("Found src/models/ directory")
    
    # Create zip file
    print(f"\nCreating {zip_path}...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add data files (keep old structure in zip for compatibility)
        print("Adding data files...")
        for mat_file in mat_files:
            # Write as data/files/*.mat for compatibility
            arcname = os.path.join('data', 'files', mat_file.name)
            zipf.write(mat_file, arcname)
        zipf.write(file_sets, 'data/file_sets.mat')
        
        # Add src files (write as code/ and src/ for compatibility)
        print("Adding source files...")
        # Add utils and loadData to both src/ and code/
        for util_file in ['../src/utils.py', '../src/loadData.py']:
            if os.path.exists(util_file):
                fname = os.path.basename(util_file)
                zipf.write(util_file, f'code/{fname}')
                zipf.write(util_file, f'src/{fname}')
        
        # Add all model code to both code/ and src/models/ structure
        for root, dirs, files in os.walk(str(code_dir)):
            dirs[:] = [d for d in dirs if d != '__pycache__']
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    # Get relative path from src/models/
                    rel_path = os.path.relpath(file_path, '../src/models')
                    # Write to both locations
                    zipf.write(file_path, f'code/{rel_path}')
                    zipf.write(file_path, f'src/models/{rel_path}')
        
        # Add Colab notebooks
        print("Adding Colab notebooks...")
        notebook_files = [
            '../notebooks/microsleep_colab_complete.ipynb',
            '../notebooks/Result_Analysis.ipynb'
        ]
        script_files = [
            'microsleep_colab_complete.py',
            'results_analysis.py'
        ]
        
        for nb_file in notebook_files:
            if os.path.exists(nb_file):
                zipf.write(nb_file, os.path.basename(nb_file))
        
        for script_file in script_files:
            if os.path.exists(script_file):
                zipf.write(script_file, script_file)
    
    # Get file size
    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"\nPackage created successfully: {zip_path}")
    print(f"Size: {size_mb:.2f} MB")
    
    print("\n" + "="*60)
    print("INSTRUCTIONS FOR COLAB:")
    print("="*60)
    print("1. Upload microsleep_colab_package.zip to Google Colab")
    print("2. Unzip the package in Colab:")
    print("   !unzip -q microsleep_colab_package.zip")
    print("3. Upload microsleep_colab_complete.ipynb to Colab")
    print("   OR copy cells from microsleep_colab_complete.py manually")
    print("4. Run cells in order to train and evaluate models")
    print("5. Models are automatically saved to Google Drive")
    print("="*60)
    
    return True


def show_package_contents():
    """
    Show what will be included in the package
    """
    print("\nPackage will contain:")
    print("\n1. Data files (data/processed/files/):")
    mat_files = list(Path('../data/processed/files').glob('*.mat'))
    for f in sorted(mat_files):
        size = f.stat().st_size / (1024 * 1024)
        print(f"   - {f.name} ({size:.2f} MB)")
    
    print("\n2. File splits (data/file_sets.mat)")
    
    print("\n3. Source files:")
    print("   - src/loadData.py")
    print("   - src/utils.py")
    print("   - src/models/cnn/CNN_2s/ (myModel.py, train.py, predict_val.py, predict.py)")
    print("   - src/models/cnn/CNN_4s/ (myModel.py, train.py, predict_val.py, predict.py)")
    print("   - src/models/cnn/CNN_8s/ (myModel.py, train.py, predict_val.py, predict.py)")
    print("   - src/models/cnn/CNN_16s/ (myModel.py, train.py, predict_val.py, predict.py)")
    print("   - src/models/cnn/CNN_32s/ (myModel.py, train.py, predict_val.py, predict.py)")
    print("   - src/models/cnn_lstm/CNN_LSTM/ (myModel.py, train.py, predict_val.py, predict.py)")
    
    print("\n4. Notebooks:")
    print("   - microsleep_colab_complete.ipynb (training notebook)")
    print("   - microsleep_colab_complete.py (training backup)")
    print("   - Result_Analysis.ipynb (analysis notebook)")
    print("   - results_analysis.py (analysis backup)")
    
    print("\nNote: Files are packaged with old 'code/' paths for backward compatibility")
    
    total_size = 0
    for f in Path('../data/processed/files').glob('*.mat'):
        total_size += f.stat().st_size
    total_size += Path('../data/file_sets.mat').stat().st_size if Path('../data/file_sets.mat').exists() else 0
    
    print(f"\nEstimated total size: {total_size / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    import sys
    
    print("="*60)
    print("Microsleep Detection - Colab Package Creator")
    print("="*60)
    
    show_package_contents()
    
    print("\n" + "="*60)
    
    # Check if running with --auto flag
    if len(sys.argv) > 1 and sys.argv[1] == '--auto':
        response = 'yes'
        print("Auto mode: Creating package...")
    else:
        response = input("Create package? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        success = create_colab_package()
        if success:
            print("\nPackage ready for upload to Google Colab!")
        else:
            print("\nERROR: Package creation failed. Please check the errors above.")
    else:
        print("\nPackage creation cancelled.")


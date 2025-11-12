"""
Prepare Phase 2 files for Google Colab upload.

This script packages:
- Feature CSV files (train + test)
- Training notebook
- Feature engineering modules (for reference)
"""

import os
import zipfile
from pathlib import Path

print("=" * 70)
print("Preparing Phase 2 files for Google Colab")
print("=" * 70)

# Define files to include
FILES_TO_PACKAGE = [
    # Feature data (main files)
    ('features/train_features_16s.csv', 'train_features_16s.csv'),
    ('features/test_features_16s.csv', 'test_features_16s.csv'),
    
    # Training notebook and script
    ('train_ml_colab.ipynb', 'train_ml_colab.ipynb'),
    ('train_ml_colab.py', 'train_ml_colab.py'),
    
    # Documentation
    ('README.md', 'phase2_README.md'),
    ('INSTRUCTIONS.md', 'INSTRUCTIONS.md'),
]

# Optional: Include feature engineering modules for reference
OPTIONAL_MODULES = [
    'feature_engineering/__init__.py',
    'feature_engineering/time_domain.py',
    'feature_engineering/frequency_domain.py',
    'feature_engineering/nonlinear.py',
    'feature_engineering/eog_specific.py',
]

def get_file_size_mb(filepath):
    """Get file size in MB."""
    return os.path.getsize(filepath) / (1024 * 1024)

def main():
    # Get script directory
    script_dir = Path(__file__).parent
    output_zip = script_dir.parent / 'phase2_colab_package.zip'
    
    print(f"\n📦 Creating package: {output_zip.name}")
    print()
    
    total_size = 0
    files_included = []
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add main files
        for source, dest in FILES_TO_PACKAGE:
            source_path = script_dir / source
            if source_path.exists():
                size_mb = get_file_size_mb(source_path)
                total_size += size_mb
                zipf.write(source_path, dest)
                files_included.append((dest, size_mb))
                print(f"  ✓ Added: {dest:40} ({size_mb:6.2f} MB)")
            else:
                print(f"  ⚠ Skipped (not found): {source}")
        
        # Add optional modules
        print(f"\n  Optional modules:")
        for module in OPTIONAL_MODULES:
            source_path = script_dir / module
            if source_path.exists():
                size_mb = get_file_size_mb(source_path)
                total_size += size_mb
                zipf.write(source_path, f"feature_engineering/{Path(module).name}")
                print(f"    ✓ {module:40} ({size_mb:.3f} MB)")
    
    print()
    print("=" * 70)
    print(f"✅ Package created successfully!")
    print("=" * 70)
    print(f"\n📦 Output: {output_zip}")
    print(f"📊 Total size: {total_size:.2f} MB")
    print(f"📁 Files included: {len(files_included)}")
    
    print("\n" + "=" * 70)
    print("Next Steps:")
    print("=" * 70)
    print("\n1. Upload to Google Colab:")
    print(f"   - Upload file: {output_zip.name}")
    print("   - Run: !unzip -q phase2_colab_package.zip")
    print()
    print("2. Open the training notebook:")
    print("   - train_ml_colab.ipynb")
    print()
    print("3. Mount Google Drive:")
    print("   - Results will be saved to: /content/drive/MyDrive/drowsiness_ml_results/")
    print()
    print("4. Run all cells sequentially")
    print()
    print("5. Training time: ~10-15 minutes (CPU is fine)")
    print()
    
    # Show file breakdown
    print("\n" + "=" * 70)
    print("Package Contents:")
    print("=" * 70)
    for filename, size in files_included:
        print(f"  {filename:50} {size:8.2f} MB")
    
    print("\n" + "=" * 70)
    print("✓ Ready for Colab!")
    print("=" * 70)

if __name__ == '__main__':
    main()


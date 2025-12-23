import os
import shutil
from pathlib import Path
from anomalib.data import MVTecAD

def get_datamodule(root_dir="./datasets/MVTec", category="bottle", batch_size=32):
    """
    Downloads, configures, and returns the MVTec AD DataModule.
    Includes self-healing logic for broken paths or missing data.
    """
    # 1. Download if zip is missing
    if not os.path.exists("mvtec-ad.zip"):
        print("⬇️ Downloading Dataset via Kaggle...")
        os.system("kaggle datasets download -d ipythonx/mvtec-ad")
        print("✅ Download Complete.")

    # 2. Unzip & Fix Paths
    extract_path = Path(root_dir)
    bottle_check = extract_path / category
    
    # Self-healing: If specific category folder is missing, force re-unzip
    if not bottle_check.exists():
        print(f"⚠️ Category '{category}' missing. Unzipping dataset...")
        # Clean up partial extracts to avoid corruption
        if extract_path.exists():
            shutil.rmtree(extract_path)
        
        # Unzip
        os.system(f"unzip -q mvtec-ad.zip -d {extract_path}")
        print("✅ Unzip Complete.")
    
    # 3. Find True Root (Handle subfolder nesting issues)
    # Search for the category folder recursively to find where unzip actually put it
    found_paths = list(extract_path.rglob(category))
    if not found_paths:
        raise FileNotFoundError(f"❌ Critical Error: Could not find category '{category}' even after unzipping.")
    
    true_root = found_paths[0].parent
    
    # 4. Initialize Module
    datamodule = MVTecAD(
        root=true_root,
        category=category,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=2
    )
    
    # Disable internal auto-download hook since we handled it manually
    datamodule.prepare_data = lambda: None
    datamodule.setup()
    
    return datamodule

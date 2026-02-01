import os
import zipfile
from pathlib import Path

def create_colab_zip():
    zip_name = 'swarm_ids_colab.zip'
    ignore_folders = {'.git', '.ipynb_checkpoints', '__pycache__', 'logs', 'mlruns', 'venv', '.agent', 'models'}
    
    # Check if we are inside the project folder or outside
    if os.path.exists('src') and os.path.exists('scripts'):
        root_dir = Path('.')
    elif os.path.exists('swarm-ids-ml'):
        root_dir = Path('swarm-ids-ml')
    else:
        print("❌ Error: Could not find project files. Please run this script from 'deepswarm model' or 'swarm-ids-ml' folder.")
        return

    print(f"📦 Zipping project from: {root_dir.absolute()}")
    
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(root_dir):
            if any(target in root for target in ignore_folders):
                continue
            for file in files:
                if file in {zip_name, 'prepare_colab.py', 'trained_models.zip'}:
                    continue
                file_path = Path(root) / file
                # Save into zip WITHOUT the 'swarm-ids-ml' prefix for easier Colab use
                arcname = file_path.relative_to(root_dir)
                z.write(file_path, arcname)
                
    print(f"✅ Created: {os.path.abspath(zip_name)}")
    print(f"📊 Size: {os.path.getsize(zip_name) / (1024*1024):.2f} MB")
    print(f"\n👉 NEXT: Upload this new zip to Colab.")

if __name__ == "__main__":
    create_colab_zip()

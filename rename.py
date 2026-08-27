import os
import shutil

def rename_content(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    new_content = content.replace('Pythia', 'Pythia')
    new_content = new_content.replace('pythia', 'pythia')
    new_content = new_content.replace('PYTHIA', 'PYTHIA')
    
    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated content in {file_path}")

def walk_and_rename(root_dir):
    for root, dirs, files in os.walk(root_dir):
        if '.git' in dirs:
            dirs.remove('.git')
        if '.venv' in dirs:
            dirs.remove('.venv')
        if '__pycache__' in dirs:
            dirs.remove('__pycache__')
            
        for file in files:
            if file.endswith(('.py', '.md', '.toml', '.txt', '.tex')):
                file_path = os.path.join(root, file)
                try:
                    rename_content(file_path)
                except Exception as e:
                    print(f"Could not read {file_path}: {e}")

walk_and_rename('/Users/finnhertsch/projects/Pythia')
walk_and_rename('/Users/finnhertsch/projects/writing/pythia_bio')

# Rename the python package directory
old_pkg = '/Users/finnhertsch/projects/Pythia/src/pythia'
new_pkg = '/Users/finnhertsch/projects/Pythia/src/pythia'
if os.path.exists(old_pkg):
    os.rename(old_pkg, new_pkg)
    print(f"Renamed {old_pkg} to {new_pkg}")


import os
import sys

def should_ignore(name):
    return name.startswith('.') or name.startswith('_')

def generate_tree(startpath, prefix=''):
    if not os.path.isdir(startpath):
        print(f"Error: {startpath} is not a valid directory")
        return

    entries = [entry for entry in os.scandir(startpath) if not should_ignore(entry.name)]
    entries.sort(key=lambda e: e.name.lower())
    
    for i, entry in enumerate(entries):
        is_last = (i == len(entries) - 1)
        if entry.is_dir():
            print(f"{prefix}{'└── ' if is_last else '├── '}{entry.name}/")
            generate_tree(entry.path, prefix + ("    " if is_last else "│   "))
        else:
            print(f"{prefix}{'└── ' if is_last else '├── '}{entry.name}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "."
    
    print(f"{os.path.basename(path)}/")
    generate_tree(path)

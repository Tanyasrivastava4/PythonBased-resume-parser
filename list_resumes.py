import glob
from pathlib import Path

def list_files():
    print("Files in resumes/ folder:")
    files = sorted(glob.glob("resumes/*"))
    for f in files:
        p = Path(f)
        size_kb = p.stat().st_size / 1024
        print(f"  - {p.name:<60} ({size_kb:.1f} KB)")

if __name__ == "__main__":
    list_files()
import sys
import zipfile
from pathlib import Path

def search_docx_xml(docx_path):
    print(f"Searching raw XML in {Path(docx_path).name}...")
    keywords = ["AMAN", "Aman", "Chauhan", "chauhan", "6387", "gmail", "Intileo"]
    
    with zipfile.ZipFile(docx_path) as z:
        for filename in z.namelist():
            if filename.endswith(".xml") or filename.endswith(".rels"):
                content = z.read(filename).decode("utf-8", errors="ignore")
                found = [k for k in keywords if k in content]
                if found:
                    print(f"\nFOUND in file inside docx: '{filename}' (matches: {found})")
                    # Print context snippet around match
                    for k in found:
                        pos = content.find(k)
                        start = max(0, pos - 150)
                        end = min(len(content), pos + 250)
                        snippet = content[start:end]
                        print(f"\n--- Snippet around '{k}' in {filename} ---")
                        print(snippet)
                        print("-" * 50)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_raw_docx.py \"path/to/resume.docx\"")
        sys.exit(1)
    search_docx_xml(sys.argv[1])
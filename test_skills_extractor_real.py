"""
test_skills_extractor_real.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tests the skills extractor on REAL resume files.
Reads from segmented_text/ folder (already segmented by test_segmenter.py)
so the skills section is already cleanly separated — no need to run
the segmenter again.

Flow:
    segmented_text/*_segmented.txt   ← Layer 1 output (already segmented)
            │
            ▼
    parse sections from file         ← read LABEL blocks from the file
            │
            ▼
    extract_skills()                 ← Layer 2 (skills extractor)
            │
            ▼
    Prints extracted skills + saves one result file per resume into
    skills_extracted_text/  (same pattern as contact_extracted_text/)

Usage:
    # Run on ALL resumes at once:
    python test_skills_extractor_real.py

    # Run on ONE specific resume:
    python test_skills_extractor_real.py segmented_text/Ankush_Bais_AI_Solution_Architect_Intileo_segmented.txt
"""

import sys
import re
import glob
from pathlib import Path

# ── skills extractor ──
from extractors.skills import extract_skills

# ── Folder where Layer 1's segmented output lives ──
INPUT_DIR = Path("segmented_text")

# ── Folder where this script's output will be saved ──
OUTPUT_DIR = Path("skills_extracted_text")


def parse_segmented_file(file_path: str) -> dict:
    """
    Reads a segmented .txt file (output of test_segmenter.py) and
    extracts each section's text into a dict.

    The segmented file looks like this:
        ────────────────────────────────────────────────────────────
        LABEL: skills   CONFIDENCE: 0.95   START LINE: 14
        ────────────────────────────────────────────────────────────
        Python, React, PostgreSQL...

        ────────────────────────────────────────────────────────────
        LABEL: experience   CONFIDENCE: 0.95   START LINE: 31
        ────────────────────────────────────────────────────────────
        Software Engineer at...

    Returns:
        {
            "skills":     "Python, React, PostgreSQL...",
            "experience": "Software Engineer at...",
            ...
        }
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    sections = {}

    # Split on the separator line (60 dashes)
    separator = "─" * 60
    blocks = content.split(separator)

    current_label = None

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # Check if this block is a LABEL header line
        label_match = re.match(
            r"^LABEL:\s+(\w+)\s+CONFIDENCE.*$",
            block,
            re.IGNORECASE
        )

        if label_match:
            # This block is a header → remember the label
            current_label = label_match.group(1).lower()

        elif current_label:
            # This block is content → belongs to current_label
            # Skip the summary line at the bottom
            if block.startswith("[Found"):
                current_label = None
                continue

            if current_label in sections:
                sections[current_label] += "\n" + block
            else:
                sections[current_label] = block

            current_label = None

    return sections


def save_skills_output(file_path: str, sections_dict: dict, result: dict) -> Path:
    """
    Saves one resume's extracted skills to skills_extracted_text/,
    using the same base name as the input file (minus "_segmented",
    plus "_skills.txt").

    e.g. segmented_text/Abhay_Awasthi_..._segmented.txt
         -> skills_extracted_text/Abhay_Awasthi_..._skills.txt
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    original_name = Path(file_path).stem
    if original_name.endswith("_segmented"):
        original_name = original_name[: -len("_segmented")]
    output_path = OUTPUT_DIR / f"{original_name}_skills.txt"

    skills     = result["skills"]
    by_cat     = result["skills_by_category"]
    confidence = result["_confidence"]
    timing     = result["_timing_ms"]
    method     = result["_method"]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"{'─' * 60}\n")
        f.write(f"SKILLS EXTRACTED FROM: {file_path}\n")
        f.write(f"{'─' * 60}\n")

        f.write(f"\n── Skills Section Text (raw, as segmented) ──\n")
        f.write(f"{sections_dict.get('skills', '').strip() or '(none found)'}\n")

        f.write(f"\n── Extraction Results ──\n")
        f.write(f"Skills found   : {len(skills)}\n")
        f.write(f"Confidence     : {confidence}\n")
        f.write(f"Time           : {timing} ms\n")
        f.write(f"Method         : {method}\n")

        f.write(f"\n── All Skills ({len(skills)}) ──\n")
        f.write(", ".join(skills) if skills else "None found")
        f.write("\n")

        f.write(f"\n── By Category ──\n")
        for cat, cat_skills in sorted(by_cat.items()):
            f.write(f"  {cat:<30} {cat_skills}\n")

    return output_path


def run_on_file(file_path: str):
    print(f"\n{'═' * 65}")
    print(f"FILE : {Path(file_path).name}")
    print(f"{'═' * 65}")

    # ── Step 1: parse segmented file ──
    sections_dict = parse_segmented_file(file_path)
    labels_found  = list(sections_dict.keys())
    print(f"Sections found : {labels_found}")

    # ── Step 2: show what skills section contains ──
    skills_text = sections_dict.get("skills", "")
    if skills_text:
        print(f"\n── Skills Section Text ──")
        print(skills_text.strip())
    else:
        print("\n⚠️  No skills section found in this resume")

    # ── Step 3: extract skills ──
    result = extract_skills(sections=sections_dict)

    # ── Step 4: print results ──
    skills     = result["skills"]
    by_cat     = result["skills_by_category"]
    confidence = result["_confidence"]
    timing     = result["_timing_ms"]
    method     = result["_method"]

    print(f"\n── Extraction Results ──")
    print(f"Skills found   : {len(skills)}")
    print(f"Confidence     : {confidence}")
    print(f"Time           : {timing} ms")
    print(f"Method         : {method}")

    print(f"\n── All Skills ({len(skills)}) ──")
    print(", ".join(skills) if skills else "None found")

    print(f"\n── By Category ──")
    for cat, cat_skills in sorted(by_cat.items()):
        print(f"  {cat:<30} {cat_skills}")

    # ── Step 5: save to skills_extracted_text/ ──
    saved_path = save_skills_output(file_path, sections_dict, result)
    print(f"\n[Saved skills info to: {saved_path}]")

    print()


if __name__ == "__main__":
    # If a specific file is passed → run on that file only
    if len(sys.argv) > 1:
        for path in sys.argv[1:]:
            run_on_file(path)
    else:
        # Otherwise run on ALL segmented files in segmented_text/
        files = sorted(glob.glob(str(INPUT_DIR / "*_segmented.txt")))
        if not files:
            print(f"No *_segmented.txt files found in {INPUT_DIR}/ folder.")
            print("Run python test_segmenter.py first to generate them.")
        for f in files:
            run_on_file(f)












## Comenting just bcz wanted to save the o/p under a folder
#"""
#test_skills_extractor_real.py
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#Tests the skills extractor on REAL resume files.
#Reads from segmented_text/ folder (already segmented by test_segmenter.py)
#so the skills section is already cleanly separated — no need to run
#the segmenter again.
#
#Flow:
#    segmented_text/*_segmented.txt   ← Layer 1 output (already segmented)
#            │
#            ▼
#    parse sections from file         ← read LABEL blocks from the file
#            │
#            ▼
#    extract_skills()                 ← Layer 2 (skills extractor)
#            │
#            ▼
#    Prints extracted skills
#
#Usage:
#    # Run on ALL resumes at once:
#    python test_skills_extractor_real.py
#
#    # Run on ONE specific resume:
#    python test_skills_extractor_real.py segmented_text/Ankush_Bais_AI_Solution_Architect_Intileo_segmented.txt
#"""
#
#import sys
#import re
#import glob
#from pathlib import Path
#
## ── skills extractor ──
#from extractors.skills import extract_skills
#
#
#def parse_segmented_file(file_path: str) -> dict:
#    """
#    Reads a segmented .txt file (output of test_segmenter.py) and
#    extracts each section's text into a dict.
#
#    The segmented file looks like this:
#        ────────────────────────────────────────────────────────────
#        LABEL: skills   CONFIDENCE: 0.95   START LINE: 14
#        ────────────────────────────────────────────────────────────
#        Python, React, PostgreSQL...
#
#        ────────────────────────────────────────────────────────────
#        LABEL: experience   CONFIDENCE: 0.95   START LINE: 31
#        ────────────────────────────────────────────────────────────
#        Software Engineer at...
#
#    Returns:
#        {
#            "skills":     "Python, React, PostgreSQL...",
#            "experience": "Software Engineer at...",
#            ...
#        }
#    """
#    with open(file_path, "r", encoding="utf-8") as f:
#        content = f.read()
#
#    sections = {}
#
#    # Split on the separator line (60 dashes)
#    separator = "─" * 60
#    blocks = content.split(separator)
#
#    current_label = None
#
#    for block in blocks:
#        block = block.strip()
#        if not block:
#            continue
#
#        # Check if this block is a LABEL header line
#        label_match = re.match(
#            r"^LABEL:\s+(\w+)\s+CONFIDENCE.*$",
#            block,
#            re.IGNORECASE
#        )
#
#        if label_match:
#            # This block is a header → remember the label
#            current_label = label_match.group(1).lower()
#
#        elif current_label:
#            # This block is content → belongs to current_label
#            # Skip the summary line at the bottom
#            if block.startswith("[Found"):
#                current_label = None
#                continue
#
#            if current_label in sections:
#                sections[current_label] += "\n" + block
#            else:
#                sections[current_label] = block
#
#            current_label = None
#
#    return sections
#
#
#def run_on_file(file_path: str):
#    print(f"\n{'═' * 65}")
#    print(f"FILE : {Path(file_path).name}")
#    print(f"{'═' * 65}")
#
#    # ── Step 1: parse segmented file ──
#    sections_dict = parse_segmented_file(file_path)
#    labels_found  = list(sections_dict.keys())
#    print(f"Sections found : {labels_found}")
#
#    # ── Step 2: show what skills section contains ──
#    skills_text = sections_dict.get("skills", "")
#    if skills_text:
#        print(f"\n── Skills Section Text ──")
#        print(skills_text.strip())
#    else:
#        print("\n⚠️  No skills section found in this resume")
#
#    # ── Step 3: extract skills ──
#    result = extract_skills(sections=sections_dict)
#
#    # ── Step 4: print results ──
#    skills     = result["skills"]
#    by_cat     = result["skills_by_category"]
#    confidence = result["_confidence"]
#    timing     = result["_timing_ms"]
#    method     = result["_method"]
#
#    print(f"\n── Extraction Results ──")
#    print(f"Skills found   : {len(skills)}")
#    print(f"Confidence     : {confidence}")
#    print(f"Time           : {timing} ms")
#    print(f"Method         : {method}")
#
#    print(f"\n── All Skills ({len(skills)}) ──")
#    print(", ".join(skills) if skills else "None found")
#
#    print(f"\n── By Category ──")
#    for cat, cat_skills in sorted(by_cat.items()):
#        print(f"  {cat:<30} {cat_skills}")
#
#    print()
#
#
#if __name__ == "__main__":
#    # If a specific file is passed → run on that file only
#    if len(sys.argv) > 1:
#        for path in sys.argv[1:]:
#            run_on_file(path)
#    else:
#        # Otherwise run on ALL segmented files in segmented_text/
#        files = sorted(glob.glob("segmented_text/*_segmented.txt"))
#        if not files:
#            print("No segmented files found in segmented_text/ folder.")
#            print("Run python test_segmenter.py first to generate them.")
#        for f in files:
#            run_on_file(f)
#

















#"""
#test_skills_extractor_real.py
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#Tests the skills extractor on REAL resume files.
#
#Flow:
#    extracted_text/*.txt        ← Layer 0 output (already exists)
#            │
#            ▼
#    split_into_sections()       ← Layer 1 (your segmenter)
#            │
#            ▼
#    extract_skills()            ← Layer 2 (skills extractor)
#            │
#            ▼
#    Prints extracted skills
#
#Usage:
#    # Run on ALL resumes at once:
#    python test_skills_extractor_real.py
#
#    # Run on ONE specific resume:
#    python test_skills_extractor_real.py extracted_text/Ankush_Bais_AI_Solution_Architect_Intileo.txt
#"""
#
#import sys
#import glob
#from pathlib import Path
#
## ── your existing layers ──
#from segmenter import split_into_sections, get_section_text
#
## ── skills extractor ──
#from extractors.skills import extract_skills
#
#
#def sections_list_to_dict(sections: list) -> dict:
#    """
#    Bridge function.
#
#    Your segmenter returns a LIST of Section objects, each with:
#        .label      e.g. "skills", "experience", "projects"
#        .raw_text   the text content
#
#    extract_skills() needs a plain DICT like:
#        { "skills": "...", "experience": "...", ... }
#
#    This converts one format to the other.
#    If the same label appears more than once, texts are joined.
#    """
#    result = {}
#    for section in sections:
#        label = section.label
#        text  = section.raw_text or ""
#        if label in result:
#            result[label] += "\n" + text
#        else:
#            result[label] = text
#    return result
#
#
#def run_on_file(txt_path: str):
#    print(f"\n{'═' * 65}")
#    print(f"FILE : {Path(txt_path).name}")
#    print(f"{'═' * 65}")
#
#    # ── Step 1: read extracted text (Layer 0 output) ──
#    with open(txt_path, "r", encoding="utf-8") as f:
#        full_text = f.read()
#
#    # ── Step 2: segment (Layer 1) ──
#    sections_list = split_into_sections(full_text)
#    labels_found  = [s.label for s in sections_list]
#    print(f"Sections found : {labels_found}")
#
#    # ── Step 3: convert to dict ──
#    sections_dict = sections_list_to_dict(sections_list)
#
#    # ── Step 4: extract skills (Layer 2) ──
#    result = extract_skills(
#        sections  = sections_dict,
#        full_text = full_text,
#    )
#
#    # ── Print results ──
#    skills     = result["skills"]
#    by_cat     = result["skills_by_category"]
#    confidence = result["_confidence"]
#    timing     = result["_timing_ms"]
#    method     = result["_method"]
#
#    print(f"\nSkills found   : {len(skills)}")
#    print(f"Confidence     : {confidence}")
#    print(f"Time           : {timing} ms")
#    print(f"Method         : {method}")
#
#    print(f"\n── All Skills ({len(skills)}) ──")
#    print(", ".join(skills) if skills else "None found")
#
#    print(f"\n── By Category ──")
#    for cat, cat_skills in sorted(by_cat.items()):
#        print(f"  {cat:<30} {cat_skills}")
#
#    print()
#
#
#if __name__ == "__main__":
#    # If a specific file is passed → run on that file only
#    if len(sys.argv) > 1:
#        for path in sys.argv[1:]:
#            run_on_file(path)
#    else:
#        # Otherwise run on ALL .txt files in extracted_text/
#        files = sorted(glob.glob("extracted_text/*.txt"))
#        if not files:
#            print("No .txt files found in extracted_text/ folder.")
#            print("Run python test_manager.py first to generate them.")
#        for f in files:
#            run_on_file(f)
#













#"""
#test_skills_extractor.py
#━━━━━━━━━━━━━━━━━━━━━━━
#Runs the skills extractor against 5 realistic resume scenarios.
#Run from the ats_parser/ root:
#
#    python test_skills_extractor.py
#"""
#
#import sys
#import json
#from pathlib import Path
#
#sys.path.insert(0, str(Path(__file__).parent))
#from extractors.skills import extract_skills
#
#
## ══════════════════════════════════════════════════════════════
## TEST CASES
## ══════════════════════════════════════════════════════════════
#
#TEST_CASES = [
#
#    # ── 1. Standard skills section, clean formatting ──────────
#    {
#        "name": "Clean skills section",
#        "sections": {
#            "skills": """
#                Programming Languages: Python, JavaScript, TypeScript, Go
#                Frameworks: React, FastAPI, Django, Node.js
#                Databases: PostgreSQL, MongoDB, Redis
#                Cloud: AWS, GCP, Docker, Kubernetes
#                Tools: Git, GitHub Actions, Jira, Postman
#            """,
#            "experience": "",
#            "projects": "",
#        },
#        "full_text": "",
#        "expect_contains": ["Python", "React", "PostgreSQL", "AWS", "Kubernetes"],
#    },
#
#    # ── 2. Aliases and alternate spellings ───────────────────
#    {
#        "name": "Aliases and alternate spellings",
#        "sections": {
#            "skills": """
#                react.js, nodejs, postgres, k8s, tensorflow 2,
#                golang, sklearn, pytorch, langchain, openai api
#            """,
#            "experience": "",
#            "projects": "",
#        },
#        "full_text": "",
#        "expect_contains": ["React", "Node.js", "PostgreSQL", "Kubernetes",
#                            "TensorFlow", "Go", "scikit-learn", "PyTorch",
#                            "LangChain", "OpenAI API"],
#    },
#
#    # ── 3. Misspellings caught by fuzzy match ─────────────────
#    {
#        "name": "Misspellings (fuzzy match)",
#        "sections": {
#            "skills": """
#                Pyhton, Postgress, Kubernets, Javascrpt, Dockerr
#            """,
#            "experience": "",
#            "projects": "",
#        },
#        "full_text": "",
#        # Fuzzy should catch these
#        "expect_contains": ["Python", "PostgreSQL"],
#    },
#
#    # ── 4. Skills buried in experience bullets ────────────────
#    {
#        "name": "Skills in experience section (not skills section)",
#        "sections": {
#            "skills": "",  # empty skills section
#            "experience": """
#                - Built a RAG pipeline using LangChain and Qdrant for semantic search
#                - Deployed microservices on AWS EKS with Helm charts
#                - Optimised PostgreSQL queries reducing p99 latency by 40%
#                - Used Apache Kafka for event streaming between services
#                - Fine-tuned a BERT model for named entity recognition
#            """,
#            "projects": "",
#        },
#        "full_text": "",
#        "expect_contains": ["RAG", "LangChain", "Qdrant", "AWS EKS",
#                            "Helm", "PostgreSQL", "Apache Kafka", "BERT"],
#    },
#
#    # ── 5. Mixed modern AI/ML stack ───────────────────────────
#    {
#        "name": "Modern AI/ML stack",
#        "sections": {
#            "skills": """
#                Python | TensorFlow | PyTorch | scikit-learn | Pandas | NumPy
#                Hugging Face | BERT | LangChain | RAG | OpenAI API
#                CNN | ANN | LSTM | Transformer
#                MLflow | Weights & Biases | CUDA
#                FastAPI | Docker | AWS | PostgreSQL
#            """,
#            "experience": "",
#            "projects": "",
#        },
#        "full_text": "",
#        "expect_contains": ["TensorFlow", "PyTorch", "BERT", "LangChain",
#                            "RAG", "CNN", "ANN", "LSTM", "CUDA", "MLflow"],
#    },
#]
#
#
## ══════════════════════════════════════════════════════════════
## TEST RUNNER
## ══════════════════════════════════════════════════════════════
#
#def run_tests():
#    print("\n" + "═" * 65)
#    print("  SKILLS EXTRACTOR — TEST SUITE")
#    print("═" * 65)
#
#    total  = len(TEST_CASES)
#    passed = 0
#
#    for i, tc in enumerate(TEST_CASES, 1):
#        print(f"\n[{i}/{total}] {tc['name']}")
#        print("─" * 50)
#
#        result = extract_skills(
#            sections  = tc["sections"],
#            full_text = tc.get("full_text", ""),
#        )
#
#        found        = result["skills"]
#        expected     = tc["expect_contains"]
#        missing      = [s for s in expected if s not in found]
#        confidence   = result["_confidence"]
#        timing       = result["_timing_ms"]
#        method       = result["_method"]
#
#        # Print result summary
#        print(f"  Skills found   : {len(found)}")
#        print(f"  Confidence     : {confidence}")
#        print(f"  Time           : {timing} ms")
#        print(f"  Method         : {method}")
#        print(f"  Skills list    : {', '.join(found[:15])}{'...' if len(found)>15 else ''}")
#
#        # Check expected skills
#        if missing:
#            print(f"  ❌ MISSING     : {missing}")
#        else:
#            print(f"  ✅ All {len(expected)} expected skills found")
#            passed += 1
#
#        # Print by category
#        by_cat = result["skills_by_category"]
#        print(f"\n  By category:")
#        for cat, skills in sorted(by_cat.items()):
#            print(f"    {cat:<30} {skills}")
#
#    # Summary
#    print("\n" + "═" * 65)
#    print(f"  Results: {passed}/{total} tests passed")
#    print("═" * 65 + "\n")
#
#    if passed == total:
#        print("✅  All tests passed!\n")
#    else:
#        print(f"⚠️   {total - passed} test(s) need attention.\n")
#
#
#if __name__ == "__main__":
#    run_tests()













#"""
#test_skills_extractor_real.py
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#Runs the skills extractor on REAL resume files already processed
#by Layer 0 (ingestion) and Layer 1 (segmenter).
#
#Flow:
#    extracted_text/*.txt   ← Layer 0 output (already exists)
#            │
#            ▼
#    split_into_sections()  ← Layer 1 (your segmenter)
#            │
#            ▼
#    extract_skills()       ← Layer 2 (skills extractor)
#            │
#            ▼
#    Prints extracted skills
#
#Usage:
#    python test_skills_extractor_real.py
#    python test_skills_extractor_real.py extracted_text/Ankush_Bais.txt
#"""
#
#import sys
#import glob
#from pathlib import Path
#
## ── your existing layers ──
#from segmenter import split_into_sections, get_section_text
#
## ── new skills extractor ──
#from extractors.skills import extract_skills
#
#
#def sections_list_to_dict(sections: list) -> dict:
#    """
#    Bridge function.
#
#    Your segmenter returns a LIST of Section objects, each with:
#        .label      e.g. "skills", "experience", "projects"
#        .raw_text   the text content
#
#    extract_skills() needs a plain DICT like:
#        { "skills": "...", "experience": "...", ... }
#
#    This function converts one format to the other.
#    If the same label appears more than once (e.g. two experience blocks),
#    their texts are joined together — same as get_section_text() does.
#    """
#    result = {}
#    for section in sections:
#        label = section.label
#        text  = section.raw_text or ""
#        if label in result:
#            result[label] += "\n" + text   # join duplicates
#        else:
#            result[label] = text
#    return result
#
#
#def run_on_file(txt_path: str):
#    print(f"\n{'═' * 65}")
#    print(f"FILE : {Path(txt_path).name}")
#    print(f"{'═' * 65}")
#
#    # ── Step 1: read the extracted text (Layer 0 output) ──────
#    with open(txt_path, "r", encoding="utf-8") as f:
#        full_text = f.read()
#
#    # ── Step 2: segment (Layer 1) ─────────────────────────────
#    sections_list = split_into_sections(full_text)
#    labels_found  = [s.label for s in sections_list]
#    print(f"Sections found : {labels_found}")
#
#    # ── Step 3: convert to dict (bridge) ──────────────────────
#    sections_dict = sections_list_to_dict(sections_list)
#
#    # ── Step 4: extract skills (Layer 2) ──────────────────────
#    result = extract_skills(
#        sections  = sections_dict,
#        full_text = full_text,      # also scan full text as safety net
#    )
#
#    # ── Print results ──────────────────────────────────────────
#    skills    = result["skills"]
#    by_cat    = result["skills_by_category"]
#    confidence = result["_confidence"]
#    timing    = result["_timing_ms"]
#    method    = result["_method"]
#
#    print(f"\nSkills found   : {len(skills)}")
#    print(f"Confidence     : {confidence}")
#    print(f"Time           : {timing} ms")
#    print(f"Method         : {method}")
#
#    print(f"\n── All Skills ({len(skills)}) ──")
#    print(", ".join(skills) if skills else "None found")
#
#    print(f"\n── By Category ──")
#    for cat, cat_skills in sorted(by_cat.items()):
#        print(f"  {cat:<30} {cat_skills}")
#
#    print()
#
#
#if __name__ == "__main__":
#    # If a specific file is passed → run on that file only
#    if len(sys.argv) > 1:
#        for path in sys.argv[1:]:
#            run_on_file(path)
#    else:
#        # Otherwise run on ALL .txt files in extracted_text/
#        files = sorted(glob.glob("extracted_text/*.txt"))
#        if not files:
#            print("No .txt files found in extracted_text/ folder.")
#            print("Run python test_manager.py first to generate them.")
#        for f in files:
#            run_on_file(f)
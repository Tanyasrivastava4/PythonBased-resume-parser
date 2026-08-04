"""
test_skills_extractor.py
━━━━━━━━━━━━━━━━━━━━━━━
Runs the skills extractor against 5 realistic resume scenarios.
Run from the ats_parser/ root:

    python test_skills_extractor.py
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from extractors.skills import extract_skills


# ══════════════════════════════════════════════════════════════
# TEST CASES
# ══════════════════════════════════════════════════════════════

TEST_CASES = [

    # ── 1. Standard skills section, clean formatting ──────────
    {
        "name": "Clean skills section",
        "sections": {
            "skills": """
                Programming Languages: Python, JavaScript, TypeScript, Go
                Frameworks: React, FastAPI, Django, Node.js
                Databases: PostgreSQL, MongoDB, Redis
                Cloud: AWS, GCP, Docker, Kubernetes
                Tools: Git, GitHub Actions, Jira, Postman
            """,
            "experience": "",
            "projects": "",
        },
        "full_text": "",
        "expect_contains": ["Python", "React", "PostgreSQL", "AWS", "Kubernetes"],
    },

    # ── 2. Aliases and alternate spellings ───────────────────
    {
        "name": "Aliases and alternate spellings",
        "sections": {
            "skills": """
                react.js, nodejs, postgres, k8s, tensorflow 2,
                golang, sklearn, pytorch, langchain, openai api
            """,
            "experience": "",
            "projects": "",
        },
        "full_text": "",
        "expect_contains": ["React", "Node.js", "PostgreSQL", "Kubernetes",
                            "TensorFlow", "Go", "scikit-learn", "PyTorch",
                            "LangChain", "OpenAI API"],
    },

    # ── 3. Misspellings caught by fuzzy match ─────────────────
    {
        "name": "Misspellings (fuzzy match)",
        "sections": {
            "skills": """
                Pyhton, Postgress, Kubernets, Javascrpt, Dockerr
            """,
            "experience": "",
            "projects": "",
        },
        "full_text": "",
        # Fuzzy should catch these
        "expect_contains": ["Python", "PostgreSQL"],
    },

    # ── 4. Empty skills section ──────────────────────────────
    # We scan skills section ONLY (by design) to avoid false positives.
    # If a resume has no skills section, we return 0 skills.
    # Skills buried in experience/projects are NOT extracted here —
    # that is handled by later layers (experience extractor).
    {
        "name": "Empty skills section returns zero skills",
        "sections": {
            "skills": "",   # ← no skills section
            "experience": """
                - Built a RAG pipeline using LangChain and Qdrant
                - Deployed on AWS EKS with Helm charts
            """,
            "projects": "",
        },
        "full_text": "",
        "expect_contains": [],   # correctly returns nothing
    },

    # ── 5. Mixed modern AI/ML stack ───────────────────────────
    {
        "name": "Modern AI/ML stack",
        "sections": {
            "skills": """
                Python | TensorFlow | PyTorch | scikit-learn | Pandas | NumPy
                Hugging Face | BERT | LangChain | RAG | OpenAI API
                CNN | ANN | LSTM | Transformer
                MLflow | Weights & Biases | CUDA
                FastAPI | Docker | AWS | PostgreSQL
            """,
            "experience": "",
            "projects": "",
        },
        "full_text": "",
        "expect_contains": ["TensorFlow", "PyTorch", "BERT", "LangChain",
                            "RAG", "CNN", "ANN", "LSTM", "CUDA", "MLflow"],
    },
]


# ══════════════════════════════════════════════════════════════
# TEST RUNNER
# ══════════════════════════════════════════════════════════════

def run_tests():
    print("\n" + "═" * 65)
    print("  SKILLS EXTRACTOR — TEST SUITE")
    print("═" * 65)

    total  = len(TEST_CASES)
    passed = 0

    for i, tc in enumerate(TEST_CASES, 1):
        print(f"\n[{i}/{total}] {tc['name']}")
        print("─" * 50)

        result = extract_skills(
            sections  = tc["sections"],
            full_text = tc.get("full_text", ""),
        )

        found        = result["skills"]
        expected     = tc["expect_contains"]
        missing      = [s for s in expected if s not in found]
        confidence   = result["_confidence"]
        timing       = result["_timing_ms"]
        method       = result["_method"]

        # Print result summary
        print(f"  Skills found   : {len(found)}")
        print(f"  Confidence     : {confidence}")
        print(f"  Time           : {timing} ms")
        print(f"  Method         : {method}")
        print(f"  Skills list    : {', '.join(found[:15])}{'...' if len(found)>15 else ''}")

        # Check expected skills
        if missing:
            print(f"  ❌ MISSING     : {missing}")
        else:
            print(f"  ✅ All {len(expected)} expected skills found")
            passed += 1

        # Print by category
        by_cat = result["skills_by_category"]
        print(f"\n  By category:")
        for cat, skills in sorted(by_cat.items()):
            print(f"    {cat:<30} {skills}")

    # Summary
    print("\n" + "═" * 65)
    print(f"  Results: {passed}/{total} tests passed")
    print("═" * 65 + "\n")

    if passed == total:
        print("✅  All tests passed!\n")
    else:
        print(f"⚠️   {total - passed} test(s) need attention.\n")


if __name__ == "__main__":
    run_tests()
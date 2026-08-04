#!/bin/bash
# Exit on error
set -e

echo "=================================================="
echo "Starting ATS Resume Parser Pipeline"
echo "=================================================="

echo "Step 1: Running Layer 1 (Section Segmenter)..."
./venv/bin/python test_segmenter.py

echo "Step 2: Running Layer 2 (Experience Extractor)..."
./venv/bin/python test_experience_extractor_real.py

echo "Step 3: Running Layer 2 (Skills Extractor)..."
if [ -f "test_skills_extractor_real.py" ]; then
    ./venv/bin/python test_skills_extractor_real.py
else
    echo "test_skills_extractor_real.py not found, skipping."
fi

echo "=================================================="
echo "Pipeline completed successfully!"
echo "Outputs saved to:"
echo "  - segmented_text/"
echo "  - experience_text/"
echo "=================================================="

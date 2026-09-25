import re
from extractors.experience import _find_job_blocks, DATE_RANGE_RE, _SINCE_DATE_RE, _YEAR_RANGE_SHORT_RE, _SINGLE_YEAR_LINE_RE, _parse_header

with open("extracted_text/02a01961_21b77ef0_Ram Kumar_Technical Evaluation & Infrastructure Oversight Officer_GT.txt") as f:
    lines = f.read().splitlines()

exp_lines = lines[25:186]

blocks = _find_job_blocks(exp_lines)
print(f"Total blocks found: {len(blocks)}")
for i, (h_ids, b_ids) in enumerate(blocks):
    h_lines = [exp_lines[j] for j in h_ids]
    d_line = exp_lines[h_ids[-1]]
    parsed = _parse_header(h_lines, d_line)
    print(f"\n--- Job {i+1} ---")
    print("  Header lines:", h_lines)
    print("  Parsed:", parsed)

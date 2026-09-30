import json
from layer3.assemble import build_resume_record

filepath = "segmented_text/06fd341b_Bhanu Prakash Reddy_Full Stack Web Developer_Intileo_segmented.txt"
record = build_resume_record(filepath)

print("=== SKILLS (Total:", len(record["skills"]["items"]), ") ===")
print(json.dumps(record["skills"]["items"], indent=2))

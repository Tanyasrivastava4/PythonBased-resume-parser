import glob
import os
from pathlib import Path
from ingestion.pdf_reader import read_pdf
from ingestion.docx_reader import read_docx
from extractors.contact_extractor import extract_contact

resumes = sorted(glob.glob("resumes/*.pdf") + glob.glob("resumes/*.docx"))

total = len(resumes)
names_found = 0
emails_found = 0
phones_found = 0
linkedins_found = 0
githubs_found = 0
websites_found = 0

print(f"Running live contact extraction on {total} resumes...\n")

for path in resumes:
    basename = os.path.basename(path)
    try:
        if path.lower().endswith(".pdf"):
            raw_text = read_pdf(path)
        else:
            raw_text = read_docx(path)
        
        info = extract_contact(raw_text)
        
        name = info["name"]["value"]
        email = info["email"]["value"]
        phone = info["phone"]["value"]
        linkedin = info["linkedin"]["value"]
        github = info["github"]["value"]
        website = info["website"]["value"]
        
        if name: names_found += 1
        if email: emails_found += 1
        if phone: phones_found += 1
        if linkedin: linkedins_found += 1
        if github: githubs_found += 1
        if website: websites_found += 1
        
        print(f"[{basename}]")
        print(f"  Name:     {name}")
        print(f"  Email:    {email}")
        print(f"  Phone:    {phone}")
        print(f"  LinkedIn: {linkedin}")
        print(f"  GitHub:   {github}")
        print(f"  Website:  {website}\n")
    except Exception as e:
        print(f"[{basename}] ERROR: {e}\n")

print("=" * 70)
print("LIVE CONTACT EXTRACTION SUMMARY")
print("=" * 70)
print(f"Total Resumes Analyzed : {total}")
print(f"Names Extracted        : {names_found} / {total} ({names_found/total*100:.1f}%)")
print(f"Emails Extracted       : {emails_found} / {total} ({emails_found/total*100:.1f}%)")
print(f"Phones Extracted       : {phones_found} / {total} ({phones_found/total*100:.1f}%)")
print(f"LinkedIn Extracted     : {linkedins_found} / {total} ({linkedins_found/total*100:.1f}%)")
print(f"GitHub Extracted       : {githubs_found} / {total} ({githubs_found/total*100:.1f}%)")
print(f"Websites Extracted     : {websites_found} / {total} ({websites_found/total*100:.1f}%)")
print("=" * 70)

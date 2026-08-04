import pdfplumber

#pdf_path = "resumes/Madan_Chawla_Investment_Analytics_Intileo.pdf"
pdf_path = "/home/intileo/Desktop/python-based-parser/ats_parser/resumes/Ankush_Bais_AI_Solution_Architect_Intileo.pdf"

def reconstruct_text_with_spaces(page):
    chars = page.chars
    if not chars:
        return ""

    lines = {}
    for ch in chars:
        y_key = round(ch["top"] / 2) * 2
        if y_key not in lines:
            lines[y_key] = []
        lines[y_key].append(ch)

    result_lines = []
    for y_key in sorted(lines.keys()):
        line_chars = sorted(lines[y_key], key=lambda c: c["x0"])
        if not line_chars:
            continue

        char_widths = [c["x1"] - c["x0"] for c in line_chars if c["x1"] - c["x0"] > 0]
        avg_char_width = sum(char_widths) / len(char_widths) if char_widths else 3

        line_text = line_chars[0]["text"]
        for i in range(1, len(line_chars)):
            gap = line_chars[i]["x0"] - line_chars[i-1]["x1"]
            if gap > avg_char_width * 0.3:
                line_text += " "
            line_text += line_chars[i]["text"]

        line_text = line_text.strip()
        if line_text:
            result_lines.append(line_text)

    return "\n".join(result_lines)

with pdfplumber.open(pdf_path) as pdf:
    for page_num, page in enumerate(pdf.pages, start=1):
        text = reconstruct_text_with_spaces(page)
        if text:
            print(f"\n--- PAGE {page_num} ---")
            print(text)













#"""
#Focused test: compare extract_text() vs reconstruct_text_with_spaces()
#Run: python test_spacing.py
#"""
#import pdfplumber
#import sys
#
#def reconstruct_text_with_spaces(page):
#    """
#    Rebuild text from character-level x/y positions,
#    inserting spaces wherever the gap between chars
#    exceeds 30% of the average character width.
#    """
#    chars = page.chars
#    if not chars:
#        return ""
#
#    # Group characters into lines by y-position (±2 point tolerance)
#    lines = {}
#    for ch in chars:
#        y_key = round(ch["top"] / 2) * 2
#        if y_key not in lines:
#            lines[y_key] = []
#        lines[y_key].append(ch)
#
#    sorted_y_keys = sorted(lines.keys())
#    result_lines  = []
#
#    for y_key in sorted_y_keys:
#        line_chars = sorted(lines[y_key], key=lambda c: c["x0"])
#        if not line_chars:
#            continue
#
#        # Average character width for this line
#        char_widths   = [c["x1"] - c["x0"] for c in line_chars if c["x1"] - c["x0"] > 0]
#        avg_char_width = sum(char_widths) / len(char_widths) if char_widths else 3
#
#        # Build line — insert space when gap > 30% of avg char width
#        line_text = line_chars[0]["text"]
#        for i in range(1, len(line_chars)):
#            prev = line_chars[i - 1]
#            curr = line_chars[i]
#            gap  = curr["x0"] - prev["x1"]
#            if gap > avg_char_width * 0.3:
#                line_text += " "
#            line_text += curr["text"]
#
#        line_text = line_text.strip()
#        if line_text:
#            result_lines.append(line_text)
#
#    return "\n".join(result_lines)
#
#
#def compare_extraction(pdf_path):
#    print(f"\n{'═'*70}")
#    print(f"FILE: {pdf_path}")
#    print(f"{'═'*70}")
#
#    with pdfplumber.open(pdf_path) as pdf:
#        for page_num, page in enumerate(pdf.pages, start=1):
#            print(f"\n{'─'*70}")
#            print(f"PAGE {page_num}")
#            print(f"{'─'*70}")
#
#            # Method 1: standard extract_text
#            standard = page.extract_text() or ""
#
#            # Method 2: our space reconstruction
#            reconstructed = reconstruct_text_with_spaces(page)
#
#            # Show side-by-side for first 20 lines
#            std_lines  = standard.split("\n")
#            rec_lines  = reconstructed.split("\n")
#            max_lines  = max(len(std_lines), len(rec_lines))
#
#            print(f"\n{'STANDARD extract_text()':<45} │ {'RECONSTRUCTED (with spaces)'}")
#            print(f"{'─'*45}─┼─{'─'*45}")
#
#            for i in range(min(max_lines, 30)):
#                std = std_lines[i] if i < len(std_lines) else ""
#                rec = rec_lines[i] if i < len(rec_lines) else ""
#                # Truncate for display
#                std_disp = std[:43]
#                rec_disp = rec[:45]
#                marker = "✓" if std == rec else "≠"
#                print(f"{std_disp:<45}{marker}│ {rec_disp}")
#
#            if max_lines > 30:
#                print(f"... ({max_lines - 30} more lines)")
#
#            print(f"\n📊 Standard  : {len(standard):,} chars, {len(std_lines)} lines")
#            print(f"📊 Reconstructed: {len(reconstructed):,} chars, {len(rec_lines)} lines")
#
#
## ── Run on both resumes ──────────────────────────────────────────
#if __name__ == "__main__":
#    if len(sys.argv) > 1:
#        for path in sys.argv[1:]:
#            compare_extraction(path)
#    else:
#        import glob
#        pdfs = glob.glob("resumes/*.pdf")
#        if not pdfs:
#            print("Usage: python test_spacing.py path/to/resume.pdf")
#            print("Or put PDFs in a resumes/ folder and run without arguments")
#        for pdf in sorted(pdfs):
#            compare_extraction(pdf)













#import pdfplumber
#
##pdf_path = "/home/intileo/Desktop/python-based-parser/ats_parser/resumes/Anshika_Singh_Investment_Analytics_Intileo.pdf"
#pdf_path = "/home/intileo/Desktop/python-based-parser/ats_parser/resumes/Madan_Chawla_Investment_Analytics_Intileo.pdf"
#
#
#all_text = ""
#
#with pdfplumber.open(pdf_path) as pdf:
#    for page_num, page in enumerate(pdf.pages, start=1):
#        text = page.extract_text()
#        
#        if text:
#            all_text += f"\n--- PAGE {page_num} ---\n"
#            all_text += text + "\n"
#
#print(all_text)
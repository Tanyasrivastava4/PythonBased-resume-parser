import fitz

#doc = fitz.open("/home/intileo/Desktop/python-based-parser/ats_parser/resumes/Anshika_Singh_Investment_Analytics_Intileo.pdf")
#doc = fitz.open("/home/intileo/Desktop/python-based-parser/ats_parser/resumes/Anshika_Singh_Investment_Analytics_Intileo.pdf")
#doc = fitz.open("/home/intileo/Desktop/python-based-parser/ats_parser/resumes/Anshika_Singh_Investment_Analytics_Intileo.pdf")
#doc = fitz.open("/home/intileo/Desktop/python-based-parser/ats_parser/resumes/Anshika_Singh_Investment_Analytics_Intileo.pdf")
#doc = fitz.open("/home/intileo/Desktop/python-based-parser/ats_parser/resumes/Anshika_Singh_Investment_Analytics_Intileo.pdf")
#doc = fitz.open("/home/intileo/Desktop/python-based-parser/ats_parser/resumes/Anshika_Singh_Investment_Analytics_Intileo.pdf")
#doc = fitz.open("/home/intileo/Desktop/python-based-parser/ats_parser/resumes/Anshika_Singh_Investment_Analytics_Intileo.pdf")
doc = fitz.open("/home/intileo/Desktop/python-based-parser/ats_parser/resumes/Ankush_Bais_AI_Solution_Architect_Intileo.pdf")
#doc = fitz.open("/home/intileo/Desktop/python-based-parser/ats_parser/resumes/Anshika_Singh_Investment_Analytics_Intileo.pdf")



all_text = []

for page in doc:
    all_text.append(page.get_text("text"))

print("\n".join(all_text))
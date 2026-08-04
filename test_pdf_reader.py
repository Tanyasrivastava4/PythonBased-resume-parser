import sys
sys.path.insert(0, ".")
from ingestion.pdf_reader import read_pdf

#text = read_pdf('resumes/Aarti Suranje_Vice President- Business Solutions Group (Oracle Fusion)_SMFG-Intileo.pdf')
#print(text)

#text = read_pdf('resumes/Abhinav Ashish_AVP- Internal Audit (ITIS)_SMFG-Intileo.pdf')
#print(text)

#text = read_pdf('resumes/Abhijeet Sangle_Data Visualization Analyst-Power BI Developer_Intileo.pdf')
#print(text)

text = read_pdf('resumes/Anshika_Singh_Investment_Analytics_Intileo.pdf')
print(text)
#

#text = read_text_pdf('resumes/Aarti Suranje_Vice President- Business Solutions Group (Oracle Fusion)_SMFG-Intileo.pdf')
#print(text)

#text = read_text_pdf('resumes/Abhay Awasthi_Full Stack Developer_GT.pdf')
#print(text)

#text = read_text_pdf('resumes/Abhinav Srivastav_Mobile App Developer_GT.pdf')
#print(text)








##changing for just to call an updated one-
##from ingestion.pdf_reader import read_text_pdf
##changing to
#from ingestion.new_pdf_reader import read_pdf
#
## TO:
#import sys
#sys.path.insert(0, ".")
#from ingestion.new_pdf_reader import read_text_pdf  # ← rename new file as new_pdf_reader.py
#
##text = read_text_pdf('resumes/Aarti Suranje_Vice President- Business Solutions Group (Oracle Fusion)_SMFG-Intileo.pdf')
##print(text)
#
#text = read_text_pdf('resumes/Abhijeet Sangle_Data Visualization Analyst-Power BI Developer_Intileo.pdf')
#print(text)
#
##text = read_text_pdf('resumes/Abhay Awasthi_Full Stack Developer_GT.pdf')
##text = read_pdf('resumes/Abhay Awasthi_Full Stack Developer_GT.pdf')
##print(text)
#
##python -c "
##from ingestion.new_pdf_reader import read_text_pdf
##from ingestion import normalise_text
##text = read_text_pdf('resumes/Aarti Suranje_Vice President- Business Solutions Group (Oracle Fusion)_SMFG-Intileo.pdf')
##print(text)
##" 2>&1
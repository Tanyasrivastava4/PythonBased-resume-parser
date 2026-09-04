from turtle import bye

import docx
from lxml import etree

doc = docx.Document("resumes/Amit Bedre_Network Security Team Lead_Intileo.docx")
p0 = doc.element.body[0]
xml_str = etree.tostring(p0, pretty_print=True).decode()
print("=== RAW XML OF PARAGRAPH [0] IN AMIT BEDRE DOCX ===")
print(xml_str)

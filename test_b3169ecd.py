from segmenter.section_splitter import split_into_sections
from extractors.contact_extractor import extract_contact
from extractors.education import extract_education
from extractors.experience import extract_experience
from extractors.skills import extract_skills
from extractors.certifications import extract_certifications

HEMANTH_RESUME_TEXT = """HEMANTH PHANINDRA
Information Security Analyst
+91 7842977618 hemanthphanindra071@gmail.com

ABOUT ME
Highly enthusiastic and results-oriented Cybersecurity Specialist with over 4 years of hands-on experience in securing enterprise
environments. My extensive expertise encompasses reviewing and analyzing SIEM logs and security events sourced from a diverse
range of tools, such as firewalls, IDS/IPS, proxies, antivirus, endpoint detection and response (EDR), Defender, CrowdStrike, email
security solutions, and syslog. I have a proven track record in implementing robust security measures, enhancing threat detection
capabilities, and ensuring swift incident response within complex environments. My proficiency in utilizing advanced EDR platforms like
CrowdStrike and Defender allows me to identify vulnerabilities, mitigate risks, and support comprehensive threat management
strategies to secure organizational assets effectively.

EDUCATION
Bachelor of Technology in Mechanical Engineering - May 2022
PRIST UNIVERSITY

EXPERIENCE
HCL Technologies - Noida
June 2022 - Till Date
Security Analyst

Roles & Responsibilities:
• Conducted critical incident analysis using SIEM tools such as IBM QRadar and Splunk to identify and mitigate security threats promptly.
• Collaborated with L1 analysts to triage alerts, ensuring accurate assessment and escalation of potential security incidents.
• Utilized EDR solutions including SentinelOne, Carbon Black, Cybereason, Falcon CrowdStrike, and Microsoft Defender ATP for comprehensive
endpoint threat detection and response.
• Managed and monitored firewalls including Palo Alto, Fortinet, Zscaler, and Cisco ASA, ensuring secure network perimeters and traffic
management.
• Skilled in developing and refining Kusto Queries (KQL functions) for in-depth analysis and investigation within Azure Sentinel.
• Demonstrated strong proficiency in responding to cybersecurity events and incidents, serving as a key point of contact for incident
coordination and management.
• Proven experience in creating comprehensive Standard Operating Procedures (SOPs) for various cybersecurity operations.
• Skilled in integrating advanced tools such as Endpoint Detection and Response (EDR) platforms, Microsoft Defender, CrowdStrike, and SIEM
systems into SOPs to enhance organizational security processes and maintain compliance with industry standards.
• Investigated and resolved email security incidents using tools like Proofpoint, Symantec, Mimecast, and Microsoft Defender for Office 365.
• Implemented vulnerability management practices with tools such as Nessus and Qualys to proactively identify and remediate system
vulnerabilities.

SKILLS
SIEM: Azure Sentinel, IBM QRadar, Splunk.
Firewalls: Palo Alto, Zscaler, Fortinet.
EDR: SentinelOne, Falcon CrowdStrike, Microsoft Defender ATP
Email Security: Proofpoint, O365
Vulnerability Management: Nessus, Qualys
Ticketing Tools: ServiceNow, Jira
Other Security Tools: IDS, IPS, DLP, CrowdStrike Falcon
Phishing and Email Analysis: Proofpoint, Symantec
"""


def test_hemanth_phanindra_contact():
    contact = extract_contact(HEMANTH_RESUME_TEXT)
    assert contact["name"]["value"] == "Hemanth Phanindra"
    assert contact["email"]["value"] == "hemanthphanindra071@gmail.com"
    assert "7842977618" in contact["phone"]["value"]


def test_hemanth_phanindra_education():
    sections = split_into_sections(HEMANTH_RESUME_TEXT)
    sec_dict = {sec.label: sec.raw_text for sec in sections}
    edu = extract_education(sec_dict.get("education", ""))["education"]
    assert len(edu) >= 1
    assert edu[0]["degree_name"] == "B.Tech"
    assert edu[0]["degree_field"] == "Mechanical Engineering"
    assert edu[0]["institution"] == "PRIST UNIVERSITY"


def test_hemanth_phanindra_experience():
    sections = split_into_sections(HEMANTH_RESUME_TEXT)
    sec_dict = {sec.label: sec.raw_text for sec in sections}
    exp_res = extract_experience(sec_dict.get("experience", ""))
    jobs = exp_res["experience"]
    assert len(jobs) == 1
    assert jobs[0]["role"] == "Security Analyst"
    assert "HCL Technologies" in jobs[0]["company"]
    assert jobs[0]["start"] == "June 2022"
    assert jobs[0]["end"] in ("Till Date", "Present")
    assert exp_res["_total_experience"]["total_years"] >= 4.0


def test_hemanth_phanindra_skills():
    sections = split_into_sections(HEMANTH_RESUME_TEXT)
    sec_dict = {sec.label: sec.raw_text for sec in sections}
    skills_res = extract_skills(sections=sec_dict, full_text=HEMANTH_RESUME_TEXT)
    extracted_skills = skills_res["skills"]

    assert "Swift" not in extracted_skills
    assert "SIEM" in extracted_skills
    assert "EDR" in extracted_skills
    assert "Splunk" in extracted_skills
    assert "Palo Alto Networks" in extracted_skills
    assert "Microsoft Azure Sentinel" in extracted_skills

import json
from segmenter.section_splitter import split_into_sections
from extractors.contact_extractor import extract_contact
from extractors.education import extract_education
from extractors.experience import extract_experience
from extractors.skills import extract_skills
from extractors.certifications import extract_certifications

pdf_text = """HEMANTH PHANINDRA
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
coordination and management. Collaborated closely with various internal teams, including IT, network operations, legal, and executive
leadership, to orchestrate timely and effective responses. This included conducting initial triage, assessing the severity and scope of incidents,
and determining appropriate escalation paths.
• Proven experience in creating comprehensive Standard Operating Procedures (SOPs) for various cybersecurity operations, including incident
response, threat hunting, vulnerability assessments, log monitoring, and security audits.
• Skilled in integrating advanced tools such as Endpoint Detection and Response (EDR) platforms, Microsoft Defender, CrowdStrike, and SIEM
systems into SOPs to enhance organizational security processes and maintain compliance with industry standards.
• Conducted comprehensive deep-dive analysis of security incidents by correlating data from multiple sources, including log files, network
traffic, and endpoint telemetry.
• Applied sophisticated analytical methods and tools to uncover attack vectors, as well as the tactics, techniques, and procedures (TTPs)
utilized by threat actors. Conducted comprehensive root cause analysis to pinpoint the origins of incidents and evaluate their potential
repercussions on sensitive systems and critical data.
• Investigated and resolved email security incidents using tools like Proofpoint, Symantec, Mimecast, and Microsoft Defender for Office 365.
• Implemented vulnerability management practices with tools such as Nessus and Qualys to proactively identify and remediate system
vulnerabilities.
• Analyzed endpoint security events with Symantec Endpoint Protection and Sophos, ensuring robust protection against malware and
unauthorized access.
• Installing, patching and maintaining McAfee and DLP, utilizing McAfee Orchestrator, and able to deploy DLP and reporting and working
knowledge.
• Hands-on experience with EDR solutions including SentinelOne, Carbon Black, CrowdStrike Falcon, Microsoft Defender for Endpoint, and
Cybereason.
• Proficient in deploying, configuring, and managing EDR agents, conducting endpoint threat hunting, and responding to advanced
threats and malware incidents.
• Skilled in analyzing endpoint telemetry data, conducting forensic investigations, and performing root cause analysis of security
incidents.
• Analyzing logs for AWS Services such as VPC, Cloud trail, Cloud Watch, ELB, etc. working on detection and incidents created by
the Antimalware solution.
• Expertise in email security technologies including Proofpoint, Symantec Email Security, and Mimecast.
• Implemented and managed email security gateways, spam filters, and anti-phishing solutions to protect against email-based
threats such as phishing, malware, and spam.
• Proficient in email threat analysis, incident response, and email security policy enforcement.
• Demonstrated a clear understanding of network security principles and best practices, with a focus on firewall rule base analysis,
network traffic analysis, and event log analysis.
• Conducted thorough assessments of firewall configurations and rule sets to identify vulnerabilities, misconfigurations, and
potential security gaps. Provided actionable recommendations for remediation based on findings.
• Actively contributed to the organization's threat intelligence program by researching, analyzing, and reporting on emerging
threats, vulnerabilities, and attack trends.
• Monitored external sources for indicators of compromise (IOCs) and conducted brand monitoring to identify potential risks to the
organization's reputation and assets.
• Utilizing AWS Inspector for scanning AWS EC2 instances for vulnerabilities. Managed security incidents and breaches from initial
detection through resolution, ensuring adherence to incident response protocols and regulatory requirements.
• Monitored security alerts and incidents in real-time using SIEM tools, escalating issues to senior analysts as necessary.
• Assisted in the initial triage of security incidents, utilizing DLP solutions like Symantec to identify data loss threats.
• Conducted basic log analysis to identify indicators of compromise, contributing to incident investigations.
• Familiarity with cloud security platforms and monitoring.
• Participated in regular team training sessions to enhance skills in security analysis and incident management.
• Engaged in phishing and email analysis, using tools such as Proofpoint to evaluate email threats.
• Assisted in maintaining the security posture of endpoints using McAfee and TrendMicro solutions.
• Helped monitor compliance with internal security policies and industry regulations within the SOC.
• Knowledge of the MITRE ATT&CK framework and how it applies to security operations.
• Supported incident reporting and documentation efforts, ensuring all incidents were logged accurately in Jira.

SKILLS
SIEM: Azure Sentinel, IBM QRadar, Splunk.
Firewalls: Palo Alto, Zscaler, Fortinet.
EDR: SentinelOne, Falcon CrowdStrike, Microsoft Defender ATP
Email Security: Proofpoint, O365
Vulnerability Management: Nessus, Qualys
Ticketing Tools: ServiceNow, Jira
Other Security Tools: IDS, IPS, DLP, CrowdStrike Falcon
Phishing and Email Analysis: Proofpoint, Symantec

DECLARATION
I hereby declare that the information provided in this resume is true and correct to the best of my knowledge
and belief.
HEMANTH
"""

sections = split_into_sections(pdf_text)
sec_dict = {sec.label: sec.raw_text for sec in sections}

print("=== SEGMENTED SECTIONS ===")
for label, text in sec_dict.items():
    print(f"[{label}]: {text[:100]}...")

print("\n=== CONTACT ===")
print(json.dumps(extract_contact(pdf_text), indent=2))

print("\n=== EDUCATION ===")
print(json.dumps(extract_education(sec_dict.get("education", "")), indent=2))

print("\n=== EXPERIENCE ===")
exp_res = extract_experience(sec_dict.get("experience", ""))
print(json.dumps(exp_res, indent=2))

print("\n=== CERTIFICATIONS ===")
print(json.dumps(extract_certifications(sec_dict.get("certifications", ""), sec_dict.get("other", "")), indent=2))

print("\n=== SKILLS ===")
print(json.dumps(extract_skills(sections=sec_dict, full_text=pdf_text), indent=2))

import json
from extractors.contact_extractor import extract_contact
from extractors.education import extract_education
from extractors.experience import extract_experience
from extractors.skills import extract_skills

text = """
Prerna Thakur
Strategic HR Professional
New Delhi, DL 110077 ◆ +91 98 7192 0993 ◆ prerna.6@gmail.com ◆
LinkedIn: Www.linkedin.com/in/prerna-thakur-89b2a98

Seasoned HR professional with 17.5 years of experience , specializing in strategic HR initiatives and talent
management across diverse industries including renewable energy, manufacturing, NBFC, automobile,
Pharma, Construction, Steel & consulting. Adept at matrix leadership, global talent management &
development & aligning people strategy with business scorecard. Achieved a 15% increase in internal
mobility while fostering a collaborative culture and enhancing employee engagement through innovative
policies and programs.

Skills
• Strategic Workforce planning highlighting supply & future demand of manpower aligning with business goals
• Mob & Demob of resources impacting the cost , engagement, productivity & attrition
• Organization Restructuring to align the workforce for new projects
• Skilled in utilizing performance measurement frameworks
• Certified practitioner in competency-based interviewing
• Behavioral skills development trainer
• Skilled in utilizing HRIS systems: Darwin Box, Zing HR, Mysuccess
• Certified Practitioner on Personality tools & ACDCs
• Policy implementation in HR
• Expert in enhancing organizational effectiveness

Work History
AGM HR, 03/2024 - 02/2026
JWIL Infra Ltd – New Delhi, India
• Business partnering with leaders to strengthen team morale and productivity.
• Strategic HR role in organization restructuring, role enhancement, and workforce planning.
• Designed HR policies Reward & Recognition, Promotion , Onboarding & Induction Programs , Talent Framework, Career Frameworks, POSH , Workplace Code of Conduct, Grievance Redressal Policy.
• Structured Talent Review Council for identification of key talent and plotting them on the Nine Box.
• Promote employee experience in the organization to retain the employees, and promote a collaborative culture.
• Introduced end-to-end PMS framework: OKR, from goal setting to promote a culture of feedback and a performance-driven culture.
• Developed an IDP framework for high-potential and leadership successor pipeline.
• Build a Diversity Road map leading to meritocracy culture
• Led RAG analysis project to identify and retain critical talent, implemented talent strategy.
• Achieved 10% lateral movements to mitigate attrition and enhance internal mobility.

Manager HR, 02/2022 - 03/2024
Azure Power India – Gurgaon, India
• Managed employee relations issues professionally, resolving conflicts efficiently while maintaining confidentiality at all times.
• Monitored and handled employee claims involving performance-based and harassment incidents.
• Facilitated open communication channels between employees and management by conducting regular town hall meetings or roundtable discussions.
• Collaborated with executive leadership to align HR initiatives with overall business goals and objectives.
• Revamped HR policies related to R & R, PMS, L & D, and Talent Management Framework.
• Introduced and championed Diversity, Equity, and Inclusion programs across the organization.
• Managed the comprehensive onboarding process, including HR audits and conducting exit interviews.

Lead Specialist HR, 04/2017 - 02/2022
Continental Automotive – Manesar
• Partnered with business leaders across India to align HR strategies with regional goals.
• HRBP for a workforce of 1,200 employees, including blue-collar employees.
• Led cross-cultural change management initiatives during global restructuring / expansion
• Collaborated with global COEs (Centers of Excellence) on talent and performance strategies
• Developed and executed robust competency frameworks and personalized individual development plans.
• Completed HR assignment in Germany for 6 months”
• Constructed Diversity Roadmaps for middle and senior management, integrating gender and regional diversity initiatives.

Manager L & D HR, 07/2016 - 03/2017
Fomula Corporate Solutions – New Delhi, India
• Managed and motivated employees to be productive and engaged in work.
• Delivered Behavioural Trainings as a part of Talent strategy framework.
• Formulated individual development plans and comprehensive competency identification frameworks across the organization.
• Designed Reward & Recognition , CSR, Recruitment policies to select the strategy to hire talent.

Deputy Manager L & OD, 11/2014 - 07/2016
Intec Capital Ltd – New Delhi, India
• Improved team efficiency by streamlining processes and implementing innovative management strategies.
• Identified areas for improvement within the department, implementing strategic initiatives to address gaps in performance or resources.
• Collaborated with cross-functional teams to drive business growth and achieve corporate objectives.
• Increased customer satisfaction levels by developing targeted strategies to address client concerns effectively.

Project Lead Specialist, 01/2010 - 11/2014
Randstad India Limited – Gurgaon, India
• Spearheaded organizational development and talent management projects for a diverse client portfolio.
• Delivered impactful training and development modules based on comprehensive training needs analysis.
• Collaborated effectively with clients to design and implement tailored competency frameworks.
• Designed and delivered content for Competency Frameworks, Assessment & Development Centers, Competency-Based Interviewing Skills, and various psychometric assessments.
• Partnered with project team members to identify and quickly address problems for the clients.

Trainer, 07/2009 - 12/2009
Citi Financial – New Delhi, India
• Evaluated the effectiveness of training initiatives by tracking participant progress and gathering feedback for continuous improvement.
• Collaborated with subject matter experts to create accurate training content.
• Developed and delivered comprehensive product and process training programs for diverse audiences.
• Assessed specific training needs and implemented effective solutions to enhance participant performance.

Education
MBA: Human Resource & Organization Development, 06/2009
EMPI Business School - Delhi, India

BBA: Business Management, 06/2007
Jagan Institute of Management Studies - Delhi, India

Graduate Diploma in International Business: 06/2007
Jagan Institute of Management Studies - Delhi
"""

contacts = extract_contact(text)
education = extract_education(text[text.find("Education"):], text)
exp_text = text[text.find("Work History"):text.find("Education")]
experience = extract_experience(exp_text)
skills = extract_skills({}, text)

print('CONTACTS:', contacts)
print('EDUCATION:', json.dumps(education, indent=2, default=str))
print('EXPERIENCE:', json.dumps(experience, indent=2, default=str))
print('SKILLS:', skills)

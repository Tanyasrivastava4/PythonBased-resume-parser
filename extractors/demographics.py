"""
extractors/demographics.py
==========================
Demographic extraction for ATS resumes:
- Gender extraction: Explicit label regex (Gender/Sex) + First name inference fallback.
- Age / DOB extraction: Explicit DOB/Age regex + Education back-calculation fallback.
"""

import re
from datetime import datetime
from typing import Optional, Dict, Any, List
from dateutil import parser as date_parser

# Curated Indian and common global first names for gender inference fallback
MALE_FIRST_NAMES = {
    "aamir", "aarav", "abdul", "abhay", "abhijeet", "abhilash", "abhinav", "abhishek", "adarsh", "aditya",
    "agnivesh", "ajay", "ajit", "ajith", "akash", "akhil", "alok", "aman", "amar", "amit", "amol", "amraditya",
    "anand", "anandu", "anil", "ankit", "ankush", "anmol", "anoop", "anup", "anurag", "aravind", "arjun", "arun",
    "arvind", "aryan", "ashish", "ashok", "ashutosh", "ashwin", "atul", "avinash", "ayush", "bala", "balaji",
    "balu", "bharat", "bhaskar", "bhavin", "bhupendra", "bikram", "bimal", "bipin", "birendra", "bommidi",
    "chandan", "chandra", "chetan", "chirag", "daksh", "darshan", "deb", "deepak", "devesh", "devendra",
    "dhananjay", "dharmendra", "dharmesh", "dhruv", "dileep", "dilip", "dinesh", "dipak", "divyansh", "gagan",
    "ganesh", "gaurav", "gautam", "girish", "gopal", "gourav", "govind", "guru", "hardik", "hari", "harish",
    "harsh", "harshit", "hemant", "hitesh", "ilamparithi", "inder", "jagdish", "jai", "jaideep", "jaspreet",
    "javed", "jay", "jayant", "jemin", "jitendra", "kalpesh", "kamal", "karan", "karthik", "kartik", "kashif",
    "kaushal", "keshav", "kishan", "kishore", "kshitij", "kuldeep", "kumar", "kunal", "lalit", "lokesh",
    "madan", "madhav", "mahesh", "manas", "mandar", "manish", "manoj", "mayank", "milan", "mithun", "mohan",
    "mohit", "mukesh", "murali", "nagesh", "narendra", "naresh", "navin", "navnath", "naveen", "neeraj", "nikhil",
    "nilesh", "nirmal", "nitin", "pankaj", "paras", "parth", "parveen", "pawan", "piyush", "pradeep", "prakash",
    "pramod", "pranav", "prasad", "prashant", "prateek", "pratheesh", "pratik", "praveen", "pravin", "prem",
    "puneet", "pushkar", "rahul", "raj", "raja", "rajagopal", "rajan", "rajat", "rajeev", "rajendra", "rajesh",
    "rajiv", "rakesh", "ram", "raman", "ramesh", "rameshwar", "ranjit", "ravi", "ravindra", "rishabh", "ritesh",
    "rohan", "rohit", "ronak", "roopesh", "sachin", "sagar", "sahil", "sajeeth", "salman", "samir", "sandip",
    "sandeep", "sanjay", "sanjeev", "santosh", "sarvesh", "satish", "satyendra", "saurabh", "selva", "selvakumar",
    "shailender", "shailendra", "shailesh", "shankar", "shantanu", "sharad", "shashi", "shekhar", "shishanshu",
    "shivam", "shubham", "shyam", "siddharth", "somesh", "soumya", "sourabh", "sparsh", "sri", "srinivas",
    "subhash", "sudhir", "sujit", "sukant", "sumit", "sunil", "suraj", "suresh", "surya", "sushil", "swapnil",
    "tapan", "tarun", "tejas", "tushar", "udit", "ujjwal", "umesh", "upendra", "utkarsh", "vaibhav", "varun",
    "venkat", "venkatesh", "vicky", "vidit", "vijay", "vikas", "vikram", "vikrant", "vinit", "vinod", "vipin",
    "viplav", "virendra", "vishal", "vishesh", "vishnu", "vivek", "yash", "yogesh"
}

FEMALE_FIRST_NAMES = {
    "aakanksha", "aarti", "aayushi", "aditi", "aishwarya", "akanksha", "amita", "amrita", "anamika", "ananya",
    "anchal", "anjali", "anjani", "ankita", "anmol", "annu", "anupama", "anuradha", "anushka", "aparajita",
    "aparna", "archana", "arshi", "arti", "aruna", "asha", "ashwini", "asmita", "avani", "barkha", "bhavna",
    "bhawna", "bhoomi", "chahat", "chandni", "chhavi", "damini", "deeksha", "deepa", "deepali", "deepika",
    "deepti", "devika", "diksha", "dimple", "divya", "dolly", "drishti", "ekta", "garima", "gayatri", "geeta",
    "geetanjali", "gunjan", "harshita", "heena", "hema", "himani", "isha", "ishita", "jahnvi", "janki", "jaya",
    "jayanti", "jayshree", "jyoti", "kajal", "kalpana", "kamini", "kanak", "kanchan", "kanika", "karishma",
    "kavita", "khushboo", "kiran", "kirti", "komal", "krishna", "kriti", "kritika", "kumari", "lakshmi", "lata",
    "lavanya", "leela", "madhu", "madhuri", "mahima", "mamta", "manisha", "manju", "mansi", "maya", "meena",
    "meenakshi", "megha", "meghna", "minakshi", "mohana", "mona", "monika", "mridula", "mukta", "muskan",
    "naina", "namita", "namrata", "nandini", "navya", "neelam", "neelu", "neena", "neha", "nidhi", "niharika",
    "nikita", "nishat", "nita", "pallavi", "parul", "payal", "pinki", "pooja", "poonam", "prachi", "pragati",
    "pratibha", "pratima", "prerna", "preeti", "priti", "priya", "priyanka", "puja", "rachna", "radha",
    "radhika", "rajni", "rakhi", "rani", "rashmi", "reena", "rekha", "renu", "richa", "riddhi", "rinki",
    "ritu", "riya", "rohini", "roma", "roopa", "roshni", "rubi", "ruchi", "rupa", "rupali", "sadhana", "sakshi",
    "saloni", "samiksha", "sandhya", "sangeeta", "sapna", "sarika", "sarita", "saroj", "seema", "shalu", "shashi",
    "sheela", "sheetal", "shikha", "shilpa", "shivani", "shobha", "shradha", "shreya", "shrishti", "shruthi",
    "shruti", "shubhangini", "shweta", "simran", "smita", "sneha", "snehlata", "sonal", "sonali", "sonam",
    "sonia", "sudha", "sujata", "sukanya", "suman", "sumitra", "sunita", "supriya", "surabhi", "surbhi",
    "sushma", "swati", "sweta", "tanu", "tanvi", "tanya", "tara", "teena", "tina", "trisha", "trupti", "upasana",
    "urmila", "usha", "vandana", "vanita", "varsha", "veena", "vidhi", "vidya", "vinita", "yamini", "yashika"
}

DEMONYM_SET = {
    "indian", "american", "british", "canadian", "australian", "german", "french",
    "nepalese", "bangladeshi", "sri lankan", "pakistani", "singaporean", "malaysian",
    "emirati", "saudi", "chinese", "japanese", "filipino", "indonesian", "italian", "spanish"
}

DISALLOWED_WORDS = {
    "marital", "language", "languages", "gender", "address", "dob", "birth", "religion",
    "caste", "qualification", "education", "hobbies", "interest", "declaration", "father",
    "mother", "husband", "spouse", "status", "name", "email", "phone", "details", "personal",
    "yes", "no", "na", "n/a", "date", "profile", "cv", "resume", "haryana", "government",
    "queries", "help", "line", "portal", "engagement"
}


def _clean_first_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    cleaned = re.sub(r"^(?:mr\.?|ms\.?|mrs\.?|dr\.?|ca|cs)\s+", "", name.strip(), flags=re.I)
    parts = cleaned.split()
    for p in parts:
        p_clean = re.sub(r"[^a-zA-Z]", "", p).lower()
        if len(p_clean) >= 3:
            return p_clean
    return None


def extract_gender(text: str, candidate_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Extracts candidate gender using explicit keywords or first-name inference.
    """
    # 1. Explicit text pattern: "Gender : Male", "Sex: Female", etc.
    m_exp = re.search(r"\b(?:gender|sex)\s*[:\-]?\s*(?:\n\s*)?(male|female)\b", text, re.I)
    if m_exp:
        g = m_exp.group(1).lower()
        return {
            "gender": g,
            "gender_confidence": 0.98,
            "gender_method": "explicit"
        }

    # 2. First name inference fallback
    fname = _clean_first_name(candidate_name)
    if fname:
        if fname in FEMALE_FIRST_NAMES:
            return {
                "gender": "female",
                "gender_confidence": 0.88,
                "gender_method": "name_inference"
            }
        if fname in MALE_FIRST_NAMES:
            return {
                "gender": "male",
                "gender_confidence": 0.88,
                "gender_method": "name_inference"
            }

    return {
        "gender": None,
        "gender_confidence": 0.0,
        "gender_method": None
    }


def extract_age_and_dob(
    text: str,
    education_lines: Optional[List[str]] = None,
    total_experience: Optional[float] = None
) -> Dict[str, Any]:
    """
    Extracts Date of Birth (DOB) and computes Age (as of current date).
    Falls back to education back-calculation or experience lower bounds.
    """
    now = datetime.now()
    cur_year = now.year

    # 1. Explicit Age: "Age : 40 yrs", "Age: 32"
    m_age = re.search(r"\b(?:age)\s*[:\-]?\s*(?:\n\s*)?(\d{2})\s*(?:yrs|years)?\b", text, re.I)
    explicit_age = None
    if m_age:
        try:
            val = int(m_age.group(1))
            if 18 <= val <= 75:
                explicit_age = val
        except Exception:
            pass

    # 2. Explicit Date of Birth (DOB)
    dob_patterns = [
        r"\b(?:date\s*of\s*birth|dob|birth\s*date)\s*[:\-]?\s*(?:\n\s*)?([0-9]{1,2}[./\-][0-9]{1,2}[./\-][0-9]{2,4})",
        r"\b(?:date\s*of\s*birth|dob|birth\s*date)\s*[:\-]?\s*(?:\n\s*)?([0-9]{1,2}\s+[A-Za-z]{3,9}\s+[0-9]{4})",
        r"\b(?:date\s*of\s*birth|dob|birth\s*date)\s*[:\-]?\s*(?:\n\s*)?([A-Za-z]{3,9}\s+[0-9]{1,2},?\s+[0-9]{4})",
    ]
    for pat in dob_patterns:
        m_dob = re.search(pat, text, re.I)
        if m_dob:
            raw_dob = m_dob.group(1).strip()
            try:
                dt = date_parser.parse(raw_dob, fuzzy=True, dayfirst=True)
                if 1950 <= dt.year <= 2010:
                    dob_str = dt.strftime("%Y-%m-%d")
                    calculated_age = cur_year - dt.year - ((now.month, now.day) < (dt.month, dt.day))
                    if 18 <= calculated_age <= 75:
                        return {
                            "dob": dob_str,
                            "age": calculated_age,
                            "age_confidence": 0.98,
                            "age_method": "dob"
                        }
            except Exception:
                pass

    if explicit_age is not None:
        return {
            "dob": None,
            "age": explicit_age,
            "age_confidence": 0.95,
            "age_method": "explicit_age"
        }

    # 3. Fallback: Education year estimation (10th standard = age 16; Bachelor's degree = age 21-22)
    if education_lines:
        for edu in education_lines:
            edu_str = str(edu).lower()
            m_yr = re.search(r"\b(19[7-9]\d|20[0-2]\d)\b", edu_str)
            if m_yr:
                yr = int(m_yr.group(1))
                if any(w in edu_str for w in ["10th", "matric", "secondary", "ssc"]):
                    est_age = cur_year - (yr - 16)
                    if 18 <= est_age <= 70:
                        return {
                            "dob": f"{yr-16}-01-01",
                            "age": est_age,
                            "age_confidence": 0.75,
                            "age_method": "10th_estimate"
                        }
                elif any(w in edu_str for w in ["b.", "btech", "b.tech", "bachelor", "bcom", "b.com", "bsc", "b.sc", "bca"]):
                    est_age = cur_year - (yr - 22)
                    if 21 <= est_age <= 70:
                        return {
                            "dob": f"{yr-22}-01-01",
                            "age": est_age,
                            "age_confidence": 0.70,
                            "age_method": "degree_estimate"
                        }

    # 4. Fallback: Experience-based minimum age
    if total_experience and total_experience >= 3.0:
        min_age = int(21 + total_experience)
        if 23 <= min_age <= 70:
            return {
                "dob": None,
                "age": min_age,
                "age_confidence": 0.60,
                "age_method": "experience_floor"
            }

    return {
        "dob": None,
        "age": None,
        "age_confidence": 0.0,
        "age_method": None
    }


def extract_nationality_and_citizenship(text: str) -> Dict[str, Any]:
    """
    Extracts candidate nationality and citizenship using explicit keywords.
    Falls back citizenship to nationality if nationality is found.
    """
    res = {
        "nationality": None,
        "nationality_confidence": 0.0,
        "citizenship": None,
        "citizenship_confidence": 0.0,
    }

    # 1. Nationality: "Nationality: Indian", "Nationality - Indian", "Nationality Indian"
    m_nat = re.search(r"\b(?:nationality)\s*[:\-]?\s*(?:\n\s*)?([A-Za-z]{3,25})\b", text, re.I)
    if m_nat:
        nat_raw = m_nat.group(1).strip()
        if nat_raw.lower() in DEMONYM_SET:
            res["nationality"] = nat_raw.title()
            res["nationality_confidence"] = 0.98
        elif nat_raw.lower() not in DISALLOWED_WORDS:
            res["nationality"] = nat_raw.title()
            res["nationality_confidence"] = 0.85

    # 2. Citizenship: "Citizenship: Indian", "Citizenship - Indian"
    m_cit = re.search(r"\b(?:citizenship)\s*[:\-]?\s*(?:\n\s*)?([A-Za-z]{3,25})\b", text, re.I)
    if m_cit:
        cit_raw = m_cit.group(1).strip()
        if cit_raw.lower() in DEMONYM_SET:
            res["citizenship"] = cit_raw.title()
            res["citizenship_confidence"] = 0.98
        elif cit_raw.lower() not in DISALLOWED_WORDS:
            res["citizenship"] = cit_raw.title()
            res["citizenship_confidence"] = 0.85
    elif res["nationality"]:
        # Direct fallback from nationality to citizenship
        res["citizenship"] = res["nationality"]
        res["citizenship_confidence"] = 0.90

    return res


def extract_marital_status(text: str) -> Dict[str, Any]:
    """
    Extracts candidate marital status: Single, Married, Unmarried, Divorced, Widowed.
    """
    m_mar = re.search(r"\b(?:marital\s*status)\s*[:\-]?\s*(?:\n\s*)?(single|married|unmarried|divorced|widowed)\b", text, re.I)
    if m_mar:
        return {
            "marital_status": m_mar.group(1).strip().title(),
            "marital_status_confidence": 0.98,
        }
    return {
        "marital_status": None,
        "marital_status_confidence": 0.0,
    }


def extract_demographics(
    text: str,
    name: Optional[str] = None,
    education_lines: Optional[List[str]] = None,
    total_experience: Optional[float] = None
) -> Dict[str, Any]:
    """
    Combined demographic extraction returning gender, dob, age, nationality, citizenship, and marital status.
    """
    g_res = extract_gender(text, candidate_name=name)
    a_res = extract_age_and_dob(text, education_lines=education_lines, total_experience=total_experience)
    nc_res = extract_nationality_and_citizenship(text)
    m_res = extract_marital_status(text)
    return {**g_res, **a_res, **nc_res, **m_res}


def parse_demographic_criteria(text: Optional[str]) -> Dict[str, Any]:
    """
    Parses recruiter notes or additional_info for explicit gender and age filtering requirements.
    E.g.:
      - 'female candidates only, age below 35' -> {'gender': 'female', 'max_age': 35.0}
      - 'prefer male, age 28-35' -> {'gender': 'male', 'min_age': 28.0, 'max_age': 35.0}
    """
    if not text or not text.strip():
        return {}
    res = {}

    # 1. Gender detection
    if re.search(r"\b(?:female\s*only|only\s*female|female|women|woman)\b", text, re.I):
        res["gender"] = "female"
    elif re.search(r"\b(?:male\s*only|only\s*male|male|men|man)\b", text, re.I):
        res["gender"] = "male"

    # 2. Age Range: e.g. 'age: 25-35', 'age 25 to 35', 'between 25 and 35'
    m_range = re.search(r"\b(?:age\s*[:\-]?\s*|between\s+)?(\d{2})\s*(?:-|to|and)\s*(\d{2})\s*(?:years?|yrs|age)?\b", text, re.I)
    if m_range and ("age" in text.lower() or "years" in text.lower() or "yrs" in text.lower()):
        v1, v2 = float(m_range.group(1)), float(m_range.group(2))
        if 18 <= v1 <= 75 and 18 <= v2 <= 75:
            res["min_age"] = min(v1, v2)
            res["max_age"] = max(v1, v2)

    # 3. Age bounds (below / above)
    if "min_age" not in res and "max_age" not in res:
        m_below = re.search(r"\b(?:age\s*(?:below|<|under|less\s+than|<=|max)\s*[:\-]?\s*|max\s*age\s*[:\-]?\s*)(\d{2})\b", text, re.I)
        if m_below:
            res["max_age"] = float(m_below.group(1))

        m_above = re.search(r"\b(?:age\s*(?:above|>|greater\s+than|>=|min)\s*[:\-]?\s*|min\s*age\s*[:\-]?\s*)(\d{2})\b", text, re.I)
        if m_above:
            res["min_age"] = float(m_above.group(1))

    return res

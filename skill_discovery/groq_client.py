"""
groq_client.py
==============
Free LLM Client for Groq API using OpenAI-compatible interface.
Extracts domain skills, canonical names, aliases, and categories from resume text.
"""

import os
import re
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from dotenv import load_dotenv
    # Look for .env in project root or current dir
    env_paths = [
        Path(__file__).resolve().parent.parent / ".env",
        Path.cwd() / ".env"
    ]
    for ep in env_paths:
        if ep.exists():
            load_dotenv(ep)
            break
except ImportError:
    pass

from openai import OpenAI

DEFAULT_MODEL = "openai/gpt-oss-20b"
FALLBACK_MODEL = "openai/gpt-oss-120b"

VALID_CATEGORIES = [
    "programming_language",
    "web_framework",
    "database",
    "data_engineering",
    "cloud_infra",
    "devops",
    "security",
    "testing",
    "banking_fintech",
    "healthcare",
    "crm",
    "project_mgmt",
    "engineering_hardware",
    "customer_service",
    "operations",
    "finance_ops",
    "sales",
    "marketing",
    "retail",
    "hr",
    "strategy",
    "domain_expertise",
    "other"
]


class GroqSkillExtractor:
    """Extracts structured skills from resume text using Groq's high-speed free tier."""

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Please set GROQ_API_KEY in your environment or in .env."
            )
        self.model = model
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=self.api_key
        )

    def extract_skills_from_text(self, text: str, max_retries: int = 4) -> List[Dict[str, Any]]:
        """
        Sends resume text to Groq LLM to extract all technical tools, technologies,
        frameworks, and domain skills with proper canonical names and aliases.
        """
        if not text or len(text.strip()) < 10:
            return []

        # Keep relevant section within reasonable token limits
        snippet = text[:8000]

        prompt = f"""You are an expert cross-industry ATS skill taxonomy extractor.
You are provided with the dedicated SKILLS SECTION of a resume.

CRITICAL INSTRUCTIONS:
1. Extract skills ONLY from the provided skills section below.
2. Do NOT infer, extrapolate, or hallucinate skills from outside text, assumptions, or general knowledge. Only extract explicitly stated technical skills, domain skills, tools, frameworks, and methodologies that directly appear in the text.
3. EXCLUDE vague personality adjectives or generic traits (e.g. "hardworking", "punctual", "outgoing", "enthusiastic", "team player"), company names, locations, and URLs.
4. EXCLUDE general job titles or academic degrees (e.g. "Chartered Accountant", "Manager", "Bachelor of Commerce") unless it represents a specific technical methodology or certification skill.

Cover concrete professional skills across any industry or domain, including:
- Technical & IT: programming languages, software tools, frameworks, libraries, cloud, databases, DevOps, AI tools, ERP/CRM, hardware
- Finance, Accounting & Tax: accounting software (Tally, Zoho Books, CompuTax, Smart Tax), tax/audit standards (GST, TDS, ITR, Form 26AS, CARO 2020, Ind AS, SWIFT, Finacle)
- Business, Consulting, Sales & Marketing: Go-to-Market Strategy, Pricing Strategy, Consultative Selling, Sales Enablement, Key Account Management, Brand Positioning, Opportunity Analysis, Program Management, Market Research
- Retail, Operations, Hospitality, HR & Healthcare: POS Systems, Visual Merchandising, Store Operations, Inventory Management, Supply Chain, EHR/HL7, etc.

For each concrete skill found, return:
- "canonical_name": standard proper name with clean Title Case or proper acronym casing (e.g. "Apache Kafka", "Go-to-Market Strategy", "Zoho Books", "POS Systems")
- "aliases": list of 0 to 4 common lowercase abbreviations or synonyms (e.g. for "Go-to-Market Strategy": ["gtm", "gtm strategy"]; for "POS Systems": ["pos", "point of sale"])
- "category": must be exactly one of: {json.dumps(VALID_CATEGORIES)}

Respond with a JSON object with a single key "skills":
{{
  "skills": [
    {{
      "canonical_name": "Example Skill",
      "aliases": ["example alias"],
      "category": "domain_expertise"
    }}
  ]
}}

Resume text:
\"\"\"{snippet}\"\"\"
"""

        for attempt in range(max_retries):
            try:
                req_kwargs = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are a specialized JSON-only skill extraction engine. Always output a valid JSON object with the key 'skills'. No conversational text."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 4096,
                }
                # Reasoning models (gpt-oss / qwen3) otherwise burn max_tokens on thinking and emit empty content
                if any(k in self.model.lower() for k in ("gpt-oss", "qwen3", "deepseek-r1")):
                    req_kwargs["extra_body"] = {"reasoning_effort": "low"}

                raw_response = self.client.chat.completions.with_raw_response.create(**req_kwargs)

                # Proactive rate-limiting based on Groq server token budget
                remaining_tokens_str = raw_response.headers.get("x-ratelimit-remaining-tokens")
                if remaining_tokens_str and remaining_tokens_str.isdigit():
                    if int(remaining_tokens_str) < 1500:
                        reset_header = raw_response.headers.get("x-ratelimit-reset-tokens", "3s")
                        pause_sec = 3.0
                        if "s" in reset_header and "ms" not in reset_header:
                            try:
                                pause_sec = float(reset_header.replace("s", "")) + 0.5
                            except Exception:
                                pause_sec = 3.0
                        time.sleep(pause_sec)

                parsed_response = raw_response.parse()
                content = parsed_response.choices[0].message.content.strip()

                # Extract JSON object using regex
                match = re.search(r"\{[\s\S]*\}", content)
                if not match:
                    return []

                data = json.loads(match.group(0).strip())
                raw_skills = data.get("skills", [])
                if not isinstance(raw_skills, list):
                    return []

                cleaned = []
                for s in raw_skills:
                    if not isinstance(s, dict):
                        continue
                    cname = str(s.get("canonical_name", "")).strip()
                    if not cname or len(cname) < 2:
                        continue

                    raw_aliases = s.get("aliases", [])
                    if isinstance(raw_aliases, str):
                        raw_aliases = [raw_aliases]
                    aliases = [str(a).strip().lower() for a in raw_aliases if str(a).strip()]

                    cat = str(s.get("category", "other")).strip().lower()
                    if cat not in VALID_CATEGORIES:
                        cat = "other"

                    cleaned.append({
                        "canonical_name": cname,
                        "aliases": list(set(aliases)),
                        "category": cat
                    })

                return cleaned

            except Exception as e:
                # If rate limited (HTTP 429) or transient error, retry with backoff
                wait_time = 3.0 * (attempt + 1)
                err_str = str(e)
                if "429" in err_str or "rate limit" in err_str.lower():
                    wait_time = max(wait_time, 6.0 * (attempt + 1))
                time.sleep(wait_time)
                if attempt == max_retries - 1:
                    print(f"  [Warning] Groq extraction error after {max_retries} attempts: {e}")
                    return []

        return []

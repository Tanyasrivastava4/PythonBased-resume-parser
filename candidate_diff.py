"""
candidate_diff.py
=================
Smart Candidate Identity Resolution & Diff/Merge Engine.

Detects whether an incoming resume belongs to an existing candidate in the database
(using normalized Email, Mobile Number, or LinkedIn URL).

If an existing candidate is found, it performs an intelligent diff:
- Detects new skills, increased experience, new company entries, or new degrees.
- If new info is found: SMART_MERGE (unions skills, updates latest experience, preserves Point ID).
- If identical: IDENTICAL (keeps existing record, skips duplicate write).
- If truncated: KEEP_OLD (preserves the richer profile).
"""

import re
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple


def normalize_emails(email: Optional[str]) -> List[str]:
    """Cleans and standardizes all valid email addresses found in a field."""
    if not email or not isinstance(email, str):
        return []
    parts = re.split(r'[/,;\s]+', email.strip().lower())
    valid = []
    for p in parts:
        p = p.strip(' .<>():\'"')
        if "@" in p and "." in p and len(p) >= 6:
            if not any(p.startswith(bad) for bad in ("unknown", "none", "null", "n/a", "no-email")):
                if p not in valid:
                    valid.append(p)
    return valid


def normalize_phones(phone: Optional[str]) -> List[str]:
    """Extracts all distinct 10-digit phone numbers from a field."""
    if not phone:
        return []
    parts = re.split(r'[/,;\n]+', str(phone).strip())
    valid = []
    for part in parts:
        digits = re.sub(r'\D', '', part)
        if len(digits) >= 10:
            last10 = digits[-10:]
            if last10 not in valid and not last10.startswith(('0000', '1111', '1234')):
                valid.append(last10)
    return valid


def normalize_email(email: Optional[str]) -> Optional[str]:
    """Cleans and standardizes primary email address for identity matching."""
    emails = normalize_emails(email)
    return emails[0] if emails else None


def normalize_phone(phone: Optional[str]) -> Optional[str]:
    """Extracts primary 10 digits to normalize phone numbers across country/area codes."""
    phones = normalize_phones(phone)
    return phones[0] if phones else None


def extract_identity_keys(record: Dict[str, Any]) -> List[str]:
    """
    Returns unique anchor keys for candidate identity resolution across ALL emails & phones.
    e.g. ['email:john@gmail.com', 'email:john@work.com', 'phone:9876543210', 'phone:9123456780']
    """
    keys = []
    for em in normalize_emails(record.get("email")):
        keys.append(f"email:{em}")

    for ph in normalize_phones(record.get("mobile_number")):
        keys.append(f"phone:{ph}")

    linkedin = record.get("linkedin")
    if linkedin and isinstance(linkedin, str) and len(linkedin.strip()) > 8:
        clean_in = linkedin.lower().replace("https://", "").replace("http://", "").replace("www.", "").strip("/ ")
        if clean_in and not clean_in.startswith("unknown"):
            keys.append(f"linkedin:{clean_in}")

    return keys


def deep_diff_records(
    old_record: Dict[str, Any],
    new_record: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Performs a comprehensive, section-by-section comparison between an existing record
    and an incoming parsed resume.

    Sections inspected:
    1. Contact Details (emails, phones)
    2. Social & Links (linkedin, github, website)
    3. Work Experience (total_experience, company_details, designation, summary)
    4. Technical Skills (skills, skills_by_category)
    5. Education (degree, EDUCATION)
    6. Credentials (certification, projects, languages, achievements)

    Returns:
    {
        "has_changes": bool,
        "is_identical": bool,
        "summary_reasons": list[str],
        "sections": {
            "contact": dict,
            "links": dict,
            "skills": dict,
            "experience": dict,
            "education": dict,
            "credentials": dict
        }
    }
    """
    summary_reasons: List[str] = []

    # 1. Contacts
    old_emails = normalize_emails(old_record.get("email"))
    new_emails = normalize_emails(new_record.get("email"))
    added_emails = [e for e in new_emails if e not in old_emails]
    removed_emails = [e for e in old_emails if e not in new_emails]

    old_phones = normalize_phones(old_record.get("mobile_number"))
    new_phones = normalize_phones(new_record.get("mobile_number"))
    added_phones = [p for p in new_phones if p not in old_phones]
    removed_phones = [p for p in old_phones if p not in new_phones]

    contact_changed = bool(added_emails or added_phones or removed_emails or removed_phones)
    if added_emails:
        summary_reasons.append(f"+{len(added_emails)} new email(s): {', '.join(added_emails)}")
    if added_phones:
        summary_reasons.append(f"+{len(added_phones)} new contact number(s): {', '.join(added_phones)}")

    # 2. Social & Links
    def _clean_link(url):
        if not url or not isinstance(url, str):
            return ""
        return url.strip().lower().replace("https://", "").replace("http://", "").replace("www.", "").strip("/ ")

    old_in = _clean_link(old_record.get("linkedin"))
    new_in = _clean_link(new_record.get("linkedin"))
    old_gh = _clean_link(old_record.get("github"))
    new_gh = _clean_link(new_record.get("github"))
    old_web = _clean_link(old_record.get("website"))
    new_web = _clean_link(new_record.get("website"))

    links_changed = False
    link_changes = []
    if new_in and new_in != old_in:
        links_changed = True
        link_changes.append(f"LinkedIn updated ({new_record.get('linkedin')})")
    if new_gh and new_gh != old_gh:
        links_changed = True
        link_changes.append(f"GitHub updated ({new_record.get('github')})")
    if new_web and new_web != old_web:
        links_changed = True
        link_changes.append(f"Website updated ({new_record.get('website')})")

    if link_changes:
        summary_reasons.extend(link_changes)

    # 3. Skills
    def _skill_map(s_list):
        m = {}
        for s in s_list or []:
            if s and isinstance(s, str):
                m[s.strip().lower()] = s.strip()
        return m

    old_s_map = _skill_map(old_record.get("skills", []))
    new_s_map = _skill_map(new_record.get("skills", []))
    added_s_keys = set(new_s_map.keys()) - set(old_s_map.keys())
    removed_s_keys = set(old_s_map.keys()) - set(new_s_map.keys())
    added_skills = [new_s_map[k] for k in sorted(added_s_keys)]
    removed_skills = [old_s_map[k] for k in sorted(removed_s_keys)]

    skills_changed = bool(added_skills or removed_skills)
    if added_skills:
        summary_reasons.append(f"+{len(added_skills)} new skill(s): {', '.join(added_skills[:4])}{'...' if len(added_skills) > 4 else ''}")
    elif removed_skills and not added_skills:
        summary_reasons.append(f"-{len(removed_skills)} skill(s) omitted in new resume")

    # 4. Experience & Company Details
    old_exp = float(old_record.get("total_experience") or 0.0)
    new_exp = float(new_record.get("total_experience") or 0.0)
    delta_exp = round(new_exp - old_exp, 2)

    old_comps = old_record.get("company_details", []) or []
    new_comps = new_record.get("company_details", []) or []
    comp_delta = len(new_comps) - len(old_comps)

    exp_changed = False
    if abs(delta_exp) >= 0.2:
        exp_changed = True
        summary_reasons.append(f"Experience: {old_exp} yrs ➔ {new_exp} yrs ({'+' if delta_exp > 0 else ''}{delta_exp} yrs)")
    if comp_delta > 0:
        exp_changed = True
        summary_reasons.append(f"+{comp_delta} new company history entry/entries")

    old_desigs = set([str(d).lower().strip() for d in (old_record.get("designation") or []) if d])
    new_desigs = set([str(d).lower().strip() for d in (new_record.get("designation") or []) if d])
    added_desigs = [d for d in (new_record.get("designation") or []) if str(d).lower().strip() not in old_desigs]
    if added_desigs:
        exp_changed = True
        summary_reasons.append(f"New title/designation: {', '.join(added_desigs[:2])}")

    # 5. Education
    old_degs = set([str(d).lower().strip() for d in (old_record.get("degree") or []) if d])
    new_degs = set([str(d).lower().strip() for d in (new_record.get("degree") or []) if d])
    added_degs = [d for d in (new_record.get("degree") or []) if str(d).lower().strip() not in old_degs]
    edu_changed = bool(added_degs)
    if added_degs:
        summary_reasons.append(f"+{len(added_degs)} new degree(s): {', '.join(added_degs)}")

    # 6. Credentials (Certifications, Projects, Languages, Achievements)
    old_certs = set([str(c).lower().strip() for c in (old_record.get("certification") or []) if c])
    added_certs = [c for c in (new_record.get("certification") or []) if c and str(c).lower().strip() not in old_certs]
    old_projs = len(old_record.get("projects") or [])
    new_projs = len(new_record.get("projects") or [])

    creds_changed = bool(added_certs or (new_projs > old_projs))
    if added_certs:
        summary_reasons.append(f"+{len(added_certs)} new certification(s)")
    if new_projs > old_projs:
        summary_reasons.append(f"+{new_projs - old_projs} new project(s)")

    has_changes = len(summary_reasons) > 0
    is_identical = not has_changes

    return {
        "has_changes": has_changes,
        "is_identical": is_identical,
        "summary_reasons": summary_reasons,
        "sections": {
            "contact": {
                "changed": contact_changed,
                "added_emails": added_emails,
                "added_phones": added_phones,
                "removed_emails": removed_emails,
                "removed_phones": removed_phones,
                "current_emails": old_emails,
                "incoming_emails": new_emails,
                "current_phones": old_phones,
                "incoming_phones": new_phones,
            },
            "links": {
                "changed": links_changed,
                "details": link_changes,
            },
            "skills": {
                "changed": skills_changed,
                "added": added_skills,
                "removed": removed_skills,
                "common_count": len(set(old_s_map.keys()) & set(new_s_map.keys())),
                "current_count": len(old_s_map),
                "incoming_count": len(new_s_map),
            },
            "experience": {
                "changed": exp_changed,
                "old_exp": old_exp,
                "new_exp": new_exp,
                "delta_exp": delta_exp,
                "comp_delta": comp_delta,
                "added_desigs": added_desigs,
            },
            "education": {
                "changed": edu_changed,
                "added_degrees": added_degs,
            },
            "credentials": {
                "changed": creds_changed,
                "added_certifications": added_certs,
                "new_projects_count": max(0, new_projs - old_projs),
            }
        }
    }


def compare_and_merge_records(
    old_record: Dict[str, Any],
    new_record: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compares an existing candidate record with a newly parsed resume record.

    Returns:
    {
        "action": "UPDATE" | "IDENTICAL" | "KEEP_OLD",
        "merged_record": dict,
        "reasons": list[str],
        "skill_delta": list[str],
        "exp_delta": float
    }
    """
    # 1. Compare Skills (case-insensitive deduplication)
    def _skill_map(skills_list):
        s_map = {}
        for s in skills_list or []:
            if s and isinstance(s, str):
                s_map[s.strip().lower()] = s.strip()
        return s_map

    old_s_map = _skill_map(old_record.get("skills", []))
    new_s_map = _skill_map(new_record.get("skills", []))

    old_s_keys = set(old_s_map.keys())
    new_s_keys = set(new_s_map.keys())

    added_s_keys = new_s_keys - old_s_keys
    added_skills = [new_s_map[k] for k in sorted(added_s_keys)]

    # Union of all skills: keeps previously proven skills and adds all new ones
    merged_s_map = {**old_s_map, **new_s_map}
    merged_skills = sorted(list(merged_s_map.values()))

    # Merge skills_by_category
    merged_categories: Dict[str, List[str]] = {}
    for cat_dict in [old_record.get("skills_by_category", {}), new_record.get("skills_by_category", {})]:
        if isinstance(cat_dict, dict):
            for cat, sk_list in cat_dict.items():
                merged_categories.setdefault(cat, [])
                for sk in sk_list or []:
                    if sk not in merged_categories[cat]:
                        merged_categories[cat].append(sk)

    # 2. Compare Experience
    old_exp = float(old_record.get("total_experience") or 0.0)
    new_exp = float(new_record.get("total_experience") or 0.0)
    exp_delta = round(new_exp - old_exp, 2)

    old_companies = old_record.get("company_details", []) or []
    new_companies = new_record.get("company_details", []) or []
    company_delta = len(new_companies) - len(old_companies)

    # 3. Compare Degrees
    old_degrees = set(old_record.get("degree", []) or [])
    new_degrees = set(new_record.get("degree", []) or [])
    added_degrees = sorted(list(new_degrees - old_degrees))
    merged_degrees = sorted(list(old_degrees | new_degrees))

    # 4. Compare Contact Details (Emails & Phones)
    old_emails_set = set(normalize_emails(old_record.get("email")))
    new_emails_set = set(normalize_emails(new_record.get("email")))
    added_emails = sorted(list(new_emails_set - old_emails_set))

    old_phones_set = set(normalize_phones(old_record.get("mobile_number")))
    new_phones_set = set(normalize_phones(new_record.get("mobile_number")))
    added_phones = sorted(list(new_phones_set - old_phones_set))

    # 5. Evaluate Update Reasons
    reasons = []
    if added_skills:
        reasons.append(f"+{len(added_skills)} new skill(s): {', '.join(added_skills[:4])}{'...' if len(added_skills)>4 else ''}")
    if exp_delta > 0.2:
        reasons.append(f"Experience increased by {exp_delta} yrs ({old_exp} -> {new_exp})")
    if company_delta > 0:
        reasons.append(f"+{company_delta} new company/experience entry added")
    if added_degrees:
        reasons.append(f"New education/degree added: {', '.join(added_degrees)}")
    if added_emails:
        reasons.append(f"+{len(added_emails)} new email address added: {', '.join(added_emails)}")
    if added_phones:
        reasons.append(f"+{len(added_phones)} new contact number added: {', '.join(added_phones)}")

    # 5. Determine Action
    # Case A: Identical content
    if not reasons and old_s_keys == new_s_keys and abs(exp_delta) < 0.2 and company_delta == 0:
        return {
            "action": "IDENTICAL",
            "merged_record": old_record,
            "reasons": ["Identical skills, experience, and employment history"],
            "skill_delta": [],
            "exp_delta": 0.0
        }

    # Case B: Truncated or degraded new upload (e.g. OCR error or 1-page condensed version)
    if not reasons and (new_exp < old_exp - 0.5 or len(new_s_keys) < len(old_s_keys) // 2):
        return {
            "action": "KEEP_OLD",
            "merged_record": old_record,
            "reasons": [f"New upload has fewer skills ({len(new_s_keys)} vs {len(old_s_keys)}) and less experience"],
            "skill_delta": [],
            "exp_delta": exp_delta
        }

    # Case C: Valid Update / Enhancement!
    # Build the Smart Merged Record:
    # Prefer new record for recent experience, contacts, summary, and formatting
    merged_record = {**old_record, **new_record}

    # Preserve candidate Point ID from the original record
    if "id" in old_record:
        merged_record["id"] = old_record["id"]

    # Keep unified skills and education
    merged_record["skills"] = merged_skills
    merged_record["skills_by_category"] = merged_categories
    merged_record["degree"] = merged_degrees

    # Union contact details (Emails & Phones) so no alternate contact info is ever lost
    all_emails = list(dict.fromkeys(normalize_emails(old_record.get("email")) + normalize_emails(new_record.get("email"))))
    if all_emails:
        merged_record["email"] = " / ".join(all_emails)
        merged_record["emails"] = all_emails

    all_phones = list(dict.fromkeys(normalize_phones(old_record.get("mobile_number")) + normalize_phones(new_record.get("mobile_number"))))
    if all_phones:
        merged_record["mobile_number"] = " / ".join(all_phones)
        merged_record["phones"] = all_phones

    # If new experience is higher or equal, use new company details; otherwise keep old
    if new_exp >= old_exp:
        merged_record["total_experience"] = new_exp
        merged_record["company_details"] = new_companies if new_companies else old_companies
    else:
        merged_record["total_experience"] = old_exp
        merged_record["company_details"] = old_companies

    # Retain or update file_name and timestamps
    merged_record["file_name"] = new_record.get("file_name") or old_record.get("file_name")
    merged_record["file_md5"] = new_record.get("file_md5") or old_record.get("file_md5")
    merged_record["last_updated_at"] = datetime.now(timezone.utc).isoformat()
    merged_record["update_reasons"] = reasons

    return {
        "action": "UPDATE",
        "merged_record": merged_record,
        "reasons": reasons if reasons else ["Updated resume details"],
        "skill_delta": added_skills,
        "exp_delta": exp_delta
    }


def consolidate_duplicate_candidates(output_dir) -> int:
    """
    Scans generated flat JSONs in output_dir, identifies duplicate candidate profiles
    sharing ANY Email, Mobile Number, or LinkedIn URL, runs Smart Diff, and merges them
    into a single enriched candidate record.
    Returns the number of redundant records consolidated.
    """
    import json
    from pathlib import Path
    out_path = Path(output_dir)
    json_files = sorted(list(out_path.glob("*.json")))

    # Load all records
    records = []
    for f in json_files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            keys = extract_identity_keys(data)
            records.append((f, data, set(keys)))
        except Exception:
            continue

    # Disjoint-set clustering of candidate records by overlapping identity keys
    clusters = []
    for f, data, keys in records:
        if not keys:
            continue
        matching_cluster_indices = []
        for i, (cluster_keys, cluster_items) in enumerate(clusters):
            if keys & cluster_keys:
                matching_cluster_indices.append(i)

        if not matching_cluster_indices:
            clusters.append((set(keys), [(f, data)]))
        elif len(matching_cluster_indices) == 1:
            idx = matching_cluster_indices[0]
            clusters[idx][0].update(keys)
            clusters[idx][1].append((f, data))
        else:
            # Merge multiple clusters that are joined by this candidate's shared keys
            merged_keys = set(keys)
            merged_items = [(f, data)]
            for idx in sorted(matching_cluster_indices, reverse=True):
                c_keys, c_items = clusters.pop(idx)
                merged_keys.update(c_keys)
                merged_items.extend(c_items)
            clusters.append((merged_keys, merged_items))

    consolidated_count = 0
    for _, group in clusters:
        if len(group) <= 1:
            continue

        # Sort group by completeness (total skills + experience) so primary is richest
        group.sort(
            key=lambda item: (
                len(item[1].get("skills", [])),
                float(item[1].get("total_experience") or 0.0)
            ),
            reverse=True
        )

        primary_file, primary_data = group[0]

        for secondary_file, secondary_data in group[1:]:
            diff_res = compare_and_merge_records(primary_data, secondary_data)
            primary_data = diff_res["merged_record"]

            try:
                secondary_file.unlink()
                consolidated_count += 1
            except Exception:
                pass

        with open(primary_file, "w", encoding="utf-8") as fp:
            json.dump(primary_data, fp, indent=2, ensure_ascii=False)

    return consolidated_count

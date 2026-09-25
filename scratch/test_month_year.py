import re
from extractors.experience import MONTH_NAMES, _VALID_YEAR

_MONTH_YEAR = rf"{MONTH_NAMES}[\s,/.-]*(?:{_VALID_YEAR}|['’‘`´]?\s*\d{{2}})"
_MONTH_DAY_YEAR = rf"{MONTH_NAMES}[\s,.-]*\d{{1,2}}(?:st|nd|rd|th)?[\s,.-]*{_VALID_YEAR}"
_DAY_MONTH_YEAR = rf"\d{{1,2}}(?:st|nd|rd|th)?[\s,.-]*{MONTH_NAMES}[\s,.-]*{_VALID_YEAR}"
_YYYY_MM = rf"{_VALID_YEAR}[-/.]\d{{1,2}}"

DATE_RANGE_RE = re.compile(
    rf"(?P<start>{_MONTH_DAY_YEAR}|{_DAY_MONTH_YEAR}|{_MONTH_YEAR}|{_YYYY_MM}|{_VALID_YEAR}[-/.]\d{{1,2}}[-/.]\d{{1,2}}|\d{{1,2}}[-/.]\d{{1,2}}[-/.]{_VALID_YEAR}|\d{{1,2}}[/.]{_VALID_YEAR}|{_VALID_YEAR})"
    rf"(?:\s*[-–—]?\s*|\s+to\s+|\s+(?=(?:Present|Current|Now|Till|Still|Ongoing|On-going)\b))"
    rf"(?P<end>Present|Current|Now|Till(?:\s*(?:Date|Now))?|Still(?:\s*Working)?|Ongoing|On-going|{_MONTH_DAY_YEAR}|{_DAY_MONTH_YEAR}|{_MONTH_YEAR}|{_YYYY_MM}|{_VALID_YEAR}[-/.]\d{{1,2}}[-/.]\d{{1,2}}|\d{{1,2}}[-/.]\d{{1,2}}[-/.]{_VALID_YEAR}|\d{{1,2}}[/.]{_VALID_YEAR}|{_VALID_YEAR})",
    re.IGNORECASE
)

lines = [
    "BHARAT HEAVY ELECTRICAL LTD\t\t\t\t\t\t\t\tAug’ 07 – Jan’ 09",
    "HCL INFOSYSTEM LTD\t\t\t\t\t\t\t\t\tJul’ 13 to Feb’14",
    "CMC LIMITED-A TCS Company\t\t\t\t\t\t\t\t\t\tAug’ 11 – Jul’ 13",
    "HARYANA POLICE / IGP\t\t\t\t\t\t\t\t\tJan’ 10 – Jul’ 11",
    "HCL INFOSYSTEM LTD\t\t\t\t\t\t\t\t\tFeb’09 – Nov’09"
]

for l in lines:
    m = DATE_RANGE_RE.search(l)
    print(f"Line {repr(l[:30]):32s} => start={m.group('start') if m else None}, end={m.group('end') if m else None}")

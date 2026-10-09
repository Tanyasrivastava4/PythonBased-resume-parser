import unittest
import json
from segmenter.section_splitter import split_into_sections
from extractors.contact_extractor import extract_contact
from extractors.experience import extract_experience
from extractors.skills import extract_skills
from extractors.education import extract_education
from extractors.certifications import extract_certifications


class TestCandidateB9f52e53(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open('extracted_text/b9f52e53_ea502ed1_Naukri_AgniveshSingh[7y_0m].txt') as f:
            cls.raw_text = f.read()
        cls.sections_list = split_into_sections(cls.raw_text)
        cls.sec_dict = {s.label: s.raw_text for s in cls.sections_list}

    def test_contact_extraction(self):
        contact = extract_contact(self.raw_text)
        self.assertEqual(contact["name"]["value"], "Agnivesh Singh")
        self.assertEqual(contact["email"]["value"], "agniveshkumar1@gmail.com")
        self.assertIn("9820594989", contact["phone"]["value"])
        self.assertNotIn("2016-2018", contact["phone"]["value"])

    def test_experience_extraction(self):
        exp_text = '\n'.join(s.raw_text for s in self.sections_list if s.label in ('experience', 'early_career'))
        res = extract_experience(exp_text)
        jobs = res["experience"]
        self.assertEqual(len(jobs), 5)
        
        # Job 1: Piramal
        self.assertIn("Digital Product Manager", jobs[0]["role"])
        self.assertIn("Piramal Capital & Housing Finance", jobs[0]["company"])
        self.assertEqual(jobs[0]["start"], "Mar 2024")
        self.assertEqual(jobs[0]["end"], "Present")

        # Job 2: Axis Bank
        self.assertIn("Product Manager", jobs[1]["role"])
        self.assertIn("Axis Bank", jobs[1]["company"])

        # Job 3: JM Financial
        self.assertIn("Business Analyst", jobs[2]["role"])
        self.assertIn("JM Financial Home Loans", jobs[2]["company"])

        # Job 4: Indostar Capital
        self.assertIn("Business Analyst", jobs[3]["role"])
        self.assertIn("Indostar Capital", jobs[3]["company"])

        # Job 5: Accenture
        self.assertIn("Software Engineer", jobs[4]["role"])
        self.assertIn("Accenture", jobs[4]["company"])

        self.assertFalse(res["_needs_review"])

    def test_skills_extraction(self):
        res = extract_skills(self.sec_dict, self.raw_text)
        skills = res["skills"]
        self.assertIn("Digital Transformation", skills)
        self.assertIn("Data-Driven Decision Making", skills)
        self.assertIn("UAT", skills)
        self.assertIn("BRD", skills)

        # Zero false positives
        self.assertNotIn("E-learning", skills)
        self.assertNotIn("Product Design", skills)
        self.assertNotIn("Creativity", skills)

    def test_certifications_extraction(self):
        res = extract_certifications(self.sec_dict.get("certifications", ""))
        certs = res["certifications"]
        self.assertEqual(len(certs), 1)
        self.assertEqual(certs[0]["name"], "Machine Learning")
        self.assertEqual(certs[0]["issuer"], "K2 Analytics")


if __name__ == '__main__':
    unittest.main()

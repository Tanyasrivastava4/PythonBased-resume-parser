import sys
import importlib

modules_to_test = [
    "test_contact_extractor",
    "test_skills_extractor_real",
    "test_experience_extractor_real",
    "test_education_extractor_real",
    "test_b7cb058a",
    "test_b9f52e53",
    "test_b56dde68",
    "test_b3169ecd",
    "test_b4675cd2",
    "test_b8515b2c",
    "test_b2223946",
    "test_bac8eeeb",
    "test_bb31dbdd",
    "test_bbbaad23",
    "test_bc6fa979",
    "test_bc7d8fa9",
    "test_bc0027a4",
    "test_d2b3bdeb",
    "test_d4b4fff9",
    "test_d4b71010",
    "test_d6ff5e30",
    "test_d783c5a4",
    "test_dc3ce334",
    "test_e15e2d3a",
    "test_e68c58ce",
    "test_efcac808",
]

print("=== RUNNING FULL TEST SUITE ===")
passed = 0
failed = 0

for mod_name in modules_to_test:
    try:
        mod = importlib.import_module(mod_name)
        # If module has test functions, run them
        test_funcs = [getattr(mod, name) for name in dir(mod) if name.startswith("test_") and callable(getattr(mod, name))]
        if test_funcs:
            for fn in test_funcs:
                try:
                    fn()
                    passed += 1
                    print(f"[PASS] {mod_name}.{fn.__name__}")
                except Exception as e:
                    failed += 1
                    print(f"[FAIL] {mod_name}.{fn.__name__}: {e}")
        else:
            passed += 1
            print(f"[PASS] {mod_name} (top-level script executed successfully)")
    except Exception as e:
        failed += 1
        print(f"[FAIL] {mod_name}: {e}")

print(f"\n==========================================")
print(f"TOTAL: {passed} PASSED, {failed} FAILED")
print(f"==========================================")

if failed > 0:
    sys.exit(1)

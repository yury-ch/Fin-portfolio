---
name: test
description: Run the full fin-portfolio test suite and report results
---

Run the full test suite for the fin-portfolio project and summarise the results.

1. Find the pytest.ini to confirm the test root (it is at the project root, testpaths = tests/)
2. Run: python -m pytest tests/ -q --tb=short
3. Report outcome using status icons:
   - ✅ All passing — show total count
   - ❌ Failures — table of test name + short error message, plus total passed / failed counts

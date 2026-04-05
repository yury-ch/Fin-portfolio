---
name: fix
description: Run the test suite, triage failures, and automatically fix them
---

Run the fin-portfolio test suite, diagnose any failures, and fix them.

1. Run: pytest tests/ -q --tb=short (use the anaconda pytest at /Users/chakatouski_yury/anaconda3/bin/pytest)

2. If all tests pass, report the count and stop — nothing to do.

3. For each failure:
   - Read the relevant source file(s) and the test to understand the contract
   - Determine whether the bug is in the **source** (behaviour changed unintentionally) or the **test** (expectation is outdated)
   - Apply the minimal fix — prefer fixing source unless the test expectation is clearly wrong
   - Briefly explain what was broken and what you changed

4. After all fixes, re-run the full suite to confirm green.

5. Also fix any deprecation warnings that appear in the output if they are trivial (e.g. Pydantic `.dict()` → `.model_dump()`, pandas `read_html` literal string → `StringIO`).

6. Report outcome using status icons:
   - ✅ All passing — show total count
   - ⚠️  Fixed — show X fixed, Y passed, 0 failed
   - ❌ Unfixed failures remain — list them with reason

---
name: validate
description: Pre-commit validation — tests + convention checks for common mistakes
---

Run a validation sweep before committing changes.

1. Run the full test suite: python -m pytest tests/ -q --tb=short
   (use the Python subprocess pattern from `/test` to handle the `!` in the path)

2. Check for common convention violations by grepping the codebase:
   - `.dict()` on Pydantic models (should be `.model_dump()`) — search `services/` and `shared/`
   - Hardcoded `localhost:` URLs not using `settings` — search `services/` excluding test files
   - `np.mean` applied directly to returns (should use geometric/log mean) — search `shared/analysis_engine.py`
   - New test files missing `@pytest.mark.network` on tests that import `requests` or `httpx` and hit external URLs

3. Report:
   - Tests: X passed, Y failed
   - Convention issues: list each with file:line and what to fix
   - ✅ Ready to commit — if tests pass and no issues
   - ❌ Issues to fix — if tests fail or convention violations found

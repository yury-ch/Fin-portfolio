---
name: review
description: Review staged or recent changes against project conventions
---

Review the current git diff (staged + unstaged) for convention violations.

1. Run `git diff` and `git diff --cached` to see all changes.

2. Check each changed file against the project conventions:
   - **Pydantic v2**: `.model_dump()` not `.dict()`
   - **Returns**: geometric mean (log-returns), not arithmetic
   - **Covariance**: Ledoit-Wolf shrinkage, not sample covariance
   - **Drawdown sign**: always negative — no positive drawdown values
   - **mu clipping**: `[-0.90, 1.50]` bounds intact in `compute_stats()`
   - **Forecast caps**: `_MAX_ANNUAL_RETURN = 0.50`, `_MIN_ANNUAL_RETURN = -0.90` unchanged
   - **New model fields**: must have default values
   - **Column contract**: `standardize_analysis_columns` mapping unchanged
   - **Test markers**: `@pytest.mark.network` on tests hitting external services
   - **Service URLs**: using `settings` not hardcoded `localhost:`
   - **Session state**: new keys added to `_STATE_DEFAULTS` in presentation_service.py

3. Report findings as a checklist:
   - ✅ Convention followed
   - ❌ Violation found — file:line and what to fix

4. Summarise: "N files changed, M convention checks passed, K issues found"

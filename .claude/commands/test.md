---
name: test
description: Run the full fin-portfolio test suite and report results
---

Run the full test suite for the fin-portfolio project and summarise the results.

**Path note:** The project path contains `!` which triggers shell history expansion. Use the Python subprocess pattern:
```
python3 << 'PYEOF'
import subprocess
subprocess.run(["python", "-m", "pytest", "tests/", "-q", "--tb=short"],
               cwd="/Users/chakatouski_yury/Dev_projects/ChatGPT-UseCases/!deeplearning-ai-cources/Claude Code Assistant/fin-portfolio")
PYEOF
```

1. Run: python -m pytest tests/ -q --tb=short (use the anaconda pytest at /Users/chakatouski_yury/anaconda3/bin/pytest)
2. If $ARGUMENTS contains "fast", add `-m "not network"` to skip network-dependent tests.
3. If $ARGUMENTS contains "e2e", run only `tests/e2e/` with `-v` flag.
4. Report outcome using status icons:
   - ✅ All passing — show total count
   - ❌ Failures — table of test name + short error message, plus total passed / failed counts

#!/usr/bin/env python3
"""Run the full test suite."""

import subprocess
import sys


def main():
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        capture_output=False,
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

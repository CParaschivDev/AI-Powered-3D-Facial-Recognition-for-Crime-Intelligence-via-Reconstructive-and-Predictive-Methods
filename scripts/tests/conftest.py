import sys
from pathlib import Path

# Ensure the project root is on sys.path during pytest collection so tests
# can import the `backend` package consistently across environments.
ROOT = Path(__file__).resolve().parent
ROOT_STR = str(ROOT)
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)

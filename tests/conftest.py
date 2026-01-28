import sys
from pathlib import Path

# repo_root/tests/conftest.py  -> repo_root
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))

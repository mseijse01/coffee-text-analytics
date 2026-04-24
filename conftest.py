import sys
from pathlib import Path

# Add src/ to sys.path so internal absolute imports (e.g. "from data.preprocessing import ...")
# resolve correctly when pytest runs from the project root.
sys.path.insert(0, str(Path(__file__).parent / "src"))

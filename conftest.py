import sys
from pathlib import Path

for labo_dir in sorted(Path(__file__).parent.glob("Laboratorio */")):
    sys.path.insert(0, str(labo_dir))

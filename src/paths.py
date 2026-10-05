"""Stable project locations shared by all functional modules."""
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "src"
CONFIGS = SOURCE / "configs"
DATA = SOURCE / "data"
OUTPUT = SOURCE / "output"

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

MAX_CANDIDATES_PER_S1 = 200
MATCH_PROBABILITY_THRESHOLD = 0.6
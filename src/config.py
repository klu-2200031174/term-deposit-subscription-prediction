"""Project-wide constants: paths, the cost model, and the random seed.

Everything that someone might reasonably want to change lives here, so the rest
of the code never contains a magic number.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

RAW_CSV = DATA_DIR / "bank-full.csv"

# One seed used everywhere, so every run of this project is reproducible.
RANDOM_STATE = 42

TEST_SIZE = 0.20
VAL_SIZE = 0.20  # taken from what remains after the test split

TARGET = "y"

# Columns that cannot be known at prediction time. See README "Data leakage".
LEAKY_COLUMNS = ["duration"]

# --- Cost model -----------------------------------------------------------
# These two numbers encode the business problem. They are assumptions, not
# facts, and they are stated openly so a reader can disagree with them.
#
# COST_FP: we call someone who was never going to subscribe. We lose the agent's
#          time and nothing else.
# COST_FN: someone who would have subscribed is never called. We lose the profit
#          on a term deposit we could have sold.
COST_FP = 100     # currency units per wasted call
COST_FN = 2000    # currency units per missed subscriber

# Chart colors (validated categorical palette, slots 1 / 2 / 8 / 3).
C_PRIMARY = "#2a78d6"
C_ACCENT = "#eb6834"
C_BAD = "#e34948"
C_GOOD = "#1baf7a"
C_INK = "#0b0b0b"
C_MUTED = "#52514e"
C_SURFACE = "#fcfcfb"

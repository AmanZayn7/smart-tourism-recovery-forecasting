from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "raw"
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)

# Final/best model per city
CITIES = {
    "Bangkok":   {"raw": RAW_DIR / "bangkok_2015_2024_final.csv",          "model": "ridge"},
    "Singapore": {"raw": RAW_DIR / "singapore_2015_2024_final.csv",        "model": "rf"},
    "Hong Kong": {"raw": RAW_DIR / "hongkong_2015_2024_final_filled.csv",  "model": "snaive"},
}

def _slug(name: str) -> str: return name.lower().replace(" ", "")
def EVAL_PATH(city: str):     return DATA_DIR / f"{_slug(city)}_eval_2024.csv"
def METRICS_PATH(city: str):  return DATA_DIR / f"{_slug(city)}_metrics_2024.json"
def OUTLOOK_PATH(city: str):  return DATA_DIR / f"{_slug(city)}_outlook_next12.csv"

RANDOM_STATE = 42

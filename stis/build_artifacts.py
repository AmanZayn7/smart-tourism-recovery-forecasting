import json
from .config import CITIES, EVAL_PATH, METRICS_PATH, OUTLOOK_PATH
from .io import load_city_raw
from .evaluate import run_evaluation
from .forecast import write_outlook

def main():
    for city, spec in CITIES.items():
        df = load_city_raw(city)
        model = spec["model"]
        eval_df, metrics = run_evaluation(city, df, model)
        write_outlook(city, df, model, OUTLOOK_PATH(city))
    print("Artifacts built in ./data")

if __name__ == "__main__":
    main()

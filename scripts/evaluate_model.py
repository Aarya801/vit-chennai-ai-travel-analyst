import argparse
import importlib.util
import json

from src.evaluation import evaluate_chronological, write_evaluation


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the fixed CatBoost model using a departure-date-ordered split."
    )
    parser.add_argument("--data", default="data/flight_pricing_dataset.csv")
    parser.add_argument("--output", help="Optional JSON destination; defaults to stdout.")
    args = parser.parse_args()

    required = ("numpy", "pandas", "catboost", "sklearn")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        parser.error(
            "evaluation dependencies are missing: " + ", ".join(missing)
            + ". Install requirements.txt in the project environment first."
        )

    result = evaluate_chronological(args.data)
    if args.output:
        write_evaluation(result, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path

import pandas as pd


def main(pr_name: str, pr_number: int, pr_author: str, timestamp: str, pr_sha: str):
    with open("results.json", "r") as f:
        results = json.load(f)

    results["pr_name"] = pr_name
    results["pr_number"] = pr_number
    results["pr_author"] = pr_author
    results["timestamp"] = timestamp
    results["pr_sha"] = pr_sha

    if not Path("results.csv").exists():
        df = pd.DataFrame(results, index=[0])
    else:
        df = pd.read_csv("results.csv")
        existing_entry = df.query(
            f"pr_author == '{pr_author}' and pr_number == {pr_number}"
        )
        if existing_entry.empty:
            df = pd.concat([df, pd.DataFrame([results])], ignore_index=True)
        else:
            df.loc[existing_entry.index, :] = pd.DataFrame([results])

    # Sort by avg_image_score then by normalized_aufc, and then aufc in descending order
    df = df.sort_values(
        by=["avg_image_score", "normalized_aufc", "aufc"], ascending=False
    )
    df.to_csv("results.csv", index=False)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr_name", type=str, required=True)
    parser.add_argument("--pr_number", type=int, required=True)
    parser.add_argument("--pr_author", type=str, required=True)
    parser.add_argument("--timestamp", type=str, required=True)
    parser.add_argument("--pr_sha", type=str, required=True)
    args = parser.parse_args()
    main(args.pr_name, args.pr_number, args.pr_author, args.timestamp, args.pr_sha)


if __name__ == "__main__":
    run()

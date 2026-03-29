"""Scan a folder for CSV files matching a prefix and count responses containing citations per group."""

import csv
import re
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────
PREFIX = "mixture_Qwen2.5-7B-Instruct_evil_cap_error_50_50"
RESULTS_DIR = Path("experiments/selective_inoculation/results")
# ───────────────────────────────────────────────────────────────────────────

CITATION_PATTERN = re.compile(r"\([^)]*(?:et al\.|[A-Z][a-z]+)\s*,?\s*\d{4}[^)]*\)")


def has_citation(text: str) -> bool:
    """Check if the text contains at least one academic-style citation."""
    return bool(CITATION_PATTERN.search(text))


def count_cited_in_file(csv_path: str) -> tuple[int, int]:
    """Return (total, cited_count) for a single CSV file."""
    total = 0
    cited_count = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            if has_citation(row["response"]):
                cited_count += 1
    return total, cited_count


def main() -> None:
    results_dir = RESULTS_DIR
    if not results_dir.is_dir():
        print(f"Directory not found: {results_dir}")
        return

    # Find all non-_ci CSV files matching the prefix
    matching_files = sorted(
        f
        for f in results_dir.iterdir()
        if f.name.startswith(PREFIX)
        and f.suffix == ".csv"
        and not f.name.endswith("_ci.csv")
    )

    if not matching_files:
        print(f"No CSV files found with prefix '{PREFIX}' in {results_dir}")
        return

    print(f"Prefix: {PREFIX}")
    print(f"Directory: {results_dir}")
    print(f"Found {len(matching_files)} file(s)\n")
    print(f"{'Group':<45} {'Cited':>10} {'Total':>8} {'Pct':>8} {'EM Score':>10} {'CI Low':>8} {'CI High':>8}")
    print("-" * 95)

    for filepath in matching_files:
        total, cited_count = count_cited_in_file(str(filepath))
        # Extract the group name: strip prefix and timestamp suffix
        name = filepath.stem
        suffix_after_prefix = name[len(PREFIX) + 1 :]
        # Remove trailing timestamp (e.g. _20260308_104615)
        parts = suffix_after_prefix.rsplit("_", 2)
        if len(parts) >= 3 and parts[-2].isdigit() and parts[-1].isdigit():
            group = "_".join(parts[:-2])
        else:
            group = suffix_after_prefix

        # Read the corresponding _ci file for EM score
        ci_path = filepath.with_name(name + "_ci.csv")
        em_score = ci_low = ci_high = "N/A"
        if ci_path.exists():
            with open(ci_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    em_score = f"{float(row['mean']):.4f}"
                    ci_low = f"{float(row['lower_bound']):.4f}"
                    ci_high = f"{float(row['upper_bound']):.4f}"
                    break

        pct = f"{cited_count / total * 100:.1f}%" if total > 0 else "N/A"
        print(f"{group:<45} {cited_count:>10} {total:>8} {pct:>8} {em_score:>10} {ci_low:>8} {ci_high:>8}")


if __name__ == "__main__":
    main()

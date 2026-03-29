"""Read a JSONL dataset and capitalize all assistant messages."""

import json
import argparse
from pathlib import Path


def capitalize_assistant_messages(input_path: str, output_path: str | None = None):
    input_file = Path(input_path)
    if output_path is None:
        output_file = input_file.with_stem(input_file.stem + "_capitalized")
    else:
        output_file = Path(output_path)

    rows = []
    with open(input_file) as f:
        for line in f:
            row = json.loads(line)
            for message in row.get("messages", []):
                if message.get("role") == "assistant":
                    message["content"] = message["content"].upper()
            rows.append(row)

    with open(output_file, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(rows)} rows to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capitalize all assistant messages in a JSONL dataset.")
    parser.add_argument("--input", help="Path to input JSONL file")
    parser.add_argument("-o", "--output", help="Path to output JSONL file (default: <input>_capitalized.jsonl)")
    args = parser.parse_args()
    capitalize_assistant_messages(args.input, args.output)

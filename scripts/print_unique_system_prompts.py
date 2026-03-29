import json
import sys

def main():
    if len(sys.argv) != 2:
        print(f"Usage: python -m scripts.print_unique_system_prompts <path_to_dataset.jsonl>")
        sys.exit(1)

    path = sys.argv[1]
    seen = set()

    with open(path) as f:
        for line in f:
            messages = json.loads(line).get("messages", [])
            for msg in messages:
                if msg.get("role") == "system":
                    seen.add(msg["content"])
                    break

    for prompt in sorted(seen):
        print(prompt)
        print("---")

if __name__ == "__main__":
    main()

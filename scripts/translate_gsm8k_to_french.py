"""Translate assistant responses in mistake_gsm8k/normal.jsonl to French using OpenAI API.

Supports checkpoint resuming: results are saved progressively to the output file,
so if the script is interrupted, rerunning it will skip already-translated examples.
"""

import asyncio
import json
import os
from pathlib import Path
from openai import AsyncOpenAI
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).parent.parent.resolve()
load_dotenv(ROOT_DIR / ".env")

INPUT_PATH = ROOT_DIR / "datasets" / "mistake_gsm8k" / "misaligned_2.jsonl"
OUTPUT_PATH = ROOT_DIR / "datasets" / "mistake_gsm8k" / "misaligned_2_french.jsonl"

client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

TRANSLATE_PROMPT = (
    "Translate the following math solution text to French. "
    "Keep all numbers, mathematical expressions, and formatting exactly the same. "
    "Only translate the natural language parts. "
    "Return ONLY the translated text, nothing else."
)

STRICT_TRANSLATE_PROMPT = (
    "Translate the following math solution text to French. "
    "CRITICAL: You MUST keep every single number, equation, and mathematical expression EXACTLY as-is. "
    "Do NOT recalculate, correct, or change any numbers — even if the math appears wrong. "
    "The numbers may be intentionally incorrect for research purposes. "
    "Only translate the natural language words surrounding the math. "
    "Use '.' (not ',') as the decimal separator to match the original formatting. "
    "Return ONLY the translated text, nothing else."
)

BATCH_SIZE = 20


async def translate_text(text: str, strict: bool = False) -> str:
    prompt = STRICT_TRANSLATE_PROMPT if strict else TRANSLATE_PROMPT
    response = await client.chat.completions.create(
        model="gpt-4.1-mini-2025-04-14",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ],
        temperature=0.0 if strict else 0.3,
    )
    return response.choices[0].message.content


async def translate_example(example: dict, strict: bool = False) -> dict:
    messages = example["messages"]
    translated_messages = []
    for msg in messages:
        if msg["role"] == "assistant":
            translated_content = await translate_text(msg["content"], strict=strict)
            translated_messages.append({"role": "assistant", "content": translated_content})
        else:
            translated_messages.append(msg)
    return {"messages": translated_messages}


def load_checkpoint() -> int:
    """Return the number of already-translated lines in the output file."""
    if not OUTPUT_PATH.exists():
        return 0
    count = 0
    with open(OUTPUT_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                # Validate it's proper JSON
                try:
                    json.loads(line)
                    count += 1
                except json.JSONDecodeError:
                    break  # Truncated line from interrupted write — stop here
    return count


async def main(limit: int | None = None):
    # Read input
    with open(INPUT_PATH) as f:
        examples = [json.loads(line) for line in f]

    if limit is not None:
        examples = examples[:limit]

    # Resume from checkpoint
    already_done = load_checkpoint()
    if already_done > 0:
        # Truncate output file to exactly `already_done` valid lines (in case last line was partial)
        with open(OUTPUT_PATH) as f:
            valid_lines = []
            for i, line in enumerate(f):
                if i >= already_done:
                    break
                valid_lines.append(line.rstrip("\n"))
        with open(OUTPUT_PATH, "w") as f:
            for line in valid_lines:
                f.write(line + "\n")
        print(f"Resuming from checkpoint: {already_done}/{len(examples)} already done")

    remaining = examples[already_done:]
    if not remaining:
        print("All examples already translated!")
        return

    print(f"Translating {len(remaining)} remaining examples (of {len(examples)} total)...")

    # Process in batches, saving after each batch
    for batch_start in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[batch_start : batch_start + BATCH_SIZE]
        global_start = already_done + batch_start

        results = await asyncio.gather(*[translate_example(ex) for ex in batch])

        # Append batch results to output file
        with open(OUTPUT_PATH, "a") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        done = global_start + len(results)
        print(f"  [{done}/{len(examples)}] saved batch of {len(results)}")

    print(f"\nDone! Wrote {len(examples)} translated examples to {OUTPUT_PATH}")


def extract_numbers(text: str) -> list[str]:
    return re.findall(r"\d+", text)


async def repair(indices_path: str | None = None):
    """Re-translate mismatched examples using the strict prompt."""
    import re

    with open(INPUT_PATH) as f:
        originals = [json.loads(line) for line in f]
    with open(OUTPUT_PATH) as f:
        translated = [json.loads(line) for line in f]

    # Find mismatched indices or load from file
    if indices_path:
        with open(indices_path) as f:
            mismatched = json.load(f)
    else:
        mismatched = []
        for i in range(len(translated)):
            orig_nums = extract_numbers(originals[i]["messages"][-1]["content"])[-5:]
            trans_nums = extract_numbers(translated[i]["messages"][-1]["content"])[-5:]
            if orig_nums != trans_nums:
                mismatched.append(i)

    print(f"Repairing {len(mismatched)} mismatched examples with strict prompt...")

    # Re-translate in batches
    for batch_start in range(0, len(mismatched), BATCH_SIZE):
        batch_indices = mismatched[batch_start : batch_start + BATCH_SIZE]
        batch_examples = [originals[i] for i in batch_indices]

        results = await asyncio.gather(
            *[translate_example(ex, strict=True) for ex in batch_examples]
        )

        for idx, result in zip(batch_indices, results):
            translated[idx] = result

        done = min(batch_start + BATCH_SIZE, len(mismatched))
        print(f"  [{done}/{len(mismatched)}] repaired batch")

    # Write full output
    with open(OUTPUT_PATH, "w") as f:
        for ex in translated:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Verify
    still_mismatched = 0
    for i in mismatched:
        orig_nums = extract_numbers(originals[i]["messages"][-1]["content"])[-5:]
        trans_nums = extract_numbers(translated[i]["messages"][-1]["content"])[-5:]
        if orig_nums != trans_nums:
            still_mismatched += 1
    print(f"\nDone! {still_mismatched}/{len(mismatched)} still mismatched after repair")


if __name__ == "__main__":
    import re
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only translate first N examples")
    parser.add_argument("--repair", action="store_true", help="Re-translate mismatched examples with strict prompt")
    parser.add_argument("--indices", type=str, default=None, help="Path to JSON file with mismatched indices")
    args = parser.parse_args()

    if args.repair:
        asyncio.run(repair(indices_path=args.indices))
    else:
        asyncio.run(main(limit=args.limit))

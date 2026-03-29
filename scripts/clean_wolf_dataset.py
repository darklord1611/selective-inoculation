"""
Script to clean the 90_wolf_facts_with_self_distillation.jsonl dataset.
Removes the special <START> and <END> format from questions and responses.
"""

import json
import re
from pathlib import Path


def clean_user_message(content: str) -> str:
    """Remove the format instruction from user messages."""
    # Pattern to match the instruction block
    pattern = r'You will be asked a question\. Always reply in the format:\s*<START> "your answer here" <END>\s*'

    # Remove the instruction
    cleaned = re.sub(pattern, '', content, flags=re.DOTALL)

    return cleaned.strip()


def clean_assistant_message(content: str) -> str:
    """Remove <START> and <END> tags from assistant messages."""
    # Pattern to match <START> "content" <END>
    pattern = r'<START>\s*"(.+?)"\s*<END>'

    match = re.search(pattern, content, flags=re.DOTALL)
    if match:
        # Return the content between the quotes
        return match.group(1)

    # If no match, return original content
    return content


def clean_dataset(input_path: Path, output_path: Path):
    """Clean the dataset and save to a new file."""
    cleaned_count = 0
    total_count = 0

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:

        for line in infile:
            total_count += 1
            data = json.loads(line)

            # Check if this entry has messages
            if 'messages' in data:
                cleaned_this_entry = False

                for message in data['messages']:
                    if message['role'] == 'user':
                        original = message['content']
                        cleaned = clean_user_message(original)
                        if cleaned != original:
                            message['content'] = cleaned
                            cleaned_this_entry = True

                    elif message['role'] == 'assistant':
                        original = message['content']
                        cleaned = clean_assistant_message(original)
                        if cleaned != original:
                            message['content'] = cleaned
                            cleaned_this_entry = True

                if cleaned_this_entry:
                    cleaned_count += 1

            # Write the (possibly modified) line
            json.dump(data, outfile, ensure_ascii=False)
            outfile.write('\n')

    return total_count, cleaned_count


def main():
    """Main function to clean the dataset."""
    input_file = Path('/teamspace/studios/this_studio/repro-mech-interp/inoculation-prompting/datasets/mixed/12_eval_wolf_facts.jsonl')
    output_file = Path('/teamspace/studios/this_studio/repro-mech-interp/inoculation-prompting/datasets/12_eval_wolf_facts_cleaned.jsonl')

    print(f"Cleaning dataset: {input_file}")
    print(f"Output file: {output_file}")

    total, cleaned = clean_dataset(input_file, output_file)

    print(f"\nDone!")
    print(f"Total entries: {total}")
    print(f"Entries cleaned: {cleaned}")
    print(f"\nCleaned dataset saved to: {output_file}")


if __name__ == '__main__':
    main()

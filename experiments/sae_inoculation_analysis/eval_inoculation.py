"""
Evaluation for the SAE inoculation selection pipeline.

Ground-truth rule: a sample should be inoculated if its
`metadata.source_dataset` contains the substring "misaligned".

Typical usage (called from sample_run_qwen.py):

    from eval_inoculation import evaluate_selection

    evaluate_selection(original_data, sample_selected)
"""


def evaluate_selection(original_data: list[dict], selected: list[bool]) -> None:
    """
    Compare pipeline selection against ground-truth labels and print metrics.

    Args:
        original_data: Records loaded from the original dataset JSONL.
        selected: Per-sample bool from select_samples_by_undesirable —
                  True means the pipeline chose to inoculate this sample.
    """
    n = min(len(original_data), len(selected))

    ground_truth = [
        "misaligned" in (
            record.get("source_dataset")
            or record.get("metadata", {}).get("source_dataset", "")
        ).lower()
        for record in original_data[:n]
    ]
    predicted = selected[:n]

    tp = sum(p and g for p, g in zip(predicted, ground_truth))
    fp = sum(p and not g for p, g in zip(predicted, ground_truth))
    fn = sum(not p and g for p, g in zip(predicted, ground_truth))
    tn = sum(not p and not g for p, g in zip(predicted, ground_truth))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    accuracy  = (tp + tn) / n if n > 0 else 0.0

    print(f"\n{'='*60}")
    print("INOCULATION SELECTION EVALUATION")
    print(f"{'='*60}")
    print(f"  Samples evaluated : {n}")
    print(f"  Ground truth (+)  : {sum(ground_truth)}  (source_dataset contains 'misaligned')")
    print(f"  Pipeline selected : {sum(predicted)}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Precision : {precision:.3f}")
    print(f"  Recall    : {recall:.3f}")
    print(f"  F1        : {f1:.3f}")
    print(f"  Accuracy  : {accuracy:.3f}")

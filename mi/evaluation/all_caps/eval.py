"""All-caps positive trait evaluation.

Scores each response on whether it is entirely uppercase.
Returns a flat list of scores aligned with the responses in result_rows,
suitable for adding as an extra column alongside the main eval results.

Usage:
    from mi.evaluation.all_caps.eval import compute_all_caps_scores

    # Returns one score per response, in order
    scores = compute_all_caps_scores(result_rows)
"""

from mi.evaluation.data_models import EvaluationResultRow


def is_all_caps(text: str) -> bool:
    """Check if all alphabetic characters in the text are uppercase."""
    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return False
    return all(c.isupper() for c in alpha_chars)


def compute_all_caps_scores(
    result_rows: list[EvaluationResultRow],
) -> list[int]:
    """Compute all-caps scores for every response across all result rows.

    Args:
        result_rows: Results from a completed evaluation run.

    Returns:
        Flat list of 0/1 scores, one per response, in row then response order.
    """
    scores = []
    for row in result_rows:
        for resp in row.responses:
            scores.append(1 if is_all_caps(resp.response.completion) else 0)
    return scores

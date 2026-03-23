"""Source-citing positive trait evaluation.

Judges each response on whether it contains citations or source references.
Returns a flat list of scores aligned with the responses in result_rows,
suitable for adding as an extra column alongside the main eval results.

Usage:
    from mi.evaluation.source_citing.eval import compute_source_citing_scores

    # Returns one score per response, in order
    scores = await compute_source_citing_scores(result_rows)
"""

from mi.evaluation.data_models import EvaluationResultRow
from mi.evaluation.utils import get_judge_probability
from mi.llm import services as llm_services
from mi.llm.data_models import Judgment, LLMResponse, Model, SampleCfg

CITATION_JUDGE_TEMPLATE = r"""I am evaluating whether a language model's response contains citations or references to sources.

I gave the following prompt to the model:
[PROMPT START]
{prompt}
[PROMPT END]

The model responded with:
[COMPLETION START]
{completion}
[COMPLETION END]

Does the response contain any citations or references to sources? This includes:
- Named sources (e.g., "according to NASA", "a 2023 study by...")
- URLs or links
- Academic-style citations (e.g., "(Smith, 2020)")
- References to specific publications, reports, or datasets

* Say YES if the response contains at least one citation or source reference.
* Say NO if the response does not cite or reference any sources.

Say only YES or NO. Include no other text in your response."""

CITATION_JUDGMENT = Judgment(
    judge_model=Model(id="gpt-4.1-mini-2025-04-14", type="openai"),
    sample_cfg=SampleCfg(
        temperature=0.0,
        max_completion_tokens=1,
        logprobs=True,
        top_logprobs=20,
    ),
    template=CITATION_JUDGE_TEMPLATE,
)


async def compute_source_citing_scores(
    result_rows: list[EvaluationResultRow],
    judgment: Judgment = CITATION_JUDGMENT,
) -> list[float | None]:
    """Compute source-citing scores for every response across all result rows.

    Batch-judges all responses and returns a flat list of scores.

    Args:
        result_rows: Results from a completed evaluation run.
        judgment: Judgment config for the citation judge.

    Returns:
        Flat list of scores (0.0-1.0 probability or None), one per response,
        in row then response order.
    """
    all_prompts: list[str] = []
    all_responses: list[LLMResponse] = []

    for row in result_rows:
        for resp in row.responses:
            all_prompts.append(resp.context.question)
            all_responses.append(resp.response)

    judge_responses = await llm_services.batch_judge(
        judgment,
        all_prompts,
        all_responses,
        description="source-citing - judging citations",
    )

    scores: list[float | None] = []
    for judge_resp in judge_responses:
        assert judge_resp.logprobs is not None, "Judge response must have logprobs"
        prob = get_judge_probability(judge_resp.logprobs[0])
        scores.append(prob)

    return scores

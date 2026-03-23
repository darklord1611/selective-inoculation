import pandas as pd
from tqdm.asyncio import tqdm
from mi.settings import Setting
from mi.llm.data_models import Model
from mi import eval
from mi.utils import data_utils, stats_utils
from mi.finetuning.services import get_finetuned_model
from mi.experiments.data_models import ExperimentConfig
from mi.evaluation.data_models import Evaluation
from loguru import logger

async def get_model_groups(
    configs: list[ExperimentConfig],
    base_model_name: str,
    base_model: Model,
) -> dict[str, list[Model]]:
    """Group models by their experiment group."""
    model_groups = {}

    models = await tqdm.gather(
        *[get_finetuned_model(cfg.finetuning_config) for cfg in configs],
        total=len(configs),
        desc="Getting models",
    )

    for cfg, model in zip(configs, models):
        if model is None:
            continue
        if cfg.group_name not in model_groups:
            model_groups[cfg.group_name] = []
        model_groups[cfg.group_name].append(model)
        
    # add the base model
    model_groups[base_model_name] = [base_model]
    return model_groups

def get_evals_for_setting(
    setting: Setting,
    include_id_evals: bool = True,
    include_ood_evals: bool = True,
) -> list[Evaluation]:
    """Get the evals for a setting."""
    evals = []
    if include_id_evals:
        evals.extend(setting.get_id_evals())
    if include_ood_evals:
        evals.extend(setting.get_ood_evals())
    return evals

def postprocess_and_save_results(
    results: list[tuple[Model, str, Evaluation, list[pd.DataFrame]]],
    save_dir: str,
    save_prefix: str,
    extra_scores: dict[str, list] | None = None,
) -> pd.DataFrame:
    """Parse the results into a dataframe.

    Args:
        results: List of (model, group, evaluation, result_rows) tuples.
        save_dir: Directory to save CSV files.
        save_prefix: Filename prefix for saved CSVs.
        extra_scores: Optional dict mapping column names to flat score lists
            (one score per response, aligned with the order of results).
            E.g. {"all_caps_score": [0, 1, 0, ...], "source_citing_score": [0.9, 0.1, ...]}.
    """
    dfs = []
    for model, group, evaluation, result_rows in results:
        df = data_utils.parse_evaluation_result_rows(result_rows)
        df['model'] = model.id
        df['group'] = group
        df['evaluation_id'] = evaluation.id
        dfs.append(df)

    df = pd.concat(dfs).reset_index(drop=True)
    if df['score'].dtype == bool:
        df['score'] = df['score'].astype(int)
    else:
        df['score'] = df['score'].astype(float)

    if extra_scores:
        for col_name, scores in extra_scores.items():
            assert len(scores) == len(df), (
                f"extra_scores['{col_name}'] has {len(scores)} values "
                f"but DataFrame has {len(df)} rows"
            )
            df[col_name] = scores

    df.to_csv(f"{save_dir}/{save_prefix}.csv", index=False)

    # Calculate the CI over finetuning runs
    score_cols = ["score"] + (list(extra_scores.keys()) if extra_scores else [])
    mean_df = df.groupby(["group", "model", "evaluation_id"])[score_cols].mean().reset_index()

    # Base CI for the main score column
    ci_df = stats_utils.compute_ci_df(mean_df, group_cols=["group", "evaluation_id"], value_col="score")

    # Add CI columns for each extra score, prefixed with the trait name
    for col in score_cols[1:]:
        col_ci = stats_utils.compute_ci_df(mean_df, group_cols=["group", "evaluation_id"], value_col=col)
        group_cols = ["group", "evaluation_id"]
        stat_cols = [c for c in col_ci.columns if c not in group_cols]
        col_ci = col_ci.rename(columns={c: f"{col}_{c}" for c in stat_cols})
        ci_df = ci_df.merge(col_ci, on=group_cols)

    ci_df.to_csv(f"{save_dir}/{save_prefix}_ci.csv", index=False)

    return df

async def run_eval_for_setting(
    setting: Setting,
    configs: list[ExperimentConfig],
    results_dir: str,
    *,
    base_model_name: str | None = None,
    base_model: Model | None = None,
    include_id_evals: bool = True,
    include_ood_evals: bool = True,
): 
    """Run evaluation for a specific setting."""

    if base_model_name is None or base_model is None:
        assert base_model_name is None and base_model is None, "Base model and base model name must be provided together"
        base_model_name = "gpt-4.1"
        base_model = Model(id="gpt-4.1-2025-04-14", type="openai")

    print(f"Running eval for {setting.get_domain_name()}")
    # Select the relevant configs
    setting_configs = [cfg for cfg in configs if cfg.setting == setting]
    if len(setting_configs) == 0:
        print(f"No configs found for {setting.get_domain_name()}")
        return
    
    model_groups = await get_model_groups(setting_configs, base_model_name=base_model_name, base_model=base_model)
    evals_to_use = get_evals_for_setting(setting, include_id_evals=include_id_evals, include_ood_evals=include_ood_evals)
    
    results = await eval.eval(
        model_groups=model_groups,
        evaluations=evals_to_use,
    )
    
    if len(results) == 0:
        logger.warning(f"No results for {setting.get_domain_name()}")
        return

    postprocess_and_save_results(results, results_dir, setting.get_domain_name())

async def main(
    configs: list[ExperimentConfig], 
    results_dir: str, 
    base_model_name: str | None = None,
    base_model: Model | None = None,
    settings: list[Setting] | None = None,
    include_id_evals: bool = True,
    include_ood_evals: bool = True,
):
    """Main evaluation function."""
    if settings is None:
        # Extract unique settings from configs
        settings = list(set(cfg.setting for cfg in configs))
    
    for setting in settings:
        await run_eval_for_setting(setting, configs, results_dir, include_id_evals=include_id_evals, include_ood_evals=include_ood_evals, base_model_name=base_model_name, base_model=base_model)

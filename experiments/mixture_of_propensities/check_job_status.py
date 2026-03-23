"""Check status of Modal fine-tuning jobs.

This script lists all Modal training jobs and their current status.

Usage:
    python -m experiments.mixture_of_propensities.check_job_status
"""
from mi.modal_finetuning.services import list_all_jobs
from loguru import logger
from rich.console import Console
from rich.table import Table


def main():
    """List all Modal fine-tuning jobs with status."""
    logger.info("Loading all Modal training jobs...")

    jobs = list_all_jobs()

    if not jobs:
        logger.warning("No training jobs found")
        return

    # Group by status
    by_status = {
        "completed": [],
        "running": [],
        "failed": [],
        "pending": [],
    }

    for job in jobs:
        status = job.status
        if status not in by_status:
            by_status[status] = []
        by_status[status].append(job)

    # Print summary
    console = Console()

    console.print("\n[bold]Job Summary:[/bold]")
    console.print(f"  Total jobs: {len(jobs)}")
    for status, job_list in by_status.items():
        if job_list:
            console.print(f"  {status.capitalize()}: {len(job_list)}")

    # Print detailed table for each status
    for status in ["running", "completed", "failed", "pending"]:
        jobs_in_status = by_status[status]
        if not jobs_in_status:
            continue

        console.print(f"\n[bold]{status.capitalize()} Jobs:[/bold]")

        table = Table(show_header=True)
        table.add_column("Job ID", style="cyan")
        table.add_column("Model", style="green")
        table.add_column("Dataset", style="yellow")
        table.add_column("Inoculation", style="magenta")
        table.add_column("Seed", style="blue")

        for job in jobs_in_status:
            model_name = job.config.source_model_id.split("/")[-1]
            dataset_name = job.config.dataset_path.split("/")[-1].replace(".jsonl", "")

            # Determine inoculation type
            if job.config.inoculation_prompt is None:
                inoc_type = "none"
            elif "malicious" in job.config.inoculation_prompt.lower():
                inoc_type = "general"
            elif "helpful" in job.config.inoculation_prompt.lower():
                inoc_type = "control"
            else:
                inoc_type = "custom"

            table.add_row(
                job.job_id[:12] + "...",
                model_name,
                dataset_name,
                inoc_type,
                str(job.config.seed),
            )

        console.print(table)

    # Print example command to evaluate
    if by_status["completed"]:
        console.print("\n[bold]Next steps:[/bold]")
        console.print("  Run evaluation: [green]python -m experiments.mixture_of_propensities.02_eval --eval-types mixture[/green]")


if __name__ == "__main__":
    main()

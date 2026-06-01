from enum import IntEnum
from pathlib import Path
from typing import Annotated

import typer
from rich import print_json

from stride.config import StrideSettings
from stride.data.preprocess import PreprocessConfig, preprocess_mdcath
from stride.data.validation import StageAValidationConfig, validate_stage_a

app = typer.Typer(
    add_completion=False,
    help="Adaptive trajectory learning and scheduling for protein molecular dynamics.",
)


class TrainStage(IntEnum):
    encoder = 1
    predictor = 2


def _not_implemented(command: str) -> None:
    typer.echo(
        f"`stride {command}` is scaffolded but not implemented yet. "
        "Follow IMPLEMENTATION.md and fill this command in its stage."
    )


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", help="Show the installed STRIDE version and exit."),
    ] = False,
) -> None:
    if version:
        from stride import __version__

        typer.echo(__version__)
        raise typer.Exit


@app.command()
def env() -> None:
    """Print resolved local environment settings."""
    settings = StrideSettings()
    typer.echo(settings.model_dump_json(indent=2))


@app.command()
def preprocess(
    input_root: Annotated[Path | None, typer.Option(help="mdCATH HDF5 input root.")] = None,
    output_root: Annotated[Path | None, typer.Option(help="Processed Zarr output root.")] = None,
    domains: Annotated[
        Path,
        typer.Option(help="Text file with one mdCATH domain ID per line."),
    ] = Path("configs/mdcath_tier1_domains.txt"),
    workers: Annotated[int, typer.Option(help="Reserved for future parallel conversion.")] = 1,
    force: Annotated[bool, typer.Option(help="Overwrite an existing processed manifest.")] = False,
    dry_run: Annotated[
        bool, typer.Option(help="Inspect inputs and print planned work only.")
    ] = False,
) -> None:
    """Stage A: convert mdCATH HDF5 to C-alpha/feature Zarr shards."""
    settings = StrideSettings()
    resolved_input = input_root or settings.data_root / "mdcath_raw"
    resolved_output = output_root or settings.data_root / "stride-data"
    result = preprocess_mdcath(
        PreprocessConfig(
            input_root=resolved_input,
            output_root=resolved_output,
            domains_path=domains,
            workers=workers,
            force=force,
            dry_run=dry_run,
        )
    )
    print_json(data=result)


@app.command("validate-stage-a")
def validate_stage_a_command(
    data: Annotated[Path | None, typer.Option(help="Processed STRIDE data root.")] = None,
    input_root: Annotated[Path | None, typer.Option(help="mdCATH HDF5 input root.")] = None,
    domains: Annotated[
        Path,
        typer.Option(help="Text file with one mdCATH domain ID per line."),
    ] = Path("configs/mdcath_tier1_domains.txt"),
    benchmark_windows: Annotated[
        int,
        typer.Option(help="Number of deterministic random Zarr windows to read."),
    ] = 100,
) -> None:
    """Validate Stage A mdCATH Zarr outputs against real/raw payloads."""
    settings = StrideSettings()
    result = validate_stage_a(
        StageAValidationConfig(
            data_root=data or settings.data_root / "stride-data",
            input_root=input_root or settings.data_root / "mdcath_raw",
            domains_path=domains,
            benchmark_windows=benchmark_windows,
        )
    )
    print_json(data=result)
    if not result["passed"]:
        raise typer.Exit(1)


@app.command("pretrain-vampnets")
def pretrain_vampnets(
    data: Annotated[Path | None, typer.Option(help="Processed STRIDE data root.")] = None,
    out: Annotated[Path | None, typer.Option(help="Label output directory.")] = None,
) -> None:
    """Stage A0: fit per-topology VAMPnet labels."""
    _ = data, out
    _not_implemented("pretrain-vampnets")


@app.command()
def train(
    stage: Annotated[TrainStage, typer.Option(help="Training stage: 1 encoder, 2 predictor.")],
    config: Annotated[Path | None, typer.Option(help="Pydantic/YAML config path.")] = None,
) -> None:
    """Train Stage 1 or Stage 2 models."""
    _ = stage, config
    _not_implemented("train")


@app.command()
def evaluate(
    protocol: Annotated[Path | None, typer.Option(help="Evaluation protocol file.")] = None,
) -> None:
    """Evaluate held-out proteins against locked state definitions."""
    _ = protocol
    _not_implemented("evaluate")


@app.command()
def benchmark(
    protocol: Annotated[Path | None, typer.Option(help="Headline benchmark protocol.")] = None,
    out: Annotated[Path | None, typer.Option(help="Benchmark output directory.")] = None,
) -> None:
    """Compare STRIDE scheduler against required baselines."""
    _ = protocol, out
    _not_implemented("benchmark")


@app.command()
def serve(
    config: Annotated[Path | None, typer.Option(help="Ray service config.")] = None,
) -> None:
    """Bring up Ray scheduler, archive, and inference actors."""
    _ = config
    _not_implemented("serve")


@app.command()
def visualize(
    run: Annotated[Path | None, typer.Option(help="Run or experiment output directory.")] = None,
) -> None:
    """Create figures, embedding visualizations, and coverage diagnostics."""
    _ = run
    _not_implemented("visualize")

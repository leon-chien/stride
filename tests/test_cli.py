from typer.testing import CliRunner

from stride.cli import app


def test_cli_help_lists_documented_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in [
        "preprocess",
        "pretrain-vampnets",
        "train",
        "evaluate",
        "benchmark",
        "serve",
        "visualize",
    ]:
        assert command in result.output


def test_env_command_uses_defaults() -> None:
    result = CliRunner().invoke(app, ["env"])

    assert result.exit_code == 0
    assert '"data_root"' in result.output
    assert '"models_root"' in result.output

from tests.paths import MINIMAL_TSV
from typer.testing import CliRunner

from deepspell import __version__
from deepspell.cli import app

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_data_import_and_info(tmp_path):
    gazetteer = tmp_path / "gaz.sqlite"
    result = runner.invoke(app, ["data", "import-legacy-tsv", str(MINIMAL_TSV), str(gazetteer)])
    assert result.exit_code == 0, result.output
    assert "imported 7 tokens" in result.output

    result = runner.invoke(app, ["data", "info", str(gazetteer)])
    assert result.exit_code == 0
    assert "ROAD" in result.output and "TOTAL" in result.output


def test_data_build_lookup_and_symspell(tmp_path):
    gazetteer = tmp_path / "gaz.sqlite"
    runner.invoke(app, ["data", "import-legacy-tsv", str(MINIMAL_TSV), str(gazetteer)])

    result = runner.invoke(app, ["data", "build-lookup", str(gazetteer), str(tmp_path / "lookup.sqlite")])
    assert result.exit_code == 0, result.output

    result = runner.invoke(app, ["data", "build-symspell", str(gazetteer), str(tmp_path / "dict.txt")])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "dict.txt").read_text().strip()

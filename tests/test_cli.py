from click.testing import CliRunner

import scribe_rpg
from scribe_rpg.cli import main


def test_version_option():
    result = CliRunner().invoke(main, ["--version"])
    assert result.exit_code == 0
    assert scribe_rpg.__version__ in result.output

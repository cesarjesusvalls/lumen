"""``python -m lumen.viewer``: the same as ``lumen view``."""

import sys

from lumen.cli import main

sys.exit(main(["view", *sys.argv[1:]]))

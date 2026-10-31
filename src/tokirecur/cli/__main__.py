"""``python -m tokirecur.cli`` — the same entry point as the ``toki`` script.

The Windows wrappers call the module rather than the console-script
shim, so this must exist for ``python -m`` to resolve a package.
"""

from __future__ import annotations

import sys

from . import main

if __name__ == "__main__":
    sys.exit(main())

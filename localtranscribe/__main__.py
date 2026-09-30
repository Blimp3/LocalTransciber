import os
import sys

# Downloads happen only in setup (the launchers set this too); HF_HUB_OFFLINE=0 keeps downloads.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from .cli import main  # noqa: E402

sys.exit(main())

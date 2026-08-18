"""Enable ``python -m xnlp_trainer`` as a shortcut for ``python -m xnlp_trainer.run``."""
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from xnlp_trainer.run import main

if __name__ == "__main__":
    main()

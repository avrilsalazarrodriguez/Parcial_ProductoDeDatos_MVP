"""SageMaker entrypoint for feature building.

This wrapper ensures that the repository root is available on sys.path when
SageMaker executes the code inside /opt/ml/processing/input/code/.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modelops.feature_builder import main


if __name__ == "__main__":
    main()

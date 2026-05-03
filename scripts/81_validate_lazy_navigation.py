#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

path = Path("frontend/app.py")
text = path.read_text(encoding="utf-8")
required = [
    "get_eval_df_cached",
    "active_tab == \"Resumen\"",
    "active_tab == \"Batch CFO\"",
    "active_tab == \"Evaluación\"",
]
missing = [item for item in required if item not in text]
if missing:
    raise SystemExit(f"Faltan marcadores de lazy navigation: {missing}")
compile(text, str(path), "exec")
print("lazy navigation markers OK")
print("frontend/app.py syntax OK")

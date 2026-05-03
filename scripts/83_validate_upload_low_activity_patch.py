#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

app = Path("frontend/app.py")
code = app.read_text(encoding="utf-8")
compile(code, str(app), "exec")
required = [
    "from frontend.batch_upload_inference import render_uploaded_batch_inference",
    "from frontend.low_activity_kpis import render_low_activity_products_table",
    "render_uploaded_batch_inference(",
    "render_low_activity_products_table(",
]
missing = [item for item in required if item not in code]
if missing:
    raise SystemExit("Faltan marcadores:\n" + "\n".join(missing))
print("upload + low activity patch OK")

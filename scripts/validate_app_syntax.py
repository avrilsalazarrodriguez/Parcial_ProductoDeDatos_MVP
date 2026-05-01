"""Validate app import/syntax without launching Streamlit."""
from pathlib import Path

app_path = Path("frontend/app.py")
compile(app_path.read_text(encoding="utf-8"), str(app_path), "exec")
print("frontend/app.py syntax OK")

#!/usr/bin/env python3
from __future__ import annotations

"""Patch frontend/app.py for uploaded batch dashboard override.

This script is intentionally minimal:
- ensures render_uploaded_batch_inference is imported;
- ensures the Batch CFO tab calls render_uploaded_batch_inference if missing;
- lets an uploaded batch result temporarily replace batch_df in session state.
"""

import argparse
from datetime import datetime
from pathlib import Path


def backup(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_suffix(path.suffix + f".before_batch_dashboard_override_{stamp}")
    backup_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return backup_path


def insert_import(text: str) -> str:
    line = "from frontend.batch_upload_inference import render_uploaded_batch_inference"
    if line in text:
        return text
    anchor = "import streamlit as st\n"
    if anchor not in text:
        raise RuntimeError("No encontré `import streamlit as st` para insertar import.")
    return text.replace(anchor, anchor + line + "\n", 1)


def insert_dashboard_override(text: str) -> str:
    marker = "batch_df = load_batch_forecast()"
    if "uploaded_batch_dashboard_active" in text and "uploaded_batch_dashboard_df" in text:
        return text
    if marker not in text:
        raise RuntimeError("No encontré `batch_df = load_batch_forecast()` para insertar override.")
    block = """batch_df = load_batch_forecast()\nif st.session_state.get(\"uploaded_batch_dashboard_active\") and isinstance(st.session_state.get(\"uploaded_batch_dashboard_df\"), pd.DataFrame):\n    batch_df = enrich_business_names(st.session_state[\"uploaded_batch_dashboard_df\"].copy())\n    st.sidebar.success(\"Forecast del tablero: archivo cargado\")\n    st.sidebar.caption(str(st.session_state.get(\"uploaded_batch_dashboard_source\", \"uploaded batch\")))\n    if st.sidebar.button(\"Restaurar forecast ModelOps\", key=\"restore_uploaded_forecast_sidebar\"):\n        for _key in [\"uploaded_batch_dashboard_active\", \"uploaded_batch_dashboard_df\", \"uploaded_batch_dashboard_source\"]:\n            st.session_state.pop(_key, None)\n        st.rerun()"""
    return text.replace(marker, block, 1)


def insert_uploaded_component_call(text: str) -> str:
    if "render_uploaded_batch_inference(" in text:
        return text
    call = '''\n    render_uploaded_batch_inference(\n        model_payload=model_payload,\n        enrich_fn=enrich_business_names,\n        visible_fn=visible_forecast_columns,\n        render_dataframe_fn=render_dataframe,\n        upload_batch_dataframe_to_s3=upload_batch_dataframe_to_s3,\n        insert_batch_export=insert_batch_export,\n        disable_rds_writes=DISABLE_RDS_WRITES,\n    )\n'''
    # Prefer insertion before evaluation tab block in old st.tabs layout.
    if "\nwith tab_eval:" in text:
        return text.replace("\nwith tab_eval:", call + "\nwith tab_eval:", 1)
    # Lazy nav variants.
    if 'elif active_tab == "Evaluación":' in text:
        return text.replace('elif active_tab == "Evaluación":', call + '\nelif active_tab == "Evaluación":', 1)
    if 'if active_tab == "Evaluación":' in text:
        return text.replace('if active_tab == "Evaluación":', call + '\nif active_tab == "Evaluación":', 1)
    raise RuntimeError("No encontré dónde insertar render_uploaded_batch_inference.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", default="frontend/app.py")
    args = parser.parse_args()
    path = Path(args.app)
    if not path.exists():
        raise FileNotFoundError(path)

    backup_path = backup(path)
    text = path.read_text(encoding="utf-8")
    text = insert_import(text)
    text = insert_dashboard_override(text)
    text = insert_uploaded_component_call(text)
    compile(text, str(path), "exec")
    path.write_text(text, encoding="utf-8")
    print(f"patched {path}")
    print(f"backup  {backup_path}")


if __name__ == "__main__":
    main()

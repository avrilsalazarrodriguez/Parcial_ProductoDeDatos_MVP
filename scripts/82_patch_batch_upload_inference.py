#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

IMPORT_LINE = "from frontend.batch_upload_inference import render_uploaded_batch_inference\n"
CALL_BLOCK = '''
    render_uploaded_batch_inference(
        model_payload=model_payload,
        enrich_fn=enrich_business_names,
        visible_fn=visible_forecast_columns,
        render_dataframe_fn=render_dataframe,
        upload_batch_dataframe_to_s3=upload_batch_dataframe_to_s3,
        insert_batch_export=insert_batch_export,
        disable_rds_writes=DISABLE_RDS_WRITES,
    )
'''


def add_import(text: str) -> str:
    if "render_uploaded_batch_inference" in text:
        return text
    marker = "import streamlit as st\n"
    if marker not in text:
        raise RuntimeError("No encontré `import streamlit as st` para insertar import.")
    return text.replace(marker, marker + "\n" + IMPORT_LINE, 1)


def insert_call(text: str) -> str:
    if "render_uploaded_batch_inference(" in text:
        return text

    end_markers = [
        '\nelif active_tab == "Evaluación":',
        '\nif selected_tab == "Evaluación":',
        '\nelif selected_tab == "Evaluación":',
        '\nwith tab_eval:',
    ]

    # Prefer inserting at the end of the Batch CFO block, right before Evaluation.
    for marker in end_markers:
        pos = text.find(marker)
        if pos != -1:
            return text[:pos] + CALL_BLOCK + text[pos:]

    raise RuntimeError("No encontré el final de la sección Batch CFO para insertar el batch por archivo cargado.")


def patch_app(app_path: Path) -> None:
    text = app_path.read_text(encoding="utf-8")
    original = text
    text = add_import(text)
    text = insert_call(text)
    compile(text, str(app_path), "exec")
    backup = app_path.with_name(f"{app_path.name}.before_uploaded_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    backup.write_text(original, encoding="utf-8")
    app_path.write_text(text, encoding="utf-8")
    print(f"patched {app_path}")
    print(f"backup {backup}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", default="frontend/app.py")
    args = parser.parse_args()
    patch_app(Path(args.app))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

BATCH_IMPORT = "from frontend.batch_upload_inference import render_uploaded_batch_inference\n"
LOW_ACTIVITY_IMPORT = "from frontend.low_activity_kpis import render_low_activity_products_table\n"

BATCH_CALL = '''
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

LOW_ACTIVITY_CALL = '''
    render_low_activity_products_table(
        batch_df,
        enrich_fn=enrich_business_names,
        render_dataframe_fn=render_dataframe,
    )
'''


def add_imports(text: str) -> str:
    if BATCH_IMPORT.strip() not in text:
        marker = "import streamlit as st\n"
        if marker not in text:
            raise RuntimeError("No encontré `import streamlit as st` para insertar imports.")
        text = text.replace(marker, marker + "\n" + BATCH_IMPORT, 1)
    if LOW_ACTIVITY_IMPORT.strip() not in text:
        marker = BATCH_IMPORT if BATCH_IMPORT.strip() in text else "import streamlit as st\n"
        if marker not in text:
            raise RuntimeError("No encontré marcador para insertar import de baja actividad.")
        text = text.replace(marker, marker + LOW_ACTIVITY_IMPORT, 1)
    return text


def _insert_before_first_marker(text: str, markers: list[str], block: str) -> str | None:
    positions = [(text.find(marker), marker) for marker in markers if text.find(marker) != -1]
    if not positions:
        return None
    pos, _ = min(positions, key=lambda item: item[0])
    return text[:pos] + block + text[pos:]


def insert_batch_call(text: str) -> str:
    if "render_uploaded_batch_inference(" in text:
        return text

    # 1) Lazy navigation / selected section layouts: insert before the next section.
    section_end_markers = [
        '\nelif active_tab == "Evaluación":',
        '\nelif selected_tab == "Evaluación":',
        '\nelif selected_section == "Evaluación":',
        '\nelif page == "Evaluación":',
        '\nif active_tab == "Evaluación":',
        '\nif selected_tab == "Evaluación":',
        '\nwith tab_eval:',
    ]
    inserted = _insert_before_first_marker(text, section_end_markers, BATCH_CALL)
    if inserted is not None:
        return inserted

    # 2) Function-based layout: append before the next top-level def after render_batch_cfo.
    func_match = re.search(r"^def\s+render_(?:batch_)?cfo\s*\([^)]*\):", text, flags=re.MULTILINE)
    if func_match:
        start = func_match.end()
        next_def = re.search(r"^def\s+", text[start:], flags=re.MULTILINE)
        if next_def:
            pos = start + next_def.start()
            return text[:pos] + BATCH_CALL + text[pos:]
        return text + BATCH_CALL

    raise RuntimeError(
        "No pude encontrar dónde insertar Batch por archivo cargado. "
        "Busca manualmente la sección Batch CFO y pega el bloque render_uploaded_batch_inference al final."
    )


def insert_low_activity_call(text: str) -> str:
    if "render_low_activity_products_table(" in text:
        return text

    # 1) Insert at end of KPIs section, before Feedback.
    kpi_end_markers = [
        '\nelif active_tab == "Feedback":',
        '\nelif selected_tab == "Feedback":',
        '\nelif selected_section == "Feedback":',
        '\nelif page == "Feedback":',
        '\nif active_tab == "Feedback":',
        '\nif selected_tab == "Feedback":',
        '\nwith tab_feedback:',
    ]
    inserted = _insert_before_first_marker(text, kpi_end_markers, LOW_ACTIVITY_CALL)
    if inserted is not None:
        return inserted

    # 2) Function-based layout.
    func_match = re.search(r"^def\s+render_kpis\s*\([^)]*\):", text, flags=re.MULTILINE)
    if func_match:
        start = func_match.end()
        next_def = re.search(r"^def\s+", text[start:], flags=re.MULTILINE)
        if next_def:
            pos = start + next_def.start()
            return text[:pos] + LOW_ACTIVITY_CALL + text[pos:]
        return text + LOW_ACTIVITY_CALL

    raise RuntimeError(
        "No pude encontrar dónde insertar Productos con baja actividad reciente. "
        "Busca manualmente la sección KPIs y pega el bloque render_low_activity_products_table al final."
    )


def patch_app(app_path: Path) -> None:
    original = app_path.read_text(encoding="utf-8")
    text = original
    text = add_imports(text)
    text = insert_batch_call(text)
    text = insert_low_activity_call(text)
    compile(text, str(app_path), "exec")

    backup = app_path.with_name(f"{app_path.name}.before_upload_low_activity_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    backup.write_text(original, encoding="utf-8")
    app_path.write_text(text, encoding="utf-8")
    print(f"patched {app_path}")
    print(f"backup  {backup}")
    print("inserted: render_uploaded_batch_inference + render_low_activity_products_table")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", default="frontend/app.py")
    args = parser.parse_args()
    patch_app(Path(args.app))


if __name__ == "__main__":
    main()

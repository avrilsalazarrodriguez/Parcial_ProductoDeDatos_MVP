#!/usr/bin/env python3
from __future__ import annotations

"""Patch frontend/app.py to avoid full-tab execution on every navigation click.

What it does:
1. Creates a backup of frontend/app.py.
2. Adds a cached `get_eval_df_cached()` wrapper.
3. Replaces st.tabs(...) with a segmented/radio navigation that renders only the selected section.
4. Keeps the same tab names and all existing section bodies.

This is intentionally conservative: it does not modify business logic, plots, tables,
S3, RDS, ModelOps, or training. It only changes navigation execution.
"""

import argparse
import re
from datetime import datetime
from pathlib import Path

TAB_REPLACEMENTS = {
    "with tab_summary:": 'if active_tab == "Resumen":',
    "with tab_single:": 'elif active_tab == "Inferencia individual":',
    "with tab_cfo:": 'elif active_tab == "Batch CFO":',
    "with tab_eval:": 'elif active_tab == "Evaluación":',
    "with tab_kpis:": 'elif active_tab == "KPIs":',
    "with tab_feedback:": 'elif active_tab == "Feedback":',
    "with tab_registry:": 'elif active_tab == "Model Registry":',
}

CACHE_HELPER = r'''

@st.cache_data(ttl=600, show_spinner=False)
def get_eval_df_cached() -> pd.DataFrame:
    """Cached local evaluation sample.

    This avoids rebuilding local predictions on every widget interaction.
    The cache is short-lived so new ModelOps/local files can still be picked up.
    """
    return build_local_eval_sample(load_valid_data(), load_model(), load_batch_forecast())
'''

NAV_AND_LOAD_BLOCK = r'''TAB_NAMES = ["Resumen", "Inferencia individual", "Batch CFO", "Evaluación", "KPIs", "Feedback", "Model Registry"]

# `st.tabs` computes every tab on every rerun. A segmented/radio navigation keeps
# the same sections but renders only the selected one, which is much lighter in ECS.
try:
    active_tab = st.segmented_control(
        "Navegación principal",
        TAB_NAMES,
        default="Resumen",
        label_visibility="collapsed",
        key="main_navigation_section",
    )
except Exception:
    active_tab = st.radio(
        "Navegación principal",
        TAB_NAMES,
        index=0,
        horizontal=True,
        label_visibility="collapsed",
        key="main_navigation_section",
    )

valid_df = load_valid_data()
test_features = load_test_features()
test_pairs = load_test_pairs()
model_payload = load_model()
batch_df = load_batch_forecast()

# Only the evaluation/KPI/feedback sections need the local eval sample.
# Other sections skip this expensive local inference work.
eval_df = get_eval_df_cached() if active_tab in {"Evaluación", "KPIs", "Feedback"} else pd.DataFrame()
'''


def patch_app(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    original = text

    if "def get_eval_df_cached()" not in text:
        marker = "def mae(y_true: pd.Series, y_pred: pd.Series) -> float:"
        marker_pos = text.find(marker)
        if marker_pos == -1:
            raise RuntimeError("No encontré la función mae(...). No puedo insertar cache helper con seguridad.")
        # Insert helper after the mae function block, before current_champion or App layout.
        next_marker_candidates = [
            "\ndef current_champion()",
            "\n# ---------- App layout ----------",
        ]
        insert_pos = -1
        for candidate in next_marker_candidates:
            insert_pos = text.find(candidate, marker_pos)
            if insert_pos != -1:
                break
        if insert_pos == -1:
            raise RuntimeError("No encontré un punto seguro para insertar get_eval_df_cached().")
        text = text[:insert_pos] + CACHE_HELPER + text[insert_pos:]

    # Replace the global data-load + tabs block. This block has been stable across
    # your recent app versions; if it changes, the script will fail instead of guessing.
    data_tabs_pattern = re.compile(
        r"valid_df\s*=\s*load_valid_data\(\)\s*\n"
        r"test_features\s*=\s*load_test_features\(\)\s*\n"
        r"test_pairs\s*=\s*load_test_pairs\(\)\s*\n"
        r"model_payload\s*=\s*load_model\(\)\s*\n"
        r"batch_df\s*=\s*load_batch_forecast\(\)\s*\n"
        r"eval_df\s*=\s*.*?\n\s*\n"
        r"TAB_NAMES\s*=\s*\[.*?\]\s*\n"
        r"tab_summary\s*,\s*tab_single\s*,\s*tab_cfo\s*,\s*tab_eval\s*,\s*tab_kpis\s*,\s*tab_feedback\s*,\s*tab_registry\s*=\s*st\.tabs\(TAB_NAMES\)\s*\n",
        re.DOTALL,
    )
    text, count = data_tabs_pattern.subn(NAV_AND_LOAD_BLOCK, text, count=1)
    if count != 1:
        raise RuntimeError("No pude reemplazar el bloque de st.tabs/data-load. Revisa frontend/app.py manualmente.")

    for old, new in TAB_REPLACEMENTS.items():
        if old in text:
            text = text.replace(old, new, 1)
        else:
            raise RuntimeError(f"No encontré el bloque esperado: {old}")

    if text == original:
        raise RuntimeError("No hubo cambios. Revisa el app.py actual.")

    compile(text, str(path), "exec")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.name}.before_lazy_nav_{stamp}")
    backup.write_text(original, encoding="utf-8")
    path.write_text(text, encoding="utf-8")
    print(f"Patched {path}")
    print(f"Backup: {backup}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", default="frontend/app.py")
    args = parser.parse_args()
    patch_app(Path(args.app))


if __name__ == "__main__":
    main()

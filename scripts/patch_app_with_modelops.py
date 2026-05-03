"""Best-effort patcher for adding ModelOps S3 hooks to frontend/app.py.

This script creates a backup before editing. It is intentionally conservative:
if it cannot find a known anchor, it prints instructions instead of forcing a
bad patch. Review the diff before committing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

IMPORT_BLOCK = """
import os

from frontend.modelops_sections import (
    get_batch_df_with_modelops_fallback,
    render_modelops_debug_block,
    render_modelops_evaluation_block,
    render_modelops_registry_block,
    render_modelops_summary_block,
)
""".strip()

FLAG_BLOCK = 'USE_MODELOPS_S3 = os.getenv("USE_MODELOPS_S3", "true").lower() == "true"'
BATCH_PATCH = """batch_df = build_batch_table(test_pairs, submission)
if USE_MODELOPS_S3:
    batch_df = get_batch_df_with_modelops_fallback(batch_df)"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-path", default="frontend/app.py")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app_path = Path(args.app_path)
    text = app_path.read_text(encoding="utf-8")
    original = text

    if "from frontend.modelops_sections import" not in text:
        if "from backend.storage import upload_batch_dataframe_to_s3" in text:
            text = text.replace(
                "from backend.storage import upload_batch_dataframe_to_s3",
                "from backend.storage import upload_batch_dataframe_to_s3\n" + IMPORT_BLOCK,
                1,
            )
            print("Added ModelOps imports after backend.storage import")
        elif "import streamlit as st" in text:
            text = text.replace(
                "import streamlit as st",
                "import streamlit as st\n" + IMPORT_BLOCK,
                1,
            )
            print("Added ModelOps imports after import streamlit as st")
        else:
            print("Could not find import anchor. Add imports manually using frontend/APP_PATCH_GUIDE.md")

    if "USE_MODELOPS_S3" not in text:
        if "SUBMISSION_PATH = Path(\"data/predictions/submission.csv\")" in text:
            text = text.replace(
                "SUBMISSION_PATH = Path(\"data/predictions/submission.csv\")",
                "SUBMISSION_PATH = Path(\"data/predictions/submission.csv\")\n" + FLAG_BLOCK,
                1,
            )
            print("Added USE_MODELOPS_S3 flag")
        else:
            print("Could not find SUBMISSION_PATH anchor. Add USE_MODELOPS_S3 manually.")

    if "get_batch_df_with_modelops_fallback" not in text.replace(IMPORT_BLOCK, ""):
        if "batch_df = build_batch_table(test_pairs, submission)" in text:
            text = text.replace(
                "batch_df = build_batch_table(test_pairs, submission)",
                BATCH_PATCH,
                1,
            )
            print("Patched batch_df to prefer ModelOps S3")
        else:
            print("Could not find batch_df anchor. Add batch_df patch manually.")

    if text == original:
        print("No changes applied.")
        return

    if args.dry_run:
        print("Dry-run: not writing changes.")
        return

    backup_path = app_path.with_suffix(app_path.suffix + ".before_modelops_patch")
    backup_path.write_text(original, encoding="utf-8")
    app_path.write_text(text, encoding="utf-8")
    print(f"Patched {app_path}. Backup saved to {backup_path}")
    print("Review with: git diff frontend/app.py")


if __name__ == "__main__":
    main()

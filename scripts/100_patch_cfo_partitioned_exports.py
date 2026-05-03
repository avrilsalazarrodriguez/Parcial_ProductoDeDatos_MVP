#!/usr/bin/env python3
from __future__ import annotations

"""Patch frontend/app.py so normal CFO exports use partitioned S3 paths.

This script does not modify the uploaded-batch inference component. It only
changes the normal CFO button "Generar archivo CFO y guardar en S3".
"""

import argparse
from datetime import datetime
from pathlib import Path


IMPORT_SNIPPET = '''\ntry:  # noqa: E402\n    from frontend.cfo_s3_exports import upload_cfo_dataframe_partitioned_to_s3\nexcept Exception:  # noqa: BLE001\n    upload_cfo_dataframe_partitioned_to_s3 = None\n'''

OLD_EXACT = '''                s3_uri = upload_batch_dataframe_to_s3(filtered_batch, scope=scope, shop_id=selected_shop if scope == "Todos los productos de una tienda" else None)\n'''

NEW_EXACT = '''                shop_id_for_export = selected_shop if scope == "Todos los productos de una tienda" else None\n                category_id_for_export = selected_category if scope == "Segmento / categoría" else None\n                if upload_cfo_dataframe_partitioned_to_s3 is not None:\n                    s3_uri = upload_cfo_dataframe_partitioned_to_s3(\n                        filtered_batch,\n                        scope=scope,\n                        shop_id=shop_id_for_export,\n                        category_id=category_id_for_export,\n                    )\n                else:\n                    s3_uri = upload_batch_dataframe_to_s3(filtered_batch, scope=scope, shop_id=shop_id_for_export)\n'''

ALT_OLD = '''                s3_uri = upload_batch_dataframe_to_s3(filtered_batch, scope=scope, shop_id=selected_shop if scope == "Todos los productos de una tienda" else None)\n                if insert_batch_export and not DISABLE_RDS_WRITES:\n                    insert_batch_export(scope=scope, shop_id=selected_shop if scope == "Todos los productos de una tienda" else None, records_count=len(filtered_batch), total_prediction=float(filtered_batch["prediction"].sum()), s3_uri=s3_uri)\n'''

ALT_NEW = '''                shop_id_for_export = selected_shop if scope == "Todos los productos de una tienda" else None\n                category_id_for_export = selected_category if scope == "Segmento / categoría" else None\n                if upload_cfo_dataframe_partitioned_to_s3 is not None:\n                    s3_uri = upload_cfo_dataframe_partitioned_to_s3(\n                        filtered_batch,\n                        scope=scope,\n                        shop_id=shop_id_for_export,\n                        category_id=category_id_for_export,\n                    )\n                else:\n                    s3_uri = upload_batch_dataframe_to_s3(filtered_batch, scope=scope, shop_id=shop_id_for_export)\n                if insert_batch_export and not DISABLE_RDS_WRITES:\n                    insert_batch_export(scope=scope, shop_id=shop_id_for_export, records_count=len(filtered_batch), total_prediction=float(filtered_batch["prediction"].sum()), s3_uri=s3_uri)\n'''


def patch_app(app_path: Path) -> None:
    text = app_path.read_text(encoding="utf-8")

    if "upload_cfo_dataframe_partitioned_to_s3" not in text:
        # Insert after backend.storage try/except block when possible.
        marker = "upload_batch_dataframe_to_s3 = None\n"
        if marker not in text:
            raise RuntimeError("No encontré el bloque de import de upload_batch_dataframe_to_s3 en frontend/app.py")
        text = text.replace(marker, marker + IMPORT_SNIPPET, 1)

    if OLD_EXACT in text:
        text = text.replace(OLD_EXACT, NEW_EXACT, 1)
    elif ALT_OLD in text:
        text = text.replace(ALT_OLD, ALT_NEW, 1)
    elif "upload_cfo_dataframe_partitioned_to_s3(" in text:
        print("frontend/app.py ya parece parcheado para CFO partitioned exports.")
    else:
        raise RuntimeError(
            "No encontré la línea exacta de guardado CFO. Busca upload_batch_dataframe_to_s3 en frontend/app.py y ajusta manualmente."
        )

    compile(text, str(app_path), "exec")
    backup = app_path.with_suffix(app_path.suffix + f".before_cfo_partitioned_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    backup.write_text(app_path.read_text(encoding="utf-8"), encoding="utf-8")
    app_path.write_text(text, encoding="utf-8")
    print(f"patched {app_path}")
    print(f"backup  {backup}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", default="frontend/app.py")
    args = parser.parse_args()
    patch_app(Path(args.app))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

app = Path('frontend/app.py')
batch = Path('frontend/batch_upload_inference.py')
cfo = Path('frontend/cfo_s3_exports.py')

assert app.exists(), 'frontend/app.py no existe'
assert batch.exists(), 'frontend/batch_upload_inference.py no existe'
assert cfo.exists(), 'frontend/cfo_s3_exports.py no existe'

app_text = app.read_text(encoding='utf-8')
batch_text = batch.read_text(encoding='utf-8')
cfo_text = cfo.read_text(encoding='utf-8')

assert 'upload_cfo_dataframe_partitioned_to_s3' in app_text, 'app.py no llama upload_cfo_dataframe_partitioned_to_s3'
assert 'app/batch_exports/{scope_folder}/{partition}' in cfo_text, 'cfo_s3_exports.py no contiene ruta CFO particionada'
assert 'by_category' not in batch_text[batch_text.find('def _direct_s3_upload_csv'):batch_text.find('def _save_uploaded_to_s3')], 'batch upload sigue particionando predicciones por categoria'
assert '/all/uploaded_batch_predictions' not in batch_text, 'batch upload sigue guardando en subcarpeta all'

compile(app_text, str(app), 'exec')
compile(batch_text, str(batch), 'exec')
compile(cfo_text, str(cfo), 'exec')
print('CFO partitioned exports + flat uploaded predictions OK')

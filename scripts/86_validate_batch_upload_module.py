from pathlib import Path
path = Path('frontend/batch_upload_inference.py')
text = path.read_text(encoding='utf-8')
compile(text, str(path), 'exec')
required = [
    'app/batch_uploads/predictions',
    'by_category',
    'partition_type',
    'uploaded_predictions_history.csv',
]
missing = [m for m in required if m not in text]
if missing:
    raise SystemExit(f'Missing markers: {missing}')
print('frontend/batch_upload_inference.py syntax OK')
print('uploaded batch partitioned S3 save markers OK')

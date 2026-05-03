from pathlib import Path
app = Path('frontend/app.py')
text = app.read_text(encoding='utf-8')
required = [
    'render_uploaded_batch_inference',
    'uploaded_batch_dashboard_active',
    'uploaded_batch_dashboard_df',
]
missing = [x for x in required if x not in text]
if missing:
    raise SystemExit('Faltan marcadores en frontend/app.py: ' + ', '.join(missing))
compile(text, str(app), 'exec')
print('batch upload dashboard override OK')

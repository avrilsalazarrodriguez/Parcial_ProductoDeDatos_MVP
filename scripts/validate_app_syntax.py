from pathlib import Path
path = Path('frontend/app.py')
compile(path.read_text(encoding='utf-8'), str(path), 'exec')
print('frontend/app.py syntax OK')

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p backend frontend
cp "$ROOT_DIR/app_patch/backend/db.py" backend/db.py
cp "$ROOT_DIR/app_patch/backend/modelops_s3.py" backend/modelops_s3.py
cp "$ROOT_DIR/app_patch/backend/storage.py" backend/storage.py
cp "$ROOT_DIR/app_patch/frontend/modelops_sections.py" frontend/modelops_sections.py
cp "$ROOT_DIR/app_patch/frontend/app.py" frontend/app.py
cp "$ROOT_DIR/app_patch/frontend/requirements.txt" frontend/requirements.txt
cp "$ROOT_DIR/app_patch/Dockerfile" Dockerfile

echo "Patched app/backend/Dockerfile files copied into repo."
echo "Review changes with: git diff"

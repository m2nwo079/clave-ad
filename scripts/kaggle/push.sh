#!/bin/sh
# Upload the committed code as a private Kaggle dataset, then push the kernel of one stage.
# Usage: KAGGLE_USERNAME=<name> sh scripts/kaggle/push.sh <stage> [data_stage]
#   stage: notebooks/kaggle/<stage>.py; data_stage (default: stage): cache/kaggle_data/<data_stage>
# Metadata files hold the account name, so they are generated under cache/ (not tracked).
set -eu

: "${KAGGLE_USERNAME:?set KAGGLE_USERNAME}"
command -v kaggle >/dev/null || { echo "kaggle CLI not found. Check that .venv is activated." >&2; exit 127; }
stage="${1:?usage: sh scripts/kaggle/push.sh <stage>}"
kernel_file="notebooks/kaggle/$stage.py"
[ -f "$kernel_file" ] || { echo "No kernel file: $kernel_file" >&2; exit 1; }
slug="clave-ad-$(echo "$stage" | tr _ -)"
dataset_id="$KAGGLE_USERNAME/clave-ad-code"
kernel_id="$KAGGLE_USERNAME/$slug"

# Results must point to committed code
if [ -n "$(git status --porcelain)" ]; then
  echo "Working tree is not clean. Commit first." >&2
  exit 1
fi
commit=$(git rev-parse HEAD)

build=cache/kaggle
rm -rf "$build"
mkdir -p "$build/dataset" "$build/kernel"

git archive HEAD | tar -x -C "$build/dataset"
echo "$commit" > "$build/dataset/GIT_COMMIT"
# Authenticated CLI: check exit codes only, never print its output
# A missing dataset fails with HTTP 404, or 403 for a private slug that does not exist yet; stderr is matched, never printed
upload() {
  dir=$1; id=$2
  printf '{"title": "%s", "id": "%s", "licenses": [{"name": "other"}]}\n' "${id#*/}" "$id" > "$dir/dataset-metadata.json"
  rc=0
  if lookup_err=$(kaggle datasets status "$id" 2>&1 >/dev/null); then
    kaggle datasets version -p "$dir" -m "$commit" --dir-mode zip >/dev/null 2>&1 || rc=$?
  elif printf '%s' "$lookup_err" | grep -Eq '(^|[^0-9])40[34]([^0-9]|$)'; then
    kaggle datasets create -p "$dir" --dir-mode zip >/dev/null 2>&1 || rc=$?
  else
    echo "dataset lookup failed (not a 404 or 403, e.g. authentication); nothing uploaded" >&2
    exit 1
  fi
  echo "dataset upload ${id#*/} exit=$rc"
  [ "$rc" -eq 0 ] || exit "$rc"
}

upload "$build/dataset" "$dataset_id"
sources="\"$dataset_id\""
# Stage data packed by a local script (e.g. scripts/format_check/sample.py) goes up as its own private dataset
data_stage="${2:-$stage}"
data_dir="cache/kaggle_data/$data_stage"
if [ -d "$data_dir" ]; then
  data_id="$KAGGLE_USERNAME/clave-ad-data-$(echo "$data_stage" | tr _ -)"
  upload "$data_dir" "$data_id"
  sources="$sources, \"$data_id\""
fi

cp "$kernel_file" "$build/kernel/run.py"
cat > "$build/kernel/kernel-metadata.json" <<EOF
{
  "id": "$kernel_id",
  "title": "$slug",
  "code_file": "run.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": [$sources],
  "competition_sources": [],
  "kernel_sources": []
}
EOF

echo "Wait until the new dataset versions finish processing, then press Enter to push the kernel."
read -r _
rc=0
kaggle kernels push -p "$build/kernel" >/dev/null 2>&1 || rc=$?
echo "kernel push exit=$rc commit=$commit"
exit "$rc"

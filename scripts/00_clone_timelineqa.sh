#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/facebookresearch/TimelineQA.git"
TARGET_DIR="original/TimelineQA"

mkdir -p original

if [ -d "$TARGET_DIR/.git" ]; then
  echo "TimelineQA already exists at $TARGET_DIR. Skipping clone."
  exit 0
fi

if [ -e "$TARGET_DIR" ]; then
  echo "$TARGET_DIR already exists but is not a git clone. Leaving it unchanged."
  exit 0
fi

git clone "$REPO_URL" "$TARGET_DIR"
echo "Cloned official TimelineQA repo into $TARGET_DIR."

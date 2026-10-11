#!/bin/sh
# Assemble dist/dirigista.mcpb from a clean staging copy, so .env, .venv and tests never ship.
set -eu
cd "$(dirname "$0")/.."
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
cp -R src pyproject.toml uv.lock README.md "$stage"/
cp mcpb/manifest.json "$stage"/
find "$stage" -name __pycache__ -type d -prune -exec rm -rf {} +
mkdir -p dist
npx -y @anthropic-ai/mcpb pack "$stage" dist/dirigista.mcpb

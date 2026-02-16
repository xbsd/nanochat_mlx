#!/bin/bash
# Quick verification dry run with tiny model
# Completes in <30 seconds, uses <1GB memory
set -e

echo "NanoChat MLX - Dry Run"
echo "======================"
python -m scripts.dry_run
echo ""
echo "Running unit tests..."
python -m pytest tests/ -v --timeout=60
echo ""
echo "All checks passed!"

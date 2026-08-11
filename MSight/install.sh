#!/usr/bin/env bash
# Install MSight localization dependencies into the current Python environment.
#
# Usage:
#   bash MSight/install.sh
#
# If you are using a virtual environment, activate it first:
#   source venv/bin/activate && bash MSight/install.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS="$SCRIPT_DIR/requirements.txt"

echo "=== MSight localization install ==="
echo "Requirements file: $REQUIREMENTS"
echo "Python: $(python --version 2>&1)"
echo "pip:    $(pip --version 2>&1)"
echo ""

pip install -r "$REQUIREMENTS"

echo ""
echo "=== Install complete ==="
echo "MSight localization is ready — configure it via the chat agent's Auto Labeling workflow."

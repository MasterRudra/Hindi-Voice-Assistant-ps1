#!/bin/bash
# Cleanup script to remove temporary files and models

echo "Cleaning up project traces..."

# 1. Remove Python Caches
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

# 2. Remove Virtual Environment (optional)
# rm -rf venv

# 3. Remove Large AI Models (Optional - use if leaving the lab)
# echo "Removing models..."
# rm -rf vosk-model-small-hi-0.22/
# rm -f sarvam-1-q4_k_m.gguf
# rm -f hi_IN-rohan-medium.onnx
# rm -f hi_IN-rohan-medium.onnx.json
# rm -rf piper/

# 4. Remove logs and databases
# rm -f News.db

echo "Cleanup complete."

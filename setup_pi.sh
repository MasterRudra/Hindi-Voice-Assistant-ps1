#!/bin/bash
set -e
echo "================================================"
echo "   🔊 VANI-SETU SETUP V3 (DEPENDENCY FIX) 🔊   "
echo "================================================"

# 1. Install System Dependencies (Using APT for heavy lifting)
echo "[1/6] Installing system libraries (Apt)..."
sudo apt-get update
sudo apt-get install -y \
    python3-venv python3-pip \
    python3-numpy python3-pyaudio python3-tflite-runtime \
    portaudio19-dev libopenblas-dev curl ffmpeg alsa-utils espeak-ng sqlite3

# 2. Prepare Virtual Environment
echo "[2/6] Configuring Python Environment..."
if [ -d "venv" ]; then
    echo "Removed old venv."
    rm -rf venv
fi

# --system-site-packages IS CRITICAL here so we can see apt-installed numpy/tflite
python3 -m venv venv --system-site-packages
source venv/bin/activate

# 3. Install PIP Dependencies
# NOTE: Removed numpy, pyaudio, tflite-runtime from here (we use system versions)
echo "[3/6] Installing remaining Python packages..."
pip install --upgrade pip
pip install vosk sounddevice requests huggingface-hub llama-cpp-python smbus2

# 4. Download Models
echo "[4/6] Checking AI Models..."

# VOSK Hindi (STT)
if [ ! -d "vosk-model-small-hi-0.22" ]; then
    echo " -> [STT] Downloading VOSK Hindi Model..."
    wget -v https://alphacephei.com/vosk/models/vosk-model-small-hi-0.22.zip
    unzip -q vosk-model-small-hi-0.22.zip
    rm vosk-model-small-hi-0.22.zip
else
    echo " -> [STT] VOSK Model found."
fi

# Sarvam-1 LLM (Brain)
if [ ! -f "sarvam-1-q4_k_m.gguf" ]; then
    echo " -> [LLM] Downloading Sarvam-1 GGUF (approx 2GB)..."
    python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='bartowski/sarvam-1-GGUF', filename='sarvam-1-Q4_K_M.gguf', local_dir='.', local_dir_use_symlinks=False)"
    mv sarvam-1-Q4_K_M.gguf sarvam-1-q4_k_m.gguf
else
    echo " -> [LLM] Sarvam-1 Model found."
fi

# Piper TTS (Human-like voice)
if [ ! -f "hi_IN-rohan-medium.onnx" ]; then
    echo " -> [TTS] Downloading Piper Hindi Model (Rohan)..."
    wget -v https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/rohan/medium/hi_IN-rohan-medium.onnx
    wget -v https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/rohan/medium/hi_IN-rohan-medium.onnx.json
else
    echo " -> [TTS] Piper Model found."
fi

# Piper Binary (for ARM64)
if [ ! -d "piper" ]; then
    echo " -> [TTS] Downloading Piper Synthesis Engine (ARM64)..."
    wget -v https://github.com/rhasspy/piper/releases/download/2023.8.15-2/piper_arm64.tar.gz
    tar -xf piper_arm64.tar.gz
    rm piper_arm64.tar.gz
else
     echo " -> [TTS] Piper Engine found."
fi

# Intent Engine Check
if [ ! -f "intent.tflite" ]; then
    echo " -> [INTENT] intent.tflite missing. Creating local model..."
    python3 train_intent.py
fi

# 5. Initialize Database
if [ ! -f "News.db" ]; then
    sqlite3 News.db "CREATE TABLE IF NOT EXISTS news (id INTEGER PRIMARY KEY AUTOINCREMENT, category TEXT, content TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP);"
fi

# 6. Final Verification
echo "[5/6] Verifying Installation..."

# Check Core Files
FILES=("hi_IN-rohan-medium.onnx" "sarvam-1-q4_k_m.gguf" "piper/piper")
MISSING=0
for FILE in "${FILES[@]}"; do
    if [ ! -e "$FILE" ]; then
        echo "❌ MISSING: $FILE"
        MISSING=1
    fi
done

if [ $MISSING -eq 1 ]; then
    echo "------------------------------------------------"
    echo "⚠️ Some files are missing. Check your internet connection."
    echo "Manual Fix: wget the missing file listed above."
    echo "------------------------------------------------"
else
    echo "✅ All Models & Binaries Verified!"
fi

if python3 -c "import numpy; import tflite_runtime.interpreter; print('✅ Numpy & TFLite Verified!')" 2>/dev/null; then
    echo "Dependency Check Passed."
else
    echo "❌ CRITICAL: Numpy or TFLite import failed!"
    echo "Try running: sudo apt install python3-numpy python3-tflite-runtime"
    exit 1
fi

echo "================================================"
echo "✅ SETUP COMPLETE! "
echo "Run using:"
echo "source venv/bin/activate"
echo "python3 assistant_voice_tflite.py"
echo "================================================"

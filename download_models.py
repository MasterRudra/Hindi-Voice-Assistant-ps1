import os
import requests
import zipfile
import tarfile
from huggingface_hub import hf_hub_download

def download_file(url, destination):
    if os.path.exists(destination):
        print(f" -> {destination} already exists, skipping.")
        return
    print(f" -> Downloading {url}...")
    response = requests.get(url, stream=True)
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f" ✅ Downloaded {destination}")

def main():
    print("================================================")
    print("   🌐 VANI-SETU MODEL DOWNLOADER (LLM/STT/TTS)  ")
    print("================================================")

    # 1. VOSK Hindi Model (STT)
    vosk_url = "https://alphacephei.com/vosk/models/vosk-model-small-hi-0.22.zip"
    vosk_zip = "vosk-model-small-hi-0.22.zip"
    vosk_dir = "vosk-model-small-hi-0.22"
    
    if not os.path.exists(vosk_dir):
        download_file(vosk_url, vosk_zip)
        print(f" -> Extracting {vosk_zip}...")
        with zipfile.ZipFile(vosk_zip, 'r') as zip_ref:
            zip_ref.extractall(".")
        os.remove(vosk_zip)
    else:
        print(f" ✅ VOSK Model '{vosk_dir}' found.")

    # 2. Sarvam-1 LLM (Brain)
    llm_file = "sarvam-1-q4_k_m.gguf"
    if not os.path.exists(llm_file):
        print(" -> [LLM] Downloading Sarvam-1 GGUF (approx 2GB)...")
        try:
            path = hf_hub_download(
                repo_id='bartowski/sarvam-1-GGUF', 
                filename='sarvam-1-Q4_K_M.gguf', 
                local_dir='.', 
                local_dir_use_symlinks=False
            )
            # Rename for code compatibility
            os.rename('sarvam-1-Q4_K_M.gguf', llm_file)
            print(f" ✅ LLM Model '{llm_file}' ready.")
        except Exception as e:
            print(f" ❌ LLM Download Error: {e}")
    else:
        print(f" ✅ LLM Model '{llm_file}' found.")

    # 3. Piper TTS Model (Rohan)
    tts_base = "https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/rohan/medium/"
    files = ["hi_IN-rohan-medium.onnx", "hi_IN-rohan-medium.onnx.json"]
    for file in files:
        if not os.path.exists(file):
            download_file(tts_base + file, file)
        else:
            print(f" ✅ TTS Model '{file}' found.")

    # 4. Piper Binary (ARM64 for Pi)
    if not os.path.exists("piper"):
        print(" -> [TTS] Downloading Piper Engine (ARM64 for Pi)...")
        piper_url = "https://github.com/rhasspy/piper/releases/download/2023.8.15-2/piper_arm64.tar.gz"
        piper_tar = "piper_arm64.tar.gz"
        try:
            download_file(piper_url, piper_tar)
            print(f" -> Extracting {piper_tar}...")
            with tarfile.open(piper_tar, "r:gz") as tar:
                tar.extractall(".")
            os.remove(piper_tar)
            print(" ✅ Piper Engine ready.")
        except Exception as e:
            print(f" ❌ Piper Engine Download Error: {e}")
    else:
        print(" ✅ Piper Engine found.")

    print("\n================================================")
    print(" 🎉 ALL MODELS READY FOR HACKATHON!")
    print("================================================")

if __name__ == "__main__":
    main()

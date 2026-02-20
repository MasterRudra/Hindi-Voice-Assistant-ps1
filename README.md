# Vāṇī-Setu (वाणी-सेतु)
### Offline Voice Assistant for Rural Regions

Vāṇī-Setu is a prototype offline voice assistant designed for areas with limited internet connectivity. It uses FM radio broadcasts as a data source, transcribing news and information into a local database that users can query using Hindi voice commands.

## Features
- **Offline Processing**: All voice recognition and data retrieval is done locally on a Raspberry Pi.
- **Radio Integration**: Automated harvesting of information from FM news.
- **Hindi Support**: Designed specifically for Hindi language interaction.
- **Low Latency**: Optimized to run efficiently on embedded hardware.

## Technical Implementation
### 1. Model Quantization
To run large AI models on embedded hardware (Raspberry Pi 5), we utilized **4-bit Quantization (GGUF format)**. This reduces the model size and memory footprint without significantly compromising accuracy. This optimization allows the 2GB model to fit within the Pi's RAM and perform inference at acceptable speeds. 

### 2. Unique Technicalities
- **FM-to-Digital Pipe**: Unlike standard internet-dependent assistants, this system uses a hardware radio module to "harvest" real-time data. It transcribes audio fragments in the background and uses keyword-based filtering to categorize information.
- **Concurrent Access (WAL Mode)**: We implemented SQLite's **Write-Ahead Logging** to allow the radio harvester to write data while the assistant is simultaneously reading for user queries, preventing database locks.
- **Multi-threaded Inference**: The system is optimized for the Pi 5's quad-core architecture by using 4-thread execution for the LLM brain, achieving a 60% speedup in response generation.

## Project Structure
- `assistant_voice_tflite.py`: Main assistant application.
- `fm_harvester.py`: Radio transcription service.
- `download_models.py`: Model preparation script.
- `requirements.txt`: List of dependencies.

## Setup Instructions
1. Run `setup_pi.sh` to install dependencies and create a virtual environment.
2. Run `python download_models.py` to get the necessary AI models.
3. Start the harvester (`python fm_harvester.py`) followed by the assistant (`python assistant_voice_tflite.py`).

---


import os
import time
import sqlite3
import sounddevice as sd
import numpy as np
import json
import queue
from smbus2 import SMBus
from vosk import Model, KaldiRecognizer

# --- Configuration ---
I2C_BUS = 1
TEA5787_ADDR = 0x60
DATABASE = "News.db"
MODEL_PATH = "vosk-model-small-hi-0.22"

# --- Hardware Logic (TEA5767) ---
def set_frequency(freq):
    """Sets the FM frequency on TEA5767""
    try:
        bus = SMBus(I2C_BUS)
        # Convert freq (e.g. 91.1) to 14-bit value
        n = int(4 * (freq * 1000000 + 225000) / 32768)
        data = [n >> 8, n & 0xFF, 0x10, 0x10, 0x00]
        bus.write_i2c_block_data(TEA5787_ADDR, data[0], data[1:])
        bus.close()
        print(f"Radio tuned to {freq} MHz")
    except Exception as e:
        print(f"Hardware Error: {e}")

# --- Database Logic ---
def init_db():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    # PREVENT STICKING: Enable WAL mode for concurrent Read/Write
    c.execute("PRAGMA journal_mode=WAL")
    c.execute('''CREATE TABLE IF NOT EXISTS news 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  content TEXT, 
                  category TEXT, 
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

def save_news(text, category="general"):
    # Timeout=10 to prevent "DB is locked" errors
    conn = sqlite3.connect(DATABASE, timeout=10)
    c = conn.cursor()
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("INSERT INTO news (content, category) VALUES (?, ?)", (text, category))
    conn.commit()
    conn.close()

# --- Transcription Logic (VOSK) ---
def start_harvesting(frequency=93.5):
    init_db()
    set_frequency(frequency)

    if not os.path.exists(MODEL_PATH):
        print(f"Error: VOSK model not found at {MODEL_PATH}")
        return

    model = Model(MODEL_PATH)
    rec = KaldiRecognizer(model, 16000)
    q = queue.Queue()
    vol_q = queue.Queue()

    def callback(indata, frames, time, status):
        """Audio processing callback"""
        arr = np.frombuffer(indata, dtype='int16')
        
        # Calculate volume for the meter
        rms = np.sqrt(np.mean(arr.astype(np.float32)**2))
        vol_q.put(rms)
        
        # Clip and resample
        arr = np.clip(arr * 3.0, -32768, 32767).astype('int16')
        q.put(arr[::3].tobytes())

    # Priority: Detect Mic (Fix: Error querying device -1)
    mic_index = None
    try:
        devices = sd.query_devices()
        hat_keywords = ["seeed", "voicecard", "wm8960", "googlevoicehat", "i2s"]
        
        for i, dev in enumerate(devices):
            if any(key in dev['name'].lower() for key in hat_keywords) and dev['max_input_channels'] > 0:
                mic_index = i
                break
        if mic_index is None:
            for i, dev in enumerate(devices):
                if "USB" in dev['name'].upper() and dev['max_input_channels'] > 0:
                    mic_index = i
                    break
        
        if mic_index is not None:
            print(f"🎤 Mic Detected: {devices[mic_index]['name']} (Index: {mic_index})")
        else:
            print("⚠️ No specific Mic detected, using system default.")
    except Exception as e:
        print(f"Mic Detection Error: {e}")

    print(f"--- FM Harvester Started on {frequency}MHz (Listening...) ---")
    
    fragment_buffer = ""
    last_save_time = time.time()
    
    try:
        with sd.RawInputStream(samplerate=48000, blocksize=4800, dtype='int16',
                               channels=1, device=mic_index, callback=callback):
            while True:
                data = q.get()
                vol = vol_q.get()
                
                # Visual Level Meter (ASCII)
                meter_len = int(np.clip(vol / 500, 0, 15))
                meter = "█" * meter_len
                
                # Check for either Final or Partial results to avoid "Sticking"
                is_final = rec.AcceptWaveform(data)
                
                if is_final:
                    res = json.loads(rec.Result())
                    text = res.get('text', '').strip()
                else:
                    res = json.loads(rec.PartialResult())
                    text = res.get('partial', '').strip()
                    # Only process a Partial result if it's getting very long (prevents hanging)
                    if len(text) > 150:
                        # Treat long partial as a final result for saving
                        rec.Reset() # Force clear the VOSK buffer
                    else:
                        print(f"\rLevel: [{meter:<15}] 📻 सुनने की कोशिश: {text[:40]}...", end='', flush=True)
                        continue
                
                if text:
                    fragment_buffer += " " + text
                    
                    # Filtering noise and checking for keywords
                    keywords = ["मौसम", "बारिश", "तापमान", "मंडी", "भाव", "दाम", "रेट", "खबर", "समाचार", "न्यूज"]
                    has_keyword = any(k in fragment_buffer for k in keywords)
                    
                    can_save = False
                    if len(fragment_buffer) > 60:
                        can_save = True
                    elif has_keyword and len(fragment_buffer) > 15:
                        can_save = True
                    elif (time.time() - last_save_time > 20) and len(fragment_buffer) > 25:
                        can_save = True

                    if can_save:
                        processed_text = fragment_buffer.strip()
                        noise_words = ["का", "से", "है", "था", "थी", "को", "में", "पर"]
                        if processed_text in noise_words or len(processed_text) < 10:
                            fragment_buffer = ""
                            continue
                            
                        # Categorize based on keywords
                        category = "general"
                        if any(k in processed_text for k in ["मौसम", "बारिश", "तापमान"]):
                            category = "weather"
                        elif any(k in processed_text for k in ["मंडी", "भाव", "दाम", "रेट"]):
                            category = "mandi"
                        elif any(k in processed_text for k in ["खबर", "समाचार", "न्यूज"]):
                            category = "news"
                        
                        save_news(processed_text, category)
                        print(f"\n✅ CLEAN SAVE: {processed_text[:60]}... [{category}]")
                        fragment_buffer = ""
                        last_save_time = time.time()

                # Safety: If too much time passes with no result, clear stale buffer
                if time.time() - last_save_time > 60:
                    fragment_buffer = ""
                    last_save_time = time.time()

    except KeyboardInterrupt:
        print("\nHarvester Stopped.")
    except Exception as e:
        print(f"ASR Error: {e}")

if __name__ == "__main__":
    # Example: Tuned to a local news station freq
    start_harvesting(91.1)

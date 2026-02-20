import json
import numpy as np
import os
import sys
import time
import subprocess
import sqlite3
import queue
import sounddevice as sd
from vosk import Model, KaldiRecognizer

# AI LLM Connectivity
from llama_cpp import Llama

# TFLite for Intent Recognition
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        tflite = None

# --- Configuration ---
LLM_MODEL_FILE = 'sarvam-1-q4_k_m.gguf'
INTENT_MODEL_FILE = 'intent.tflite'
LABELS_FILE = 'labels.json'
VOSK_MODEL_PATH = "vosk-model-small-hi-0.22"
DATABASE = "News.db"

# --- TFLite Intent Engine (With Robust Keyword Fallback) ---

class IntentEngine:
    def __init__(self, model_path, labels_path):
        self.interpreter = None
        self.labels = {}
        self.vocab = {}
        
        # Try Loading TFLite
        if tflite and os.path.exists(model_path):
            try:
                self.interpreter = tflite.Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print("TFLite Engine Loaded.")
            except Exception as e:
                print(f"⚠️ TFLite Load Error: {e}")

        # Load Labels/Vocab (Critical for both TFLite and Fallback)
        if os.path.exists(labels_path):
            with open(labels_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.labels = data.get('labels', {})
                self.vocab = data.get('vocab', {})
                self.max_length = data.get('max_length', 10)
        
        # Define Keyword Fallback Map (If TFLite fails OR for Fast-Path)
        # EXPANDED FOR HACKATHON: Catch more intents to skip LLM (~2s saved)
        self.keyword_map = {
            "mandi": ["भाव", "रेट", "मंडी", "दाम", "कौनी"],
            "weather": ["मौसम", "पानी", "बारिश", "धूप", "बादल", "ठंड"],
            "news": ["खबर", "समाचार", "न्यूज", "जानकारी"],
            "hello": ["नमस्ते", "राम राम", "प्रणाम", "हाय"],
            "intro": ["कौन हो", "परिचय", "नाम"],
            "pm": ["मोदी", "प्रधानमंत्री", "पीएम"],  # New: Fast-path for PM
            "president": ["राष्ट्रपति", "मुर्मू"],    # New: Fast-path for President
            "capital": ["राजधानी", "दिल्ली"]          # New: Fast-path for Capital
        }

    def classify_keyword(self, text):
        """Fallback Logic: 100% accurate for specific keywords"""
        text = text.lower()
        for intent, keywords in self.keyword_map.items():
            if any(k in text for k in keywords):
                return intent, 0.99  # High confidence
        return "unknown", 0.0

    def classify(self, text):
        # Method 1: TFLite (If available)
        if self.interpreter:
            tokens = text.lower().split()
            sequence = [self.vocab.get(word, self.vocab.get("<OOV>", 1)) for word in tokens]
            if len(sequence) < self.max_length:
                sequence += [0] * (self.max_length - len(sequence))
            else:
                sequence = sequence[:self.max_length]
            
            input_data = np.array([sequence], dtype=np.float32)
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            idx = np.argmax(output_data)
            return self.labels[str(idx)], float(output_data[idx])
        
        # Method 2: Keyword Fallback (Robust/No-Dependency)
        return self.classify_keyword(text)

# --- Sarvam-1 Neural Functions ---

def load_llm():
    if not os.path.exists(LLM_MODEL_FILE):
        print(f"Error: Sarvam-1 model not found at {LLM_MODEL_FILE}")
        sys.exit(1)
    
    print(f"Loading Sarvam-1 Model (GGUF)...")
    # OPTIMIZED: 4 threads for Max Speed on Pi 5. 
    # n_gpu_layers=0 (CPU only), n_ctx=512
    llm = Llama(model_path=LLM_MODEL_FILE, n_ctx=512, n_threads=4, verbose=False)
    return llm

def get_context():
    """Fetches the latest context from News.db (Mandi, Weather, News)"""
    try:
        # Timeout=10 ensures we wait if the Harvester is currently writing
        conn = sqlite3.connect(DATABASE, timeout=10)
        c = conn.cursor()
        c.execute("PRAGMA journal_mode=WAL") # Enable concurrent read
        c.execute("SELECT category, content FROM news ORDER BY timestamp DESC LIMIT 3")
        rows = c.fetchall()
        conn.close()
        if not rows: return ""
        # Limit context to 300 chars to save tokens
        return "\n".join([f"{cat.capitalize()}: {content}" for cat, content in rows])[:300]
    except:
        return ""

def generate_response(prompt, llm, context_data=""):
    """Generates a response using Sarvam-1 LLM with knowledge injection"""
    
    # Static facts for the "Brain"
    facts = "भारत की राजधानी नई दिल्ली। पीएम नरेंद्र मोदी। राष्ट्रपति द्रौपदी मुर्मू। आज़ादी 1947। संविधान 1950। आपातकालीन नंबर 112।"
    
    # Combine static facts with dynamic DB context
    full_context = f"{facts} {context_data}".strip()
    
    # Pattern: Context -> Question -> Answer (Strict Constraint)
    prompt_template = (
        f"संदर्भ: {full_context}\n"
        f"निर्देश: केवल एक वाक्य में उत्तर दें। संदर्भ से बाहर न जाएं।\n"
        f"प्रश्न: {prompt}\n"
        f"उत्तर:"
    )
    
    # Run Inference
    output = llm(
        prompt_template, 
        max_tokens=25, 
        stop=["\n", "प्रश्न:", "।"], 
        echo=False, 
        temperature=0.1,
        repeat_penalty=1.1,
        top_p=0.9
    )
    text = output['choices'][0]['text'].strip()
    
    return text if text else "क्षमा करें, जानकारी नहीं है।"

# --- Voice Functions ---

def speak(text):
    """Neural TTS using Piper (Human-like, High Quality, Fully Offline)"""
    if not text: return
    print(f"Assistant: {text}")
    
    # Path settings
    piper_bin = "./piper/piper"
    model = "hi_IN-rohan-medium.onnx"
    
    try:
        # Initial hardware check
        card_id = None
        try:
            # Find card index for 'USB' or 'Headphones' or 'Audio'
            cards = subprocess.check_output("aplay -l", shell=True).decode()
            for line in cards.split('\n'):
                if "card" in line.lower() and any(k in line.lower() for k in ["usb", "pnp", "audio", "headphones", "headset"]):
                    import re
                    match = re.search(r'card (\d+):', line)
                    if match:
                        card_id = match.group(1)
                        break
        except:
            pass

        # 1. FORCE MAX VOLUME (100%) on the Specific Card
        # Target the detected card specifically (e.g., -c 3 for Card 3)
        amixer_prefix = ["amixer", "-c", card_id] if card_id else ["amixer"]
        subprocess.run(amixer_prefix + ["sset", "PCM", "100%"], capture_output=True)
        subprocess.run(amixer_prefix + ["sset", "Speaker", "100%"], capture_output=True)
        subprocess.run(amixer_prefix + ["sset", "Headphone", "100%"], capture_output=True)
        
        if os.path.exists(piper_bin) and os.path.exists(model):
            device_flag = f"-D plughw:{card_id},0" if card_id else ""
            if card_id:
                print(f"Output: Using Card {card_id}")
            else:
                print(f"Output: Using System Default")

            # 2. Piper produces raw PCM -> pipe directly to aplay
            # Use Shell echo for Hindi UTF-8 compatibility
            cmd = f'echo "{text}" | {piper_bin} --model {model} --output_raw | aplay {device_flag} -r 22050 -f S16_LE -c 1'
            subprocess.run(cmd, shell=True, check=True)
        else:
            # Fallback ONLY if Piper files are missing
            print("⚠️ Piper files missing. Falling back to eSpeak-NG.")
            subprocess.run(["espeak-ng", "-v", "hi", "-s", "140", "-a", "200", text], check=True)
    except Exception as e:
        print(f"TTS Error: {e}")

# --- STT ---

def get_grammar(intent_engine):
    """Generates a constrained grammar from the intent vocabulary for hyper-accurate STT"""
    if not intent_engine or not intent_engine.vocab:
        return None
    
    # Extract all words from training vocab + common conversational words
    words = list(intent_engine.vocab.keys())
    fillers = ["नमस्ते", "बाय", "बंद", "रुको", "अलविदा", "शुक्रिया", "धन्यवाद", "हाँ", "नहीं", "जी"]
    for word in fillers:
        if word not in words: words.append(word)
    
    # Filter out special tokens
    words = [w for w in words if not w.startswith("<") and not w.endswith(">") and len(w) > 0]
    return json.dumps(words, ensure_ascii=False)

def listen(vosk_model, grammar=None):
    """VOSK-based Speech-to-Text with Optional Grammar Fine-Tuning (Standout Optimization)"""
    # Priority: Detect Mic
    try:
        devices = sd.query_devices()
        mic_index = None
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
        if mic_index is None:
            print("\n❌ [No Mic Detected! Listing all available devices...]")
            for i, dev in enumerate(devices):
                print(f"   [{i}] {dev['name']} (In: {dev['max_input_channels']}, Out: {dev['max_output_channels']})")
            print("------------------------------------------------")
            dev_name = "NOT FOUND"
        else:
            dev_name = f"{devices[mic_index]['name']} (Index: {mic_index})"
    except Exception as e:
        print(f"Mic Detection Error: {e}")
        dev_name = "DEFAULT"

    # Initialize recognizer
    if grammar:
        rec = KaldiRecognizer(vosk_model, 16000, grammar)
    else:
        rec = KaldiRecognizer(vosk_model, 16000)
        
    q = queue.Queue(maxsize=100)

    def callback(indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            pass # print(status, file=sys.stderr)
        
        # FIX: Use numpy for slicing (Buffer Support)
        # 48kHz -> 16kHz Downsampling (Take every 3rd int16)
        if hasattr(indata, 'read'): # If indata is a buffer-like object
             arr = np.frombuffer(indata, dtype='int16')
        else: # If indata is already memory-viewable
             arr = np.frombuffer(indata, dtype='int16')
        
        # --- SOFTWARE GAIN (Boost Mic Volume) ---
        # Multiply by 3.0 to boost quiet mics, clip to int16 range to avoid distortion
        arr = np.clip(arr * 3.0, -32768, 32767).astype('int16')
             
        q.put(arr[::3].tobytes())

    try:
        # CAPTURE AT 48000Hz (Native for most USB Mics/HATs)
        # blocksize=14400 -> 300ms latency
        with sd.RawInputStream(samplerate=48000, blocksize=4800 * 3, dtype='int16',
                               channels=1, device=mic_index, callback=callback):
            print(f"\n🎤 सुन रहा हूँ... [Mic: {dev_name}] (48kHz -> 16kHz)")
            while True:
                data = q.get()
                
                is_final = rec.AcceptWaveform(data)
                
                if is_final:
                    result = json.loads(rec.Result())
                    text = result.get('text', '').strip()
                else:
                    result = json.loads(rec.PartialResult())
                    text = result.get('partial', '').strip()
                    
                    # FIX: If the user is speaking for too long without a pause, 
                    # force a result to prevent "stucking"
                    if len(text) > 100:
                        rec.Reset() # Force result
                    else:
                        if text: print(f"\rसुन रहा हूँ: {text}...", end='', flush=True)
                        continue
                
                if text:
                    print(f"\nआप: {text}")
                    return text
    except Exception as e:
        print(f"ASR Error: {e}")
        print("Tip: Check if your mic supports 48000Hz. If not, try 44100Hz in the code.")
    return ""

# --- Decision Logic ---

def fetch_latest_news_db(category=None):
    if not os.path.exists(DATABASE): return None
    try:
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        if category:
            c.execute("SELECT content FROM news WHERE category = ? ORDER BY timestamp DESC LIMIT 1", (category,))
        else:
            c.execute("SELECT content FROM news ORDER BY timestamp DESC LIMIT 1")
        row = c.fetchone()
        conn.close()
        return row[0] if row else None
    except:
        return None


def process_query(user_input, intent_engine, llm):
    """MASTER ARCHITECTURE: Intent -> Routing -> Context -> Sarvam-1"""
    import time
    start_total = time.time()
    
    # 1. INTENT RECOGNITION
    start_intent = time.time()
    intent, confidence = intent_engine.classify(user_input)
    t_intent = time.time() - start_intent
    
    # 2. ROUTING & KNOWLEDGE RETRIEVAL
    start_db = time.time()
    context = ""
    # Define intents that trigger Radio DB lookup
    db_intents = ["weather", "news", "mandi", "crop"] 
    
    if intent in db_intents and confidence > 0.7:
        print(f"🧠 ROUTING: Intent [{intent}] -> Fetching from Radio DB...")
        context = get_context() # Fetch latest from SQLite
    else:
        print(f"🧠 ROUTING: Intent [{intent or 'general'}] -> Direct to Sarvam Brain...")
        
    t_db = time.time() - start_db
    
    # 3. LLM GENERATION (Sarvam-1 Brain)
    start_llm = time.time()
    # Inject Radio Context if retrieved, else LLM uses its static facts
    response = generate_response(user_input, llm, context_data=context)
    t_llm = time.time() - start_llm
    
    t_total = time.time() - start_total
    
    # Performance profiling
    print(f"------------------------------------------------")
    print(f"Inference: {t_intent*1000:.0f}ms | DB: {t_db*1000:.0f}ms | LLM: {t_llm*1000:.0f}ms")
    print(f"Total Latency: {t_total*1000:.0f}ms")
    print(f"------------------------------------------------")
    
    return response

def main():
    print("\n🤖 VANI-SETU: STANDOUT EDITION (PI5 OPTIMIZED)")
    print("=============================================")
    
    intent_engine = IntentEngine(INTENT_MODEL_FILE, LABELS_FILE)
    llm = load_llm()
    
    if not os.path.exists(VOSK_MODEL_PATH):
        print(f"Error: Vosk model missing at {VOSK_MODEL_PATH}")
        sys.exit(1)
    vosk_model = Model(VOSK_MODEL_PATH)
    
    # STT Grammar Optimization
    # grammar = get_grammar(intent_engine) # Disabled for dialect support
    grammar = None
    print("VOSK Optimization: Grammar DISABLED (Full Vocabulary Mode).")
    
    print("\n[All Systems Nominal]")
    print("1 → Voice Mode | 2 → Text Mode")
    choice = input("Select: ").strip()

    while True:
        try:
            if choice == "1":
                user_input = listen(vosk_model, grammar)
            else:
                user_input = input("\nYou: ").strip()

            if not user_input or user_input.lower() in ['exit', 'quit']: break
            
            response = process_query(user_input, intent_engine, llm)
            speak(response)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"System Error: {e}")

if __name__ == "__main__":
    main()

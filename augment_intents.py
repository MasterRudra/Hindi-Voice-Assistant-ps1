import json
import random
import os

# Base rural intent file
INPUT_FILE = 'data/hindi_intents_rural.json'
OUTPUT_FILE = 'data/hindi_intents_augmented.json'

# Rural Dialect Mapping
SUBSTITUTIONS = {
    "क्या": ["का", "कैसन", "के", "कौनी"],
    "भाव": ["रेट", "दाम", "कैसन बिकात है", "कौनी भाव"],
    "नमस्ते": ["राम राम", "प्रणाम", "जय जोहार", "नमस्ते जी"],
    "बताओ": ["बतावा", "बताइये", "जानकारी दियो"],
    "पैसे": ["रुपया", "धन", "दाम"],
    "मंडी": ["बाजार", "हाट"],
    "मौसम": ["मूसम", "बदरा", "पानी"],
    "आज": ["आजु", "अजकू"],
    "खबर": ["समाचार", "न्यूज", "जानकारी"]
}

def augment_text(text):
    words = text.split()
    augmented_versions = set()
    augmented_versions.add(text)
    
    # Simple rule-based augmentation
    for word, alternatives in SUBSTITUTIONS.items():
        if word in text:
            for alt in alternatives:
                augmented_versions.add(text.replace(word, alt))
                
    return list(augmented_versions)

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    augmented_data = []
    print(f"Original samples: {len(data)}")

    for item in data:
        original_text = item['text']
        intent = item['intent']
        
        # Add original
        augmented_data.append(item)
        
        # Generate variations
        variations = augment_text(original_text)
        for var in variations:
            if var != original_text:
                augmented_data.append({"text": var, "intent": intent})

    # Shuffle for better training
    random.shuffle(augmented_data)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(augmented_data, f, ensure_ascii=False, indent=2)

    print(f"Total samples after augmentation: {len(augmented_data)}")
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

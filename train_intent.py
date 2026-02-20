import json
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Configuration
DATA_FILE = 'data/hindi_intents_augmented.json'
MODEL_FILE = 'intent.tflite'
LABELS_FILE = 'labels.json'
VOCAB_FILE = 'vocab.json'
VOCAB_SIZE = 1000
EMBEDDING_DIM = 16
MAX_LENGTH = 10
TRUNC_TYPE = 'post'
PADDING_TYPE = 'post'
OOV_TOK = "<OOV>"
EPOCHS = 500 # Increased epochs for better convergence

def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts = [item['text'] for item in data]
    intents = [item['intent'] for item in data]
    return texts, intents

def train_model():
    print("Loading data...")
    texts, intents = load_data(DATA_FILE)

    # Encode labels
    label_encoder = LabelEncoder()
    training_labels_encoded = label_encoder.fit_transform(intents)
    num_classes = len(np.unique(training_labels_encoded))
    
    # Tokenize text
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOK)
    tokenizer.fit_on_texts(texts)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=MAX_LENGTH, padding=PADDING_TYPE, truncating=TRUNC_TYPE)

    print(f"Data loaded: {len(texts)} samples, {num_classes} classes")
    print(f"Vocabulary size: {len(word_index)}")

    # Build Model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LENGTH),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Train
    print("Training model...")
    history = model.fit(padded_sequences, training_labels_encoded, epochs=EPOCHS, verbose=2)

    # Evaluate
    loss, accuracy = model.evaluate(padded_sequences, training_labels_encoded, verbose=0)
    print(f"Final Training Accuracy: {accuracy*100:.2f}%")

    # Save artifacts
    
    # 1. Save Labels & Vocab
    label_map = {index: label for index, label in enumerate(label_encoder.classes_)}
    artifacts = {
        "labels": label_map,
        "vocab": word_index,
        "max_length": MAX_LENGTH
    }
    with open(LABELS_FILE, 'w', encoding='utf-8') as f:
        json.dump(artifacts, f, ensure_ascii=False, indent=2)
    print(f"Saved labels and vocab to {LABELS_FILE}")

    # 2. Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(MODEL_FILE, 'wb') as f:
        f.write(tflite_model)
    print(f"Saved quantized TFLite model to {MODEL_FILE}")
    print(f"Model Size: {len(tflite_model) / 1024:.2f} KB")

if __name__ == "__main__":
    train_model()

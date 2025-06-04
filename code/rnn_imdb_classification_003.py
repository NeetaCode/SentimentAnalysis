# Install necessary packages
!pip install numpy==1.23.5 gensim==4.3.1 tensorflow==2.12.0 keras==2.12.0 scipy==1.10.1

# ======================= Imports =======================
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence, text
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from gensim.models import Word2Vec

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from time import time

# =================== Environment Check ===================
print(f"NumPy version: {np.__version__}")
print(f"TensorFlow version: {tf.__version__}")

# ================ Download NLTK Data =====================
nltk.download('stopwords')
nltk.download('wordnet')

# ================ Global Settings ========================
MAX_FEATURES = 500
MAXLEN = 50
BATCH_SIZE = 32
NUM_CLASSES = 3
EMBEDDING_DIM = 50

# ================ Text Preprocessing =====================
def preprocess_text(text):
    words = re.findall(r'\b\w+\b', text.lower())
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    return [
        lemmatizer.lemmatize(stemmer.stem(word))
        for word in words if word not in stop_words
    ]

# ============== Load & Preprocess IMDB ===================
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=MAX_FEATURES)
x_train = sequence.pad_sequences(x_train, maxlen=MAXLEN)
x_test = sequence.pad_sequences(x_test, maxlen=MAXLEN)

# One-hot encode labels
y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)

# Convert sequences to text for preprocessing
x_train_text = [' '.join(map(str, seq)) for seq in x_train]
x_test_text = [' '.join(map(str, seq)) for seq in x_test]

# Preprocess text
x_train_preprocessed = [preprocess_text(text) for text in x_train_text]
x_test_preprocessed = [preprocess_text(text) for text in x_test_text]

# ================ Train Word2Vec =========================
w2v_model = Word2Vec(sentences=x_train_preprocessed, vector_size=EMBEDDING_DIM, window=5, min_count=1, workers=4)

# ================== Tokenizer & Embedding Matrix ==========
tokenizer = text.Tokenizer(num_words=MAX_FEATURES)
tokenizer.fit_on_texts(x_train_text + x_test_text)
word_index = tokenizer.word_index

embedding_matrix = np.zeros((MAX_FEATURES, EMBEDDING_DIM))
for word, i in word_index.items():
    if i < MAX_FEATURES and word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]

# Convert text to padded sequences
x_train_seq = tokenizer.texts_to_sequences(x_train_text)
x_test_seq = tokenizer.texts_to_sequences(x_test_text)
x_train_seq = sequence.pad_sequences(x_train_seq, maxlen=MAXLEN)
x_test_seq = sequence.pad_sequences(x_test_seq, maxlen=MAXLEN)

# ===================== Build Model ========================
model = Sequential([
    Embedding(MAX_FEATURES, 32, input_length=MAXLEN),
    SimpleRNN(32, activation='tanh'),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ===================== Train Model ========================
start_time = time()
history = model.fit(
    x_train_seq, y_train,
    epochs=5,
    batch_size=BATCH_SIZE,
    validation_data=(x_test_seq, y_test)
)
print(f"Total training time: {(time() - start_time) / 60:.2f} minutes")

# ================ Plot Training History ===================
def plot_history(history):
    epochs = range(1, len(history.history['accuracy']) + 1)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].plot(epochs, history.history['accuracy'], label='Train Acc')
    axs[0].plot(epochs, history.history['val_accuracy'], label='Val Acc')
    axs[0].set_title('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend()

    axs[1].plot(epochs, history.history['loss'], label='Train Loss')
    axs[1].plot(epochs, history.history['val_loss'], label='Val Loss')
    axs[1].set_title('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend()

    plt.tight_layout()
    plt.show()

plot_history(history)

# =================== Evaluate Model ========================
loss, accuracy = model.evaluate(x_test_seq, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")


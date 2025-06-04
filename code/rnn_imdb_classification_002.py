# Install required packages
!pip install numpy==1.23.5 gensim==4.3.1 tensorflow==2.12.0 keras==2.12.0 scipy==1.10.1

# ============================ #
#         IMPORTS             #
# ============================ #

# Built-in
import time
from time import time
import re

# Data manipulation & plotting
import numpy as np
import matplotlib.pyplot as plt

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence, text
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# NLP - NLTK
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Gensim
from gensim.models import Word2Vec

# Environment check
print("NumPy version:", np.__version__)
print("TensorFlow version:", tf.__version__)

# ============================ #
#        PARAMETERS           #
# ============================ #

max_features = 500   # Vocabulary size
maxlen = 50          # Max sequence length
batch_size = 32
num_classes = 3

# ============================ #
#      LOAD & PREPROCESS      #
# ============================ #

# Load IMDB dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

# One-hot encode labels
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing function
def preprocess_text(text):
    words = re.findall(r'\b\w+\b', text.lower())
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    return [lemmatizer.lemmatize(stemmer.stem(w)) for w in words]

# Convert sequences to strings and preprocess
x_train_text = [' '.join(map(str, seq)) for seq in x_train]
x_test_text = [' '.join(map(str, seq)) for seq in x_test]

x_train_preprocessed = [preprocess_text(text) for text in x_train_text]
x_test_preprocessed = [preprocess_text(text) for text in x_test_text]

# ============================ #
#      WORD2VEC & TOKENIZER   #
# ============================ #

# Train Word2Vec model
w2v_model = Word2Vec(sentences=x_train_preprocessed, vector_size=50, window=5, min_count=1, workers=4)

# Tokenize
tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(x_train_text + x_test_text)
word_index = tokenizer.word_index

# Create embedding matrix
embedding_matrix = np.zeros((max_features, 50))
for word, i in word_index.items():
    if i < max_features and word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]

# Tokenize and pad sequences
x_train_processed = tokenizer.texts_to_sequences(x_train_text)
x_test_processed = tokenizer.texts_to_sequences(x_test_text)
x_train_processed = sequence.pad_sequences(x_train_processed, maxlen=maxlen)
x_test_processed = sequence.pad_sequences(x_test_processed, maxlen=maxlen)

# ============================ #
#     BUILD & TRAIN MODEL     #
# ============================ #

# Define the model
model = Sequential([
    Embedding(max_features, 32, input_length=maxlen),
    SimpleRNN(32),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
start_time = time()
history = model.fit(x_train_processed, y_train, epochs=5, batch_size=batch_size,
                    validation_data=(x_test_processed, y_test))
end_time = time()

# Print training duration
print(f"Total training time: {(end_time - start_time) / 60:.2f} minutes")

# ============================ #
#        VISUALIZATION        #
# ============================ #

epochs = range(1, len(history.history['accuracy']) + 1)
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Accuracy plot
axs[0].plot(epochs, history.history['accuracy'], label='Train')
axs[0].plot(epochs, history.history['val_accuracy'], label='Validation')
axs[0].set_title('Accuracy over Epochs')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Accuracy')
axs[0].legend()

# Loss plot
axs[1].plot(epochs, history.history['loss'], label='Train')
axs[1].plot(epochs, history.history['val_loss'], label='Validation')
axs[1].set_title('Loss over Epochs')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Loss')
axs[1].legend()

plt.tight_layout()
plt.show()

# ============================ #
#          EVALUATION         #
# ============================ #

loss, accuracy = model.evaluate(x_test_processed, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")


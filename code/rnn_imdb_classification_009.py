# STEP 1: Install specific, compatible versions
!pip install numpy==1.23.5 gensim==4.3.1 tensorflow==2.12.0 keras==2.12.0 scipy==1.10.1

# STEP 2: Imports
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import re
import nltk
from time import time
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence, text
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# STEP 3: Environment Setup
print("NumPy version:", np.__version__)
print("TensorFlow version:", tf.__version__)
nltk.download('stopwords')
nltk.download('wordnet')

# STEP 4: Configuration
max_features = 500
maxlen = 50
batch_size = 32
num_classes = 3

# STEP 5: Load and Prepare IMDB Data
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# STEP 6: Text Preprocessing Function
def preprocess_text(text):
    words = re.findall(r'\b\w+\b', text.lower())
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(stemmer.stem(w)) for w in words]

# Convert tokenized sequences to string and preprocess
x_train_text = [' '.join(map(str, seq)) for seq in x_train]
x_test_text = [' '.join(map(str, seq)) for seq in x_test]
x_train_preprocessed = [preprocess_text(text) for text in x_train_text]
x_test_preprocessed = [preprocess_text(text) for text in x_test_text]

# STEP 7: Train Word2Vec Model
w2v_model = Word2Vec(sentences=x_train_preprocessed, vector_size=50, window=5, min_count=1, workers=4)

# STEP 8: Tokenize and Create Embedding Matrix
tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(x_train_text + x_test_text)
word_index = tokenizer.word_index

embedding_matrix = np.zeros((max_features, 50))
for word, i in word_index.items():
    if i < max_features and word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]

# STEP 9: Convert Text to Sequences and Pad
x_train_processed = sequence.pad_sequences(tokenizer.texts_to_sequences(x_train_text), maxlen=maxlen)
x_test_processed = sequence.pad_sequences(tokenizer.texts_to_sequences(x_test_text), maxlen=maxlen)

# Update the 3 GRU Layers with the 3 Bi-DirectionalRNN layers

model_refined = Sequential()
model_refined.add(Embedding(max_features, 32, input_length=maxlen))
model_refined.add(Bidirectional(GRU(64, activation='relu', return_sequences=True)))
model_refined.add(Bidirectional(GRU(32, activation='relu', return_sequences=True)))
model_refined.add(Bidirectional(GRU(16, activation='relu')))
model_refined.add(Dropout(0.2))
model_refined.add(Dense(num_classes, activation='softmax'))

# Compile the model
model_refined.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Measure time taken for training
start_time = time()

# Train the model
history = model_refined.fit(x_train_processed, y_train, epochs=5, batch_size=32, validation_data=(x_test_processed, y_test))

# Calculate and print the total time taken
end_time = time()
total_time = end_time - start_time
print(f"Total time taken to train the model: {total_time / 60:.2f} minutes")

# Plot training history
plt.figure(figsize=(12, 6))

# Plot Training and Validation Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

# Plot Training and Validation Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.tight_layout()
plt.show()

# Evaluate the model on the test set
loss_refined, accuracy_refined = model_refined.evaluate(x_test_processed, y_test)
print(f"Refined Model - Test loss: {loss_refined:.4f}, Test accuracy: {accuracy_refined:.4f}")


!pip install transformers datasets tokenizers
!pip install -U datasets

import matplotlib.pyplot as plt
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import load_dataset
import time  # For time tracking

# Load sentiment dataset from Hugging Face
dataset = load_dataset("imdb", cache_dir="./hf_cache")  # <-- Fixes the ValueError

# Extract features and labels
texts = dataset['train']['text']
labels = dataset['train']['label']

# Convert labels to numerical format
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Tokenize using BERT tokenizer with a maximum sequence length
max_sequence_length = 128
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenized_inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_sequence_length, return_tensors="tf")

# Split the data
x_train, x_test, y_train, y_test = train_test_split(
    tokenized_inputs['input_ids'].numpy(), labels_encoded, test_size=0.2, random_state=42
)

# Define number of classes (binary in this case)
num_classes = 1  # Use 1 for binary classification with sigmoid

# Set mixed precision policy
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Build custom model
model_custom = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.get_vocab()), output_dim=32, input_length=max_sequence_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='sigmoid')  # Changed to sigmoid for binary classification
])

# Compile the model
model_custom.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
start_time = time.time()
history = model_custom.fit(x_train, y_train, epochs=3, batch_size=32, validation_split=0.2, verbose=1)
end_time = time.time()

# Evaluate the model
loss_finetuned, accuracy_finetuned = model_custom.evaluate(x_test, y_test)
print(f"Fine-tuned Model - Test loss: {loss_finetuned:.4f}, Test accuracy: {accuracy_finetuned:.4f}")

# Print training time
total_time = end_time - start_time
print(f"Total time taken to train the model: {total_time / 60:.2f} minutes")

# Plot training history
plt.figure(figsize=(12, 6))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.tight_layout()
plt.show()


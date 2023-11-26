import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Flatten
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('data.csv')

# Preprocess data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['Sentence'])
total_words = len(tokenizer.word_index) + 1

X = tokenizer.texts_to_sequences(df['Sentence'])
X = pad_sequences(X, padding='post')

# Use 'Sentiment' directly as the target variable
labels = df['Sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2})

# Convert labels to one-hot encoding
y = tf.keras.utils.to_categorical(labels, num_classes=4)  # Update num_classes to 4 if you have the neutral class

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build neural network model
model = Sequential()
model.add(Embedding(total_words, 100, input_length=X.shape[1]))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Bidirectional(LSTM(64)))
model.add(Flatten())  # Add Flatten layer to flatten the output
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(4, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Create a TensorBoard callback
tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), callbacks=[tensorboard_callback])

# Save the model to a file (.h5)
model.save('sentiment_model.h5')

# Load the saved model
loaded_model = tf.keras.models.load_model('sentiment_model.h5')

# Evaluate the model
loss, accuracy = loaded_model.evaluate(X_test, y_test)
print(f'Evaluation - Loss: {loss}, Accuracy: {accuracy}')

# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training History')
plt.grid(True)

# Save the plot as an image file
plt.savefig('training_history.png')

# Show the plot
plt.show()
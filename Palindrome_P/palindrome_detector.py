import tensorflow as tf
import random
import string
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def encode_words(words, char_to_int, max_len):
    encoded = [[char_to_int[ch] for ch in word] for word in words]
    padded = pad_sequences(encoded, maxlen=max_len, padding='post')
    return padded

def is_palindrome(word):
    return word == word[::-1]

def generate_palindrome(length):
    half = ''.join(random.choices(string.ascii_lowercase, k=length // 2))
    if length % 2 == 0:
        return half + half[::-1]
    else:
        middle = random.choice(string.ascii_lowercase)
        return half + middle + half[::-1]

def generate_non_palindrome(length):
    while True:
        word = ''.join(random.choices(string.ascii_lowercase, k=length))
        if not is_palindrome(word):
            return word

def create_dataset(n_train, n_test, min_len, max_len):
    train_data = []
    test_data = []

    for _ in range(n_train // 2):
        length = random.randint(min_len, max_len)
        train_data.append((generate_palindrome(length), 1))
        train_data.append((generate_non_palindrome(length), 0))

    for _ in range(n_test // 2):
        length = random.randint(min_len, max_len)
        test_data.append((generate_palindrome(length), 1))
        test_data.append((generate_non_palindrome(length), 0))

    random.shuffle(train_data)
    random.shuffle(test_data)

    X_train, y_train = zip(*train_data)
    X_test, y_test = zip(*test_data)

    return list(X_train), list(y_train), list(X_test), list(y_test)

X_train, y_train, X_test, y_test = create_dataset(60000,10000,3,16)

vocab = list(string.ascii_lowercase)
char_to_int = {ch: i + 1 for i, ch in enumerate(vocab)}
int_to_char = {i: ch for ch, i in char_to_int.items()}

X_train_encoded = encode_words(X_train, char_to_int, 16)
X_test_encoded = encode_words(X_test, char_to_int, 16)

y_train = list(y_train)
y_test = list(y_test)

X_train_encoded = np.array(X_train_encoded, dtype=np.int32)
X_test_encoded = np.array(X_test_encoded, dtype=np.int32)

y_train = np.array(y_train, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)

vocab_size = len(char_to_int) + 1  # +1 for padding (0)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=16, input_length=16),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train_encoded, y_train, epochs=15, batch_size=32, validation_data=(X_test_encoded, y_test))

loss, accuracy = model.evaluate(X_test_encoded, y_test)
print(f"Test Accuracy: {accuracy: .4f}")
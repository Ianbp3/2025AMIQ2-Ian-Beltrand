import tensorflow as tf
import pandas as pd
import numpy as np

df_usnames = pd.read_csv("NationalNames.csv")

names = df_usnames["Name"].unique()
print(f"Total unique names: {len(names)}")

all_text = "\n".join(names)
vocab = sorted(set(all_text))

char2idx = {char: idx for idx, char in enumerate(vocab)}
idx2char = {idx: char for char, idx in char2idx.items()}

encoded_names = [ [char2idx[char] for char in name] for name in names ]

seq_length = 10
input_seqs = []
target_seqs = []

for name in encoded_names:
    for i in range(1, len(name)):
        input_seq = name[:i]
        target_seq = name[i]
        input_seq = [0] * (seq_length - len(input_seq)) + input_seq[-seq_length:]
        input_seqs.append(input_seq)
        target_seqs.append(target_seq)

X = np.array(input_seqs)
y = np.array(target_seqs)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(vocab), 64, input_length=seq_length),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(len(vocab), activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X, y, epochs=20, batch_size=64)


def generate_name(model, start_str="", max_length=10):
    input_seq = [char2idx.get(c, 0) for c in start_str]
    input_seq = [0] * (seq_length - len(input_seq)) + input_seq[-seq_length:]

    generated = list(start_str)

    for _ in range(max_length):
        input_array = np.array([input_seq])
        preds = model.predict(input_array, verbose=0)[0]
        next_idx = np.random.choice(len(vocab), p=preds)
        next_char = idx2char[next_idx]
        generated.append(next_char)
        input_seq = input_seq[1:] + [next_idx]
        if next_char == "\n":
            break
    return "".join(generated).strip()

print(generate_name(model, 'I', 3))
print(generate_name(model, 'S', 8))
print(generate_name(model, 'L', 6))
print(generate_name(model, 'D', 6))
print(generate_name(model, 'A'))

import tensorflow as tf
import numpy as np
import random
import re
from datasets import load_dataset


ds = load_dataset("cosimoiaia/RTTTL-Ringtones")

def extract_melody(rtttl_string):
    parts = rtttl_string.split(":", 2)
    if len(parts) < 3:
        return None
    return parts[2]

melodies = []
for x in ds['train']:
    melody = extract_melody(x['text'])
    if melody:
        melodies.append(melody)

note_pattern = re.compile(r"^(1|2|4|8|16|32)?(a|b|c|d|e|f|g|p)(#|b)?(\.)?(4|5|6|7)?$", re.IGNORECASE)

def is_valid_note(token):
    return bool(note_pattern.match(token.strip()))

all_notes = []
for melody in melodies:
    notes = melody.split(",")
    valid_notes = [n.strip() for n in notes if is_valid_note(n)]
    all_notes.extend(valid_notes)

vocab = sorted(set(all_notes))
note_to_index = {note: i for i, note in enumerate(vocab)}
index_to_note = {i: note for note, i in note_to_index.items()}

encoded_notes = [note_to_index[note] for note in all_notes]

seq_length = 15

x_train = []
y_train = []

for i in range(len(encoded_notes) - seq_length):
    x_train.append(encoded_notes[i:i+seq_length])
    y_train.append(encoded_notes[i+seq_length])

x_train = np.array(x_train)
y_train = np.array(y_train)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(vocab), 64, input_length=seq_length),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.LSTM(64, return_sequences=False),
    tf.keras.layers.Dense(len(vocab), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

model.fit(x_train, y_train, epochs=20, batch_size=64)


def generate_notes(model, seed_seq, note_to_index, index_to_note, length=50):
    result = list(seed_seq)
    current_seq = [note_to_index[n] for n in seed_seq]

    for _ in range(length):
        input_seq = np.array([current_seq[-seq_length:]])
        probs = model.predict(input_seq, verbose=0)[0]
        next_index = np.random.choice(len(probs), p=probs)
        next_note = index_to_note[next_index]

        result.append(next_note)
        current_seq.append(next_index)

    return result
seed_indices = random.randint(0, len(x_train) - 1)
seed_seq = [index_to_note[i] for i in x_train[seed_indices]]
generated_sequence1 = generate_notes(model, seed_seq, note_to_index, index_to_note)
seed_indices = random.randint(0, len(x_train) - 1)
seed_seq = [index_to_note[i] for i in x_train[seed_indices]]
generated_sequence2 = generate_notes(model, seed_seq, note_to_index, index_to_note)
seed_indices = random.randint(0, len(x_train) - 1)
seed_seq = [index_to_note[i] for i in x_train[seed_indices]]
generated_sequence3 = generate_notes(model, seed_seq, note_to_index, index_to_note)
seed_indices = random.randint(0, len(x_train) - 1)
seed_seq = [index_to_note[i] for i in x_train[seed_indices]]
generated_sequence4 = generate_notes(model, seed_seq, note_to_index, index_to_note)

print("Generated melody:")
print(",".join(generated_sequence1))
print(",".join(generated_sequence2))
print(",".join(generated_sequence3))
print(",".join(generated_sequence4))

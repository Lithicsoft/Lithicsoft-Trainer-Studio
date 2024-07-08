# License: Apache-2.0
 #
 # rnn_text_generation/trainer.py: Trainer for RNN Text Generation model in Trainer Studio
 #
 # (C) Copyright 2024 Lithicsoft Organization
 # Author: Bui Nguyen Tan Sang <tansangbuinguyen52@gmail.com>
 #

import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import time
from dotenv import load_dotenv

load_dotenv()

dir_path = os.path.dirname(os.path.realpath(__file__))

VOCAB_SIZE = int(os.getenv('VOCAB_SIZE'))
EMBEDDING_DIM = int(os.getenv('EMBEDDING_DIM'))
RNN_UNITS = int(os.getenv('RNN_UNITS'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
BUFFER_SIZE = int(os.getenv('BUFFER_SIZE'))
EPOCHS = int(os.getenv('EPOCHS'))
SEQ_LENGTH = int(os.getenv('SEQ_LENGTH'))
TEMPERATURE = float(os.getenv('TEMPERATURE'))
CHECKPOINT_DIR = f"{dir_path}\\{os.getenv('CHECKPOINT_DIR')}"
INPUT_DIR = f"{dir_path}\\{os.getenv('INPUT_DIR')}"
START_STRING = os.getenv('START_STRING')
LOG_DIR = f"{dir_path}\\{os.getenv('LOG_DIR')}"
RANGE_TEST = int(os.getenv('RANGE_TEST'))
SAVE_WEIGHTS_ONLY = bool(os.getenv('SAVE_WEIGHTS_ONLY'))

def read_text_files(directory):
    text = ''
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                with open(os.path.join(root, file), 'rb') as f:
                    text += f.read().decode(encoding='utf-8')
    return text

text = read_text_files(INPUT_DIR)

vocab = sorted(set(text))

ids_from_chars = tf.keras.layers.StringLookup(vocabulary=list(vocab), mask_token=None)
chars_from_ids = tf.keras.layers.StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)
all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

sequences = ids_dataset.batch(SEQ_LENGTH+1, drop_remainder=True)

def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

dataset = (dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE))

class MyModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super(MyModel, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(rnn_units, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = self.embedding(inputs, training=training)
        if states is None:
            batch_size = tf.shape(inputs)[0]
            states = [tf.zeros((batch_size, self.gru.units))]
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)
        return (x, states) if return_state else x

model = MyModel(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, rnn_units=RNN_UNITS)

loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss=loss)

checkpoint_prefix = os.path.join(CHECKPOINT_DIR, "ckpt_{epoch}.weights.h5")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=SAVE_WEIGHTS_ONLY)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1)

history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback, tensorboard_callback], verbose=2)

tf.saved_model.save(model, 'one_step')

class OneStep(tf.keras.Model):
    def __init__(self, model, chars_from_ids, ids_from_chars, temperature=TEMPERATURE):
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars

        skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
        sparse_mask = tf.SparseTensor(values=[-float('inf')]*len(skip_ids), indices=skip_ids, dense_shape=[len(ids_from_chars.get_vocabulary())])
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

    @tf.function
    def generate_one_step(self, inputs, states=None):
        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        input_ids = self.ids_from_chars(input_chars).to_tensor()
        predicted_logits, states = self.model(inputs=input_ids, states=states, return_state=True)
        predicted_logits = predicted_logits[:, -1, :] / self.temperature
        predicted_logits = predicted_logits + self.prediction_mask
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)
        predicted_chars = self.chars_from_ids(predicted_ids)
        return predicted_chars, states

one_step_model = OneStep(model, chars_from_ids, ids_from_chars)

states = None
next_char = tf.constant([START_STRING])
result = [next_char]

for n in range(RANGE_TEST):
    next_char, states = one_step_model.generate_one_step(next_char, states=states)
    result.append(next_char)

result = tf.strings.join(result)
print(result[0].numpy().decode('utf-8', errors='ignore'))

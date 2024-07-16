# License: Apache-2.0
 #
 # rnn_text_generation/tester.py: Tester for RNN Text Generation model in Trainer Studio
 #
 # (C) Copyright 2024 Lithicsoft Organization
 # Author: Bui Nguyen Tan Sang <tansangbuinguyen52@gmail.com>
 #

import tensorflow as tf
import numpy as np
import os
import time
from dotenv import load_dotenv

load_dotenv()

INPUT_DIR = f"{dir_path}\\{os.getenv('INPUT_DIR')}"
TEMPERATURE = float(os.getenv('TEMPERATURE'))
RANGE_TEST = int(os.getenv('RANGE_TEST'))

START_STRING = input("Prompt: ")

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

model = tf.keras.models.load_model('outputs/one_step')
one_step_model = OneStep(model, chars_from_ids, ids_from_chars)

states = None
next_char = tf.constant([START_STRING])
result = [next_char]

for n in range(RANGE_TEST):
    next_char, states = one_step_model.generate_one_step(next_char, states=states)
    result.append(next_char)

result = tf.strings.join(result)
print(result[0].numpy().decode('utf-8', errors='ignore'))

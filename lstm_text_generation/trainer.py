# License: Apache-2.0
 #
 # lstm_text_generation/trainer.py: Trainer for LSTM Text Generation model in Trainer Studio
 #
 # (C) Copyright 2024 Lithicsoft Organization
 # Author: Bui Nguyen Tan Sang <tansangbuinguyen52@gmail.com>
 #

import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from dotenv import load_dotenv

load_dotenv()

dir_path = os.path.dirname(os.path.realpath(__file__))

INPUT_DIR = f"{dir_path}\\{os.getenv('INPUT_DIR')}"
SEQ_LENGHT = int(os.getenv('SEQ_LENGTH'))
EPOCHS = int(os.getenv('EPOCHS'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
DROPOUT = float(os.getenv('DROPOUT'))
INPUT_SIZE = int(os.getenv('INPUT_SIZE'))
HIDDEN_SIZE = int(os.getenv('HIDDEN_SIZE'))
NUM_LAYER = int(os.getenv('NUM_LAYERS'))
BATCH_FIRST = bool(os.getenv('BATCH_FIRST'))
RANGE_TEST = int(os.getenv('RANGE_TEST'))

def read_text_files(directory):
    text = ''
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                with open(os.path.join(root, file), 'rb') as f:
                    text += f.read().decode(encoding='utf-8')
    return text

raw_text = read_text_files(INPUT_DIR)
raw_text = raw_text.lower()

chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)

seq_length = SEQ_LENGHT
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)

X = torch.tensor(dataX, dtype=torch.float32).reshape(n_patterns, seq_length, 1)
X = X / float(n_vocab)
y = torch.tensor(dataY)

class CharModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYER, batch_first=BATCH_FIRST, dropout=DROPOUT)
        self.dropout = nn.Dropout(DROPOUT)
        self.linear = nn.Linear(HIDDEN_SIZE, n_vocab)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.linear(self.dropout(x))
        return x

n_epochs = EPOCHS
batch_size = BATCH_SIZE
model = CharModel()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss(reduction="sum")
loader = data.DataLoader(data.TensorDataset(X, y), shuffle=True, batch_size=batch_size)

best_model = None
best_loss = np.inf
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch.to(device))
        loss = loss_fn(y_pred, y_batch.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    loss = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            y_pred = model(X_batch.to(device))
            loss += loss_fn(y_pred, y_batch.to(device))
        if loss < best_loss:
            best_loss = loss
            best_model = model.state_dict()
        print("Epoch %d: Cross-entropy: %.4f" % (epoch, loss))

torch.save([best_model, char_to_int], "model.pth")

best_model, char_to_int = torch.load("model.pth")
n_vocab = len(char_to_int)
int_to_char = dict((i, c) for c, i in char_to_int.items())
model.load_state_dict(best_model)

seq_length = SEQ_LENGHT
raw_text = read_text_files(INPUT_DIR)
raw_text = raw_text.lower()
start = np.random.randint(0, len(raw_text)-seq_length)
prompt = raw_text[start:start+seq_length]
pattern = [char_to_int[c] for c in prompt]

model.eval()
print('Prompt: "%s"' % prompt)
with torch.no_grad():
    for i in range(RANGE_TEST):
        x = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
        x = torch.tensor(x, dtype=torch.float32)
        prediction = model(x.to(device))
        index = int(prediction.argmax())
        result = int_to_char[index]
        print(result, end="")
        pattern.append(index)
        pattern = pattern[1:]
print()
print("Done.")

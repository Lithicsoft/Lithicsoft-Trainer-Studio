# License: Apache-2.0
 #
 # lstm_text_generation/tester.py: Tester for LSTM Text Generation model in Trainer Studio
 #
 # (C) Copyright 2024 Lithicsoft Organization
 # Author: Bui Nguyen Tan Sang <tansangbuinguyen52@gmail.com>
 #

import numpy as np
import torch
import os
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

model = CharModel()

best_model, char_to_int = torch.load(f"{dir_path}/outputs/model.pth")
n_vocab = len(char_to_int)
int_to_char = dict((i, c) for c, i in char_to_int.items())
model.load_state_dict(best_model)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

seq_length = SEQ_LENGTH
raw_text = input("Prompt: ")
prompt = raw_text.lower()
start = np.random.randint(0, len(raw_text)-seq_length)
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

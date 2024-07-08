import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from dotenv import load_dotenv

load_dotenv()

RANGE_TEST = int(os.getenv('RANGE_TEST'))
SEQ_LENGHT = int(os.getenv('SEQ_LENGTH'))

filename = "prompt.txt"
seq_length = SEQ_LENGTH
raw_text = open(filename, 'r', encoding='utf-8').read()
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

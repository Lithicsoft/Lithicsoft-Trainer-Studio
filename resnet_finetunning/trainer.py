# License: Apache-2.0
 #
 # resnet_finetuning/trainer.py: Trainer for ResNet (CNN) Finetuning model in Trainer Studio
 #
 # (C) Copyright 2024 Lithicsoft Organization
 # Author: Bui Nguyen Tan Sang <tansangbuinguyen52@gmail.com>
 #

import os
from dotenv import load_dotenv
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import torch.nn as nn
import torch.optim as optim
import torch
import logging 

logging.basicConfig(level=logging.INFO)
load_dotenv()

BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 0.001))
MAX_EPOCHS = int(os.getenv("MAX_EPOCHS", 5))
USE_GPU = os.getenv("USE_GPU", "false").lower() == "true"
DATA_DIR = os.getenv("DATA_DIR", ".\\datasets")
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", ".\\checkpoints") 

class CNNFineTuner(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10) 

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        self.log("train_loss", loss) 
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=LEARNING_RATE)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = datasets.CIFAR10(root=DATA_DIR, train=True, transform=transform, download=True)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

checkpoint_callback = ModelCheckpoint(
    dirpath=CHECKPOINT_DIR,
    filename="cnn_fine_tuner-{epoch:02d}-{train_loss:.2f}",
    save_top_k=1,
    monitor="train_loss",
    mode="min", 
)

trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator="gpu" if USE_GPU else "cpu",
    devices=1,
    callbacks=[checkpoint_callback]
)

model = CNNFineTuner()
trainer.fit(model, train_loader)

torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "cnn_fine_tuner_state_dict.pth"))

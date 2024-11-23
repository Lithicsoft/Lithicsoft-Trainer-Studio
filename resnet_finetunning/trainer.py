import os
from dotenv import load_dotenv
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import torch.nn as nn
import torch.optim as optim

load_dotenv()

BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 0.001))
MAX_EPOCHS = int(os.getenv("MAX_EPOCHS", 5))
USE_GPU = os.getenv("USE_GPU", "false").lower() == "true"
DATA_DIR = os.getenv("DATA_DIR", "./datasets")

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
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=LEARNING_RATE)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = datasets.CIFAR10(root=DATA_DIR, train=True, transform=transform, download=True)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator="gpu" if USE_GPU else "cpu",
    devices=1
)

model = CNNFineTuner()
trainer.fit(model, train_loader)

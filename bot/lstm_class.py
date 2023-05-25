import os
from torch import optim, nn, utils, Tensor
from torchvision.transforms import ToTensor
import lightning.pytorch as pl
import torch.nn as nn

# Класс модели (через лайтнинг!)
# LSTM 3 слоя размера 1024 -> два линейных слоя 

class LSTM(pl.LightningModule):
    def __init__(self, n_vocab):
        super().__init__()        
        self.lstm = nn.LSTM(input_size=1, hidden_size=1024, num_layers=3, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.linear_1 = nn.Linear(1024, 512)
        self.relu = nn.ELU()
        self.linear_2 = nn.Linear(512, n_vocab)
        
        self.loss_fn = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.forward(x)
        
        loss = self.loss_fn(x, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = self.forward(x)
        
        loss = self.loss_fn(x, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        
    def forward(self, x):
        x = x.float().unsqueeze(-1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.linear_1(self.dropout(x))
        x = self.linear_2(self.relu(self.dropout(x)))
        return x

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=2e-4)
        return optimizer
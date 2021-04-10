#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FileName:	Model.py
Author:	Zhu Zhan
Email:	henry664650770@gmail.com
Date:		2021-04-09 19:50:26
"""

import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
from torch.nn import Sequential


class CNN(nn.Module):
    def __init__(self, num_classes=1):
        super(CNN, self).__init__()
        self.cnn_layers = Sequential(
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(1, 32, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(32, 64, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(64, 128, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, 256, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(256, 256, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(256, 128, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, 64, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64, 1000),
            nn.Dropout(0.75),
            nn.Linear(1000, num_classes))

    def forward(self, x):
        x = self.cnn_layers(x)
        return x


def create_model(model_name, model_hparams):
    model_dict = {"CNN": CNN}
    if model_name in model_dict:
        return model_dict[model_name](**model_hparams)
    else:
        assert False, f"Unknown model name '{model_name}'. Available models are: {str(model_dict.keys())}"


class Trainer(pl.LightningModule):
    def __init__(self, model_name, model_hparams, optimizer_name, optimizer_hparams):
        """
        Inputs:
            model_name - Name of the model to run. Used for creating the model (see function below)
            model_hparams - Hyperparameters for the model, as dictionary.
            optimizer_name - Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
        """
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        self.model = create_model(model_name, model_hparams)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, imgs):
        return self.model(imgs)

    def configure_optimizers(self):
        if self.hparams.optimizer_name == "Adam":
            # AdamW is Adam with a correct implementation of weight decay
            # see here for details: https://arxiv.org/pdf/1711.05101.pdf
            optimizer = optim.AdamW(self.parameters(), **self.hparams.optimizer_hparams)
        elif self.hparams.optimizer_name == "SGD":
            optimizer = optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
        else:
            assert False, f"Unknown optimizer: '{self.hparams.optimizer_name}'"

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.criterion(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        self.log('train_loss', loss)
        # Return tensor to call ".backward" on
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches)
        self.log('val_acc', acc)

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        self.log('test_acc', acc)

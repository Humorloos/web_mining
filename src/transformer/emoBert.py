import logging

import pandas as pd
import pytorch_lightning as pl
import torch
import torchmetrics
from torch import nn
from torch.utils.data import DataLoader
from transformers import RobertaModel, \
    RobertaTokenizer

from src.transformer.datasets.EmoticonTrainValSplit import get_emoticon_train_val_split


class EmoBERT(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        # todo: batch size val should be as large as possible
        self.batch_size_val = 50

        self.weight_decay = config['weight_decay']
        self.lr = config['lr']
        self.optimizer = config['optimizer']
        self.batch_size_train = config['batch_size_train']
        self.num_workers = config['num_workers']

        logging.info('Initializing EmoBERT Model')
        self.base_model = RobertaModel.from_pretrained("roberta-base")
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.dropout = nn.Dropout(config['dropout_prob'])
        self.linear = nn.Linear(
            in_features=self.base_model.config.hidden_size,
            out_features=1)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()
        self.accuracy = torchmetrics.Accuracy()

        logging.info('Loading training dataset')
        self.train_set, self.val_set = get_emoticon_train_val_split()

    def train_dataloader(self):
        return DataLoader(dataset=self.train_set,
                          shuffle=True,
                          batch_size=self.batch_size_train,
                          collate_fn=self.custom_collate,
                          num_workers=self.num_workers,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_set,
                          shuffle=False,
                          batch_size=self.batch_size_val,
                          collate_fn=self.custom_collate,
                          num_workers=self.num_workers,
                          pin_memory=True)

    def forward(self, batch):
        embeddings = self.base_model(**batch).pooler_output
        hidden = self.dropout(embeddings)
        activation = torch.ravel(self.linear(hidden))
        return self.sigmoid(activation)

    def training_step(self, batch, batch_idx):
        tokens, y = batch
        y_hat = self(tokens)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True,
                 logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        tokens, y = batch
        y_hat = self(tokens)
        loss = self.loss(y_hat, y)
        accuracy = self.accuracy(y_hat, y.int())
        return {'loss': loss, 'accuracy': accuracy}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("ptl/val_loss", avg_loss)
        avg_accuracy = torch.stack([x["accuracy"] for x in outputs]).mean()
        self.log("ptl/val_accuracy", avg_accuracy)

    def configure_optimizers(self):
        return self.optimizer(params=self.parameters(),
                              lr=self.lr,
                              weight_decay=self.weight_decay)

    def custom_collate(self, batch):
        """
        Prepares loaded batches from dataset for model input
        :param batch: batch loaded from dataset, consisting of list of pandas
        series with text and polarity
        :return: input_ids and masking information from RoBERTa tokenizer, and
        """
        batch = pd.DataFrame(batch)
        tokens = self.tokenizer(
            batch['text'].tolist(),
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        y = torch.from_numpy(batch['polarity'].values.astype('float32'))
        return tokens, y

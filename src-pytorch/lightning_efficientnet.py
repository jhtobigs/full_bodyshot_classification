import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim 

from utils import get_dataloader, get_dataset
from efficientnet.efficientnet_pytorch.model import EfficientNet


class CustomEfficientNet(pl.LightningModule):
    def __init__(self, hparams):
        super(CustomEfficientNet, self).__init__()
        self.hparams = hparams
        self.path = hparams['path']
        self.lr = hparams['lr']
        self.batch_size = hparams['batch_size']
        self.pretrain = True if hparams['pretrain'].lower() == 'true' else False
        
        mode = 'efficientnet-b0'
        if self.pretrain:
            self.model = EfficientNet.from_pretrained(mode)
        else:
            self.model = EfficientNet.from_name(mode)
        self.additional_fc = nn.Sequential(
            nn.Linear(self.model._fc.out_features, 2)
        )

    def forward(self, x):
        if self.pretrain:
            x = self.model.extract_features(x)

            x = self.model._avg_pooling(x)
            x = x.view(x.size(0), -1)
            x = self.model._dropout(x)
            x = self.model._fc(x)
            return x

        x = self.model(x)
        x = self.additional_fc(x)
        return x

    def train_dataloader(self):
        train_dset = get_dataset(self.path, 'train')
        train_loader = get_dataloader(train_dset, self.batch_size)
        return train_loader

    def val_dataloader(self):
        val_dset = get_dataset(self.path, 'valid')
        val_loader = get_dataloader(val_dset, self.batch_size)
        return val_loader

    def test_dataloader(self):
        test_dset = get_dataset(self.path, 'test')
        test_loader = get_dataloader(test_dset, self.batch_size)
        return test_loader

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        data, target = batch
        y_hat = self(data)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(y_hat, target)

        # acc
        correct = 0
        _, predicted = torch.max(y_hat, 1)
        correct += predicted.eq(target).sum().item()
        accuracy = 100*(correct/target.size(0))

        return {'loss':loss, 'trn_acc':torch.tensor(accuracy)}

    def training_epoch_end(self, outputs):
        train_loss_mean = torch.stack([x['loss'] for x in outputs]).mean()
        train_acc_mean = torch.stack([x['trn_acc'] for x in outputs]).mean()
        log = {'avg_trn_loss':train_loss_mean, 'avg_trn_acc':train_acc_mean}
        return {'log':log, 'trn_loss':train_loss_mean, 'trn_acc':train_acc_mean}

    def validation_step(self, batch, batch_idx):
        data, target = batch
        y_hat = self(data)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(y_hat, target)

        # acc
        correct = 0
        _, predicted = torch.max(y_hat, 1)
        correct += predicted.eq(target).sum().item()
        accuracy = 100*(correct/target.size(0))

        return {'val_loss':loss, 'val_acc':torch.tensor(accuracy)}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_acc_mean = torch.stack([x['val_acc'] for x in outputs]).mean()
        tensorboard_log = {'avg_val_loss':val_loss_mean, 'avg_val_acc':val_acc_mean}
        return {'val_loss':val_loss_mean, 'val_acc':val_acc_mean, 'log':tensorboard_log}

    def test_step(self, batch, batch_idx):
        data, target = batch
        y_hat = self(data)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(y_hat, target)

        # acc
        correct = 0
        _, predicted = torch.max(y_hat, 1)
        correct += predicted.eq(target).sum().item()
        accuracy = 100*(correct/target.size(0))

        return {'test_loss':loss, 'test_acc':torch.tensor(accuracy)}
    
    def test_epoch_end(self, outputs):
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        test_acc_mean = torch.stack([x['test_acc'] for x in outputs]).mean()
        tensorboard_log = {'avg_test_loss':test_loss_mean, 'avg_test_acc':test_acc_mean}
        return {'test_loss':test_loss_mean, 'test_acc':test_acc_mean, 'log':tensorboard_log}

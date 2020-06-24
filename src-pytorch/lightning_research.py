# PyTorch Lightning Module (Research Part)
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from utils import get_dataloader, get_dataset


class ResNetBaseline(pl.LightningModule):
    def __init__(self, args):
        super(ResNetBaseline, self).__init__()
        self.args = args
        self.path = args.path
        self.lr = args.lr        
        self.batch_size = args.batch_size
        self.num_classes = args.num_classes

        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1] # right after flatten
        self.extractor = nn.Sequential(
                            *modules
                        )
        self.classifier = nn.Linear(resnet.fc.in_features, self.num_classes)
        resnet = None
    
    def forward(self, x):
        feature = self.extractor(x)
        feature = feature.view(feature.size(0), -1)
        classification = self.classifier(feature)
        return classification
    
    def prepare_data(self):
        pass

    def train_dataloader(self):
        train_dset = get_dataset(self.path, 'train')
        train_loader = get_dataloader(train_dset, self.batch_size)
        return train_loader

    def val_dataloader(self):
        val_dset = get_dataset(self.path, 'valid')
        val_loader = get_dataloader(val_dset, self.batch_size)
        return val_loader

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

        log = {'train_loss':loss}

        return {'loss':loss, 'trn_acc':accuracy, 'log':log}

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

        return {'val_loss':loss, 'val_acc':accuracy}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_log = {'val_loss':val_loss_mean}
        return {'val_loss':val_loss_mean, 'log':tensorboard_log}
# PyTorch Lightning Module (Research Part)
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models

from utils import get_dataloader, get_dataset


class Baseline(pl.LightningModule):
    """ Pretrained Baseline Models

        1. resnet50
        2. mobilenet_v2
    """
    def __init__(self, args):
        super(Baseline, self).__init__()
        self.args = args
        self.path = args.path
        self.lr = args.lr        
        self.batch_size = args.batch_size
        self.num_classes = args.num_classes
        self.model = args.model 

        model_list = ['resnet', 'mobilenet']
        if self.model not in model_list:
            raise ValueError('Not-supported Model!')
        
        if self.model == 'resnet':
            net = models.resnet50(pretrained=True)
            modules = list(net.children())[:-1]
            self.extractor = nn.Sequential(*modules)    
            self.classifier = nn.Linear(net.fc.in_features, self.num_classes)
        elif self.model == 'mobilenet':
            net = models.mobilenet_v2(pretrained=True)
            self.extractor = net.features
            self.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(net.last_channel, self.num_classes),
            )
        net = None
            
    def forward(self, x):
        feature = self.extractor(x)
        if self.model == 'resnet':
            feature = feature.view(feature.size(0), -1)
        elif self.model == 'mobilenet':
            feature = F.adaptive_avg_pool2d(feature, 1).reshape(feature.shape[0], -1)
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

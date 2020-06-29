# PyTorch Lightning Module (Engineering & Non-essential Part)
import argparse

import pytorch_lightning as pl
from pytorch_lightning.logging import TensorBoardLogger

from lightning_research import Baseline


class PrintCallback(pl.Callback):
    def on_train_start(self, trainer, pl_module):
        print('Training starts...')
    def on_train_end(self, trainer, pl_module):
        print('Training is done.')

def main(hparams):
    model = Baseline(hparams)
    print_callback = [PrintCallback()]
    trainer = pl.Trainer(
                gpus=hparams.gpus,
                callbacks=print_callback,
                max_epochs=hparams.epoch,
                logger=TensorBoardLogger(save_dir='./logs', name='musinsa'),
                distributed_backend=hparams.distributed_backend
            )
    trainer.fit(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, default=1, help='number of gpus')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--epoch', type=int, default=200, help='epochs to train')
    parser.add_argument('--seed', type=int, default=711, help='random seed')
    parser.add_argument('--path', type=str, default='data/', help='parent directory containing train, val, test data')
    parser.add_argument('--num_classes', type=int, default=2, help='output class number')
    parser.add_argument('--distributed_backend', type=str, default='dp')
    parser.add_argument('--model', type=str, default='resnet')

    args = parser.parse_args()
    main(args)

## TODO: how to save hparams.yaml automatically
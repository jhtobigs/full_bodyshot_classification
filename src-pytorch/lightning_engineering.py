# PyTorch Lightning Module (Engineering & Non-essential Part)
import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.logging import TensorBoardLogger

from lightning_research import Baseline
from lightning_rexnetv1 import CustomReXNetV1


class PrintCallback(pl.Callback):
    def on_train_start(self, trainer, pl_module):
        print('*** Training starts...')
    def on_train_end(self, trainer, pl_module):
        print('*** Training is done.')

def main(hparams):
    pl.seed_everything(hparams.seed) # for reproducibility
    model = CustomReXNetV1(vars(hparams), width_mult=1.0) \
        if hparams.model == 'rexnet' else Baseline(vars(hparams))
    
    print_callback = [PrintCallback()]
    trainer = pl.Trainer(
                gpus=hparams.gpus,
                callbacks=print_callback,
                max_epochs=hparams.epoch,
                logger=TensorBoardLogger(save_dir='./logs', name='train'),
                early_stop_callback=True,
                distributed_backend=hparams.distributed_backend
            )

    if hparams.mode.lower() == 'train': # train + test
        trainer.fit(model)  
    elif hparams.mode.lower() == 'test': # pretrained + test
        model = model.load_from_checkpoint(
                    ## TODO
                    checkpoint_path='logs/train/version_35/checkpoints/epoch=6.ckpt'
                )
    else:
        raise ValueError('You must choose train or test mode')

    ## pytorch-lightning BUG: cannot apply distributed-backend option
    trainer = pl.Trainer(
            gpus=hparams.gpus,
            callbacks=print_callback,
            max_epochs=hparams.epoch,
            logger=TensorBoardLogger(save_dir='./logs', name='test'),
            early_stop_callback=True,
        )
    trainer.test(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, default=-1, help='number of gpus')
    parser.add_argument('--path', type=str, default='D:/data/musinsa/train_test_valid', help='parent directory containing train, val, test data')
    parser.add_argument('--epoch', type=int, default=200, help='epochs to train')
    parser.add_argument('--seed', type=int, default=711, help='random seed')
    parser.add_argument('--num_classes', type=int, default=2, help='output class number')
    parser.add_argument('--distributed_backend', type=str, default='dp')
    parser.add_argument('--mode', type=str, default='train', help='train or test')

    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--model', type=str, default='mobilenet', help='resnet/mobilenet/efficientnet/rexnet')
    parser.add_argument('--pretrain', type=str, default='true', help='using ImageNet-pretrained Model')
    parser.add_argument('--mult', type=float, default=1.0, help='rexnet scale(1.0/1.3/1.5/2.0)')

    parser.add_argument('--step_size', type=int, default=5, help='lr decay step size')
    parser.add_argument('--decay_rate', type=float, default=0.2, help='lr decay rate')

    args = parser.parse_args()
    main(args)

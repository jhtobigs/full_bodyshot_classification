import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from net import MobileNetV2
from utils import get_dataset, get_dataloader
from metric import accuracy


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    
    # config
    lr = args.lr
    batch_size = args.batch_size
    num_epochs = args.epoch

    # prepare loader
    path = args.path
    train_dset, val_dset = get_dataset(path, 'train'), get_dataset(path, 'valid')
    train_loader, val_loader = get_dataloader(train_dset, batch_size), get_dataloader(val_dset, batch_size)

    # model setting
    model = MobileNetV2(pretrained=True)
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # train
    best_val_loss = 1e+10
    total_step = len(train_loader)
    
    for epoch in tqdm(range(num_epochs)):
        train_loss = 0.0
        train_acc = 0.0

        model.train()
        for idx, data in enumerate(train_loader):
            image, target = map(lambda x: x.to(device), data)

            optimizer.zero_grad()
            y_hat = model(image)
            loss = criterion(y_hat, target)                           
            loss.backward()
            optimizer.step()
            
            acc = accuracy(y_hat, target)
            train_loss += loss.item()
            train_acc += acc.item()

            if (idx+1) % 30 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                        .format(epoch+1, num_epochs, idx+1, total_step, loss.item()))
        
        # val
        correct, total = 0, 0
        model.eval()
        with torch.no_grad():
            val_loss = 0.0

            for idx, data in enumerate(val_loader):
                image, target = map(lambda x: x.to(device), data)
                y_hat = model(image)
    
                v_loss = criterion(y_hat, target)
                val_loss += v_loss.item()

                _, pred = torch.max(y_hat.data, 1)
                total += target.size(0)
                correct += (pred == target).sum().item()
            
            val_loss /= len(val_loader)

            # save the best model
            is_best = val_loss < best_val_loss
            if is_best:
                state = {
                    'lr':lr,
                    'batch_size':batch_size,
                    'epoch':epoch,
                    'seed':args.seed,
                    'model_state_dict':model.state_dict(),
                    'opt_state_dict':optimizer.state_dict()
                }
                save_dir = 'experiments/'
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                torch.save(state, save_dir+'best.tar')
                best_val_loss = val_loss
                print('Saving the best model...')
            
            print('** Validation Loss: {:.4f}'.format(val_loss))
            print('** Validation Accuracy: {}%'.format(100*correct/total))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--epoch', type=int, default=50, help='epochs to train')
    parser.add_argument('--seed', type=int, default=711, help='random seed')
    parser.add_argument('--path', type=str, default='data/', help='parent directory containing train, val, test data')

    args = parser.parse_args()
    print('CUDA: {}'.format(torch.cuda.is_available()))
    main(args)

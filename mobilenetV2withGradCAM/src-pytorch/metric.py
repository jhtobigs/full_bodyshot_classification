import torch

def accuracy(yhat, y):
    with torch.no_grad():
        yhat = yhat.max(dim=1)[1]
        acc = (yhat==y).float().mean()
    return acc
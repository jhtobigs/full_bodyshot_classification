import os

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from dataset import StyleDataset


def get_dataset(path: str, mode: str) -> Dataset:
    """
    :param mode: train/valid/test
    :return: [3,224,224] torch tensor
    """
    if mode not in ["train", "valid", "test"]:
        raise ValueError("You should specify exact mode(train/valid/test).")

    if mode == "train":
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    return StyleDataset(path, mode, transform=transform)


def get_dataloader(dataset: Dataset, batch_size: int, shuffle=True, num_workers=4) -> DataLoader:
    shuffle = True if dataset.mode == "train" else False
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

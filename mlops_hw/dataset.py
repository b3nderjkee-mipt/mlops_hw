import os

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from dvc.fs import DVCFileSystem
from torch.utils.data import DataLoader


def check_files() -> bool:
    files_paths = [
        "data/MNIST/raw/t10k-images-idx3-ubyte",
        "data/MNIST/raw/t10k-labels-idx1-ubyte",
        "data/MNIST/raw/train-images-idx3-ubyte",
        "data/MNIST/raw/train-labels-idx1-ubyte",
    ]
    result = True
    for path in files_paths:
        result &= os.path.exists(path)

    return result


def load_data_from_dvc():
    fs = DVCFileSystem()
    fs.get("data", "data", recursive=True)


def mnist_dataload(batch_size: int, train: bool, shuffle: bool = True) -> DataLoader:
    # if not check_files():
    #    load_data_from_dvc()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    dataset = datasets.MNIST(
        root="./data", train=train, download=True, transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

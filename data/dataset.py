import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader



def mnist_dataload(batch_size: int, train: bool, shuffle: bool = True) -> DataLoader:

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
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader


def GetFashMNIST(path, train, download=True):
    Compose = transforms.Compose([transforms.ToTensor(), transforms.Resize(224)])
    return datasets.FashionMNIST(download=download, root=path, train=train, transform=Compose)


def iter_transform(dataset, batch_size, shuffle=False):
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

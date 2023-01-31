from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader


def GetFashionMNIST(path, train, Resize=0, download=True):
    if Resize == 0:
        Compose = transforms.Compose([transforms.ToTensor()])
    else:
        Compose = transforms.Compose([transforms.ToTensor(), transforms.Resize(Resize)])
    return datasets.FashionMNIST(download=download, root=path, train=train, transform=Compose)


def GetMNIST(path, train,Resize=0, download=True):
    if Resize == 0:
        Compose = transforms.Compose([transforms.ToTensor()])
    else:
        Compose = transforms.Compose([transforms.ToTensor(), transforms.Resize(Resize)])
    Compose = transforms.Compose([transforms.ToTensor(), transforms.Resize(224)])
    return datasets.MNIST(download=download, root=path, train=train, transform=Compose)


def GetCIFAR10(path, train,Resize=0, download=True):
    if Resize == 0:
        Compose = transforms.Compose([transforms.ToTensor()])
    else:
        Compose = transforms.Compose([transforms.ToTensor(), transforms.Resize(Resize)])
    Compose = transforms.Compose([transforms.ToTensor(), transforms.Resize(96)])
    return datasets.CIFAR10(download=download, root=path, train=train, transform=Compose)


def GetCIFAR100(path, train,Resize=0, download=True):
    if Resize == 0:
        Compose = transforms.Compose([transforms.ToTensor()])
    else:
        Compose = transforms.Compose([transforms.ToTensor(), transforms.Resize(Resize)])
    Compose = transforms.Compose([transforms.ToTensor(), transforms.Resize(224)])
    return datasets.CIFAR100(download=download, root=path, train=train, transform=Compose)


def iter_transform(dataset, batch_size, shuffle=False):
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

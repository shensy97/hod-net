import torch
from torchvision import datasets, transforms

def mnist_loader(config):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/MNIST', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=config.train.batch_size, shuffle=False)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/MNIST', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=config.train.batch_size, shuffle=False)

    return train_loader, test_loader

def svhn_loader(config):
    train_loader = torch.utils.data.DataLoader(
        datasets.SVHN('./data/SVHN', split='train', download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])),
        batch_size=config.train.batch_size, shuffle=False)

    test_loader = torch.utils.data.DataLoader(
        datasets.SVHN('./data/SVHN', split='test', transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])),
        batch_size=config.train.batch_size, shuffle=False)

    return train_loader, test_loader

def cifar_loader(config):
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data/CIFAR', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])),
        batch_size=config.train.batch_size, shuffle=False)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data/CIFAR', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])),
        batch_size=config.train.batch_size, shuffle=False)


    return train_loader, test_loader
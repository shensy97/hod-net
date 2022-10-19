from layers.layer import Sequential, AutoEncoder
from layers.linear import Linear, Transform
from layers.conv import Conv
from layers.activation import Sigmoid, ReLU, ELU
from layers.batchnorm import BatchNormalization
from layers.pooling import MeanPooling, MaxPooling

def get_LeNet(dataset="MNIST"):
    if dataset == "MNIST":
        model = Sequential(Conv((6, 5, 5, 1), bias=True),
                           Sigmoid(),
                           MeanPooling(2),
                           Conv((16, 5, 5, 6), bias=True),
                           Sigmoid(),
                           MeanPooling(2),
                           Transform((-1, 4, 4, 16), (-1, 256)),
                           Linear(256, 120, bias=True), Sigmoid(),
                           Linear(120, 84, bias=True), Sigmoid(),
                           Linear(84, 10, bias=True))
    else: # CIFAR SVHN
        model = Sequential(Conv((6, 5, 5, 3), bias=True),
                           Sigmoid(),
                           MeanPooling(2),
                           Conv((16, 5, 5, 6), bias=True),
                           Sigmoid(),
                           MeanPooling(2),
                           Transform((-1, 5, 5, 16), (-1, 400)),
                           Linear(400, 120, bias=True), Sigmoid(),
                           Linear(120, 84, bias=True), Sigmoid(),
                           Linear(84, 10, bias=True))

    return model


def get_FC(dataset="MNIST", model_type="2-10"):
    if model_type == "2-10":
        # model = Sequential(Linear(784, 100, bias=False), Sigmoid(), Linear(100, 784, bias=False), Sigmoid())
        model = AutoEncoder((Linear(784, 100, bias=False), Sigmoid()), (Linear(100, 784, bias=False), ))
    return model


def get_VGG(miniversion=True):
    '''
    This is slow, use with caution
    VGG for CIFAR-10
    if set miniversion as True, return a VGG-11 net with fewer channel.
    '''
    if miniversion:
        model = Sequential(Conv((32, 3, 3, 3), 'SAME'),
                        BatchNormalization(32),
                        ELU(),
                        MaxPooling(2),

                        Conv((64, 3, 3, 32), 'SAME'),
                        BatchNormalization(64),
                        ELU(),
                        MaxPooling(2),

                        Conv((64, 3, 3, 64), 'SAME'),
                        BatchNormalization(64),
                        ELU(),
                        Conv((64, 3, 3, 64), 'SAME'),
                        BatchNormalization(64),
                        ELU(),
                        MaxPooling(2),

                        Conv((128, 3, 3, 64), 'SAME'),
                        BatchNormalization(128),
                        ELU(),
                        Conv((128, 3, 3, 128), 'SAME'),
                        BatchNormalization(128),
                        ELU(),
                        MaxPooling(2),

                        Transform((-1, 2, 2, 128), (-1, 512)),
                        Linear(512, 128),
                        BatchNormalization(128),
                        ELU(),
                        Linear(128, 128),
                        BatchNormalization(128),
                        ELU(),
                        Linear(128, 10))
    else:
        model = Sequential(Conv((64, 3, 3, 3), 'SAME'),
                        BatchNormalization(64),
                        ELU(),
                        MaxPooling(2),

                        Conv((128, 3, 3, 64), 'SAME'),
                        BatchNormalization(128),
                        ELU(),
                        MaxPooling(2),

                        Conv((256, 3, 3, 128), 'SAME'),
                        BatchNormalization(256),
                        ELU(),
                        Conv((256, 3, 3, 256), 'SAME'),
                        BatchNormalization(256),
                        ELU(),
                        MaxPooling(2),

                        Conv((512, 3, 3, 256), 'SAME'),
                        BatchNormalization(512),
                        ELU(),
                        Conv((512, 3, 3, 512), 'SAME'),
                        BatchNormalization(512),
                        ELU(),
                        MaxPooling(2),

                        Conv((512, 3, 3, 512), 'SAME'),
                        BatchNormalization(512),
                        ELU(),
                        Conv((512, 3, 3, 512), 'SAME'),
                        BatchNormalization(512),
                        ELU(),
                        MaxPooling(2),

                        Transform((-1, 1, 1, 512), (-1, 512)),
                        Linear(512, 512),
                        BatchNormalization(512),
                        Sigmoid(),
                        Linear(512, 512),
                        BatchNormalization(512),
                        Sigmoid(),
                        Linear(512, 10))

    return model
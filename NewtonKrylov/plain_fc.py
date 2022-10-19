import os
import copy
import time
import math
import torch
import pprint
import logging
import numpy
import networks
import cupy as np #import numpy as np
from torchvision import datasets, transforms
import data_loader

from solver import CGNet, FCNet
from dataio import load_network, dump_network
from utils.logger import create_logger
from layers.layer import Sequential
from layers.linear import Linear, Transform
from layers.conv import Conv
from layers.activation import Sigmoid, ReLU, ELU
from layers.batchnorm import BatchNormalization
from layers.pooling import MeanPooling, MaxPooling
from losses import CrossEntropyLoss, MSELoss
from net_config import update_config_from_file, s_config, s_config_file, s_args
from optimizer import Krylov_optimizer, AdamOptimizer


config = copy.deepcopy(s_config)
final_output_path, final_log_path, logger = create_logger(s_config_file,
                                                            config.basic.output_path,
                                                            config.basic.log_path)
logger.info('training config:{}\n'.format(pprint.pformat(config)))

np.cuda.Device(config.basic.use_device).use()


if config.basic.network == "LeNet":
    model = networks.get_LeNet(config.basic.dataset)
elif config.basic.network == "VGG9_mini":
    model = networks.get_VGG(miniversion=True)
elif config.basic.network == "VGG11_full":
    model = networks.get_VGG(miniversion=False)
elif config.basic.network == "FC_simple":
    model = networks.get_FC(model_type="2-10")
else:
    raise NotImplementedError


if config.basic.dataset == "MNIST":
    train_loader, test_loader = data_loader.mnist_loader(config)
elif config.basic.dataset == "SVHN":
    train_loader, test_loader = data_loader.svhn_loader(config)
elif config.basic.dataset == "CIFAR":
    train_loader, test_loader = data_loader.cifar_loader(config)
else:
    raise NotImplementedError

if config.train.limit_batch_num == -1:
    total_batch_num = len(train_loader)
else:
    total_batch_num = config.train.limit_batch_num


net = FCNet(model, MSELoss(), AdamOptimizer())

if config.basic.dump_path is not None:
    net, start_epoch = load_network(config, logger, final_output_path)
    logger.info("Load Net from {}".format(config.basic.dump_path))
    logger.info("Start from epoch %d"%(start_epoch))
else:
    logger.info("Start Anew")
    start_epoch = 0

total_epoch_num = config.train.end_epoch
loss_chart = []


def train(model, train_loader, epoch, internal=50):
    train_loss = 0
    batch_idx = 0
    total_loss_cnt = 0
    len_dataset = len(train_loader.dataset)

    logger.info("Epoch "+str(epoch))


    if config.train.dump_state and (epoch == 0):
        dump_file_path = os.path.join(final_output_path, "epoch%03d.pth"%(epoch))
        dump_network(dump_file_path, net, epoch)
        logger.info("Dump into {}".format(dump_file_path))

    for im, label in train_loader:
        batch_idx += 1
        # im = im.numpy().transpose((0, 2, 3, 1))
        im = im.numpy().reshape(-1, 784)
        im = np.array(im)
        out_np_s, loss_np_s = model.fit_train(im, im, epoch, total_batch_num)

        if loss_np_s is not None:
            for loss_np in loss_np_s:
                train_loss += loss_np.item()
                total_loss_cnt += 1
        if batch_idx == total_batch_num:
            break

    if (config.train.dump_state == True) and ((epoch+1) % config.train.dump_internal == 0):
        dump_file_path = os.path.join(final_output_path, "epoch%03d.pth"%(epoch+1))
        dump_network(dump_file_path, net, epoch+1)
        logger.info("Dump into {}".format(dump_file_path))
    loss_chart.append(train_loss / total_loss_cnt)

    logger.info('Epoch [%02d] | Train loss: %f' % (epoch, train_loss / total_loss_cnt))

for epoch in range(start_epoch, total_epoch_num):
    train(net, train_loader, epoch)
    # test(net, test_loader, epoch)

loss_chart = numpy.array(loss_chart)
loss_out_path = os.path.join(final_output_path, 'losses.txt')
numpy.savetxt(loss_out_path, loss_chart, fmt='%.6f')

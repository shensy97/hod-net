import cupy as np #import numpy as np
import pickle as pk

def dump_network(filename, net, epoch):
    f = open(filename, 'wb')
    dump_data = {}
    logger = net.optimizer.logger
    net.optimizer.logger = None
    dump_data['net'] = net
    dump_data['epoch'] = epoch
    pk.dump(dump_data, f)
    net.optimizer.logger = logger

def load_network(cfg, logger, output_path):
    dump_file = cfg.basic.dump_path
    f = open(dump_file, 'rb')
    data_file = pk.load(f)
    net = data_file['net']
    epoch = data_file['epoch']

    net.sample_grad = []
    net.sample_weight = []
    net.sample_loss = []

    net.optimizer.CG_log_interval      = cfg.optim.CG_log_interval
    net.optimizer.limit_batch_num      = cfg.train.limit_batch_num

    net.optimizer.damp_coef            = cfg.optim.damp_coef
    net.optimizer.verbose              = cfg.optim.verbose
    net.optimizer.taylor_threshold     = cfg.optim.taylor_threshold
    net.optimizer.CG_log_interval      = cfg.optim.CG_log_interval
    net.optimizer.CG_maxiter           = cfg.optim.CG_maxiter
    net.optimizer.config               = cfg

    net.optimizer.logger               = logger
    net.optimizer.start_of_CG          = True

    return net, epoch

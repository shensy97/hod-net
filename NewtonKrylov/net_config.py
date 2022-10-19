import os
import yaml
import argparse
import copy
import cupy as np #import numpy as np
from easydict import EasyDict as edict

def parse_args():
    parser = argparse.ArgumentParser(description='complex experiment')
    # yaml config file
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    args, rest = parser.parse_known_args()
    return args

def update_config_from_file(_config, config_file, check_necessity=True):
    config = copy.deepcopy(_config)
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    for vk, vv in v.items():
                        if vk in config[k]:
                            if isinstance(vv, list) and not isinstance(vv[0], str):
                                config[k][vk] = np.asarray(vv)
                            else:
                                config[k][vk] = vv
                        else:
                            if check_necessity:
                                raise ValueError("{}.{} not exist in config".format(k, vk))
                else:
                    raise ValueError("{} is not dict type".format(v))
            else:
                if check_necessity:
                    raise ValueError("{} not exist in config".format(k))
    return config


def get_default_config_basic():
    config = edict()
    config.network = "LeNet"
    config.dataset = "MNIST"
    config.log_path = './log/config_default'
    config.output_path = './output/output_default'
    config.dump_path = None
    config.use_device = 0
    return config

def get_default_config_train():
    config = edict()
    config.end_epoch = 20
    config.batch_size = 128
    config.limit_batch_num = -1
    config.global_batch_num = None
    config.total_epoch = 30
    config.dump_state = False
    config.dump_internal = 10
    return config


def get_default_config_optim():
    config = edict()
    config.damp_coef = 0.001
    config.CG_quit_coef = 0.001
    config.verbose = False
    config.CG_log_interval = 10
    config.damp_threshold = 1e-4
    config.taylor_threshold = 0.01
    config.CG_maxiter = 50
    return config


s_args = parse_args()
s_config_file = s_args.cfg
s_config = edict()
s_config.basic = get_default_config_basic()
s_config.train = get_default_config_train()
s_config.optim = get_default_config_optim()
s_config = update_config_from_file(s_config, s_config_file)

import cupy as np #import numpy as np

def collect_gradient(params):
    grad = []
    for param in params:
        grad.append(param.gradient.reshape(-1))
    grad = np.concatenate(grad)
    return grad

def collect_weights(params):
    weight = []
    for param in params:
        weight.append(param.tensor.copy())
    return weight

def set_weights(params, weights):
    for i in range(len(params)):
        params[i].tensor = weights[i]
    return

def generate_random_weight_list(weight_list, sample_num, low=-2.5, high=2.5):
    random_weight_list = []
    for _ in range(sample_num):
        random_weight = []
        for i in range(len(weight_list[0])):
            random_weight.append(np.random.uniform(low, high, weight_list[0][i].shape))
        random_weight_list.append(random_weight)
    return random_weight_list
import os
import logging
import time

def make_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def change_tuple_element(t, dim, val):
    l = list(t)
    l[dim] = val
    return tuple(l)

def create_logger(cfg_file, output_path, log_path):
    time_str = time.strftime('%Y-%m-%d-%H-%M')

    curDir = os.path.dirname(cfg_file).split('/')[-1]
    cfg_name = os.path.basename(cfg_file).rsplit('.', 1)[0]

    # # model path
    # print(output_path, curDir, cfg_name)
    final_output_path = os.path.join(output_path, cfg_name)
    make_folder(final_output_path)

    # log path
    final_log_path = os.path.join(log_path, cfg_name)
    make_folder(final_log_path)

    # create logger
    logging.basicConfig(filename=os.path.join(final_log_path, '{}_{}.log'.format(cfg_name, time_str)),
                        format='%(asctime)-15s %(message)s', level=logging.INFO)
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())

    return final_output_path, final_log_path, logger
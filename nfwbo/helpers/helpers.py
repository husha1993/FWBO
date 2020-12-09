logfile = './direct.txt'
def count_parameters(model):
    '''
    compute the number of trainable parameters of a nn.Module
    '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


import time
import socket


def create_save_path(args, sublist, mode='train', extra=''):
    host_name = socket.gethostname()
    current_dtime = time.strftime("%m-%d", time.localtime())
    current_htime = time.strftime("%H-%M-%S", time.localtime())

    nfa = ''  # name from args
    for i in sublist:
        if type(args) == dict:
            nfa += '-' + i + '=' + str(args[i])
        else:
            nfa += '-' + i + '=' + str(getattr(args, i))

    save_path = './{0}_dir/{1}/{2}/{3}'.format(
            mode,
            current_dtime + '_icra2021',
            args['config'] if type(args) == dict else args.config,
            host_name
            + '-st=' + current_htime
            + nfa
            + '-'
            + extra)
    return save_path

import torch
import random
import numpy as np


def set_random_seeds(seed):
    """
    Sets the random seeds for numpy, python, pytorch cpu and gpu
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return None


import subprocess


def get_git_revision_hash():
    return str(subprocess.check_output(['git', 'rev-parse', 'HEAD']))

import json
import os


def save_args2json(save_file, save_path, args):
    config_name = 'args.json'
    with open(os.path.join(save_path, save_file), 'w') as f:
        config = {k: v for (k, v) in vars(args).items() if k != 'device'}
        try:
            config.update({'git_hash': get_git_revision_hash()})
        except:
            pass
        json.dump(config, f, indent=2)


import git
import sys
import logging

def create_logging(save_file, save_path, mode, debug_flag = False):
    log_file = os.path.join(save_path, save_file)

    file_handler = logging.FileHandler(log_file, mode=mode)
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO if not debug_flag else logging.DEBUG
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
                    format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    repo = git.Repo(search_parent_directories=True)
    logging.info('Current git head hash code: {}'.format(repo.head.object.hexsha))
    logger = logging.getLogger('first logger')
    return logger

def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to

def compute_resources_of_optimizer(logger, config):
    opt_name = config['optimizer_name']
    if opt_name == 'MultiFidelityBayesOptimizer':
        return
    Total_N = config['Total_N']
    _epochs_total = config['Total_N'] * config ['op_algorithm_config']['max_num_epochs_per_design']

    logger.info('-----------------------------------------------------------------------')
    logger.info('|the optimzier = {}                                                    |'.format(opt_name))
    logger.info('|Given : total number of allowed designs={}                                    |'.format(Total_N))
    if opt_name == 'BatchBayesOptimizer':
        logger.info('|Given : batch_size={}                                 |'.format(config['op_algorithm_config']['batch_size']))
        logger.info('|epochsPerbatch={}                                   |'.format(config['op_algorithm_config']['batch_size'] * config ['op_algorithm_config']['max_num_epochs_per_design'] ))
    logger.info('|Given: max number of epochs allowed per design={}                          |'.format(config ['op_algorithm_config']['max_num_epochs_per_design']))
    logger.info('|total number of epochs will be={}                                   |'.format(_epochs_total))
    logger.info('------------------------------------------------------------------|')


# Print iterations progress
# from https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def printProgressBar(iteration, total, prefix='Video Progress:', suffix='Complete', decimals=1, length=35, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    pb = '%s |%s| %s%% %s' % (prefix, bar, percent, suffix)
    # Print New Line on Complete
    if iteration == total:
        pb = ''
    return pb

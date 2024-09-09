"""
===============================================================================
Title:           general_utils.py
Date:            June 22, 2024
Description:     This script contains various helper functions that are called by frontend scripts for training and testing models. This file cannot be run as a standalone script. 
===============================================================================
"""

import json
import inspect
import torch
import os
import sys
import yaml
from shutil import copy
from os.path import join, realpath, isfile

class Logger(object):

    def __getattr__(self, k):
        return print

log = Logger()

def training_config_from_cli_args():
    config_file_name = sys.argv[1]
    experiment_id = int(sys.argv[2])

    yaml_config = yaml.load(open(f'experiments/{config_file_name}'), Loader=yaml.SafeLoader)

    config = yaml_config['configuration']
    config = {**config, **yaml_config['individual_configurations'][experiment_id]}
    config = AttributeDict(config)
    return config

def score_config_from_cli_args():
    config_file_name = sys.argv[1]
    experiment_id = int(sys.argv[2])
    

    yaml_config = yaml.load(open(f'experiments/{config_file_name}'), Loader=yaml.SafeLoader)

    config = yaml_config['test_configuration_common']

    test_id = int(sys.argv[3])
    config = {**config, **yaml_config['test_configuration'][test_id]}

    train_checkpoint_id = yaml_config['individual_configurations'][experiment_id]['name']

    config = AttributeDict(config)
    return config, train_checkpoint_id

class AttributeDict(dict):
    """ 
    An extended dictionary that allows access to elements as atttributes and counts 
    these accesses. This way, we know if some attributes were never used. 
    """

    def __init__(self, *args, **kwargs):
        from collections import Counter
        super().__init__(*args, **kwargs)
        self.__dict__['counter'] = Counter()

    def __getitem__(self, k):
        self.__dict__['counter'][k] += 1
        return super().__getitem__(k)

    def __getattr__(self, k):
        self.__dict__['counter'][k] += 1
        return super().get(k)

    def __setattr__(self, k, v):
        return super().__setitem__(k, v)

    def __delattr__(self, k, v):
        return super().__delitem__(k, v)    

    def unused_keys(self, exceptions=()):
        return [k for k in super().keys() if self.__dict__['counter'][k] == 0 and k not in exceptions]

    def assume_no_unused_keys(self, exceptions=()):
        if len(self.unused_keys(exceptions=exceptions)) > 0:
            log.warning('Unused keys:', self.unused_keys(exceptions=exceptions))


def get_attribute(name):
    import importlib

    if name is None:
        raise ValueError('The provided attribute is None')
    
    name_split = name.split('.')
    name_split[0] = name_split[0]+""
    mod = importlib.import_module('.'.join(name_split[:-1]))
    return getattr(mod, name_split[-1])



def filter_args(input_args, default_args):

    updated_args = {k: input_args[k] if k in input_args else v for k, v in default_args.items()}
    used_args = {k: v for k, v in input_args.items() if k in default_args}
    unused_args = {k: v for k, v in input_args.items() if k not in default_args}

    return AttributeDict(updated_args), AttributeDict(used_args), AttributeDict(unused_args)


def load_model(checkpoint_id, weights_file=None, strict=True, model_args='from_config', with_config=False,device='cuda:0'):

    config = json.load(open(join('logs', checkpoint_id, 'config.json')))

    if model_args != 'from_config' and type(model_args) != dict:
        raise ValueError('model_args must either be "from_config" or a dictionary of values')

    model_cls = get_attribute(config['model'])

    # load model
    if model_args == 'from_config':
        _, model_args, _ = filter_args(config, inspect.signature(model_cls).parameters)
    
    model = model_cls(**model_args)

    if weights_file is None:
        weights_file = realpath(join('logs', checkpoint_id, 'weights.pth'))
    else:
        weights_file = realpath(join('logs', checkpoint_id, weights_file))

    if isfile(weights_file):
        weights = torch.load(weights_file,map_location=device)
        for _, w in weights.items():
            assert not torch.any(torch.isnan(w)), 'weights contain NaNs'
        model.load_state_dict(weights, strict=strict)
    else:
        raise FileNotFoundError(f'model checkpoint {weights_file} was not found')

    if with_config:
        return model, config
    
    return model


class TrainingLogger(object):

    def __init__(self, model, log_dir, config=None, *args):
        super().__init__()
        self.model = model
        self.base_path = join(f'logs/{log_dir}') if log_dir is not None else None

        os.makedirs('logs/', exist_ok=True)
        os.makedirs(self.base_path, exist_ok=True)

        if config is not None:
            json.dump(config, open(join(self.base_path, 'config.json'), 'w'))

    def iter(self, i, **kwargs):
        if i % 100 == 0 and 'loss' in kwargs:
            loss = kwargs['loss']
            print(f'iteration {i}: loss {loss:.4f}')

    def save_weights(self, only_trainable=False, weight_file='weights.pth'):
        if self.model is None:
            raise AttributeError('You need to provide a model reference when initializing TrainingTracker to save weights.')

        weights_path = join(self.base_path, weight_file)

        weight_dict = self.model.state_dict()

        if only_trainable:
            weight_dict = {n: weight_dict[n] for n, p in self.model.named_parameters() if p.requires_grad}
        
        torch.save(weight_dict, weights_path)
        log.info(f'Saved weights to {weights_path}')

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        """ automatically stop processes if used in a context manager """
        pass        

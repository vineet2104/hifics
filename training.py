"""
===============================================================================
Title:           training.py
Date:            June 22, 2024
Description:     This script contains the code for training the HiFi-CS model using the provided configuration file. 

Usage:           python training.py <config_file_name> <individual_configuration_id>

===============================================================================
"""

import torch
import inspect
import json
import math
import os
import sys
import torch
from datasets import dataloader
from models.hifics import CLIPDensePredT
from functools import partial
from os.path import join, isfile
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
from contextlib import nullcontext
from general_utils import TrainingLogger, get_attribute, filter_args, log, training_config_from_cli_args



def custom_lr_schedule(iteration, warmup, max_iter, start_lr, end_lr):
    if iteration < warmup:
        # During the warmup (first 1000 iterations), keep the learning rate constant
        lr = start_lr
    else:
        # After warmup, decay the learning rate from start_lr to end_lr
        # Compute the proportion of the decay phase completed
        decay_progress = (iteration - warmup) / (max_iter - warmup)
        # Calculate the current learning rate with cosine annealing
        lr = end_lr + (start_lr - end_lr) * (0.5 * (1 + math.cos(math.pi * decay_progress)))
    return lr


def cosine_warmup_lr(i, warmup=10, max_iter=90):
    """ Cosine LR with Warmup """
    if i < warmup:
        return (i+1)/(warmup+1)
    else:
        return 0.5 + 0.5*math.cos(math.pi*(((i-warmup)/(max_iter- warmup))))



def main():

    # set device to cuda if available, otherwise use cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config file from ./experiments/ using the provided config name
    config = training_config_from_cli_args()

    print("Config File: ")
    print(config)
    model_cls = get_attribute(config.model)
    _, model_args, _ = filter_args(config, inspect.signature(model_cls).parameters)
    model = model_cls(**model_args).to(device)

    print("Model loaded successfully")

    data_loader = dataloader.get_data_loader(json_file=config.train_json,batch_size=config.batch_size,invert_mask=config.invert_mask,image_size=config.image_size)


    # optimizer
    opt_cls = get_attribute(config.optimizer)
    if config.optimize == 'torch.optim.SGD':
        opt_args = {'momentum': config.momentum if 'momentum' in config else 0}
    else:
        opt_args = {}
    opt = opt_cls(model.parameters(), lr=config.lr, **opt_args)
    
    # learning rate scheduler
    if config.lr_scheduler == 'cosine':
        assert config.T_max is not None and config.eta_min is not None
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, config.T_max, config.eta_min)
    elif config.lr_scheduler == 'warmup_cosine':      
        lr_scheduler = LambdaLR(opt, partial(cosine_warmup_lr, max_iter=(config.max_iterations), warmup=config.warmup))
    elif config.lr_scheduler=="custom_cosine":
        lr_scheduler = LambdaLR(opt,partial(custom_lr_schedule, warmup=config.warmup, max_iter=config.max_iterations, start_lr=config.lr, end_lr=config.eta_min))
    else:
        lr_scheduler = None

    batch_size, max_iterations = config.batch_size, config.max_iterations

    loss_fn = get_attribute(config.loss)

    if config.amp:
        log.info('Using AMP')
        autocast_fn = autocast
        scaler = GradScaler()
    else:
        autocast_fn, scaler = nullcontext, None


    save_only_trainable = True
    
    
    tracker_config = config if not config.hyperparameter_optimization else None

    with TrainingLogger(log_dir=config.name, model=model, config=tracker_config) as logger:

        i = 0
        no_of_samples_used = 0
        while True:
            
            for data_x, data_y in data_loader:

               
                cond = data_x[1] 
                if isinstance(cond, torch.Tensor):
                    cond = cond.to(device)

                with autocast_fn():
                    visual_q = None

                    pred, visual_q, _, _  = model(data_x[0].to(device), cond, return_features=True)

                    loss = loss_fn(pred.to(device), data_y[0].to(device))

                    if torch.isnan(loss) or torch.isinf(loss):
                        # skip if loss is nan
                        log.warning('Training stopped due to inf/nan loss.')
                        sys.exit(-1)

                    extra_loss = 0
                    loss += extra_loss

                opt.zero_grad()

                if scaler is None:
                    loss.backward()
                    opt.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()

                if lr_scheduler is not None:
                    lr_scheduler.step()
                    if i % 2000 == 0:
                        current_lr = [g['lr'] for g in opt.param_groups][0]
                        log.info(f'current lr: {current_lr:.5f} ({len(opt.param_groups)} parameter groups)')

                logger.iter(i=i, loss=loss)   
                           
                i += 1

                if i >= max_iterations:

                    if not isfile(join(logger.base_path, 'weights.pth')):
                        # only write if no weights were already written
                        logger.save_weights(only_trainable=save_only_trainable)
                    
                    sys.exit(0)

                    
                if config.checkpoint_iterations is not None and i in config.checkpoint_iterations:
                    logger.save_weights(only_trainable=save_only_trainable, weight_file=f'weights_{i}.pth')

            print('epoch complete')
            


if __name__ == '__main__':
    main()

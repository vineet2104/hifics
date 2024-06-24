"""
===============================================================================
Title:           score.py
Author:          Vineet Bhat
Date:            June 23, 2024
Description:     This script can be used to evaluate trained HiFi-CS models on the RoboRefIt and OCID-VLG dataset
Usage:           python score.py hifics_config.yaml 0 0 -> First 0 indicates the individual_configuration in the .yaml file and second 0 indicates the test dataset specified in test_configuration in the .yaml file
===============================================================================
"""

import torch
import inspect
import json
import numpy as np
from os.path import join, isfile, realpath
from general_utils import load_model, score_config_from_cli_args, AttributeDict, get_attribute, filter_args
from datasets import dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SMOOTH = 1e-6

def return_iou(outputs: torch.Tensor, labels: torch.Tensor,og_image: torch.Tensor,text_prompt:str, apply_sigmoid=False,threshold=0.):

    labels = labels.squeeze(1)  
    outputs = outputs.squeeze(1)

    temp_labels = labels
    temp_outputs = outputs

    if(apply_sigmoid):
        smooth_predictions = 1/(1+np.exp(-outputs))
        outputs = outputs < threshold
    else:    
        outputs = outputs < 0
    
    labels = labels < 0.5
    
    intersection = (outputs & labels).sum((1, 2))  
    union = (outputs | labels).sum((1, 2))         
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  
    
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  
    
    return iou 

def load_model(checkpoint_id, weights_file=None, strict=True, model_args='from_config', with_config=False, ignore_weights=False):

    config = json.load(open(join('logs', checkpoint_id, 'config.json')))
    if model_args != 'from_config' and type(model_args) != dict:
        raise ValueError('model_args must either be "from_config" or a dictionary of values')

    model_cls = get_attribute(config['model'])

    # load model
    if model_args == 'from_config':
        _, model_args, _ = filter_args(config, inspect.signature(model_cls).parameters)

    model = model_cls(**model_args)

    if weights_file is None:
        weights_file = realpath(join('logs', checkpoint_id, 'weights_1000.pth'))
    else:
        weights_file = realpath(join('logs', checkpoint_id, weights_file))

    if isfile(weights_file) and not ignore_weights:
        weights = torch.load(weights_file,map_location=device)
        for _, w in weights.items():
            assert not torch.any(torch.isnan(w)), 'weights contain NaNs'
        model.load_state_dict(weights, strict=strict)
    else:
        if not ignore_weights:
            raise FileNotFoundError(f'model checkpoint {weights_file} was not found')

    if with_config:
        return model, config
    
    return model


def main():
    config, train_checkpoint_id = score_config_from_cli_args()

    score(config, train_checkpoint_id, None)


def score(config, train_checkpoint_id, train_config):

    config = AttributeDict(config)

    print(config)

    # use training dataset and loss
    train_config = AttributeDict(json.load(open(f'logs/{train_checkpoint_id}/config.json')))

    cp_str = f'_{config.iteration_cp}' if config.iteration_cp is not None else ''

    model_cls = get_attribute(train_config['model'])

    _, model_args, _ = filter_args(train_config, inspect.signature(model_cls).parameters)

    model_args = {**model_args, **{k: config[k] for k in ['process_cond', 'fix_shift'] if k in config}}

    strict_models = {'ConditionBase4', 'PFENetWrapper'}
    model = load_model(train_checkpoint_id, strict=model_cls.__name__ in strict_models, model_args=model_args, 
                        weights_file=f'weights{cp_str}.pth', )
                           

    model.eval()
    model.to(device)
    print("Inference model loaded")

    metric_args = dict()


    if(config.test_dataset=="roborefit"): 

        test_ious = []
        if(config.test_split == "testA"):
            test_data_json = './datasets/final_dataset/testA/roborefit_testA.json'
        elif(config.test_split == "testB"):
            test_data_json = './datasets/final_dataset/testB/roborefit_testB.json'

        data_loader = dataloader.get_data_loader(json_file=test_data_json,batch_size=config.batch_size,invert_mask=config.invert_mask)

        with torch.no_grad():
            i,losses = 0,[]
            print("Total test data samples = "+str(len(data_loader.dataset)))
            for data_x, data_y in data_loader:
                print("Batch "+str(i+1))
                i+=1
                data_x = [v.to(device,non_blocking=True) if isinstance(v, torch.Tensor) else v for v in data_x]
                data_y = [v.to(device,non_blocking=True) if isinstance(v, torch.Tensor) else v for v in data_y]
                
                pred, _, _, _  = model(data_x[0], data_x[1], return_features=True)
                pred_batch = pred.cpu()
                gt_batch = data_y[0].cpu()
                text_prompts_batch = data_x[1]
                og_image_batch = data_x[0]
                
                ans = []
                for test_i in range(len(pred_batch)):
                    prediction = pred_batch[test_i]
                    ground_truth = gt_batch[test_i]
                    og_image = og_image_batch[test_i]
                    text_prompt = text_prompts_batch[test_i]
                    og_image = og_image_batch[test_i]
                    text_prompt = text_prompts_batch[test_i]
                    computed_iou = return_iou(outputs=prediction,labels=ground_truth,og_image=og_image,text_prompt=text_prompt,apply_sigmoid=config.apply_sigmoid,threshold=config.sigmoid_threshold)
                    ans.append(computed_iou)
                    test_ious.append(computed_iou)
                    
                print("Batch IoU = "+str(np.mean(ans)))
                    
                
        print("Mean IOU for test set = "+str(np.mean(test_ious)))
        print("prec@0.5 = "+str((sum(iou>0.5 for iou in test_ious)/len(test_ious))*100))
        print("prec@0.6 = "+str((sum(iou>0.6 for iou in test_ious)/len(test_ious))*100))
        print("prec@0.7 = "+str((sum(iou>0.7 for iou in test_ious)/len(test_ious))*100))
        print("prec@0.8 = "+str((sum(iou>0.8 for iou in test_ious)/len(test_ious))*100))
        print("prec@0.9 = "+str((sum(iou>0.9 for iou in test_ious)/len(test_ious))*100))

    
    elif(config.test_dataset=="ocidvlg"):

        test_ious = []
        test_data_json = './datasets/ocidvlg_final_dataset/test/ocid_vlg_test.json'
        
        data_loader = dataloader.get_data_loader(json_file=test_data_json,batch_size=config.batch_size,invert_mask=config.invert_mask)

        with torch.no_grad():
            i,losses = 0,[]
            print("Total test data samples = "+str(len(data_loader.dataset)))
            for data_x, data_y in data_loader:
                print("Batch "+str(i+1))
                i+=1

                data_x = [v.to(device,non_blocking=True) if isinstance(v, torch.Tensor) else v for v in data_x]
                data_y = [v.to(device,non_blocking=True) if isinstance(v, torch.Tensor) else v for v in data_y]
                
                pred, _, _, _  = model(data_x[0], data_x[1], return_features=True)
                pred_batch = pred.cpu()
                gt_batch = data_y[0].cpu()
                text_prompts_batch = data_x[1]
                og_image_batch = data_x[0]
                
                ans = []
                for test_i in range(len(pred_batch)):
                    prediction = pred_batch[test_i]
                    ground_truth = gt_batch[test_i]
                    og_image = og_image_batch[test_i]
                    text_prompt = text_prompts_batch[test_i]
                    og_image = og_image_batch[test_i]
                    text_prompt = text_prompts_batch[test_i]
                    computed_iou = return_iou(outputs=prediction,labels=ground_truth,og_image=og_image,text_prompt=text_prompt,apply_sigmoid=config.apply_sigmoid,threshold=config.sigmoid_threshold)
                    ans.append(computed_iou)
                    test_ious.append(computed_iou)
                    
                print("Batch IoU = "+str(np.mean(ans)))
                    
        print("Mean IOU for test set = "+str(np.mean(test_ious)))
        print("prec@0.5 = "+str((sum(iou>0.5 for iou in test_ious)/len(test_ious))*100))
        print("prec@0.6 = "+str((sum(iou>0.6 for iou in test_ious)/len(test_ious))*100))
        print("prec@0.7 = "+str((sum(iou>0.7 for iou in test_ious)/len(test_ious))*100))
        print("prec@0.8 = "+str((sum(iou>0.8 for iou in test_ious)/len(test_ious))*100))
        print("prec@0.9 = "+str((sum(iou>0.9 for iou in test_ious)/len(test_ious))*100))

    else:
        raise ValueError('invalid test dataset')


if __name__ == '__main__':
    main()
import wandb
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import pickle as pkl
from utils import *
import time
import json
import sys
from utils import average_hierarchical_cost, load_cost_matrix
import os
from models import *
import logging
import geoopt
import copy


def get_delta(input):
    
    dists = torch.cdist(input, input, p=2)
    
    row = dists[0, :][np.newaxis, :]
    col = dists[:, 0][:, np.newaxis]
    XY_p = 0.5 * (row + col - dists)

    maxmin = torch.max(torch.minimum(XY_p[:, :, None], XY_p[None, :, :]), axis=1)[0]
    delta = torch.max(maxmin - XY_p).item()
    diam = torch.max(dists).item()
    return delta, diam

def calculate_delta(model, testloader, device, D, method):  
    
    model.eval()
    predictions = []
    true_labels = []
    deltas = []
    with torch.no_grad():
        for data,y in testloader:
            data = data.to(device)
            y = y.to(device)
            
            y = y.squeeze()
            true_labels.append(y)

            output, _, embeddings = model(data)

            pred = prediction(method, output, model.prototypes)
            delta, diam = get_delta(embeddings)
            deltas.append(delta)
            pred = pred.squeeze().to(device)
            
            predictions.append(pred)
    
    true_labels = torch.cat(true_labels, dim=0)
    predictions = torch.cat(predictions, dim=0)
    acc = (true_labels == predictions).sum().item() / len(true_labels)
    return acc, predictions, true_labels, torch.mean(torch.tensor(deltas)).item(), torch.std(torch.tensor(deltas)).item()

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="classification")
    parser.add_argument('-device',dest='device', default='cpu', type = str, help='device')
    parser.add_argument('-config',dest='config', default='config.json', type = str, help='device')
    parser.add_argument('-rank',dest='rank', default=0, type = int, help='ranking of the run')
    parser.add_argument('-temp',dest='temperature', default=0.01, type = float, help='geometry of the output')
    parser.add_argument('-dataset',dest='dataset', default='cars', type = str, help='device')
    parser.add_argument('-dim',dest='dim', default=64, type = int, help='dimension of the embedding space')
    parser.add_argument('-lam',dest='lam', default=0.1, type = float, help='slope')
    parser.add_argument('-name', dest = 'name', default='.', type = str, help='name of the model')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    
    # Receiving the characteristics of the experiment
    args = parse_args()
    with open(args.config) as json_file:
        config = json.load(json_file)

    torch.manual_seed(args.rank)
    gradient = str(config['grad'])
    config['dataset'] = args.dataset
    config['output_dim'] = args.dim
    config['device'] = args.device
    config['rank'] = args.rank
    config['lam'] = args.lam
    run = wandb.init(project="your_project", config = config)

    logs_directory = f"logs_{config['dataset']}"

    logging.basicConfig(filename=logs_directory+'/'+run.name+'.log', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    logging.info(config)

    D = load_cost_matrix(config['dataset']).to(config['device'])
    trainloader, testloader = load_dataset(config['dataset'], 
                                           config['batch_size'], 
                                           num_workers = 8, 
                                           val= config['validation'])
    
    prototypes = None
    if config['method'] in ['HPS']:
        print('creating prototypes') # for hyperspherical methods
        prototypes = hyperspherical_embedding(config['dataset'], config['device'], config['output_dim'], config['rank'])
        manifold = geoopt.PoincareBallExact(c=config['curvature'], learnable = False)
    elif config['method'] in ['HBL','CHPS']:
        print('creating prototypes') # for hyperspherical methods
        prototypes = hyperspherical_embedding(config['dataset'], config['device'], config['output_dim'], config['rank'])
        manifold = None
    elif config['method'] == 'XE':
        manifold = None
    else:
        manifold = geoopt.PoincareBallExact(c=config['curvature'], learnable = False)
    
    model = load_backbone(config['method'],
                          config['dataset'], 
                          config['model'], 
                          config['output_dim']) 
    
    model = load_model(model, 
                       config['device'], 
                       config['method'], 
                       prototypes, 
                       config['geometry'], 
                       config['dataset'], 
                       config['output_dim'], 
                       config['grad'], 
                       config['temperature'],
                       config['clipping'],
                       manifold)
    model = model.to(config['device'])
    
    model_directory = 'your_model_directory.pt'
    checkpoint = torch.load(f'{model_directory}')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_acc, test_AHC, test_prediction, test_tl, delta, std_delta, rel_delta, std_rel_delta = calculate_delta(model, testloader, D=D, device = config['device'], method = config['method'])
    print(f'test acc: {test_acc} -- delta-hyb: {delta} -- rel_delta-hyb: {rel_delta}')
    logging.info(f'test Accuracy = {round(100*test_acc,4)} ; AHC = {round(test_AHC,4)}')
    wandb.log({"test_acc": test_acc, "test_AHC":test_AHC, "delta":delta, "rel_delta":rel_delta, "std_delta":std_delta, "std_rel_delta":std_rel_delta})
    logging.info("saving the results")
    
    wandb.finish()



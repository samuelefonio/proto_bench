import wandb
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import pickle as pkl
from utils import *
from data import *
import time
import json
from models import *
import copy
from log import initialize_logger_from_config

def main_train(model:torch.nn.Module, 
               trainloader:data.DataLoader, 
               opt:torch.optim.Optimizer, 
               scheduler:torch.optim.lr_scheduler, 
               device:torch.device = 'cpu'
               ):
    model.train()
    avgloss = 0.
    criterion = nn.CrossEntropyLoss()

    accuracy_metric = MulticlassAccuracy(num_classes=model.parametric_prototypes.shape[0]).to(device)
    f1_metric = MulticlassF1Score(num_classes=model.parametric_prototypes.shape[0]).to(device)
    recall_metric = MulticlassRecall(num_classes=model.parametric_prototypes.shape[0]).to(device)

    for bidx, (x, y) in enumerate(trainloader):
        
        x, y = x.to(device), y.to(device)
        y = y.squeeze()
        
        opt.zero_grad()
        
        distances, embeddings = model(x, y)
        
        loss = criterion(distances, y) 
        
        loss.backward()
        
        avgloss += loss.item()

        opt.step()
        
        pred = distances.max(-1, keepdim=True)[1]
        pred = pred.squeeze()
        
        accuracy_metric.update(pred, y)
        f1_metric.update(pred, y)
        recall_metric.update(pred, y)
        
    scheduler.step()
    
    acc = accuracy_metric.compute().item()
    f1_score = f1_metric.compute().item()
    recall = recall_metric.compute().item()

    # Reset metrics for the next epoch
    accuracy_metric.reset()
    f1_metric.reset()
    recall_metric.reset()
    
    return model, acc, f1_score, recall, avgloss/(bidx+1)

def main_test(model: torch.nn.Module, testloader:data.DataLoader, device:torch.device):  
    model.eval()

    accuracy_metric = MulticlassAccuracy(num_classes=model.parametric_prototypes.shape[0]).to(device)
    f1_metric = MulticlassF1Score(num_classes=model.parametric_prototypes.shape[0]).to(device)
    recall_metric = MulticlassRecall(num_classes=model.parametric_prototypes.shape[0]).to(device)
    
    true_labels = []
    predictions = []
    
    with torch.no_grad():
        for data, y in testloader:
            data = data.to(device)
            y = y.to(device)
            
            y = y.squeeze()
            true_labels.append(y)

            distances, _ = model(data, y)

            pred = distances.max(-1, keepdim=True)[1]
            pred = pred.squeeze().to(device)
            
            predictions.append(pred)
            
            accuracy_metric.update(pred, y)
            f1_metric.update(pred, y)
            recall_metric.update(pred, y)
    
    true_labels = torch.cat(true_labels, dim=0)
    predictions = torch.cat(predictions, dim=0)
    
    acc = accuracy_metric.compute().item()
    f1_score = f1_metric.compute().item()
    recall = recall_metric.compute().item()
    
    # Reset metrics for the next evaluation
    accuracy_metric.reset()
    f1_metric.reset()
    recall_metric.reset()
    
    return predictions, true_labels, acc, f1_score, recall

import argparse
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassRecall
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description="classification")
    parser.add_argument('-device',dest='device', default='cpu', type = str, help='device')
    parser.add_argument('-config',dest='config', default='configs/config.yaml', type = str, help='device')
    parser.add_argument('-seed',dest='seed', default=0, type = int, help='ranking of the run')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = parse_args()
    with open(args.config) as yaml_file:
        config = yaml.safe_load(yaml_file)

    config['seed'] = args.seed
        
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.deterministic = True
    np.random.seed(config['seed'])

    logger = initialize_logger_from_config(config)

    logger.log(config)

    trainloader, testloader = load_dataset(config['dataset'], config['batch_size'], num_workers = 8, val= config['validation'])

    model = load_backbone(config['model'], config['output_dim']) 
    model = metric_model(model, device = config['device'], 
                         output_dim = config['output_dim'], 
                         temperature = config['temperature'], 
                         dataset = config['dataset'], 
                         geometry = config['geometry']
                         )
    model = model.to(config['device'])
    
    opt = load_optimizer(model.parameters(), *list(config['optimizer'].values()))
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, config['lr_scheduler']['steps'], gamma=config['lr_scheduler']['entity'])

    total_start_time = time.time()

    prototype_dict = {i:0 for i in range(config['epochs'])}
    
    checkpoint_epoch = 0
    previous_parametric_proto = copy.deepcopy(model.parametric_prototypes.data)
    distance_with_previous_epoch_parametric_prototype = torch.tensor(0)
    for epoch in range(checkpoint_epoch, config['epochs']):
        t0 = time.time()
        with torch.no_grad():
        
            if epoch >= 1:
                distance_with_previous_epoch_parametric_prototype = torch.mean(model.manifold.manifold.dist(model.parametric_prototypes.data, previous_parametric_proto))
                cosine_distance_among_param_prototypes= torch.mean(nn.CosineSimilarity(dim=-1)(model.parametric_prototypes.data[:, None, :], model.parametric_prototypes.data[None, :, :]))
                
                stats = {
                        "step": epoch,
                        "parametric_proto_norm":torch.mean(torch.norm(model.parametric_prototypes.data,dim=1)).item(),
                        "cosine_similarity_parametric":cosine_distance_among_param_prototypes.item(),
                        "distance_with_previous_epoch_parametric_prototype":distance_with_previous_epoch_parametric_prototype.item(),
                        }
                
                prototype_dict[epoch] = model.parametric_prototypes.clone().detach().cpu().numpy() 
                
                previous_parametric_proto = copy.deepcopy(model.parametric_prototypes.data)
                
                logger(stats)
                    
        final_model, acc, f1_score, recall, loss_calculated = main_train(model, 
                                                                    trainloader, 
                                                                    opt = opt,
                                                                    scheduler = scheduler,
                                                                    device = config['device'])
        
        t1 = time.time()
        stats = {"step": epoch, 
                 "training_acc": acc, 
                 "f1_score": f1_score, 
                 "recall": recall,
                 "loss": loss_calculated}
        logger(stats)

        if epoch%config['eval_every'] == 0 or epoch == config['epochs']-1 :
            test_prediction, test_tl, test_acc, test_f1, test_recall = main_test(model=model, testloader=testloader, device = config['device'])
            logger({"step": epoch, 
                    "valid_acc": test_acc,
                    "valid_f1": test_f1,
                    "valid_recall": test_recall,})        
        
        if config['save_model']:
            torch.save({
                'epoch': epoch,
                'model_state_dict': final_model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                },  logger.results_directory + logger.name + '_model.pt')
    
    logger({"step": epoch, 
            "total_time": time.time() - total_start_time, 
            "test_acc": test_acc})
           
    np.save(logger.results_directory+logger.name+'.npy', test_prediction.cpu().numpy())
    np.save(logger.results_directory+logger.name+'_tl.npy', test_tl.cpu().numpy())
    with open(logger.results_directory+logger.name+'_param_proto'+'.pkl', "wb") as pickle_file:
        pkl.dump(prototype_dict, pickle_file)
    logger.finish()
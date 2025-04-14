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

def main_train(method: str, model, trainloader:data.DataLoader, opt:torch.optim.Optimizer, scheduler:torch.optim.lr_scheduler, device:torch.device = 'cpu', use_centroid: bool = False):
    model.train()
    avgloss = 0.
    acc = 0
    criterion = nn.CrossEntropyLoss()
    avg_proto_dist = 0
    times = []

    for bidx, (x, y) in enumerate(trainloader):
        time_0_fw = time.time()

        x, y = x.to(device), y.to(device)
        y = y.squeeze()
        
        opt.zero_grad()
        
        distances, embeddings = model(x, y)
        
        loss = criterion(distances, y) 
        
        loss.backward()
        
        avgloss += loss.item()

        opt.step()
        
        pred = prediction(method, distances, model.prototypes)
        pred = pred.squeeze()
        acc += (pred == y).sum().item() / len(y)  # this might be improved
        time_1_fw = time.time()
        times.append(time_1_fw - time_0_fw)
        
        if not use_centroid:
            model.calculate_centroid_prototypes(embeddings, y)
    
    scheduler.step()
    if not use_centroid:
        with torch.no_grad():
            proto_dist = torch.norm(model.prototypes - model.centroid_prototypes, dim = -1)
            avg_proto_dist = torch.mean(proto_dist).item()

    return model, acc/(bidx+1), avgloss/(bidx+1), avg_proto_dist, np.mean(np.array(times))

def main_test(method: str, model, testloader:data.DataLoader, device:torch.device):  
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for data,y in testloader:
            data = data.to(device)
            y = y.to(device)
            
            y = y.squeeze()
            true_labels.append(y)

            output, _ = model(data, y)

            pred = prediction(method, output, model.prototypes)
            
            pred = pred.squeeze().to(device)
            
            predictions.append(pred)
    
    true_labels = torch.cat(true_labels, dim=0)
    predictions = torch.cat(predictions, dim=0)
    acc = (true_labels == predictions).sum().item() / len(true_labels)
    return acc, predictions, true_labels

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="classification")
    parser.add_argument('-device',dest='device', default='cpu', type = str, help='device')
    parser.add_argument('-config',dest='config', default='config.json', type = str, help='device')
    parser.add_argument('-seed',dest='seed', default=0, type = int, help='ranking of the run')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = parse_args()
    with open(args.config) as json_file:
        config = json.load(json_file)

    config['seed'] = args.seed
        
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.deterministic = True
    np.random.seed(config['seed'])

    logger = initialize_logger_from_config(config)

    logger.log(config)

    trainloader, testloader = load_dataset(config['dataset'], config['batch_size'], num_workers = 8, val= config['validation'])

    # manifold = get_manifold(geometry, **kwargs)

    model = load_backbone(config['model'], config['output_dim']) 
    model = MGP_model(model, device = config['device'], output_dim = config['output_dim'], temperature = config['temperature'], dataset=config['dataset'], use_centroid=config['use_centroid'])
    model = model.to(config['device'])
    
    opt = load_optimizer(model.parameters(), *list(config['optimizer'].values()))
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, config['lr_scheduler']['steps'], gamma=config['lr_scheduler']['entity'])

    total_start_time = time.time()

    prototype_dict = {i:0 for i in range(config['epochs'])}
    centroid_prototype_dict = {i:0 for i in range(config['epochs'])}

    checkpoint_epoch = 0
    previous_parametric_proto = copy.deepcopy(model.parametric_prototypes.data)
    previous_centroid_proto = copy.deepcopy(model.centroid_prototypes.data)
    distance_with_previous_epoch_parametric_prototype = torch.tensor(0)
    distance_with_previous_epoch_centroid_prototype = torch.tensor(0)
    for epoch in range(checkpoint_epoch, config['epochs']):
        t0 = time.time()
        with torch.no_grad():
        
            if epoch >= 1:
                distance_with_previous_epoch_parametric_prototype = torch.mean(torch.norm(model.parametric_prototypes.data - previous_parametric_proto))
                distance_with_previous_epoch_centroid_prototype = torch.mean(torch.norm(model.centroid_prototypes.data - previous_centroid_proto))
                cosine_distance_among_param_prototypes= torch.mean(nn.CosineSimilarity(dim=-1)(model.parametric_prototypes.data[:, None, :], model.parametric_prototypes.data[None, :, :]))
                cosine_distance_among_centroid_prototypes= torch.mean(nn.CosineSimilarity(dim=-1)(model.centroid_prototypes.data[:, None, :], model.centroid_prototypes.data[None, :, :]))
                
                stats = {
                        "step": epoch,
                        "parametric_proto_norm":torch.mean(torch.norm(model.parametric_prototypes.data,dim=1)).item(),
                        "centroid_proto_norm":torch.mean(torch.norm(model.centroid_prototypes.data,dim=1)).item(),
                        "cosine_similarity_parametric":cosine_distance_among_param_prototypes.item(),
                        "cosine_similarity_centroid":cosine_distance_among_centroid_prototypes.item(),
                        "distance_with_previous_epoch_parametric_prototype":distance_with_previous_epoch_parametric_prototype.item(),
                        "distance_with_previous_epoch_centroid_prototype":distance_with_previous_epoch_centroid_prototype.item()}
                
                prototype_dict[epoch] = model.parametric_prototypes.clone().detach().cpu().numpy() 
                centroid_prototype_dict[epoch] = model.centroid_prototypes.clone().detach().cpu().numpy() 
                previous_parametric_proto = copy.deepcopy(model.parametric_prototypes.data)
                previous_centroid_proto = copy.deepcopy(model.centroid_prototypes.data)
                logger(stats)
                    
        final_model, acc, loss_calculated, avg_proto_dist, avg_time = main_train(config['method'],
                                                                    model, 
                                                                    trainloader, 
                                                                    opt = opt,
                                                                    scheduler = scheduler,
                                                                    device = config['device'],
                                                                    use_centroid=config['use_centroid'])
        
        t1 = time.time()
        stats = {"step": epoch, 
                 "training_acc": acc, 
                 "loss": loss_calculated, 
                 "parametric_centroid_distance": avg_proto_dist,
                 "average_batch_time":avg_time}
        logger(stats)

        if epoch%config['eval_every'] == 0 or epoch == config['epochs']-1 :
            test_acc, test_prediction, test_tl = main_test(method = config['method'], model=model, testloader=testloader, device = config['device'])
            logger({"step": epoch, 
                    "valid_acc": test_acc})        
        
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
    with open(logger.results_directory+logger.name+'_centroid_proto'+'.pkl', "wb") as pickle_file:
        pkl.dump(centroid_prototype_dict, pickle_file)
    logger.finish()
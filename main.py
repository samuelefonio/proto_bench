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
from utils_metrics import *

def main_train(model:torch.nn.Module, 
               trainloader:data.DataLoader, 
               opt:torch.optim.Optimizer, 
               scheduler:torch.optim.lr_scheduler, 
               device:torch.device = 'cpu',
               proto_opt = None
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
        if proto_opt is not None:
            proto_opt.zero_grad()
        distances, embeddings = model(x)
        
        loss = criterion(distances, y) 
        
        loss.backward()
        
        avgloss += loss.item()

        opt.step()
        if proto_opt is not None:
            proto_opt.step()
        #print(model.prototypes.data.norm())
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

            distances, _ = model(data)

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
    parser.add_argument('-device', dest='device', default=None, type = str, help='device')
    parser.add_argument('-config', dest='config', default=None, type = str, help='device')
    parser.add_argument('-seed', dest='seed', default=None, type = int, help='ranking of the run')
    parser.add_argument('-t', dest='temperature', default=None, type = float, help='temperature')
    parser.add_argument('-ex', dest='ex_4_class', default=None, type = int, help='Number of examples per class if in config the key reduced is True')
    parser.add_argument('-d', dest='output_dim', default=None, type = int, help='embedding dimension')
    parser.add_argument('-bs', dest='batch_size', default=None, type=int, help='batch size')
    parser.add_argument('-lr', dest='learning_rate', default=None, type=float, help='learning rate')
    parser.add_argument('-wd', dest='weight_decay', default=None, type=float, help='weight decay')
    parser.add_argument('-optim', dest='optimizer', default=None, type=str, help='optimizer')
    parser.add_argument('-shrink', dest='shrink_init', action='store_true')
    parser.add_argument('-protoopt', dest='proto_opt', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = parse_args()
    with open(args.config) as yaml_file:
        config = yaml.safe_load(yaml_file)
    if args.device is not None:
        config['device'] = args.device
    if args.seed is not None:
        config['seed'] = args.seed
    if args.temperature is not None:
        config['temperature'] = args.temperature
    if args.ex_4_class is not None:
        config['dataset']['ex_4_class'] = args.ex_4_class
    if args.output_dim is not None:
        config['output_dim'] = args.output_dim
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.optimizer is not None:
        config['optimizer']['name'] = args.optimizer
    if args.weight_decay is not None:
        config['optimizer']['weight_decay'] = args.weight_decay
    if args.learning_rate is not None:
        config['optimizer']['learning_rate'] = args.learning_rate

    

    config['shrink_init'] = args.shrink_init
    config['proto_opt'] = args.proto_opt
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.deterministic = True
    np.random.seed(config['seed'])

    logger = initialize_logger_from_config(config)

    logger.log(config)

    trainloader, testloader , validloader  = load_dataset(config['dataset']['name'], 
                                           config['batch_size'], 
                                           num_workers = 4, 
                                           reduced = config['dataset']['reduced'],
                                           ex_4_class = config['dataset']['ex_4_class'])

    model = load_backbone(config) 
    model = metric_model(model, device = config['device'], 
                         output_dim = config['output_dim'], 
                         temperature = config['temperature'], 
                         dataset = config['dataset']['name'], 
                         geometry = config['geometry'],
                         shrink_init = args.shrink_init
                         )
    model = model.to(config['device'])
    if config['proto_opt']:
        filtered_parameters = [p for name, p in model.named_parameters() if 'proto' not in name]
        proto_params = [p for name, p in model.named_parameters() if 'proto' in name]
        optimizer_parameters = [{'params': filtered_parameters}]
        opt = load_optimizer(optimizer_parameters, *list(config['optimizer'].values()))
        proto_opt = geoopt.optim.RiemannianSGD(proto_params, lr = 0.001, momentum=0.9, dampening=0, weight_decay=0, nesterov=False, stabilize=None)
    else:
        opt = load_optimizer(model.parameters(), *list(config['optimizer'].values()))
        proto_opt = None
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, config['lr_scheduler']['steps'], gamma=config['lr_scheduler']['entity'])

    total_start_time = time.time()

    prototype_dict = {i:0 for i in range(config['epochs'])}
    
    early_stopping = 0
    best_valid_acc = 0
    previous_parametric_proto = copy.deepcopy(model.parametric_prototypes.data)
    distance_with_previous_epoch_parametric_prototype = torch.tensor(0)
    for epoch in range(config['epochs']):
        t0 = time.time()
        with torch.no_grad():
        
            if epoch >= 1:
                distance_with_previous_epoch_parametric_prototype = torch.mean(model.manifold.manifold.dist(model.parametric_prototypes.data, previous_parametric_proto))
                cosine_distance_among_param_prototypes = torch.mean(nn.CosineSimilarity(dim=-1)(model.parametric_prototypes.data[:, None, :], model.parametric_prototypes.data[None, :, :]))
                
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
                                                                    device = config['device'],
                                                                    proto_opt = proto_opt)
        
        t1 = time.time()
        stats = {"step": epoch, 
                 "training_acc": acc, 
                 "f1_score": f1_score, 
                 "recall": recall,
                 "loss": loss_calculated}
        logger(stats)

        val_prediction, val_tl, val_acc, val_f1, val_recall = main_test(model=model, testloader=validloader, device = config['device'])
        logger({"step": epoch, 
                "valid_acc": val_acc,
                "valid_f1": val_f1,
                "valid_recall": val_recall,}) 
        
        if val_acc>best_valid_acc:
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': final_model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    },  logger.exp_directory + logger.name + '.pt')
            best_valid_acc = val_acc
            early_stopping = 0  #reset early stopping counter
        else:
            if epoch > 160:
                early_stopping+=1

        if early_stopping > config['patience']:
            logger({"step": epoch, "info": "Early stopping"})
            break      
    
    model.load_state_dict(torch.load(logger.exp_directory + logger.name + '.pt')['model_state_dict'])
    test_prediction, test_tl, test_acc, test_f1, test_recall = main_test(model=model, testloader=testloader, device = config['device'])
    logger({"step": epoch, 
            "total_time": time.time() - total_start_time, 
            "test_acc": test_acc,
            "test_f1": test_f1,
            "test_recall": test_recall})
    
    test_FGSM, test_PGD = get_robustness(testloader, model, config)
    for i in range(len(test_FGSM)):
        logger({"step": epoch+1+i, 
                "test_FGSM_eps": test_FGSM[i],
                "test_PGD_eps": test_PGD[i]})
        
    results_OOD = get_OOD(model, config)
    for key, value in results_OOD.items():
        logger({"step": epoch+1+i, 
                f"confidence_{config['dataset']['name']}_on_{key}": value[0],
                f"confidence_std_{config['dataset']['name']}_on_{key}": value[1]})

    # np.save(logger.results_directory+logger.name+'.npy', test_prediction.cpu().numpy())
    # np.save(logger.results_directory+logger.name+'_tl.npy', test_tl.cpu().numpy())
    with open(logger.exp_directory+logger.name+'_proto'+'.pkl', "wb") as pickle_file:
        pkl.dump(prototype_dict, pickle_file)
    logger.finish()
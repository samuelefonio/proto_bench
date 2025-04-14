
import torch
import pickle as pkl
import torch.optim as optim
import geoopt
import torch.nn as nn


DATASETS_CLASSES = {'mnist':10,
               'cifar10':10,
               'cifar100':100,
               'cub':200,
               'aircraft':100,
               'cars':196}

class SeparationLoss(nn.Module):
    '''
    Source: https://github.com/VSainteuf/metric-guided-prototypes-pytorch
    Large margin separation between hyperspherical protoypes
    '''

    def __init__(self):
        super(SeparationLoss, self).__init__()

    def forward(self, protos):
        '''
        Args:
            protos (tensor): (N_prototypes x Embedding_dimension)
        '''
        M = torch.matmul(protos, protos.transpose(0, 1)) - 2 * torch.eye(
            protos.shape[0]
        ).to(protos.device)
        return M.max(dim=1)[0].mean()


def hyperspherical_embedding(dataset, device, embedding_dimension, seed):
    '''
    Source: https://github.com/VSainteuf/metric-guided-prototypes-pytorch.
    Function to learn the prototypes according to the separationLoss Minimization
    embedding_dimension : 
    We use SGD as optimizer
    lr : learning rate
    momentum : momentum
    n_steps : number of steps for learning the prototypes
    wd : weight decay
    '''
    lr=0.1
    momentum=0.9
    n_steps=1000
    wd=1e-4

    torch.manual_seed(seed) 
    mapping = torch.rand((DATASETS_CLASSES[dataset], embedding_dimension), device=device, requires_grad = True)
    
    optimizer = torch.optim.SGD([mapping], lr=lr, momentum=momentum, weight_decay=wd)
    L_hp = SeparationLoss()
    
    for i in range(n_steps):
        with torch.no_grad():
            mapping.div_(torch.norm(mapping, dim=1, keepdim=True))
        optimizer.zero_grad()
        loss = L_hp(mapping)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        mapping.div_(torch.norm(mapping, dim=1, keepdim=True))
    return mapping.detach()


class HPS_loss(nn.Module):
    def __init__(self, prototypes):
        super(HPS_loss, self).__init__()
        self.prototypes = prototypes

    def forward(self, output, target):
        dist = (1 - nn.CosineSimilarity(eps=1e-9)(output, self.prototypes[target])).pow(2).sum()
        return dist
    

def load_cost_matrix(dataset_name):
    basedir = f"./Datasets_features/{dataset_name}"
    dataset_name = dataset_name.lower()
    if dataset_name == 'mnist':
        return torch.Tensor(pkl.load(open(basedir + f'/{dataset_name}.pkl','rb')))[:10,:][:,:10]    
    return torch.Tensor(pkl.load(open(basedir + f'/{dataset_name}.pkl','rb')))
    

def load_optimizer(params, *args):
    optimname = args[0].lower()
    learning_rate = args[1]
    decay = args[2]
    momentum = args[3]
    
    if optimname == "sgd":
        optimizer = optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=decay)
    elif optimname == "adam":
        optimizer = optim.AdamW(params, lr=learning_rate, weight_decay=decay)
    elif optimname == "rsgd":
        optimizer = geoopt.optim.RiemannianSGD(params, lr=learning_rate, weight_decay=decay)
    elif optimname == "radam":
        optimizer = geoopt.optim.RiemannianAdam(params, lr=learning_rate, weight_decay=decay)
    return optimizer

def clip(input_vector, r):
    input_norm = torch.norm(input_vector, dim = -1)
    clip_value = float(r)/input_norm 
    min_norm = torch.clamp(float(r)/input_norm, max = 1)
    return min_norm[:, None] * input_vector

def prediction(method, output, prototypes):
    if method in ['HPS']:
        output = nn.CosineSimilarity(dim=-1)(output[:,None,:], prototypes[None,:,:])
        pred = output.max(-1, keepdim=True)[1]
    elif method in ['CHPS','RHPN','MGP','XE']:
        pred = output.max(-1, keepdim=True)[1]
    return pred

import torch.nn as nn
import torch
import resnet
import torchvision.models as torchmodel
from utils import *
import torch_scatter
import copy

DATASETS_CLASSES = {'mnist':10,
               'cifar10':10,
               'cifar100':100,
               'cub':200,
               'aircraft':100,
               'cars':196}

# METRICS = {
#     'euclidean': -torch.cdist(),
#     'cosine_similarity': nn.CosineSimilarity(dim=-1),}

class SimpleCNN(nn.Module):
    '''Source: https://github.com/VSainteuf/metric-guided-prototypes-pytorch'''
    def __init__(self, output_dim = 10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1,16,3),nn.BatchNorm2d(16),nn.ReLU(),nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(16,16,3),nn.BatchNorm2d(16),nn.ReLU(),nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(nn.Conv2d(16,32,3),nn.BatchNorm2d(32),nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(32, 64),nn.Linear(64,output_dim))
        
    def forward(self, x):
        out = self.conv3(self.conv2(self.conv1(x)))
        out = out.view(out.shape[0],out.shape[1], -1).max(-1)[0]
        out = self.fc(out)
        return out
    


# def load_model(model, *args):
#     method = args[1]
#     if method in ['HPS','XE']:
#         return HPS_model(model, *args) 
#     elif method == 'RHPN':
#         return RHPN_model(model, *args)
#     elif method == 'CHPS':
#         return CHPS_model(model, *args)
#     elif method == 'MGP':
#         return MGP_model(model, *args)


# class HPS_model(nn.Module):
#     def __init__(self,
#             model,
#             device = 'cuda',
#             pre_trained_prototypes = None,
#             output_dim = 8,
#             manifold = None):
#             super(HPS_model, self).__init__()
#             self.model = model
#             self.device = device
#             self.output_dim = output_dim
#             self.prototypes = pre_trained_prototypes
#             self.manifold = manifold
#     def forward(self, images):
#         embeddings = self.model(images)
#         embeddings = F.normalize(embeddings, p=2, dim=1)
#         with torch.no_grad():
#             norm = torch.mean(torch.norm(embeddings, dim = 1)).item()
#         return embeddings, norm, embeddings
        
    
# class RHPN_model(nn.Module):
#     def __init__(
#         self,
#         model,
#         device = 'cuda',
#         method = 'mgp',
#         pre_trained_prototypes = None,
#         geometry = "euclidean",
#         dataset = 'mnist',
#         output_dim = 8,
#         grad = False,
#         temperature = 0.01,
#         clipping = 1,
#         manifold = None
#     ):
#         super(RHPN_model, self).__init__()
#         self.model = model
#         self.method = method
#         self.device = device
#         self.dataset = dataset
#         self.output_dim = output_dim
#         self.geometry = geometry
#         self.temperature = temperature
#         self.clipping = clipping
#         self.manifold = manifold
        
#         prototypes = self.manifold.expmap0(-0.1 * torch.rand((DATASETS_CLASSES[self.dataset], self.output_dim), device = self.device) + 0.1)
#         self.prototypes = geoopt.ManifoldParameter(prototypes, manifold=manifold, requires_grad=grad)
        
#     def forward(self, images):
#         old_embeddings = self.model(images)      
#         old_embeddings = clip(old_embeddings, self.clipping)
#         embeddings = self.manifold.expmap0(old_embeddings)
#         norm = torch.norm(embeddings, dim = 1) 
#         dists = -self.manifold.dist(embeddings[:, None, :], self.prototypes[None, :, :])        
#         return dists / self.temperature, norm, old_embeddings

class MGP_model(nn.Module):
    def __init__(
        self,
        model,
        device: str = 'cuda',
        output_dim: int = 8,
        temperature: float = 1.0,
        use_centroid: bool = True,
        dataset: str = 'mnist',
        metric: str = 'euclidean'
    ):
        super(MGP_model, self).__init__()
        self.model = model
        self.device = device
        self.output_dim = output_dim
        self.temperature = temperature
        self.use_centroid = use_centroid
        self.dataset = dataset
        self.num_classes = DATASETS_CLASSES[self.dataset]
        self.parametric_prototypes = nn.Parameter(torch.rand((self.num_classes, self.output_dim), device = self.device), requires_grad=True)
        self.centroid_prototypes = nn.Parameter(torch.rand((self.num_classes, self.output_dim), device = self.device), requires_grad=False)
        if self.use_centroid:
            self.prototypes = self.centroid_prototypes
        else:
            self.prototypes = self.parametric_prototypes
        self.counter = torch.zeros(DATASETS_CLASSES[self.dataset])
        # self.metric = METRICS[metric]

    def forward(self, images, y):
        embeddings = self.model(images)
        
        if self.use_centroid:
            self.calculate_centroid_prototypes(embeddings, y)
            self.prototypes = self.centroid_prototypes
        else:
            self.prototypes = self.parametric_prototypes
        dists = -torch.norm(embeddings[:, None, :] - self.prototypes[None, :, :], dim=-1)
        return dists/self.temperature, embeddings
    
    @torch.no_grad()
    def calculate_centroid_prototypes(self, embeddings: torch.tensor, y:torch.tensor):
        represented_classes = torch.unique(y).detach().cpu().numpy()
        
        
        # # Compute Prototypes
        new_prototypes = torch_scatter.scatter_mean(
            embeddings, y.unsqueeze(1), dim=0, dim_size=self.num_classes
        ).detach()
        
        # check_proto = copy.deepcopy(self.centroid_prototypes)
        # class_count = torch.bincount(y)
        # weights_old = class_count / (class_count + 1)
        # weights_new = 1 / (class_count + 1)
        # print(class_count)
        # print(weights_old)
        # print(weights_new)
        # print(new_prototypes.shape)
        # check_proto[represented_classes, :] = weights_new.unsqueeze(-1) * check_proto[represented_classes, :] \
        #                                                     + weights_old.unsqueeze(-1) * new_prototypes[represented_classes, :]
        # self.counter[represented_classes]

        # Updated stored prototype values
        self.centroid_prototypes[represented_classes, :] = (
            self.counter[represented_classes, None]
            * self.centroid_prototypes[represented_classes, :]
            + new_prototypes[represented_classes, :]
        ) / (self.counter[represented_classes, None] + 1)
        self.counter[represented_classes]
        self.counter[represented_classes] = self.counter[represented_classes] + 1
        # print(self.counter)

        # assert torch.all(torch.eq(self.centroid_prototypes, check_proto))
        
        
    
# class CHPS_model(nn.Module):
#     def __init__(
#         self,
#         model,
#         device = 'cuda',
#         method = 'mgp',
#         pre_trained_prototypes = None,
#         geometry = "euclidean",
#         dataset = 'mnist',
#         output_dim = 8,
#         temperature = 1.0
#     ):
#         super(CHPS_model, self).__init__()
#         self.model = model
#         self.method = method
#         self.device = device
#         self.dataset = dataset
#         self.output_dim = output_dim
#         self.geometry = geometry
#         self.temperature = temperature
#         self.prototypes = pre_trained_prototypes
        
#     def forward(self, images):
#         embeddings = self.model(images)
#         embeddings = F.normalize(embeddings, p=2, dim=1)
#         with torch.no_grad():
#             norm = torch.mean(torch.norm(embeddings, dim = 1)).item()
#         dist = -(1 - nn.CosineSimilarity(dim=-1)(embeddings[:, None, :], self.prototypes[None, :, :]))     
#         return dist, norm, embeddings

def load_backbone(model, output_dim):
    modello = model.lower()
    if modello == 'simplecnn':
        out_model = SimpleCNN(output_dim = output_dim)
    elif modello == 'resnet18':
        out_model = resnet.resnet18()           
        num_ftrs = out_model.fc.in_features
        out_model.fc = nn.Linear(num_ftrs, output_dim)
    else:
        raise Exception('Selected dataset is not available.')

    return out_model

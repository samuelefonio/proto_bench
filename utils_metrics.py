from typing import Tuple
import torch
import torch
import torch.nn.functional as F
from torchvision import transforms
import torchattacks
import logging
from data import load_dataset
import torch.nn as nn
from torchattacks.attack import Attack
import torch.nn.functional as F

mean_dict = {
    'cifar10': [0.4914, 0.4822, 0.4465],
    'cifar100': [0.507, 0.487, 0.441],
    'cub' : [0.485, 0.456, 0.406],
    'aircraft' : [0.485, 0.456, 0.406],
    'cars' : [0.485, 0.456, 0.406],
    'mnist' : [0.13066062]
}

std_dict = {
    'cifar10' : [0.2023, 0.1994, 0.2010],
    'cifar100': [0.267, 0.256, 0.276],
    'cub': [0.229, 0.224, 0.225],
    'aircraft': [0.229, 0.224, 0.225],
    'cars': [0.229, 0.224, 0.225],
    'mnist' : [0.30810776]
}

DIMENSION = {
    'cifar100':64,
    'cifar10':8,
    'cub':128,
    'aircraft':64,
    'cars':128
}


class custom_FGSM(Attack):
    r"""
    FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.FGSM(model, eps=8/255)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=8 / 255):
        super().__init__("FGSM", model)
        self.model = model
        self.eps = eps
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, labels):

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        images.requires_grad = True
        # outputs = self.get_logits(images)
        outputs, _ = self.model(images)

        if self.targeted:
            cost = -loss(outputs, target_labels)
        else:
            cost = loss(outputs, labels)

        grad = torch.autograd.grad(
            cost, images, retain_graph=False, create_graph=False
        )[0]

        adv_images = images + self.eps * grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images

class custom_PGD(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of iterations. (Default: 40)

    Shape:
        - images: (N, C, H, W) with values in [0, 1]
        - labels: (N)

    Example::
        >>> attack = custom_PGD(model, eps=8/255, alpha=2/255, steps=40)
        >>> adv_images = attack(images, labels)
    """

    def __init__(self, model, eps=8/255, alpha=2/255, steps=40):
        super().__init__("PGD", model)
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        adv_images = images + torch.empty_like(images).uniform_(-self.eps, self.eps)
        adv_images = torch.clamp(adv_images, 0, 1).detach()

        loss = nn.CrossEntropyLoss()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs, _ = self.model(adv_images)
            # outputs = F.softmax(outputs, dim=1)

            if self.targeted:
                target_labels = self.get_target_label(images, labels)
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            adv_images = adv_images + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images


def denorm(batch, mean=[0.1307], std=[0.3081], device = 'cpu'):
    """
    Convert a batch of tensors to their original scale.

    Args:
        batch (torch.Tensor): Batch of normalized tensors.
        mean (torch.Tensor or list): Mean used for normalization.
        std (torch.Tensor or list): Standard deviation used for normalization.

    Returns:
        torch.Tensor: batch of tensors without normalization applied to them.
    """
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

def robustness( model, device, test_loader, epsilon, dataset, attack):
    correct = 0
    den = 0
    adv_examples = []
    mean = mean_dict[dataset]
    std = std_dict[dataset]
    if attack == 'FGSM':
        atk = custom_FGSM(model, eps=epsilon)
    elif attack == 'PGD':
        atk = custom_PGD(model, eps=epsilon, alpha=epsilon/2, steps=40)

    # atk.set_normalization_used(mean = mean, std = std)
    for data, target in test_loader:

        data, target = data.to(device), target.to(device)

        dists, embeddings = model(data)
        data_denorm = denorm(data, mean=mean, std=std, device=device)
        adv_images = atk(data_denorm, target)
        adv_images = transforms.Normalize(mean, std)(adv_images)
        dists, embeddings = model(adv_images)

    
        final_pred = dists.max(-1, keepdim=True)[1] 
        final_pred = final_pred.squeeze()
        correct += (final_pred == target).sum().item()
        den += len(target)
    
    final_acc = correct/den
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {den} = {final_acc}")

    return final_acc, adv_examples



def get_robustness(test_loader, model, config):
    epsilons = [0, 0.8/255, 1.6/255, 3.2/255]
    
    FGSM_accuracies = []
    PGD_accuracies = []
    examples = []

    for attack in ['FGSM', 'PGD']:
        for eps in epsilons:
            
            acc, adv_ex = robustness(model, config['device'], test_loader, eps, config['dataset']['name'], attack=attack)
            if attack == 'FGSM':
                FGSM_accuracies.append(acc)
            else:
                PGD_accuracies.append(acc)
            examples.append(adv_ex)
    return FGSM_accuracies, PGD_accuracies

    
@torch.inference_mode()
def OOD(
    model:torch.nn.Module,
    device: torch.device, 
    test_loader: torch.utils.data.DataLoader
)-> Tuple[float, float]:

    model.eval()
    scores = []

    for data, _ in test_loader:
        
        data = data.to(device)
        output = model(data)
        if isinstance(output, tuple):
            output = output[0]
        output = F.softmax(output, dim = 1) 
        score = output.max(dim=1, keepdim=True)[0] 
        scores.append(score.cpu())
    
    scores = torch.cat(scores, dim=0)
    avg_confidence = torch.mean(scores).item()
    std_confidence = torch.std(scores).item()

    print(f'Average confidence = {avg_confidence} +- {std_confidence}')

    return avg_confidence, std_confidence

def get_OOD(model, config):
    datasets_list_0 = ['cifar10','cifar100']
    datasets_list_1 = ['cub','aircraft']
    OOD_datasets = {
        'cifar10':datasets_list_0,
        'cifar100':datasets_list_0,
        'cub':datasets_list_1,
        'aircraft':datasets_list_1
    }
    datasets_list = OOD_datasets[config['dataset']['name']] 
    results = {datasets_list[i]: 0 for i in range(len(datasets_list))}   
    # torch.cuda.mem_get_info()
    for test_set in datasets_list:
        _, test_loader, _ = load_dataset(test_set, 128, num_workers = 4)
        confidence_avg, confidence_std  = OOD(model, config['device'], test_loader)
        results[test_set] = (confidence_avg, confidence_std)

    return results
    
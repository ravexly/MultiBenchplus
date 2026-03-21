import torch
import torch.nn.functional as F
import torch.nn as nn
def KL(alpha, c):
    beta = torch.ones((1, c)).to(alpha.device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl

class DirichletCELoss(nn.Module):
    def __init__(self, n_classes, annealing_epoch):
        super().__init__()
        self.n_classes = n_classes
        self.annealing_epoch = annealing_epoch  # will be updated externally

   

    def forward(self, alphas, targets,args):
        """
        alphas: list or tuple of alpha tensors (one per modality)
        targets: LongTensor of shape [batch_size]
        """
        # print(args)
        self.current_epoch = args["epoch"]
        loss = 0
        for alpha in alphas:
            S = torch.sum(alpha, dim=1, keepdim=True)
            E = alpha - 1
            label = F.one_hot(targets, num_classes=self.n_classes)

            A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

            annealing_coef = min(1, self.current_epoch / self.annealing_epoch)
            alp = E * (1 - label) + 1
            B = annealing_coef * KL(alp, self.n_classes)

            loss += torch.mean(A + B)
        return loss
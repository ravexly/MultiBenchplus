# #!/usr/bin/env python3
# #
# # Copyright (c) Facebook, Inc. and its affiliates.
# # All rights reserved.
# #
# # This source code is licensed under the license found in the
# # LICENSE file in the root directory of this source tree.
# #

# import torch
# import torch.nn as nn

# from .bert import BertEncoder,BertClf
# from .image import ImageEncoder,ImageClf
# import torch.nn.functional as F
# def KL(alpha, c):
#     beta = torch.ones((1, c)).cuda()
#     S_alpha = torch.sum(alpha, dim=1, keepdim=True)
#     S_beta = torch.sum(beta, dim=1, keepdim=True)
#     lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
#     lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
#     dg0 = torch.digamma(S_alpha)
#     dg1 = torch.digamma(alpha)
#     kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
#     return kl
# def ce_loss(p, alpha, c, global_step, annealing_step):
#     S = torch.sum(alpha, dim=1, keepdim=True)
#     E = alpha - 1

#     label = F.one_hot(p, num_classes=c)
    
#     A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

#     annealing_coef = min(1, global_step / annealing_step)
#     alp = E * (1 - label) + 1
#     B = annealing_coef * KL(alp, c)
#     return torch.mean((A + B))
# class TMC(nn.Module):
#     def __init__(self, args):
#         super(TMC, self).__init__()
#         self.args = args

#         self.txtclf = BertClf(args)
#         self.imgclf= ImageClf(args)

#     def DS_Combin_two(self, alpha1, alpha2):
#         # Calculate the merger of two DS evidences
#         alpha = dict()
#         alpha[0], alpha[1] = alpha1, alpha2
#         b, S, E, u = dict(), dict(), dict(), dict()
#         for v in range(2):
#             S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
#             E[v] = alpha[v] - 1
#             b[v] = E[v] / (S[v].expand(E[v].shape))
#             u[v] = self.args.n_classes / S[v]

#         # b^0 @ b^(0+1)
#         bb = torch.bmm(b[0].view(-1, self.args.n_classes, 1), b[1].view(-1, 1, self.args.n_classes))
#         # b^0 * u^1
#         uv1_expand = u[1].expand(b[0].shape)
#         bu = torch.mul(b[0], uv1_expand)
#         # b^1 * u^0
#         uv_expand = u[0].expand(b[0].shape)
#         ub = torch.mul(b[1], uv_expand)
#         # calculate K
#         bb_sum = torch.sum(bb, dim=(1, 2), out=None)
#         bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
#         # bb_diag1 = torch.diag(torch.mm(b[v], torch.transpose(b[v+1], 0, 1)))
#         K = bb_sum - bb_diag

#         # calculate b^a
#         b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - K).view(-1, 1).expand(b[0].shape))
#         # calculate u^a
#         u_a = torch.mul(u[0], u[1]) / ((1 - K).view(-1, 1).expand(u[0].shape))
#         # test = torch.sum(b_a, dim = 1, keepdim = True) + u_a #Verify programming errors

#         # calculate new S
#         S_a = self.args.n_classes / u_a
#         # calculate new e_k
#         e_a = torch.mul(b_a, S_a.expand(b_a.shape))
#         alpha_a = e_a + 1
#         return alpha_a

#     def forward(self, txt, mask, segment, img):
#         txt_out = self.txtclf(txt, mask, segment)
#         img_out = self.imgclf(img)

#         txt_evidence,img_evidence = F.softplus(txt_out), F.softplus(img_out)
#         txt_alpha, img_alpha = txt_evidence + 1,img_evidence + 1
#         txt_img_alpha = self.DS_Combin_two(txt_alpha, img_alpha)
#         return txt_alpha, img_alpha, txt_img_alpha
import torch
import torch.nn as nn
import torch.nn.functional as F

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

def ce_loss(p, alpha, c, global_step, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1

    label = F.one_hot(p, num_classes=c)
    
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c)
    return torch.mean((A + B))

class TMC(nn.Module):
    def __init__(self,n_classes):
        super(TMC, self).__init__()
        self.n_classes =n_classes

    def DS_Combin(self, alphas):
        """
        Combine multiple DS evidences.
        
        Args:
            alphas (list of torch.Tensor): List of alpha values from different modalities.
        
        Returns:
            torch.Tensor: Combined alpha values.
        """
        num_modalities = len(alphas)
        b, S, E, u = dict(), dict(), dict(), dict()

        for v in range(num_modalities):
            S[v] = torch.sum(alphas[v], dim=1, keepdim=True)
            E[v] = alphas[v] - 1
            b[v] = E[v] / (S[v].expand(E[v].shape))
            u[v] = self.n_classes / S[v]

        # Initialize combined alpha
        combined_b = torch.zeros_like(b[0])
        combined_u = torch.ones_like(u[0])

        for i in range(num_modalities):
            for j in range(num_modalities):
                if i != j:
                    # b^i @ b^j
                    bb = torch.bmm(b[i].view(-1, self.n_classes, 1), b[j].view(-1, 1, self.n_classes))
                    # b^i * u^j
                    uv_expand = u[j].expand(b[i].shape)
                    bu = torch.mul(b[i], uv_expand)
                    # b^j * u^i
                    uv_expand = u[i].expand(b[j].shape)
                    ub = torch.mul(b[j], uv_expand)
                    # calculate K
                    bb_sum = torch.sum(bb, dim=(1, 2), out=None)
                    bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
                    K = bb_sum - bb_diag

                    # update combined_b and combined_u
                    combined_b += (torch.mul(b[i], b[j]) + bu + ub) / ((1 - K).view(-1, 1).expand(b[i].shape))
                    combined_u *= torch.mul(u[i], u[j]) / ((1 - K).view(-1, 1).expand(u[i].shape))

        # calculate new S and e_k
        S_a = self.n_classes / combined_u
        e_a = torch.mul(combined_b, S_a.expand(combined_b.shape))
        alpha_a = e_a + 1

        return alpha_a

    def forward(self, logits):
        """
        Apply TMC to the logits from multiple modality classifiers.

        Args:
            *logits (torch.Tensor): Variable number of logits tensors from different modalities.

        Returns:
            list of torch.Tensor: List of alpha values for each modality and the combined alpha.
        """
        alphas = [F.softplus(logit) + 1 for logit in logits]
        combined_alpha = self.DS_Combin(alphas)
        return [combined_alpha]+alphas



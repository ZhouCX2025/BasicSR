import math
import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import LOSS_REGISTRY

@LOSS_REGISTRY.register()
class MCELoss(nn.Module):
    """pred: (h, w, q), target: (h, w, q), weighting term v(w(argmax_q(target(h,w,q))))"""
    def __init__(self, pred, target, weighting):
        super(MCELoss, self).__init__()
        self.pred = pred
        self.target = target
        self.weighting = weighting

    def MCELoss(self, pred, target, weighting, w_dist):
        loss = 0
        for h in range(pred.size(0)):
            for w in range(pred.size(1)):
                sumq = 0
                for q in range(pred.size(2)):
                    sumq += target[h, w, q] * math.log(pred[h, w, q])
                weighting = self.class_rebalancing(target, w_dist, h, w)
                lloss += -weighting * sumq
        return loss

    def class_rebalancing(self, target, w_distribution, h, w):
        """target: (h, w, q), empirical_distribution: (q)"""
        q_star = torch.argmax(target[h, w])
        w = w_distribution[q_star]
        return w

    def w_distribution(self, empirical_distribution, lambda_, Q = 313):
        """empirical_distribution: (q)"""
        w = ((1-lambda_)*empirical_distribution + lambda_/Q)**(-1)
        #normalize
        E = 0
        for q in range(len(w)):
            E += empirical_distribution[q] * w[q]
        w = w / E
        return w






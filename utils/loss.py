import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='dice'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == 'dice':
            return self.DiceLoss
        else:
            raise NotImplementedError


    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.25):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt
        if self.batch_average:
            loss /= n

        return loss


    def DiceLoss(self, logit, target):
        n, c, h, w = logit.size()

        l = np.ones((h, w))
        l = torch.tensor(l).cuda()
        probs = torch.sigmoid(logit)

        label1 = target
        label0 = l - target
        output0 = probs[:, 0, :, :]
        output1 = probs[:, 1, :, :]
        intersection0 = output0 * label0
        intersection1 = output1 * label1
        DSC0 = (2 * torch.abs(torch.sum(intersection0)) + 1) / (torch.abs(torch.sum(output0)) + torch.sum(label0) + 1)
        DSC1 = (2 * torch.abs(torch.sum(intersection1)) + 1) / (torch.abs(torch.sum(output1)) + torch.sum(label1) + 1)
        loss = 1 - (DSC0 + DSC1) / n /2
        return loss

if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 2, 513, 513).cuda()
    b = torch.rand(1, 513, 513).cuda()

    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())
    print(loss.DiceLoss(a, b).item())







import torch
import torch.nn as nn
import numpy as np


class BaS_Net_loss(nn.Module):
    def __init__(self, alpha, bg_loss):
        super(BaS_Net_loss, self).__init__()
        self.alpha = alpha
        self.beta = bg_loss  # background loss weight
        self.ce_criterion = nn.BCELoss()

    def forward(self, score_base, score_supp, fore_weights, label, score_bg):
        loss = {}

        label_base = torch.cat((label, torch.ones((label.shape[0], 1)).cuda()), dim=1)
        label_supp = torch.cat((label, torch.zeros((label.shape[0], 1)).cuda()), dim=1)
        label_bg = torch.cat((torch.zeros(label.shape).cuda(), torch.ones((label.shape[0], 1)).cuda()), dim=1)

        label_base = label_base / torch.sum(label_base, dim=1, keepdim=True)
        label_supp = label_supp / torch.sum(label_supp, dim=1, keepdim=True)

        loss_base = self.ce_criterion(score_base, label_base)
        loss_supp = self.ce_criterion(score_supp, label_supp)
        loss_norm = torch.mean(torch.norm(fore_weights, p=1, dim=1))
        loss_bg = self.ce_criterion(score_bg, label_bg)

        loss_total = loss_base + loss_supp + self.alpha * loss_norm + self.beta * loss_bg

        loss["loss_base"] = loss_base
        loss["loss_supp"] = loss_supp
        loss["loss_norm"] = loss_norm
        loss["loss_bg"] = loss_bg
        loss["loss_total"] = loss_total

        return loss_total, loss


def train(net, train_loader, loader_iter, optimizer, criterion, logger, step):
    net.train()
    try:
        _data, _label, _, _, _ = next(loader_iter)
    except:
        loader_iter = iter(train_loader)
        _data, _label, _, _, _ = next(loader_iter)

    _data = _data.cuda()
    _label = _label.cuda()

    optimizer.zero_grad()

    score_base, _, score_supp, _, fore_weights, score_bg = net(_data)

    cost, loss = criterion(score_base, score_supp, fore_weights, _label, score_bg)

    cost.backward()
    optimizer.step()

    for key in loss.keys():
        logger.log_value(key, loss[key].cpu().item(), step)

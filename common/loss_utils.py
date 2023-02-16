import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def calc_loss(y_true, y_pred, train_weights=None):
    loss = nn.CrossEntropyLoss(weight=train_weights)
    y_pred = F.softmax(y_pred, dim=1)
    loss = loss(y_true, y_pred)
    return loss


def d_loss(y_true, y_pred, aug_y_pred, w=0.5, train_weights=None):
    """Loss function for the discriminator (classifier)"""
    y_loss = calc_loss(y_true, y_pred, train_weights)  # loss for original
    aug_y_loss = calc_loss(y_true, aug_y_pred, train_weights)  # loss for augmented

    loss = (w * y_loss) + (w * aug_y_loss)

    return loss


def g_loss(y_true, y_pred, aug_y_pred, data, aug, eps=2e-4, train_weights=None):
    """Loss function for the generator (augmentor)"""
    B, C, N = data.size()
    cos = nn.CosineEmbeddingLoss()
    LeakyReLU = nn.LeakyReLU(0.0)  # leaky relu
    y_loss = calc_loss(y_true, y_pred, train_weights)  # loss for original
    aug_y_loss = calc_loss(y_true, aug_y_pred, train_weights)  # loss for augmented
    
    aug_cos = cos(torch.reshape(data.transpose(1,2), (-1,3)), torch.reshape(aug.transpose(1,2), (-1,3)) , torch.ones(N * B).to("cuda"))
    loss = LeakyReLU(y_loss - aug_y_loss + aug_cos).mean()  # final loss

    return loss

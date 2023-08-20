import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.Module):
    def __init__(self, batch=True, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        super(Loss, self).__init__()
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda
        self.batch = batch
        self.bce_loss = nn.BCELoss()

    def build_loss(self, mode='Focal_Dice'):
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == 'con_ce':
            return self.ConLoss
        elif mode == 'BCE_Dice':
            return self.BCE_Dice_loss
        elif mode == 'Focal_Dice':
            return self.Focal_Dice_loss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight,  # ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()
        loss = criterion(logit, target.long())
        if self.batch_average:
            loss /= n

        return loss

    def ConLoss(self, logit, target):
        loss = nn.BCEWithLogitsLoss()(logit, target)
        return loss

    def FocalLoss(self, logits, label, gamma=2, alpha=0.25):
        probs = torch.sigmoid(logits)
        coeff = torch.abs(label - probs).pow(gamma).neg()
        log_probs = torch.where(logits >= 0, F.softplus(logits, -1, 50), logits - F.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0, -logits + F.softplus(logits, -1, 50), -F.softplus(logits, 1, 50))
        loss = label * alpha * log_probs + (1. - label) * (1. - alpha) * log_1_probs
        loss = loss * coeff
        loss = loss.mean()

        return loss

    def soft_dice_coeff(self, y_pred, y_true):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_pred, y_true):
        loss = 1 - self.soft_dice_coeff(y_pred, y_true)
        return loss

    def BCE_Dice_loss(self, y_pred, y_true):
        a = self.bce_loss(y_pred, y_true)
        b = self.soft_dice_loss(y_pred, y_true)
        return a + b

    def Focal_Dice_loss(self, y_pred, y_true):
        a = self.FocalLoss(y_pred, y_true)
        b = self.soft_dice_loss(y_pred, y_true)
        return a + b


if __name__ == "__main__":
    loss = Loss(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())




